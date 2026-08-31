// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "vmm_slab_provider.h"

#include <stdio.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/buffer.h"

typedef struct hrx_vmm_slab_t hrx_vmm_slab_t;

typedef struct hrx_vmm_slab_provider_t {
  // Base provider header for vtable dispatch and reference counting.
  iree_hal_slab_provider_t base;

  // Allocator retained for virtual and physical memory operations.
  iree_hal_allocator_t* allocator;

  // Allocator used for provider and slab-state bookkeeping.
  iree_allocator_t host_allocator;

  // Parameters used to allocate physical backing for every slab.
  iree_hal_buffer_params_t buffer_params;

  // Granularity required for virtual reservations and physical allocations.
  iree_device_size_t page_size;

  // Cumulative number of slabs acquired through the VMM allocator interface.
  iree_atomic_int64_t total_acquired;

  // Cumulative number of slabs released through the VMM allocator interface.
  iree_atomic_int64_t total_released;

  // Slabs whose native teardown failed and must retain their live handles.
  // Protected by |mutex| and retried by provider trimming and destruction.
  hrx_vmm_slab_t* failed_release_head;

  // Serializes access to |failed_release_head|.
  iree_slim_mutex_t mutex;
} hrx_vmm_slab_provider_t;

struct hrx_vmm_slab_t {
  // Virtual address reservation retained until the slab is released.
  iree_hal_buffer_t* virtual_buffer;

  // Physical allocation mapped through |virtual_buffer|.
  iree_hal_physical_memory_t* physical_memory;

  // Mapped byte length, rounded up to the VMM page granularity.
  iree_device_size_t allocation_size;

  // True after the physical allocation has been mapped into the reservation.
  bool is_mapped;

  // True after this slab has been returned to the owning HAL pool.
  bool is_published;

  // Next slab retained after a failed release attempt.
  hrx_vmm_slab_t* next_failed_release;
};

static const iree_hal_slab_provider_vtable_t hrx_vmm_slab_provider_vtable;

static iree_status_t hrx_vmm_slab_provider_retry_failed_releases(
    hrx_vmm_slab_provider_t* provider);

static hrx_vmm_slab_provider_t* hrx_vmm_slab_provider_cast(
    iree_hal_slab_provider_t* base_provider) {
  return (hrx_vmm_slab_provider_t*)base_provider;
}

static const hrx_vmm_slab_provider_t* hrx_vmm_slab_provider_const_cast(
    const iree_hal_slab_provider_t* base_provider) {
  return (const hrx_vmm_slab_provider_t*)base_provider;
}

static void hrx_vmm_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  hrx_vmm_slab_provider_t* provider = hrx_vmm_slab_provider_cast(base_provider);
  iree_status_t status = hrx_vmm_slab_provider_retry_failed_releases(provider);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }
  iree_slim_mutex_lock(&provider->mutex);
  const bool has_failed_releases = provider->failed_release_head != NULL;
  iree_slim_mutex_unlock(&provider->mutex);
  if (has_failed_releases) {
    fprintf(stderr,
            "VMM slab provider destroyed with live resources retained after "
            "native teardown failure\n");
    return;
  }
  iree_slim_mutex_deinitialize(&provider->mutex);
  iree_hal_allocator_release(provider->allocator);
  iree_allocator_free(provider->host_allocator, provider);
}

static iree_status_t hrx_vmm_slab_provider_release_slab_state(
    hrx_vmm_slab_provider_t* provider, hrx_vmm_slab_t* slab) {
  if (!slab) return iree_ok_status();

  iree_status_t status = iree_ok_status();
  if (slab->is_mapped) {
    iree_status_t unmap_status = iree_hal_allocator_virtual_memory_unmap(
        provider->allocator, slab->virtual_buffer, /*virtual_offset=*/0,
        slab->allocation_size);
    if (iree_status_is_ok(unmap_status)) {
      slab->is_mapped = false;
    } else {
      status = iree_status_join(status, unmap_status);
    }
  }

  // Mapped resources must stay alive when unmapping fails. Once unmapped, the
  // physical allocation and virtual reservation have independent lifetimes.
  if (!slab->is_mapped && slab->physical_memory) {
    iree_status_t free_status = iree_hal_allocator_physical_memory_free(
        provider->allocator, slab->physical_memory);
    if (iree_status_is_ok(free_status)) {
      slab->physical_memory = NULL;
    } else {
      status = iree_status_join(status, free_status);
    }
  }
  if (!slab->is_mapped && slab->virtual_buffer) {
    iree_status_t release_status = iree_hal_allocator_virtual_memory_release(
        provider->allocator, slab->virtual_buffer);
    if (iree_status_is_ok(release_status)) {
      slab->virtual_buffer = NULL;
    } else {
      status = iree_status_join(status, release_status);
    }
  }
  if (!slab->is_mapped && !slab->physical_memory && !slab->virtual_buffer) {
    iree_allocator_free(provider->host_allocator, slab);
  }
  return status;
}

static void hrx_vmm_slab_provider_retain_failed_release(
    hrx_vmm_slab_provider_t* provider, hrx_vmm_slab_t* slab) {
  iree_slim_mutex_lock(&provider->mutex);
  slab->next_failed_release = provider->failed_release_head;
  provider->failed_release_head = slab;
  iree_slim_mutex_unlock(&provider->mutex);
}

static iree_status_t hrx_vmm_slab_provider_retry_failed_releases(
    hrx_vmm_slab_provider_t* provider) {
  iree_slim_mutex_lock(&provider->mutex);
  hrx_vmm_slab_t* pending = provider->failed_release_head;
  provider->failed_release_head = NULL;
  iree_slim_mutex_unlock(&provider->mutex);

  iree_status_t status = iree_ok_status();
  while (pending) {
    hrx_vmm_slab_t* slab = pending;
    pending = slab->next_failed_release;
    slab->next_failed_release = NULL;
    const bool is_published = slab->is_published;
    iree_status_t release_status =
        hrx_vmm_slab_provider_release_slab_state(provider, slab);
    if (iree_status_is_ok(release_status)) {
      if (is_published) {
        iree_atomic_fetch_add(&provider->total_released, 1,
                              iree_memory_order_relaxed);
      }
    } else {
      hrx_vmm_slab_provider_retain_failed_release(provider, slab);
      status = iree_status_join(status, release_status);
    }
  }
  return status;
}

static iree_status_t hrx_vmm_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab) {
  IREE_ASSERT_ARGUMENT(out_slab);
  *out_slab = (iree_hal_slab_t){0};
  if (IREE_UNLIKELY(min_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VMM slab allocations must be non-empty");
  }

  hrx_vmm_slab_provider_t* provider = hrx_vmm_slab_provider_cast(base_provider);
  iree_device_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_align(
          min_length, provider->page_size, &allocation_size))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VMM slab allocation size overflows page alignment");
  }

  hrx_vmm_slab_t* slab = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(provider->host_allocator,
                                             sizeof(*slab), (void**)&slab));
  *slab = (hrx_vmm_slab_t){
      .allocation_size = allocation_size,
  };

  iree_status_t status = iree_hal_allocator_virtual_memory_reserve(
      provider->allocator, provider->buffer_params.queue_affinity,
      allocation_size, &slab->virtual_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_physical_memory_allocate(
        provider->allocator, provider->buffer_params, allocation_size,
        provider->host_allocator, &slab->physical_memory);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_virtual_memory_map(
        provider->allocator, slab->virtual_buffer, /*virtual_offset=*/0,
        slab->physical_memory, /*physical_offset=*/0, allocation_size);
    slab->is_mapped = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_virtual_memory_protect(
        provider->allocator, slab->virtual_buffer, /*virtual_offset=*/0,
        allocation_size, provider->buffer_params.queue_affinity,
        IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
        IREE_HAL_MEMORY_PROTECTION_READ_WRITE);
  }

  iree_hal_external_buffer_t external_buffer;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_export_buffer(
        provider->allocator, slab->virtual_buffer,
        IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
        IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer);
  }
  if (iree_status_is_ok(status) &&
      !external_buffer.handle.device_allocation.ptr) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "VMM allocator returned a reservation without a device address");
  }

  if (!iree_status_is_ok(status)) {
    iree_status_t release_status =
        hrx_vmm_slab_provider_release_slab_state(provider, slab);
    if (!iree_status_is_ok(release_status)) {
      hrx_vmm_slab_provider_retain_failed_release(provider, slab);
      status = iree_status_join(status, release_status);
    }
    return status;
  }

  slab->is_published = true;
  out_slab->base_ptr =
      (uint8_t*)(uintptr_t)external_buffer.handle.device_allocation.ptr;
  out_slab->length = allocation_size;
  out_slab->provider_handle = (uint64_t)(uintptr_t)slab;
  iree_atomic_fetch_add(&provider->total_acquired, 1,
                        iree_memory_order_relaxed);
  return iree_ok_status();
}

static void hrx_vmm_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  if (!slab || !slab->provider_handle) return;
  hrx_vmm_slab_provider_t* provider = hrx_vmm_slab_provider_cast(base_provider);
  hrx_vmm_slab_t* virtual_slab =
      (hrx_vmm_slab_t*)(uintptr_t)slab->provider_handle;
  iree_status_t status =
      hrx_vmm_slab_provider_release_slab_state(provider, virtual_slab);
  if (iree_status_is_ok(status)) {
    iree_atomic_fetch_add(&provider->total_released, 1,
                          iree_memory_order_relaxed);
  } else {
    hrx_vmm_slab_provider_retain_failed_release(provider, virtual_slab);
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }
}

static iree_status_t hrx_vmm_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(slab);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  hrx_vmm_slab_provider_t* provider = hrx_vmm_slab_provider_cast(base_provider);
  hrx_vmm_slab_t* virtual_slab =
      (hrx_vmm_slab_t*)(uintptr_t)slab->provider_handle;
  if (!virtual_slab || !virtual_slab->virtual_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VMM slab has no virtual reservation");
  }

  const iree_hal_memory_type_t required_type =
      params.type & ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(virtual_slab->virtual_buffer),
      required_type));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(virtual_slab->virtual_buffer),
      params.access));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(virtual_slab->virtual_buffer),
      params.usage));

  // The callback returns this range to the pool. Tie it to the final subspan
  // reference so TLSF cannot reuse the range while a caller still owns it.
  return iree_hal_subspan_buffer_create_with_callback(
      virtual_slab->virtual_buffer, slab_offset, allocation_size,
      release_callback, provider->host_allocator, out_buffer);
}

static iree_status_t hrx_vmm_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* base_provider,
    const iree_hal_asan_pool_options_t* options) {
  (void)base_provider;
  (void)options;
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "VMM slab provider does not support HAL ASAN range advice");
}

static void hrx_vmm_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  (void)base_provider;
  (void)slab;
  (void)backing_offset;
  (void)advice_flags;
  (void)layout;
  IREE_ASSERT(false, "VMM slab provider cannot advise ASAN ranges");
}

static void hrx_vmm_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, iree_hal_slab_t* slab) {
  (void)base_provider;
  (void)slab;
}

static void hrx_vmm_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_trim_flags_t flags) {
  (void)flags;
  hrx_vmm_slab_provider_t* provider = hrx_vmm_slab_provider_cast(base_provider);
  iree_status_t status = hrx_vmm_slab_provider_retry_failed_releases(provider);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }
}

static void hrx_vmm_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  const hrx_vmm_slab_provider_t* provider =
      hrx_vmm_slab_provider_const_cast(base_provider);
  if (iree_hal_slab_provider_visited(visited, base_provider)) return;
  out_stats->total_acquired += (uint64_t)iree_atomic_load(
      (iree_atomic_int64_t*)&provider->total_acquired,
      iree_memory_order_relaxed);
  out_stats->total_released += (uint64_t)iree_atomic_load(
      (iree_atomic_int64_t*)&provider->total_released,
      iree_memory_order_relaxed);
}

static void hrx_vmm_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  const hrx_vmm_slab_provider_t* provider =
      hrx_vmm_slab_provider_const_cast(base_provider);
  out_properties->memory_type = provider->buffer_params.type;
  out_properties->supported_usage = provider->buffer_params.usage;
}

iree_status_t hrx_vmm_slab_provider_create(
    iree_hal_allocator_t* allocator,
    iree_hal_buffer_params_t physical_buffer_params,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  if (!iree_hal_allocator_supports_virtual_memory(allocator)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "allocator does not support virtual memory");
  }

  iree_hal_buffer_params_canonicalize(&physical_buffer_params);
  const iree_hal_memory_type_t required_type =
      physical_buffer_params.type & ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  if (!iree_all_bits_set(required_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) ||
      iree_any_bit_set(required_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                          IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                                          IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "VMM slab provider requires device-local memory without host access");
  }

  iree_device_size_t minimum_page_size = 0;
  iree_device_size_t recommended_page_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_virtual_memory_query_granularity(
      allocator, physical_buffer_params, &minimum_page_size,
      &recommended_page_size));
  const iree_device_size_t page_size =
      recommended_page_size ? recommended_page_size : minimum_page_size;
  if (IREE_UNLIKELY(!iree_device_size_is_valid_alignment(page_size))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocator returned an invalid VMM page size");
  }

  hrx_vmm_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  *provider = (hrx_vmm_slab_provider_t){
      .allocator = allocator,
      .host_allocator = host_allocator,
      .buffer_params = physical_buffer_params,
      .page_size = page_size,
      .total_acquired = IREE_ATOMIC_VAR_INIT(0),
      .total_released = IREE_ATOMIC_VAR_INIT(0),
      .failed_release_head = NULL,
  };
  iree_slim_mutex_initialize(&provider->mutex);
  iree_hal_slab_provider_initialize(&hrx_vmm_slab_provider_vtable,
                                    &provider->base);
  iree_hal_allocator_retain(provider->allocator);
  *out_provider = &provider->base;
  return iree_ok_status();
}

static const iree_hal_slab_provider_vtable_t hrx_vmm_slab_provider_vtable = {
    .destroy = hrx_vmm_slab_provider_destroy,
    .acquire_slab = hrx_vmm_slab_provider_acquire_slab,
    .release_slab = hrx_vmm_slab_provider_release_slab,
    .wrap_buffer = hrx_vmm_slab_provider_wrap_buffer,
    .validate_asan_options = hrx_vmm_slab_provider_validate_asan_options,
    .advise_asan_range = hrx_vmm_slab_provider_advise_asan_range,
    .prefault = hrx_vmm_slab_provider_prefault,
    .trim = hrx_vmm_slab_provider_trim,
    .query_stats = hrx_vmm_slab_provider_query_stats,
    .query_properties = hrx_vmm_slab_provider_query_properties,
};
