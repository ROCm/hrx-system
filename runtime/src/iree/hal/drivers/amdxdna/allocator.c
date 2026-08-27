// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/allocator.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdxdna/buffer.h"
#include "iree/hal/drivers/amdxdna/util.h"

static const iree_hal_allocator_vtable_t iree_hal_amdxdna_allocator_vtable;

enum { kAmdxdnaAllocatorCacheCapacity = 64 };

typedef struct iree_hal_amdxdna_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_native_device_t* native_device;
  iree_slim_mutex_t cache_mutex;
  // Free-list of native buffers for reuse. NOTE: the reuse key is size-only and
  // every generic allocation here is host_only; this is safe today because the
  // generic allocator path allocates a single buffer type. If instruction /
  // cacheable / device-visible buffers start flowing through this path, the
  // cache key must include the native buffer type/visibility before reuse.
  iree_hal_amdxdna_native_buffer_t*
      cached_buffers[kAmdxdnaAllocatorCacheCapacity];
  iree_host_size_t cached_buffer_count;
  iree_host_size_t cached_buffer_capacity;
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_amdxdna_allocator_t;

static iree_status_t iree_hal_amdxdna_allocator_unimplemented(
    const char* operation) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "%s is not implemented",
                          operation);
}

static void iree_hal_amdxdna_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
#if IREE_STATISTICS_ENABLE
  iree_hal_amdxdna_allocator_t* allocator =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_allocator,
                                           iree_hal_amdxdna_allocator_vtable,
                                           iree_hal_amdxdna_allocator_t);
  *out_statistics = allocator->statistics;
#else
  (void)base_allocator;
  memset(out_statistics, 0, sizeof(*out_statistics));
#endif  // IREE_STATISTICS_ENABLE
}

static void iree_hal_amdxdna_allocator_release_cached_buffer(
    void* user_data, iree_hal_buffer_t* base_buffer) {
  iree_hal_amdxdna_allocator_t* allocator =
      (iree_hal_amdxdna_allocator_t*)user_data;
  if (!allocator) return;

  iree_hal_amdxdna_native_buffer_t* native_buffer =
      iree_hal_amdxdna_buffer_steal_native_buffer(base_buffer);
  if (native_buffer) {
    iree_slim_mutex_lock(&allocator->cache_mutex);
    if (allocator->cached_buffer_count < allocator->cached_buffer_capacity) {
      allocator->cached_buffers[allocator->cached_buffer_count++] =
          native_buffer;
      native_buffer = NULL;
    }
    iree_slim_mutex_unlock(&allocator->cache_mutex);
  }
  if (native_buffer) {
    iree_hal_amdxdna_native_buffer_c_destroy(native_buffer);
  }
  iree_hal_allocator_release((iree_hal_allocator_t*)allocator);
}

static iree_hal_buffer_release_callback_t
iree_hal_amdxdna_allocator_make_release_callback(
    iree_hal_amdxdna_allocator_t* allocator) {
  iree_hal_allocator_retain((iree_hal_allocator_t*)allocator);
  return (iree_hal_buffer_release_callback_t){
      iree_hal_amdxdna_allocator_release_cached_buffer, allocator};
}

static void iree_hal_amdxdna_allocator_drop_release_callback(
    iree_hal_buffer_release_callback_t release_callback) {
  if (release_callback.fn) {
    iree_hal_allocator_release(
        (iree_hal_allocator_t*)release_callback.user_data);
  }
}

static void iree_hal_amdxdna_allocator_trim_cache(
    iree_hal_amdxdna_allocator_t* allocator) {
  iree_hal_amdxdna_native_buffer_t*
      cached_buffers[kAmdxdnaAllocatorCacheCapacity] = {0};
  iree_host_size_t cached_buffer_count = 0;
  iree_slim_mutex_lock(&allocator->cache_mutex);
  cached_buffer_count = allocator->cached_buffer_count;
  for (iree_host_size_t i = 0; i < cached_buffer_count; ++i) {
    cached_buffers[i] = allocator->cached_buffers[i];
    allocator->cached_buffers[i] = NULL;
  }
  allocator->cached_buffer_count = 0;
  iree_slim_mutex_unlock(&allocator->cache_mutex);

  for (iree_host_size_t i = 0; i < cached_buffer_count; ++i) {
    iree_hal_amdxdna_native_buffer_c_destroy(cached_buffers[i]);
  }
}

static iree_status_t iree_hal_amdxdna_allocator_take_cached_buffer(
    iree_hal_amdxdna_allocator_t* allocator, iree_device_size_t allocation_size,
    iree_hal_amdxdna_native_buffer_t** out_buffer) {
  *out_buffer = NULL;
  iree_slim_mutex_lock(&allocator->cache_mutex);
  for (iree_host_size_t i = 0; i < allocator->cached_buffer_count; ++i) {
    iree_hal_amdxdna_native_buffer_t* candidate = allocator->cached_buffers[i];
    if (iree_hal_amdxdna_native_buffer_c_size(candidate) != allocation_size) {
      continue;
    }
    *out_buffer = candidate;
    const iree_host_size_t tail_count = allocator->cached_buffer_count - i - 1;
    if (tail_count > 0) {
      memmove(&allocator->cached_buffers[i], &allocator->cached_buffers[i + 1],
              tail_count * sizeof(allocator->cached_buffers[0]));
    }
    --allocator->cached_buffer_count;
    allocator->cached_buffers[allocator->cached_buffer_count] = NULL;
    break;
  }
  iree_slim_mutex_unlock(&allocator->cache_mutex);
  return iree_ok_status();
}

static const iree_hal_memory_type_t iree_hal_amdxdna_host_only_memory_type =
    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_CACHED |
    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;

static const iree_hal_memory_type_t iree_hal_amdxdna_host_local_bit =
    IREE_HAL_MEMORY_TYPE_HOST_LOCAL & ~IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
static const iree_hal_memory_type_t iree_hal_amdxdna_device_local_bit =
    IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL & ~IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;

static iree_hal_buffer_compatibility_t
iree_hal_amdxdna_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)base_allocator;

  const bool optimal =
      iree_any_bit_set(params->type, IREE_HAL_MEMORY_TYPE_OPTIMAL);
  iree_hal_memory_type_t required_type =
      params->type & ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  if (optimal) {
    // HOST_LOCAL / DEVICE_LOCAL extra bits are placement preferences under
    // OPTIMAL. HOST_COHERENT is a coherence contract, not a placement hint:
    // HOST_ONLY BOs still fail it so callers flush/invalidate.
    required_type &=
        ~(iree_hal_amdxdna_host_local_bit | iree_hal_amdxdna_device_local_bit);
  }

  // HOST_ONLY allocations are cached host DRAM that the NPU can snoop. They
  // are host-visible and device-visible, but not host-coherent: callers must
  // flush/invalidate. They are also not HOST_LOCAL or DEVICE_LOCAL.
  const iree_hal_memory_type_t rejected_type =
      IREE_HAL_MEMORY_TYPE_HOST_COHERENT | iree_hal_amdxdna_host_local_bit |
      iree_hal_amdxdna_device_local_bit | IREE_HAL_MEMORY_TYPE_DEVICE_UNCACHED;
  if (iree_any_bit_set(required_type, rejected_type)) {
    IREE_TRACE_ZONE_END(z0);
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }

  params->type = iree_hal_amdxdna_host_only_memory_type;
  if (!iree_all_bits_set(params->type, required_type)) {
    IREE_TRACE_ZONE_END(z0);
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }

  params->usage |=
      IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_TRANSFER;

  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;
  // Transfer usage remains legal buffer metadata, but queue-transfer
  // compatibility is intentionally not advertised until amdxdna has a native
  // blit path.
  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
  }

  if (*allocation_size == 0) *allocation_size = 4;
  *allocation_size = iree_host_align(*allocation_size, 4);

  IREE_TRACE_ZONE_END(z0);
  return compatibility;
}

static iree_status_t iree_hal_amdxdna_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  (void)base_allocator;
  const iree_host_size_t count = 1;
  if (out_count) *out_count = count;
  if (capacity < count) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  heaps[0] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_CACHED |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                       IREE_HAL_BUFFER_USAGE_DISPATCH |
                       IREE_HAL_BUFFER_USAGE_MAPPING,
      .max_allocation_size = IREE_DEVICE_SIZE_MAX,
      .min_alignment = 4,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_allocator_t* allocator =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_allocator,
                                           iree_hal_amdxdna_allocator_vtable,
                                           iree_hal_amdxdna_allocator_t);
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_amdxdna_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
  }

  iree_hal_amdxdna_native_buffer_t* native_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_allocator_take_cached_buffer(
              allocator, allocation_size, &native_buffer));
  if (!native_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_device_c_alloc_buffer(
                allocator->native_device, allocation_size,
                IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY, &native_buffer));
  }

  iree_hal_buffer_t* buffer = NULL;
  const iree_hal_buffer_placement_t placement = {
      .device = NULL,
      .queue_affinity = params->queue_affinity ? params->queue_affinity
                                               : IREE_HAL_QUEUE_AFFINITY_ANY,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  iree_hal_buffer_release_callback_t release_callback =
      iree_hal_amdxdna_allocator_make_release_callback(allocator);
  iree_status_t status = iree_hal_amdxdna_buffer_wrap(
      native_buffer, placement, compat_params.type, compat_params.access,
      compat_params.usage, allocation_size, /*byte_offset=*/0,
      /*byte_length=*/allocation_size, release_callback,
      allocator->host_allocator, &buffer);

  if (iree_status_is_ok(status)) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    native_buffer = NULL;
    *out_buffer = buffer;
  } else {
    if (buffer) {
      native_buffer = NULL;
      iree_hal_buffer_release(buffer);
    } else {
      iree_hal_amdxdna_allocator_drop_release_callback(release_callback);
    }
  }
  if (native_buffer) {
    iree_hal_amdxdna_native_buffer_c_destroy(native_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdxdna_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_allocator_t* allocator =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_allocator,
                                           iree_hal_amdxdna_allocator_vtable,
                                           iree_hal_amdxdna_allocator_t);
  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_allocation_size(base_buffer)));
  iree_hal_buffer_destroy(base_buffer);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdxdna_allocator_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t* native_device,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  memset(allocator, 0, sizeof(*allocator));
  iree_hal_resource_initialize(&iree_hal_amdxdna_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->native_device = native_device;
  iree_slim_mutex_initialize(&allocator->cache_mutex);
  // Fail closed: retaining native allocations is permitted only when the
  // backend explicitly guarantees that reuse is safe for its object lifecycle.
  allocator->cached_buffer_capacity = 0;
  iree_hal_amdxdna_native_c_device_caps_t native_caps = {0};
  if (native_device &&
      iree_status_is_ok(iree_hal_amdxdna_native_device_c_query_caps(
          native_device, &native_caps))) {
    allocator->cached_buffer_capacity = native_caps.supports_host_buffer_reuse
                                            ? kAmdxdnaAllocatorCacheCapacity
                                            : 0;
  }

  *out_allocator = (iree_hal_allocator_t*)allocator;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdxdna_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_allocator_t* allocator =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_allocator,
                                           iree_hal_amdxdna_allocator_vtable,
                                           iree_hal_amdxdna_allocator_t);
  iree_allocator_t host_allocator = allocator->host_allocator;
  iree_hal_amdxdna_allocator_trim_cache(allocator);
  iree_slim_mutex_deinitialize(&allocator->cache_mutex);
  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_amdxdna_allocator_supports_virtual_memory(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  // The native backend allocates whole BOs directly, not via a
  // virtual-address-reservation API. A real implementation would need native
  // support for partial-population / page-level mapping that the amdxdna
  // backends do not provide today.
  return false;
}

static iree_allocator_t iree_hal_amdxdna_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_amdxdna_allocator_t* allocator =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_allocator,
                                           iree_hal_amdxdna_allocator_vtable,
                                           const iree_hal_amdxdna_allocator_t);

  IREE_TRACE_ZONE_END(z0);
  return allocator->host_allocator;
}

static iree_status_t iree_hal_amdxdna_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_amdxdna_allocator_t* allocator =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_allocator,
                                           iree_hal_amdxdna_allocator_vtable,
                                           iree_hal_amdxdna_allocator_t);
  iree_hal_amdxdna_allocator_trim_cache(allocator);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  (void)base_allocator;
  (void)params;
  (void)external_buffer;
  (void)release_callback;
  (void)out_buffer;
  return iree_hal_amdxdna_allocator_unimplemented("buffer import");
}

static iree_status_t iree_hal_amdxdna_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  (void)base_allocator;
  (void)buffer;
  (void)requested_type;
  (void)requested_flags;
  (void)out_external_buffer;
  return iree_hal_amdxdna_allocator_unimplemented("buffer export");
}

static iree_status_t
iree_hal_amdxdna_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params,
    iree_device_size_t* IREE_RESTRICT out_minimum_page_size,
    iree_device_size_t* IREE_RESTRICT out_recommended_page_size) {
  (void)base_allocator;
  (void)params;
  (void)out_minimum_page_size;
  (void)out_recommended_page_size;
  return iree_hal_amdxdna_allocator_unimplemented(
      "virtual memory granularity query");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
    iree_hal_buffer_t** IREE_RESTRICT out_virtual_buffer) {
  (void)base_allocator;
  (void)queue_affinity;
  (void)size;
  (void)out_virtual_buffer;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory reserve");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_release(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer) {
  (void)base_allocator;
  (void)virtual_buffer;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory release");
}

static iree_status_t iree_hal_amdxdna_allocator_physical_memory_allocate(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params, iree_device_size_t size,
    iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** IREE_RESTRICT out_physical_memory) {
  (void)base_allocator;
  (void)params;
  (void)size;
  (void)host_allocator;
  (void)out_physical_memory;
  return iree_hal_amdxdna_allocator_unimplemented("physical memory allocate");
}

static iree_status_t iree_hal_amdxdna_allocator_physical_memory_free(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory) {
  (void)base_allocator;
  (void)physical_memory;
  return iree_hal_amdxdna_allocator_unimplemented("physical memory free");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_map(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)physical_memory;
  (void)physical_offset;
  (void)size;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory map");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory unmap");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_protect(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_protection_t protection) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  (void)queue_affinity;
  (void)protection;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory protect");
}

static iree_status_t iree_hal_amdxdna_allocator_virtual_memory_advise(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_advice_t advice) {
  (void)base_allocator;
  (void)virtual_buffer;
  (void)virtual_offset;
  (void)size;
  (void)queue_affinity;
  (void)advice;
  return iree_hal_amdxdna_allocator_unimplemented("virtual memory advise");
}

static const iree_hal_allocator_vtable_t iree_hal_amdxdna_allocator_vtable = {
    .destroy = iree_hal_amdxdna_allocator_destroy,
    .host_allocator = iree_hal_amdxdna_allocator_host_allocator,
    .trim = iree_hal_amdxdna_allocator_trim,
    .query_statistics = iree_hal_amdxdna_allocator_query_statistics,
    .query_memory_heaps = iree_hal_amdxdna_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_amdxdna_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_amdxdna_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_amdxdna_allocator_deallocate_buffer,
    // Stubs that return UNIMPLEMENTED. The XDNA kernel ABI has no
    // wrap-host-pointer / userptr ioctl (no equivalent of
    // hipHostRegister/cuMemHostRegister), so a real zero-copy import is not
    // possible without kernel changes. Today the AllocatorTest.Import*
    // CTS tests SKIP at compatibility-check time because
    // query_buffer_compatibility above never sets IMPORTABLE; callers that
    // do reach iree_hal_memory_file_wrap see UNIMPLEMENTED here and fall
    // back to a HOST_LOCAL | HOST_COHERENT heap buffer; fine for transfer
    // source/target use but not DEVICE_VISIBLE so it cannot be a dispatch
    // binding. A real implementation would have to allocate a SHMEM BO and
    // memcpy host data into it (gives a dispatch-capable buffer at the cost
    // of one copy), or add a kernel userptr path (zero-copy, kernel work).
    .import_buffer = iree_hal_amdxdna_allocator_import_buffer,
    .export_buffer = iree_hal_amdxdna_allocator_export_buffer,
    // Virtual memory is not supported on XDNA. supports_virtual_memory
    // returns false so callers short-circuit before reaching the rest of
    // the VM vtable; the remaining slots are filled with UNIMPLEMENTED
    // stubs to keep the vtable complete (NULL fn pointers would SEGV any
    // caller that bypasses the supports check).
    .supports_virtual_memory =
        iree_hal_amdxdna_allocator_supports_virtual_memory,
    .virtual_memory_query_granularity =
        iree_hal_amdxdna_allocator_virtual_memory_query_granularity,
    .virtual_memory_reserve = iree_hal_amdxdna_allocator_virtual_memory_reserve,
    .virtual_memory_release = iree_hal_amdxdna_allocator_virtual_memory_release,
    .physical_memory_allocate =
        iree_hal_amdxdna_allocator_physical_memory_allocate,
    .physical_memory_free = iree_hal_amdxdna_allocator_physical_memory_free,
    .virtual_memory_map = iree_hal_amdxdna_allocator_virtual_memory_map,
    .virtual_memory_unmap = iree_hal_amdxdna_allocator_virtual_memory_unmap,
    .virtual_memory_protect = iree_hal_amdxdna_allocator_virtual_memory_protect,
    .virtual_memory_advise = iree_hal_amdxdna_allocator_virtual_memory_advise,
};
