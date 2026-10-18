// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/slab_provider.h"

#include <stddef.h>

#include "iree/hal/drivers/amdgpu/allocation_queue.h"
#include "iree/hal/drivers/amdgpu/allocator.h"
#include "iree/hal/drivers/amdgpu/asan_state.h"
#include "iree/hal/drivers/amdgpu/atomic_memory.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/virtual_memory.h"
#include "iree/hal/memory/tracing.h"

typedef struct iree_hal_amdgpu_slab_provider_t {
  // Base slab-provider interface header.
  iree_hal_slab_provider_t base;

  // Retained native allocation owner; NULL only for an explicit inline source.
  iree_hal_amdgpu_allocation_queue_t* allocation_queue;

  // Borrowed HAL device used when wrapping slabs as AMDGPU buffers.
  iree_hal_device_t* device;

  // Borrowed native attachment service used only for external storage.
  iree_hal_allocator_t* import_allocator;

  // Borrowed HSA dispatch table used for memory-pool operations.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Native grants copied into this provider's metadata allocation.
  struct {
    // Number of unique agents qualified at construction.
    uint32_t count;
    // Inline agent handles used by ordinary HSA memory-pool allocation.
    hsa_agent_t* agents;
    // Inline RW descriptors for sanitizer and mapping-source VMM allocation.
    hsa_amd_memory_access_desc_t* vmem_descriptors;
  } access;

  // HSA pool this provider acquires slabs from.
  hsa_amd_memory_pool_t memory_pool;

  // Flags passed to every HSA memory-pool allocation.
  uint32_t allocation_flags;

  // Session-local physical device ordinal owning this provider.
  uint32_t physical_device_ordinal;

  // Borrowed wrapper pool used for materialized HAL buffer views.
  iree_hal_amdgpu_buffer_pool_t* buffer_pool;

  // Queue families that may access buffers materialized from this provider.
  iree_hal_queue_family_affinity_t supported_queue_family_affinity;

  // Source GPU masks for atomic operations supported by the HSA memory pool.
  iree_hal_amdgpu_atomic_memory_source_masks_t atomic_memory_source_masks;

  // Atomic memory cells supported across all queue families.
  iree_hal_amdgpu_atomic_memory_cell_flags_t atomic_memory_cells;

  // Stable named-memory stream for HSA backing allocations from this provider.
  iree_hal_memory_trace_t trace;

  // Flags selecting provider behavior.
  iree_hal_amdgpu_slab_provider_flags_t flags;

  // Minimum allocation-size granule for the active slab acquisition mode.
  iree_device_size_t allocation_granule;

  // Base-pointer alignment guaranteed by the active slab acquisition mode.
  iree_device_size_t allocation_alignment;

  // HSA VMM memory type used for sanitizer and mapping-source placement.
  iree_hal_amdgpu_vmem_memory_type_t vmem_memory_type;

  // Translated HSA VMM memory type for hsa_amd_vmem_handle_create.
  hsa_amd_memory_type_t hsa_vmem_memory_type;

  // Borrowed ASAN state used when ASAN shadow support is enabled.
  iree_hal_amdgpu_asan_state_t* asan_state;

  // Exact properties of slabs allocated from |memory_pool|.
  struct {
    // HAL memory type bits of provider-owned HSA allocations.
    iree_hal_memory_type_t memory_type;
    // HAL buffer usage bits supported by provider-owned HSA allocations.
    iree_hal_buffer_usage_t supported_usage;
  } owned;

  // Immutable HAL memory properties exposed by this provider.
  iree_hal_slab_provider_properties_t properties;

  // Cumulative slab acquisitions reported through query_stats().
  iree_atomic_int64_t total_acquired;

  // Cumulative slab releases reported through query_stats().
  iree_atomic_int64_t total_released;
} iree_hal_amdgpu_slab_provider_t;

typedef struct iree_hal_amdgpu_slab_handle_t {
  // Physical ledger retained through sanitizer quarantine and native release.
  iree_hal_memory_trace_t trace;

  // HSA allocation byte length used when releasing the slab.
  iree_device_size_t allocation_size;

  // Device-visible base pointer returned for the slab.
  IREE_AMDGPU_DEVICE_PTR void* base_ptr;

  // Physical VMM identity for sanitizer backing or explicit mapping sources.
  iree_hal_amdgpu_vmem_source_t vmem_source;

  // ASAN application range covering |base_ptr| for VMM-backed slabs.
  iree_hal_amdgpu_asan_application_range_t* application_range;

  // Intrusive ASAN quarantine node used for stale-pointer detection.
  iree_hal_amdgpu_asan_quarantine_entry_t quarantine_entry;

  // Root buffer owning a native external-storage attachment, or NULL for an
  // allocation acquired from the provider's HSA memory pool.
  iree_hal_buffer_t* imported_buffer;

  // Profiling session id owning |profile_allocation_id|.
  uint64_t profile_session_id;

  // Session-local allocation id for the slab acquire/release lifecycle.
  uint64_t profile_allocation_id;

  // True when |base_ptr| is an active HSA VMM mapping.
  uint32_t is_vmem_mapped : 1;

  // Set by independently advised subranges; read after all users retire.
  iree_atomic_int32_t has_asan_advice;

  // Borrowed HSA API table used to release slab backing allocations.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Borrowed ASAN state used to return |application_range| for VMM slabs.
  iree_hal_amdgpu_asan_state_t* asan_state;

  // Host allocator used to release this handle.
  iree_allocator_t host_allocator;
} iree_hal_amdgpu_slab_handle_t;

static const iree_hal_slab_provider_vtable_t
    iree_hal_amdgpu_slab_provider_vtable;

static const char* IREE_HAL_AMDGPU_SLAB_PROVIDER_TRACE_ID =
    "iree-hal-amdgpu-slab-provider";

static iree_hal_amdgpu_slab_provider_t* iree_hal_amdgpu_slab_provider_cast(
    iree_hal_slab_provider_t* base_provider) {
  return (iree_hal_amdgpu_slab_provider_t*)base_provider;
}

static const iree_hal_amdgpu_slab_provider_t*
iree_hal_amdgpu_slab_provider_const_cast(
    const iree_hal_slab_provider_t* base_provider) {
  return (const iree_hal_amdgpu_slab_provider_t*)base_provider;
}

static bool iree_hal_amdgpu_slab_provider_uses_asan_vmm(
    const iree_hal_amdgpu_slab_provider_t* provider) {
  return iree_any_bit_set(provider->flags,
                          IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_VMM);
}

static bool iree_hal_amdgpu_slab_provider_uses_mapping_source(
    const iree_hal_amdgpu_slab_provider_t* provider) {
  return iree_any_bit_set(provider->flags,
                          IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_MAPPING_SOURCE);
}

static bool iree_hal_amdgpu_slab_provider_uses_asan_shadow(
    const iree_hal_amdgpu_slab_provider_t* provider) {
  return iree_any_bit_set(provider->flags,
                          IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_SHADOW);
}

static bool iree_hal_amdgpu_slab_provider_record_memory_event(
    iree_hal_amdgpu_slab_provider_t* provider,
    iree_hal_profile_memory_event_type_t type,
    iree_hal_amdgpu_slab_handle_t* slab_handle, const void* backing_ptr) {
  uint64_t session_id = slab_handle->profile_session_id;
  uint64_t allocation_id = slab_handle->profile_allocation_id;
  if (type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE) {
    allocation_id =
        iree_hal_amdgpu_logical_device_allocate_profile_memory_allocation_id(
            provider->device, &session_id);
    if (allocation_id == 0) {
      return false;
    }
  } else if (allocation_id == 0) {
    return false;
  }

  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = type;
  event.allocation_id = allocation_id;
  event.pool_id = (uint64_t)(uintptr_t)provider;
  event.backing_id = (uint64_t)(uintptr_t)backing_ptr;
  event.physical_device_ordinal = provider->physical_device_ordinal;
  event.memory_type = provider->owned.memory_type;
  event.buffer_usage = provider->owned.supported_usage;
  event.length = slab_handle->allocation_size;
  event.alignment = provider->allocation_alignment;
  const bool recorded =
      iree_hal_amdgpu_logical_device_record_profile_memory_event_for_session(
          provider->device, session_id, &event);
  if (recorded && type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE) {
    slab_handle->profile_session_id = session_id;
    slab_handle->profile_allocation_id = allocation_id;
  }
  return recorded;
}

static void iree_hal_amdgpu_slab_provider_finalize_vmem_handle(
    void* user_data) {
  iree_hal_amdgpu_slab_handle_t* slab_handle =
      (iree_hal_amdgpu_slab_handle_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (slab_handle->is_vmem_mapped) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_vmem_unmap_raw(slab_handle->libhsa, slab_handle->base_ptr,
                                    slab_handle->allocation_size));
  }
  if (slab_handle->vmem_source.handle.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_vmem_handle_release_raw(slab_handle->libhsa,
                                             slab_handle->vmem_source.handle));
  }
  if (slab_handle->application_range) {
    iree_hal_amdgpu_asan_state_release_application_range(
        slab_handle->asan_state, slab_handle->application_range);
  } else if (slab_handle->base_ptr) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_vmem_address_free_raw(slab_handle->libhsa,
                                           slab_handle->base_ptr,
                                           slab_handle->allocation_size));
  }
  iree_hal_memory_trace_free(&slab_handle->trace, slab_handle);
  iree_allocator_free(slab_handle->host_allocator, slab_handle);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_slab_provider_finalize_hsa_allocation(
    void* user_data) {
  iree_hal_amdgpu_slab_handle_t* slab_handle =
      (iree_hal_amdgpu_slab_handle_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (slab_handle->base_ptr) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_memory_pool_free_raw(slab_handle->libhsa,
                                          slab_handle->base_ptr));
  }
  iree_hal_memory_trace_free(&slab_handle->trace, slab_handle);
  iree_allocator_free(slab_handle->host_allocator, slab_handle);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_slab_provider_acquire_vmem_slab(
    iree_hal_amdgpu_slab_provider_t* provider,
    iree_device_size_t allocation_size,
    iree_hal_amdgpu_slab_handle_t* slab_handle, void** out_base_ptr) {
  IREE_ASSERT_ARGUMENT(out_base_ptr);
  *out_base_ptr = NULL;
  if (IREE_UNLIKELY(
          !iree_hal_amdgpu_asan_state_is_enabled(provider->asan_state))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU ASAN VMM slab acquisition requires enabled ASAN state");
  }

  uint64_t application_address = 0;
  iree_status_t status = iree_hal_amdgpu_asan_state_reserve_application_range(
      provider->asan_state, allocation_size, provider->allocation_granule,
      &application_address, &slab_handle->application_range);
  if (iree_status_is_ok(status)) {
    slab_handle->base_ptr = (void*)(uintptr_t)application_address;
  }

  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_vmem_handle_create(
        IREE_LIBHSA(provider->libhsa), provider->memory_pool, allocation_size,
        provider->hsa_vmem_memory_type, /*flags=*/0,
        &slab_handle->vmem_source.handle);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status,
          "creating AMDGPU ASAN VMM slab: physical_device_ordinal=%" PRIu32
          ", memory_pool=0x%016" PRIx64 ", allocation_size=%" PRIu64
          ", allocation_granule=%" PRIu64 ", vmem_memory_type=%d",
          provider->physical_device_ordinal, provider->memory_pool.handle,
          (uint64_t)allocation_size, (uint64_t)provider->allocation_granule,
          (int)provider->vmem_memory_type);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_vmem_map(
        IREE_LIBHSA(provider->libhsa), slab_handle->base_ptr, allocation_size,
        /*offset=*/0, slab_handle->vmem_source.handle,
        /*flags=*/0);
    slab_handle->is_vmem_mapped = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_vmem_set_access(
        IREE_LIBHSA(provider->libhsa), slab_handle->base_ptr, allocation_size,
        provider->access.vmem_descriptors, provider->access.count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_asan_state_map_range(
        provider->asan_state, application_address, allocation_size);
  }

  if (iree_status_is_ok(status)) {
    *out_base_ptr = slab_handle->base_ptr;
  } else {
    iree_hal_amdgpu_slab_provider_finalize_vmem_handle(slab_handle);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_slab_provider_acquire_mapping_slab(
    iree_hal_amdgpu_slab_provider_t* provider,
    iree_hal_amdgpu_slab_handle_t* slab_handle, void** out_base_ptr) {
  iree_status_t status = iree_hsa_amd_vmem_handle_create(
      IREE_LIBHSA(provider->libhsa), provider->memory_pool,
      slab_handle->allocation_size, provider->hsa_vmem_memory_type, 0,
      &slab_handle->vmem_source.handle);
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_vmem_address_reserve_align(
        IREE_LIBHSA(provider->libhsa), &slab_handle->base_ptr,
        slab_handle->allocation_size, 0, provider->allocation_alignment, 0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_vmem_map(
        IREE_LIBHSA(provider->libhsa), slab_handle->base_ptr,
        slab_handle->allocation_size, 0, slab_handle->vmem_source.handle, 0);
    slab_handle->is_vmem_mapped = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_vmem_set_access(
        IREE_LIBHSA(provider->libhsa), slab_handle->base_ptr,
        slab_handle->allocation_size, provider->access.vmem_descriptors,
        provider->access.count);
  }
  if (iree_status_is_ok(status)) {
    *out_base_ptr = slab_handle->base_ptr;
  } else {
    iree_hal_amdgpu_slab_provider_finalize_vmem_handle(slab_handle);
  }
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_amd_memory_pool_t memory_pool,
    iree_hal_amdgpu_slab_provider_memory_pool_properties_t* out_properties) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_properties);
  memset(out_properties, 0, sizeof(*out_properties));

  size_t allocation_granule = 0;
  IREE_RETURN_IF_ERROR(
      iree_hsa_amd_memory_pool_get_info(
          IREE_LIBHSA(libhsa), memory_pool,
          HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &allocation_granule),
      "querying HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE for an AMDGPU "
      "slab provider");
  if (allocation_granule == 0 ||
      !iree_device_size_is_power_of_two(allocation_granule)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "invalid HSA runtime allocation granule for an AMDGPU memory pool: "
        "%" PRIhsz,
        (iree_host_size_t)allocation_granule);
  }

  size_t allocation_alignment = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT, &allocation_alignment));
  if (allocation_alignment == 0 ||
      !iree_device_size_is_power_of_two(allocation_alignment)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "invalid HSA runtime allocation alignment for an AMDGPU memory pool: "
        "%" PRIhsz,
        (iree_host_size_t)allocation_alignment);
  }

  hsa_region_segment_t segment = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
      &segment));
  if (segment != HSA_REGION_SEGMENT_GLOBAL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU slab providers require a GLOBAL HSA pool");
  }

  bool alloc_allowed = false;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed));
  if (!alloc_allowed) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU slab provider memory pool does not support runtime "
        "allocations");
  }

  uint32_t global_flags = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
      &global_flags));

  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  // Sharing hints do not affect HSA pool selection. Export is omitted because
  // it requires dedicated platform export support.
  const iree_hal_buffer_usage_t sharing_usage =
      IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE |
      IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT |
      IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE;
  iree_hal_buffer_usage_t supported_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                                            IREE_HAL_BUFFER_USAGE_DISPATCH |
                                            sharing_usage;
  if (iree_any_bit_set(
          global_flags,
          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
              HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED)) {
    memory_type |=
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
    supported_usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                       IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
                       IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
                       IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
                       IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;
  }

  out_properties->allocation_granule = (iree_device_size_t)allocation_granule;
  out_properties->allocation_alignment =
      (iree_device_size_t)allocation_alignment;
  out_properties->memory_type = memory_type;
  out_properties->supported_usage = supported_usage;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_slab_provider_create(
    iree_hal_device_t* device, iree_hal_allocator_t* import_allocator,
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_slab_provider_options_t options,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_buffer_pool_t* buffer_pool, iree_string_view_t trace_name,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(buffer_pool);
  IREE_ASSERT_ARGUMENT(out_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_provider = NULL;

  if (IREE_UNLIKELY(!options.memory_pool.handle)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU slab provider requires an HSA memory pool");
  }
  if (IREE_UNLIKELY(options.owned.memory_type == IREE_HAL_MEMORY_TYPE_NONE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU slab provider requires an owned HAL memory type");
  }
  if (IREE_UNLIKELY(options.owned.supported_usage ==
                    IREE_HAL_BUFFER_USAGE_NONE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU slab provider requires owned HAL buffer usage bits");
  }
  if (IREE_UNLIKELY(options.exposed.memory_type == IREE_HAL_MEMORY_TYPE_NONE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU slab provider requires an exposed HAL memory type");
  }
  if (IREE_UNLIKELY(options.exposed.supported_usage ==
                    IREE_HAL_BUFFER_USAGE_NONE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU slab provider requires exposed HAL buffer usage bits");
  }
  if (IREE_UNLIKELY(options.exposed.allocation_alignment &&
                    !iree_device_size_is_valid_alignment(
                        options.exposed.allocation_alignment))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU slab provider exposed alignment %" PRIu64
                            " is not a power-of-two",
                            (uint64_t)options.exposed.allocation_alignment);
  }
  if (IREE_UNLIKELY(options.access.agent_count && !options.access.agents)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU slab provider access agent array is null");
  }
  const iree_hal_amdgpu_slab_provider_flags_t unsupported_flags =
      options.flags & ~(IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_SHADOW |
                        IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_VMM |
                        IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_MAPPING_SOURCE);
  if (IREE_UNLIKELY(unsupported_flags)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU slab provider flags 0x%08x contain "
                            "unsupported bits",
                            unsupported_flags);
  }
  if (iree_any_bit_set(options.flags,
                       IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_SHADOW) &&
      IREE_UNLIKELY(!options.asan_state)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ASAN slab provider requires ASAN state");
  }
  if (iree_any_bit_set(options.flags,
                       IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_VMM) &&
      IREE_UNLIKELY(!iree_all_bits_set(
          options.flags, IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_SHADOW))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN VMM slab provider also requires ASAN shadow support");
  }
  if (iree_any_bit_set(options.flags,
                       IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_VMM) &&
      IREE_UNLIKELY(options.owned.allocation_flags !=
                    HSA_AMD_MEMORY_POOL_STANDARD_FLAG)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN VMM slabs cannot represent HSA allocation flags");
  }

  if (iree_any_bit_set(options.flags,
                       IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_MAPPING_SOURCE) &&
      (iree_any_bit_set(options.flags,
                        IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_SHADOW) ||
       options.owned.allocation_flags != HSA_AMD_MEMORY_POOL_STANDARD_FLAG)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mapping sources require ordinary native VMM");
  }

  iree_hal_amdgpu_slab_provider_t* provider = NULL;
  const iree_host_size_t agents_size =
      options.access.agent_count * sizeof(hsa_agent_t);
  const iree_host_size_t descriptors_size =
      iree_any_bit_set(options.flags,
                       IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_VMM |
                           IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_MAPPING_SOURCE)
          ? options.access.agent_count * sizeof(hsa_amd_memory_access_desc_t)
          : 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*provider) + agents_size + descriptors_size,
                            (void**)&provider));
  memset(provider, 0, sizeof(*provider));
  iree_hal_slab_provider_initialize(&iree_hal_amdgpu_slab_provider_vtable,
                                    host_allocator, &provider->base);
  provider->allocation_queue = options.allocation_queue;
  iree_hal_amdgpu_allocation_queue_retain(provider->allocation_queue);
  provider->device = device;
  provider->import_allocator = import_allocator;
  provider->libhsa = libhsa;
  provider->access.count = options.access.agent_count;
  provider->access.agents = (hsa_agent_t*)(provider + 1);
  if (agents_size) {
    memcpy(provider->access.agents, options.access.agents, agents_size);
  }
  if (descriptors_size) {
    provider->access.vmem_descriptors =
        (hsa_amd_memory_access_desc_t*)(provider->access.agents +
                                        options.access.agent_count);
    for (uint32_t i = 0; i < options.access.agent_count; ++i) {
      provider->access.vmem_descriptors[i] = (hsa_amd_memory_access_desc_t){
          .agent_handle = options.access.agents[i],
          .permissions = HSA_ACCESS_PERMISSION_RW,
      };
    }
  }
  provider->memory_pool = options.memory_pool;
  provider->allocation_flags = options.owned.allocation_flags;
  provider->physical_device_ordinal = (uint32_t)physical_device_ordinal;
  provider->buffer_pool = buffer_pool;
  provider->supported_queue_family_affinity =
      options.access.queue_family_affinity;
  provider->atomic_memory_source_masks = options.access.atomic_source_masks;
  provider->atomic_memory_cells =
      iree_hal_amdgpu_atomic_memory_select_device_cells(
          &provider->atomic_memory_source_masks,
          provider->supported_queue_family_affinity);
  provider->flags = options.flags;
  provider->vmem_memory_type = options.vmem_memory_type;
  provider->asan_state = options.asan_state;
  provider->owned.memory_type = options.owned.memory_type;
  provider->owned.supported_usage = options.owned.supported_usage;
  provider->total_acquired = IREE_ATOMIC_VAR_INIT(0);
  provider->total_released = IREE_ATOMIC_VAR_INIT(0);

  iree_status_t status = iree_hal_memory_trace_initialize(
      trace_name, IREE_HAL_AMDGPU_SLAB_PROVIDER_TRACE_ID, &provider->trace);
  iree_hal_amdgpu_slab_provider_memory_pool_properties_t properties;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
        libhsa, options.memory_pool, &properties);
  }
  if (iree_status_is_ok(status) &&
      (iree_hal_amdgpu_slab_provider_uses_asan_vmm(provider) ||
       iree_hal_amdgpu_slab_provider_uses_mapping_source(provider))) {
    status = iree_hal_amdgpu_vmem_translate_memory_type(
        options.vmem_memory_type, &provider->hsa_vmem_memory_type);
  }
  if (iree_status_is_ok(status)) {
    if (iree_hal_amdgpu_slab_provider_uses_asan_vmm(provider) ||
        iree_hal_amdgpu_slab_provider_uses_mapping_source(provider)) {
      iree_hal_amdgpu_vmem_granularity_t granularity;
      status = iree_hal_amdgpu_vmem_query_alloc_granularity(
          libhsa, options.memory_pool, &granularity);
      if (iree_status_is_ok(status)) {
        provider->allocation_granule =
            iree_hal_amdgpu_slab_provider_uses_asan_vmm(provider)
                ? granularity.recommended
                : granularity.minimum;
        provider->allocation_alignment = provider->allocation_granule;
      }
    } else {
      provider->allocation_granule = properties.allocation_granule;
      provider->allocation_alignment = properties.allocation_alignment;
    }
  }
  if (iree_status_is_ok(status)) {
    provider->properties.memory_type = options.exposed.memory_type;
    provider->properties.supported_usage = options.exposed.supported_usage;
    provider->properties.queue_family_affinity =
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
    provider->properties.allocation_alignment =
        options.exposed.allocation_alignment
            ? options.exposed.allocation_alignment
            : provider->allocation_alignment;
    provider->properties.max_allocation_alignment =
        provider->allocation_alignment;
    provider->properties.maintenance_alignment = 1;
    provider->properties.atomic_operations = options.exposed.atomic_operations;
    *out_provider = &provider->base;
  } else {
    iree_hal_slab_provider_release(&provider->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base_provider);
  iree_allocator_t host_allocator = provider->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_memory_trace_deinitialize(&provider->trace);
  iree_hal_amdgpu_allocation_queue_release(provider->allocation_queue);
  iree_allocator_free(host_allocator, provider);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_device_size_t alignment, iree_hal_slab_t* out_slab) {
  (void)alignment;
  IREE_ASSERT_ARGUMENT(out_slab);
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_slab, 0, sizeof(*out_slab));

  if (IREE_UNLIKELY(min_length == 0)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "AMDGPU slab allocations must be non-empty"));
  }
  iree_device_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_align(
          min_length, provider->allocation_granule, &allocation_size))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "AMDGPU slab allocation size overflow aligning %" PRIu64
                " bytes to a %" PRIu64 "-byte HSA allocation granule",
                (uint64_t)min_length, (uint64_t)provider->allocation_granule));
  }

  iree_hal_amdgpu_slab_handle_t* slab_handle = NULL;
  iree_status_t status =
      iree_allocator_malloc(provider->base.host_allocator, sizeof(*slab_handle),
                            (void**)&slab_handle);
  if (iree_status_is_ok(status)) {
    memset(slab_handle, 0, sizeof(*slab_handle));
    slab_handle->allocation_size = allocation_size;
    slab_handle->vmem_source.byte_length = allocation_size;
    slab_handle->libhsa = provider->libhsa;
    slab_handle->asan_state = provider->asan_state;
    slab_handle->host_allocator = provider->base.host_allocator;
  }

  void* base_ptr = NULL;
  bool slab_handle_consumed = false;
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_slab_provider_uses_asan_vmm(provider)) {
    status = iree_hal_amdgpu_slab_provider_acquire_vmem_slab(
        provider, allocation_size, slab_handle, &base_ptr);
    slab_handle_consumed = !iree_status_is_ok(status);
  } else if (iree_status_is_ok(status) &&
             iree_hal_amdgpu_slab_provider_uses_mapping_source(provider)) {
    status = iree_hal_amdgpu_slab_provider_acquire_mapping_slab(
        provider, slab_handle, &base_ptr);
    slab_handle_consumed = !iree_status_is_ok(status);
  } else if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(provider->libhsa), provider->memory_pool,
        (size_t)allocation_size, provider->allocation_flags, &base_ptr);
    if (iree_status_is_ok(status)) {
      slab_handle->base_ptr = base_ptr;
    }
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_amdgpu_slab_provider_uses_asan_vmm(provider) &&
      !iree_hal_amdgpu_slab_provider_uses_mapping_source(provider)) {
    status = iree_hsa_amd_agents_allow_access(
        IREE_LIBHSA(provider->libhsa), provider->access.count,
        provider->access.agents, /*flags=*/NULL, base_ptr);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_slab_provider_uses_asan_shadow(provider) &&
      !iree_hal_amdgpu_slab_provider_uses_asan_vmm(provider)) {
    status = iree_hal_amdgpu_asan_state_map_range(
        provider->asan_state, (uint64_t)(uintptr_t)base_ptr, allocation_size);
  }

  if (iree_status_is_ok(status)) {
    out_slab->base_ptr = (uint8_t*)base_ptr;
    // Preserve the requested logical slab length. The hidden HSA allocation may
    // be larger due to runtime granule rounding, but exposing that padding here
    // would incorrectly inflate HAL buffer byte lengths in pass-through pools.
    out_slab->length = min_length;
    out_slab->provider_handle = (uint64_t)(uintptr_t)slab_handle;
    out_slab->native_allocation.byte_length = allocation_size;
    out_slab->native_allocation.granularity = provider->allocation_granule;
    out_slab->maintenance_range =
        iree_hal_make_buffer_range(0, allocation_size);
    out_slab->host_cacheability =
        iree_any_bit_set(provider->owned.memory_type,
                         IREE_HAL_MEMORY_TYPE_HOST_CACHED)
            ? IREE_HAL_HOST_CACHEABILITY_WRITE_BACK
            : IREE_HAL_HOST_CACHEABILITY_UNKNOWN;
    out_slab->allocation_alignment = provider->allocation_alignment;
    slab_handle->trace = provider->trace;
    iree_hal_memory_trace_alloc(&slab_handle->trace, slab_handle,
                                allocation_size);
    iree_atomic_fetch_add(&provider->total_acquired, 1,
                          iree_memory_order_relaxed);
    iree_hal_amdgpu_slab_provider_record_memory_event(
        provider, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE, slab_handle,
        base_ptr);
  } else if (base_ptr) {
    status = iree_status_join(
        status,
        iree_hsa_amd_memory_pool_free(IREE_LIBHSA(provider->libhsa), base_ptr));
  }
  if (!iree_status_is_ok(status) && !slab_handle_consumed) {
    iree_allocator_free(provider->base.host_allocator, slab_handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_slab_provider_import_slab(
    iree_hal_slab_provider_t* base_provider,
    const iree_hal_external_storage_t* storage,
    iree_hal_imported_slab_t* out_import) {
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_external_storage_t registration;
  iree_device_size_t storage_offset = 0;
  if (storage->storage_class.type ==
      IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
    storage_offset = storage->maintenance_range.byte_offset;
    registration = iree_hal_slab_provider_registration_view(storage);
    storage = &registration;
  }

  if (!provider->import_allocator) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_UNAVAILABLE,
                "AMDGPU slab provider has no external attachment service"));
  }

  const iree_hal_amdgpu_buffer_spec_t buffer_spec = {
      .type = provider->properties.memory_type,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = provider->properties.supported_usage,
      .queue_family_affinity = provider->supported_queue_family_affinity,
      .min_alignment = provider->properties.allocation_alignment,
  };
  iree_hal_buffer_t* imported_buffer = NULL;
  iree_status_t status = iree_hal_amdgpu_allocator_import_direct_buffer(
      provider->import_allocator, &buffer_spec, storage, provider->access.count,
      provider->access.agents, &imported_buffer);

  iree_hal_amdgpu_slab_handle_t* slab_handle = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(provider->base.host_allocator,
                                   sizeof(*slab_handle), (void**)&slab_handle);
  }
  if (iree_status_is_ok(status)) {
    memset(slab_handle, 0, sizeof(*slab_handle));
    slab_handle->allocation_size = storage->byte_length;
    slab_handle->imported_buffer = imported_buffer;
    slab_handle->host_allocator = provider->base.host_allocator;

    const iree_hal_amdgpu_buffer_native_t* native =
        iree_hal_amdgpu_buffer_native(imported_buffer);
    const uint64_t device_address =
        native->bindings[IREE_HAL_AMDGPU_BUFFER_BINDING_DEVICE_ADDRESS]
            .device_address;
    if (IREE_UNLIKELY(!device_address)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU external attachment has no device execution address");
    } else {
      slab_handle->base_ptr = (void*)(uintptr_t)device_address;
      out_import->slab.base_ptr = (uint8_t*)(uintptr_t)device_address;
      out_import->slab.length = storage->byte_length;
      out_import->slab.provider_handle = (uint64_t)(uintptr_t)slab_handle;
      out_import->storage_offset = storage_offset;
      out_import->properties = (iree_hal_slab_provider_properties_t){
          .memory_type = iree_hal_buffer_memory_type(imported_buffer),
          .supported_usage = iree_hal_buffer_allowed_usage(imported_buffer),
          .queue_family_affinity = provider->properties.queue_family_affinity,
          .allocation_alignment = provider->properties.allocation_alignment,
          .max_allocation_alignment = provider->properties.allocation_alignment,
          .maintenance_alignment = 1,
          .atomic_operations =
              iree_hal_amdgpu_atomic_memory_expand_capabilities(
                  native->atomic_memory_cells),
      };
      out_import->slab.maintenance_range = storage->maintenance_range;
      out_import->slab.host_cacheability = storage->host_cacheability;
      out_import->slab.physical_backing_id = storage->physical_backing_id;
      out_import->slab.allocation_alignment =
          out_import->properties.allocation_alignment;
      iree_atomic_fetch_add(&provider->total_acquired, 1,
                            iree_memory_order_relaxed);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(imported_buffer);
    iree_allocator_free(provider->base.host_allocator, slab_handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_amdgpu_slab_provider_query_import_requirements(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_import_requirements_t* out_requirements) {
  const iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_const_cast(base_provider);
  if ((storage_class.type != IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER &&
       storage_class.type != IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS) ||
      storage_class.provenance[0] || storage_class.provenance[1]) {
    return false;
  }
  *out_requirements = (iree_hal_pool_import_requirements_t){
      .min_byte_length = 1,
      .max_byte_length = 0,
      .base_alignment = provider->properties.allocation_alignment,
      .byte_length_granularity = 1,
  };
  return true;
}

static void iree_hal_amdgpu_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  IREE_ASSERT_ARGUMENT(slab);
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (slab->provider_handle) {
    iree_hal_amdgpu_slab_handle_t* slab_handle =
        (iree_hal_amdgpu_slab_handle_t*)(uintptr_t)slab->provider_handle;
    if (slab_handle->imported_buffer) {
      iree_hal_buffer_release(slab_handle->imported_buffer);
      iree_allocator_free(provider->base.host_allocator, slab_handle);
      iree_atomic_fetch_add(&provider->total_released, 1,
                            iree_memory_order_relaxed);
      IREE_TRACE_ZONE_END(z0);
      return;
    }
    iree_hal_amdgpu_slab_provider_record_memory_event(
        provider, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE, slab_handle,
        slab->base_ptr);
    if (iree_hal_amdgpu_slab_provider_uses_asan_shadow(provider) &&
        iree_atomic_load(&slab_handle->has_asan_advice,
                         iree_memory_order_relaxed)) {
      iree_hal_amdgpu_asan_quarantine_release_fn_t release_fn =
          iree_hal_amdgpu_slab_provider_uses_asan_vmm(provider)
              ? iree_hal_amdgpu_slab_provider_finalize_vmem_handle
              : iree_hal_amdgpu_slab_provider_finalize_hsa_allocation;
      iree_hal_amdgpu_asan_state_quarantine_entry(
          provider->asan_state, &slab_handle->quarantine_entry,
          slab_handle->allocation_size, release_fn, slab_handle);
    } else {
      if (iree_hal_amdgpu_slab_provider_uses_asan_vmm(provider) ||
          iree_hal_amdgpu_slab_provider_uses_mapping_source(provider)) {
        iree_hal_amdgpu_slab_provider_finalize_vmem_handle(slab_handle);
      } else {
        iree_hal_amdgpu_slab_provider_finalize_hsa_allocation(slab_handle);
      }
    }
    iree_atomic_fetch_add(&provider->total_released, 1,
                          iree_memory_order_relaxed);
  }
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_amdgpu_slab_provider_borrowed_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)user_data;
  (void)buffer;
}

static iree_status_t iree_hal_amdgpu_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base_provider);
  const iree_hal_amdgpu_slab_handle_t* slab_handle =
      (const iree_hal_amdgpu_slab_handle_t*)(uintptr_t)slab->provider_handle;

  iree_hal_memory_type_t memory_type = provider->owned.memory_type;
  iree_hal_amdgpu_atomic_memory_cell_flags_t atomic_memory_cells =
      provider->atomic_memory_cells;
  uint8_t* device_pointer = slab->base_ptr;
  uint8_t* host_pointer =
      iree_any_bit_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)
          ? slab->base_ptr
          : NULL;
  if (slab_handle->imported_buffer) {
    const iree_hal_buffer_t* imported_buffer = slab_handle->imported_buffer;
    const iree_hal_amdgpu_buffer_native_t* native =
        iree_hal_amdgpu_buffer_native(imported_buffer);
    memory_type = iree_hal_buffer_memory_type(imported_buffer);
    atomic_memory_cells = native->atomic_memory_cells;
    device_pointer =
        (uint8_t*)(uintptr_t)native
            ->bindings[IREE_HAL_AMDGPU_BUFFER_BINDING_DEVICE_ADDRESS]
            .device_address;
    host_pointer =
        native->bindings[IREE_HAL_AMDGPU_BUFFER_BINDING_HOST].host_pointer;
  }

  const iree_hal_buffer_placement_t placement = {
      .device = provider->device,
      .queue_family_affinity = provider->supported_queue_family_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  if (!release_callback.fn) {
    release_callback.fn = iree_hal_amdgpu_slab_provider_borrowed_buffer_release;
  }
  iree_status_t status = iree_hal_amdgpu_buffer_create_pooled(
      provider->libhsa, placement, memory_type, allowed_access, allowed_usage,
      atomic_memory_cells, slab->length, slab_offset, allocation_size,
      device_pointer, host_pointer, release_callback, provider->buffer_pool,
      provider->base.host_allocator, out_buffer);
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_slab_provider_uses_mapping_source(provider)) {
    iree_hal_amdgpu_buffer_prepare_mapping_source(
        *out_buffer, &slab_handle->vmem_source,
        (uint32_t)provider->allocation_granule);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* base_provider,
    const iree_hal_asan_pool_options_t* options) {
  const iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_const_cast(base_provider);
  if (!iree_hal_amdgpu_slab_provider_uses_asan_shadow(provider)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HSA memory-pool slab provider does not support HAL ASAN range "
        "advice");
  }
  if (IREE_UNLIKELY(
          !iree_hal_amdgpu_asan_state_is_enabled(provider->asan_state))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU ASAN slab provider requires enabled ASAN state");
  }
  const iree_device_size_t shadow_granule_size =
      (iree_device_size_t)1ull
      << provider->asan_state->shadow_map.shadow_scale_shift;
  if (IREE_UNLIKELY(options->shadow_granule_size != shadow_granule_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN pool shadow granule size %" PRIu64
        " does not match device shadow granule size %" PRIu64,
        (uint64_t)options->shadow_granule_size, (uint64_t)shadow_granule_size);
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base_provider);
  IREE_ASSERT(iree_hal_amdgpu_slab_provider_uses_asan_shadow(provider));
  IREE_ASSERT(iree_hal_amdgpu_asan_state_is_enabled(provider->asan_state));
  iree_hal_amdgpu_slab_handle_t* slab_handle =
      (iree_hal_amdgpu_slab_handle_t*)(uintptr_t)slab->provider_handle;
  iree_atomic_store(&slab_handle->has_asan_advice, 1,
                    iree_memory_order_relaxed);

  const uint64_t mapped_address =
      (uint64_t)(uintptr_t)slab->base_ptr + backing_offset;
  switch (advice_flags) {
    case IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED:
      iree_hal_amdgpu_asan_state_publish_allocated_range_raw(
          provider->asan_state, mapped_address, layout->backing_length,
          mapped_address + layout->user_offset, layout->user_length);
      return;
    case IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED:
      iree_hal_amdgpu_asan_state_publish_released_range(
          provider->asan_state, mapped_address, layout->backing_length);
      return;
    default:
      IREE_CHECK_UNREACHABLE("invalid ASAN range advice flags: 0x%08x",
                             advice_flags);
  }
}

static void iree_hal_amdgpu_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t offset, iree_device_size_t length) {
  (void)base_provider;
  (void)slab;
}

static void iree_hal_amdgpu_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider, iree_hal_pool_trim_flags_t flags) {
  (void)base_provider;
  (void)flags;
}

static void iree_hal_amdgpu_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base_provider)) {
    return;
  }
  const iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_const_cast(base_provider);
  out_stats->total_acquired += (uint64_t)iree_atomic_load(
      &provider->total_acquired, iree_memory_order_relaxed);
  out_stats->total_released += (uint64_t)iree_atomic_load(
      &provider->total_released, iree_memory_order_relaxed);
}

static void iree_hal_amdgpu_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  const iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_const_cast(base_provider);
  *out_properties = provider->properties;
}

static void iree_hal_amdgpu_slab_provider_acquire_slabs_native(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_acquire_request_t* request) {
  iree_hal_amdgpu_slab_provider_t* native =
      iree_hal_amdgpu_slab_provider_cast(provider);
  if (!iree_hal_amdgpu_slab_provider_uses_mapping_source(native)) {
    iree_hal_slab_provider_acquire_slabs_inline(
        provider, request, iree_hal_amdgpu_slab_provider_acquire_slab,
        iree_hal_amdgpu_slab_provider_release_slab);
    return;
  }
  memset(request->slabs, 0, request->count * sizeof(*request->slabs));
  iree_status_t status = iree_ok_status();
  iree_host_size_t acquired = 0;
  while (acquired < request->count && iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_slab_provider_acquire_slab(
        provider, request->requests[acquired].min_length,
        request->requests[acquired].min_alignment, &request->slabs[acquired]);
    if (iree_status_is_ok(status)) {
      ++acquired;
    }
  }
  if (iree_status_is_ok(status) && acquired) {
    status = iree_hal_amdgpu_virtual_memory_publish(
        native->libhsa, native->memory_pool, native->access.count,
        native->access.agents);
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < acquired; ++i) {
      iree_hal_amdgpu_slab_provider_release_slab(provider, &request->slabs[i]);
    }
    memset(request->slabs, 0, request->count * sizeof(*request->slabs));
  }
  request->completion.fn(request->completion.user_data, status);
}

static void iree_hal_amdgpu_slab_provider_release_slabs_native(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_release_request_t* request) {
  iree_hal_slab_provider_release_slabs_inline(
      provider, request, iree_hal_amdgpu_slab_provider_release_slab);
}

void iree_hal_amdgpu_slab_provider_execute(
    iree_hal_slab_operation_t* operation) {
  switch (operation->type) {
    case IREE_HAL_SLAB_OPERATION_ACQUIRE:
      iree_hal_amdgpu_slab_provider_acquire_slabs_native(
          operation->provider, (iree_hal_slab_acquire_request_t*)operation);
      break;
    case IREE_HAL_SLAB_OPERATION_RELEASE:
      iree_hal_amdgpu_slab_provider_release_slabs_native(
          operation->provider, (iree_hal_slab_release_request_t*)operation);
      break;
  }
}

static void iree_hal_amdgpu_slab_provider_acquire_slabs(
    iree_hal_slab_provider_t* base, iree_hal_slab_acquire_request_t* request) {
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base);
  if (provider->allocation_queue) {
    iree_hal_amdgpu_allocation_queue_enqueue(provider->allocation_queue,
                                             &request->operation);
  } else {
    iree_hal_amdgpu_slab_provider_execute(&request->operation);
  }
}

static void iree_hal_amdgpu_slab_provider_release_slabs(
    iree_hal_slab_provider_t* base, iree_hal_slab_release_request_t* request) {
  iree_hal_amdgpu_slab_provider_t* provider =
      iree_hal_amdgpu_slab_provider_cast(base);
  // Borrowed imports must detach before final public view release returns.
  iree_host_size_t owned_count = 0;
  for (iree_host_size_t i = 0; i < request->count; ++i) {
    const iree_hal_amdgpu_slab_handle_t* handle =
        (const iree_hal_amdgpu_slab_handle_t*)(uintptr_t)request->slabs[i]
            .provider_handle;
    if (handle && handle->imported_buffer) {
      iree_hal_amdgpu_slab_provider_release_slab(base, &request->slabs[i]);
    } else {
      if (owned_count != i) {
        request->slabs[owned_count] = request->slabs[i];
      }
      ++owned_count;
    }
  }
  request->count = owned_count;
  if (!owned_count) {
    request->completion.fn(request->completion.user_data);
    return;
  }
  if (provider->allocation_queue) {
    iree_hal_amdgpu_allocation_queue_enqueue(provider->allocation_queue,
                                             &request->operation);
  } else {
    iree_hal_amdgpu_slab_provider_execute(&request->operation);
  }
}

static const iree_hal_slab_provider_vtable_t
    iree_hal_amdgpu_slab_provider_vtable = {
        .destroy = iree_hal_amdgpu_slab_provider_destroy,
        .acquire_slabs = iree_hal_amdgpu_slab_provider_acquire_slabs,
        .release_slabs = iree_hal_amdgpu_slab_provider_release_slabs,
        .wrap_buffer = iree_hal_amdgpu_slab_provider_wrap_buffer,
        .validate_asan_options =
            iree_hal_amdgpu_slab_provider_validate_asan_options,
        .advise_asan_range = iree_hal_amdgpu_slab_provider_advise_asan_range,
        .prefault = iree_hal_amdgpu_slab_provider_prefault,
        .trim = iree_hal_amdgpu_slab_provider_trim,
        .query_stats = iree_hal_amdgpu_slab_provider_query_stats,
        .query_properties = iree_hal_amdgpu_slab_provider_query_properties,
        .import_slab = iree_hal_amdgpu_slab_provider_import_slab,
        .query_import_requirements =
            iree_hal_amdgpu_slab_provider_query_import_requirements,
};
