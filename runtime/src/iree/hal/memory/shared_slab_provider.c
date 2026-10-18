// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/memory.h"
#include "iree/base/internal/shm.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/tracing.h"

typedef struct iree_hal_shared_slab_provider_t {
  // Reference-counted native source.
  iree_hal_slab_provider_t base;
  // Process-owned system backing, without a native NUMA placement guarantee.
  iree_hal_memory_trace_t trace;
} iree_hal_shared_slab_provider_t;

typedef struct iree_hal_shared_slab_t {
  // Original section and its host mapping, owned until slab release.
  iree_shm_mapping_t mapping;
  // Prepared host and mapping-source identities in slab coordinates.
  iree_hal_buffer_native_binding_t bindings[2];
} iree_hal_shared_slab_t;

const iree_hal_buffer_binding_layout_t*
iree_hal_shared_slab_provider_binding_layout(void) {
  static const uint16_t types[] = {IREE_HAL_BUFFER_INTERFACE_HOST,
                                   IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE};
  static const iree_hal_buffer_binding_layout_t layout = {
      .byte_length = 2 * sizeof(iree_hal_buffer_native_binding_t),
      .binding_count = 2,
      .host_binding_index = 0,
      .types = types,
  };
  return &layout;
}

static void iree_hal_shared_slab_provider_destroy(
    iree_hal_slab_provider_t* base) {
  iree_hal_shared_slab_provider_t* provider =
      (iree_hal_shared_slab_provider_t*)base;
  iree_allocator_free(provider->base.host_allocator, provider);
}

static iree_status_t iree_hal_shared_slab_provider_acquire(
    iree_hal_slab_provider_t* base, iree_device_size_t min_length,
    iree_device_size_t min_alignment, iree_hal_slab_t* out_slab) {
  iree_hal_shared_slab_provider_t* provider =
      (iree_hal_shared_slab_provider_t*)base;
  memset(out_slab, 0, sizeof(*out_slab));
  const iree_host_size_t page_size = iree_memory_query_info().normal_page_size;
  const iree_device_size_t alignment = iree_max(min_alignment, page_size);
  // Overallocate only the alignment slack. Every slab coordinate is also a
  // native section coordinate plus the captured prefix; no extra alias is
  // needed to align the original host view.
  iree_host_size_t native_length = 0;
  if (min_length > IREE_HOST_SIZE_MAX || alignment > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_add((iree_host_size_t)min_length,
                                  (iree_host_size_t)alignment - page_size,
                                  &native_length) ||
      native_length > IREE_HOST_SIZE_MAX - (page_size - 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "shared slab geometry exceeds host address space");
  }
  iree_hal_shared_slab_t* slab = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(provider->base.host_allocator,
                                             sizeof(*slab), (void**)&slab));
  iree_status_t status = iree_shm_create(NULL, native_length, &slab->mapping);
  if (iree_status_is_ok(status)) {
    uint8_t* pointer = (uint8_t*)iree_host_align((uintptr_t)slab->mapping.base,
                                                 (iree_host_size_t)alignment);
    const iree_host_size_t prefix = pointer - (uint8_t*)slab->mapping.base;
    slab->bindings[0].host_pointer = pointer;
    slab->bindings[1].mapping_source.descriptor = &slab->mapping;
    slab->bindings[1].mapping_source.granularity = (uint32_t)page_size;
    slab->bindings[1].mapping_source.offset = prefix;
    iree_hal_memory_trace_alloc(&provider->trace, slab, slab->mapping.size);
    *out_slab = (iree_hal_slab_t){
        .base_ptr = pointer,
        .length = slab->mapping.size - prefix,
        .provider_handle = (uint64_t)(uintptr_t)slab,
        .native_allocation = {.byte_length = slab->mapping.size,
                              .byte_offset = prefix,
                              .granularity = page_size},
        .maintenance_range = {0, slab->mapping.size - prefix},
        .host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK,
        .allocation_alignment = alignment,
    };
  } else {
    iree_allocator_free(provider->base.host_allocator, slab);
  }
  return status;
}

static void iree_hal_shared_slab_provider_release(
    iree_hal_slab_provider_t* base, const iree_hal_slab_t* slab) {
  iree_hal_shared_slab_provider_t* provider =
      (iree_hal_shared_slab_provider_t*)base;
  iree_hal_shared_slab_t* storage =
      (iree_hal_shared_slab_t*)(uintptr_t)slab->provider_handle;
  iree_shm_close(&storage->mapping);
  iree_hal_memory_trace_free(&provider->trace, storage);
  iree_allocator_free(provider->base.host_allocator, storage);
}

static iree_status_t iree_hal_shared_slab_provider_wrap(
    iree_hal_slab_provider_t* base, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_shared_slab_provider_t* provider =
      (iree_hal_shared_slab_provider_t*)base;
  const iree_hal_shared_slab_t* storage =
      (const iree_hal_shared_slab_t*)(uintptr_t)slab->provider_handle;
  const iree_hal_buffer_placement_t placement = {
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
  };
  IREE_RETURN_IF_ERROR(iree_hal_heap_buffer_wrap(
      placement, IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE, allowed_access,
      allowed_usage, slab->length,
      iree_make_byte_span(slab->base_ptr, (iree_host_size_t)slab->length),
      release_callback, provider->base.host_allocator, out_buffer));
  (*out_buffer)->byte_offset = slab_offset;
  (*out_buffer)->byte_length = allocation_size;
  (*out_buffer)->memory.bindings = storage->bindings;
  (*out_buffer)->memory.binding_offset = slab_offset;
  (*out_buffer)->memory.offset = slab_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_shared_slab_provider_validate_asan(
    const iree_hal_slab_provider_t* provider,
    const iree_hal_asan_pool_options_t* options) {
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "shared CPU storage does not support HAL ASAN range advice");
}

static void iree_hal_shared_slab_provider_prefault(
    iree_hal_slab_provider_t* provider, const iree_hal_slab_t* slab,
    iree_device_size_t offset, iree_device_size_t length) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(slab->base_ptr + offset, 0, (iree_host_size_t)length);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_shared_slab_provider_trim(
    iree_hal_slab_provider_t* provider, iree_hal_pool_trim_flags_t flags) {
  // This leaf owns no cache. Callers can share an explicit slab cache above it.
}

static void iree_hal_shared_slab_provider_query_stats(
    const iree_hal_slab_provider_t* provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  // Physical storage accounting belongs to the owning pool.
}

static void iree_hal_shared_slab_provider_query_properties(
    const iree_hal_slab_provider_t* provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  *out_properties = (iree_hal_slab_provider_properties_t){
      .memory_type = IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE,
      .supported_usage = IREE_HAL_CPU_SLAB_PROVIDER_BUFFER_USAGE,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      .allocation_alignment = iree_memory_query_info().normal_page_size,
      .max_allocation_alignment =
          iree_min(IREE_HOST_SIZE_MAX, IREE_DEVICE_SIZE_MAX) / 2 + 1,
      .maintenance_alignment = 1,
      .atomic_operations = iree_hal_atomic_operation_capabilities_for_host(
          IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL),
  };
}

static void iree_hal_shared_slab_provider_acquire_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_acquire_request_t* request) {
  iree_hal_slab_provider_acquire_slabs_inline(
      provider, request, iree_hal_shared_slab_provider_acquire,
      iree_hal_shared_slab_provider_release);
}

static void iree_hal_shared_slab_provider_release_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_release_request_t* request) {
  iree_hal_slab_provider_release_slabs_inline(
      provider, request, iree_hal_shared_slab_provider_release);
}

static const iree_hal_slab_provider_vtable_t
    iree_hal_shared_slab_provider_vtable = {
        .destroy = iree_hal_shared_slab_provider_destroy,
        .acquire_slabs = iree_hal_shared_slab_provider_acquire_slabs,
        .release_slabs = iree_hal_shared_slab_provider_release_slabs,
        .wrap_buffer = iree_hal_shared_slab_provider_wrap,
        .validate_asan_options = iree_hal_shared_slab_provider_validate_asan,
        .prefault = iree_hal_shared_slab_provider_prefault,
        .trim = iree_hal_shared_slab_provider_trim,
        .query_stats = iree_hal_shared_slab_provider_query_stats,
        .query_properties = iree_hal_shared_slab_provider_query_properties,
};

iree_status_t iree_hal_shared_slab_provider_create(
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider) {
  *out_provider = NULL;
  iree_hal_shared_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  iree_hal_slab_provider_initialize(&iree_hal_shared_slab_provider_vtable,
                                    host_allocator, &provider->base);
  iree_status_t status = iree_hal_memory_trace_initialize(
      iree_string_view_empty(), "hal-system", &provider->trace);
  if (iree_status_is_ok(status)) {
    *out_provider = &provider->base;
  } else {
    iree_hal_slab_provider_release(&provider->base);
  }
  return status;
}
