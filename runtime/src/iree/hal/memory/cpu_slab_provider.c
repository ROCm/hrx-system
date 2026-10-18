// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/cpu_slab_provider.h"

#include "iree/hal/memory/tracing.h"

#if defined(IREE_PLATFORM_LINUX)
#include <sys/mman.h>
#endif  // IREE_PLATFORM_LINUX

typedef struct iree_hal_cpu_slab_provider_t {
  // Base provider header for vtable dispatch and ref counting.
  iree_hal_slab_provider_t base;
  // Process-owned system backing, without a native NUMA placement guarantee.
  iree_hal_memory_trace_t trace;

  // Minimum absolute byte alignment of every slab allocation.
  iree_host_size_t allocation_alignment;
} iree_hal_cpu_slab_provider_t;

// Marks a borrowed external pointer attachment. Owned CPU slabs use zero.
#define IREE_HAL_CPU_SLAB_PROVIDER_IMPORTED_HANDLE UINT64_MAX

static const iree_hal_slab_provider_vtable_t iree_hal_cpu_slab_provider_vtable;

iree_status_t iree_hal_cpu_slab_provider_create(
    iree_host_size_t min_alignment, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  if (!iree_host_size_is_valid_alignment(min_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "slab alignment must be a power of two (got %" PRIhsz ")",
        min_alignment);
  }

  iree_hal_cpu_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  iree_hal_slab_provider_initialize(&iree_hal_cpu_slab_provider_vtable,
                                    host_allocator, &provider->base);
  provider->allocation_alignment =
      iree_max(min_alignment, IREE_HAL_HEAP_BUFFER_ALIGNMENT);
  iree_status_t status = iree_hal_memory_trace_initialize(
      iree_string_view_empty(), "hal-system", &provider->trace);
  if (iree_status_is_ok(status)) {
    *out_provider = &provider->base;
  } else {
    iree_hal_slab_provider_release(&provider->base);
  }
  return status;
}

static void iree_hal_cpu_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  iree_allocator_t allocator = provider->base.host_allocator;
  iree_allocator_free(allocator, provider);
}

static iree_status_t iree_hal_cpu_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_device_size_t min_alignment, iree_hal_slab_t* out_slab) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  memset(out_slab, 0, sizeof(*out_slab));
  if (min_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "CPU slab length exceeds host address space");
  }
  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      provider->base.host_allocator, min_length,
      iree_max(min_alignment, provider->allocation_alignment),
      /*offset=*/0, &ptr));
  iree_hal_memory_trace_alloc(&provider->trace, ptr, min_length);
  out_slab->base_ptr = (uint8_t*)ptr;
  out_slab->length = min_length;
  out_slab->provider_handle = 0;
  out_slab->native_allocation.byte_length = min_length;
  out_slab->native_allocation.granularity = 1;
  out_slab->maintenance_range = iree_hal_make_buffer_range(0, min_length);
  out_slab->host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
  out_slab->allocation_alignment =
      iree_max(min_alignment, provider->allocation_alignment);
  return iree_ok_status();
}

static iree_status_t iree_hal_cpu_slab_provider_import_slab(
    iree_hal_slab_provider_t* base_provider,
    const iree_hal_external_storage_t* storage,
    iree_hal_imported_slab_t* out_import) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  if (storage->storage_class.type !=
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER ||
      storage->storage_class.provenance[0] ||
      storage->storage_class.provenance[1]) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "CPU slabs import portable host pointers only");
  }
  if (!storage->payload.host_pointer ||
      storage->byte_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid host pointer storage extent");
  }
  if (storage->host_cacheability != IREE_HAL_HOST_CACHEABILITY_WRITE_BACK) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "CPU slab import requires write-back host storage");
  }

  const uintptr_t address = (uintptr_t)storage->payload.host_pointer;
  const iree_device_size_t exact_alignment =
      (iree_device_size_t)(address & (~address + 1));
  out_import->slab = (iree_hal_slab_t){
      .base_ptr = storage->payload.host_pointer,
      .length = storage->byte_length,
      .provider_handle = IREE_HAL_CPU_SLAB_PROVIDER_IMPORTED_HANDLE,
      .maintenance_range = storage->maintenance_range,
      .host_cacheability = storage->host_cacheability,
      .physical_backing_id = storage->physical_backing_id,
      .allocation_alignment = exact_alignment,
  };
  out_import->properties = (iree_hal_slab_provider_properties_t){
      .memory_type = IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE,
      .supported_usage = IREE_HAL_CPU_SLAB_PROVIDER_BUFFER_USAGE,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      .allocation_alignment = exact_alignment,
      .max_allocation_alignment = exact_alignment,
      .maintenance_alignment = 1,
      .atomic_operations = iree_hal_atomic_operation_capabilities_for_host(
          IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL),
  };
  (void)provider;
  return iree_ok_status();
}

static bool iree_hal_cpu_slab_provider_query_import_requirements(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_import_requirements_t* out_requirements) {
  (void)base_provider;
  if (storage_class.type != IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER ||
      storage_class.provenance[0] || storage_class.provenance[1]) {
    return false;
  }
  *out_requirements = (iree_hal_pool_import_requirements_t){
      .min_byte_length = 1,
      .max_byte_length = IREE_HOST_SIZE_MAX,
      .base_alignment = 1,
      .byte_length_granularity = 1,
  };
  return true;
}

static void iree_hal_cpu_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  if (slab->provider_handle != IREE_HAL_CPU_SLAB_PROVIDER_IMPORTED_HANDLE) {
    iree_hal_memory_trace_free(&provider->trace, slab->base_ptr);
    iree_allocator_free_aligned(provider->base.host_allocator, slab->base_ptr);
  }
}

static iree_status_t iree_hal_cpu_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  iree_byte_span_t data = {
      .data = slab->base_ptr,
      .data_length = (iree_host_size_t)slab->length,
  };
  const iree_hal_buffer_placement_t placement = {
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
  };
  iree_status_t status = iree_hal_heap_buffer_wrap(
      placement, IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE, allowed_access,
      allowed_usage, slab->length, data, release_callback,
      provider->base.host_allocator, out_buffer);
  if (iree_status_is_ok(status)) {
    (*out_buffer)->byte_offset = slab_offset;
    (*out_buffer)->byte_length = allocation_size;
    (*out_buffer)->memory.binding_offset = slab_offset;
    (*out_buffer)->memory.offset = slab_offset;
  }
  return status;
}

static iree_status_t iree_hal_cpu_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* base_provider,
    const iree_hal_asan_pool_options_t* options) {
  (void)base_provider;
  (void)options;
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "CPU slab provider does not support HAL ASAN range advice");
}

static void iree_hal_cpu_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  (void)base_provider;
  (void)slab;
  (void)backing_offset;
  (void)advice_flags;
  (void)layout;
  IREE_ASSERT(false, "CPU slab provider cannot advise ASAN ranges");
}

// First touch runs on the captured native owner. Only the owned range may be
// written: neighboring bytes can belong to independently active pools.
static void iree_hal_cpu_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t offset, iree_device_size_t length) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);
  uint8_t* data = slab->base_ptr + offset;
  bool populated = false;
#if defined(IREE_PLATFORM_LINUX)
#ifdef MADV_HUGEPAGE
  if (length >= 2 * 1024 * 1024) {
    madvise(data, (size_t)length, MADV_HUGEPAGE);
  }
#endif  // MADV_HUGEPAGE
#ifdef MADV_POPULATE_WRITE
  // This hint may fail for an interior range that is not page aligned. The
  // bounded write below still provides first touch without widening ownership.
  populated = madvise(data, (size_t)length, MADV_POPULATE_WRITE) == 0;
#endif  // MADV_POPULATE_WRITE
#endif  // IREE_PLATFORM_LINUX
  if (!populated) {
    memset(data, 0, (size_t)length);
  }
  IREE_TRACE_ZONE_END(z0);
}

// The CPU provider has no cache or freelist; nothing to trim.
static void iree_hal_cpu_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider, iree_hal_pool_trim_flags_t flags) {
}

// The CPU provider tracks no statistics beyond what the allocator itself
// provides. Leaf provider; no inner provider to recurse into.
static void iree_hal_cpu_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base_provider)) {
    return;
  }
}

static void iree_hal_cpu_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  const iree_hal_cpu_slab_provider_t* provider =
      (const iree_hal_cpu_slab_provider_t*)base_provider;
  out_properties->memory_type = IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE;
  out_properties->allocation_alignment = provider->allocation_alignment;
  out_properties->max_allocation_alignment =
      iree_min(IREE_HOST_SIZE_MAX, IREE_DEVICE_SIZE_MAX) / 2 + 1;
  out_properties->maintenance_alignment = 1;
  out_properties->supported_usage = IREE_HAL_CPU_SLAB_PROVIDER_BUFFER_USAGE;
  out_properties->queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  out_properties->atomic_operations =
      iree_hal_atomic_operation_capabilities_for_host(
          IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
}

static void iree_hal_cpu_slab_provider_acquire_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_acquire_request_t* request) {
  iree_hal_slab_provider_acquire_slabs_inline(
      provider, request, iree_hal_cpu_slab_provider_acquire_slab,
      iree_hal_cpu_slab_provider_release_slab);
}

static void iree_hal_cpu_slab_provider_release_slabs(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_release_request_t* request) {
  iree_hal_slab_provider_release_slabs_inline(
      provider, request, iree_hal_cpu_slab_provider_release_slab);
}

static const iree_hal_slab_provider_vtable_t iree_hal_cpu_slab_provider_vtable =
    {
        .destroy = iree_hal_cpu_slab_provider_destroy,
        .acquire_slabs = iree_hal_cpu_slab_provider_acquire_slabs,
        .import_slab = iree_hal_cpu_slab_provider_import_slab,
        .release_slabs = iree_hal_cpu_slab_provider_release_slabs,
        .wrap_buffer = iree_hal_cpu_slab_provider_wrap_buffer,
        .validate_asan_options =
            iree_hal_cpu_slab_provider_validate_asan_options,
        .advise_asan_range = iree_hal_cpu_slab_provider_advise_asan_range,
        .prefault = iree_hal_cpu_slab_provider_prefault,
        .trim = iree_hal_cpu_slab_provider_trim,
        .query_stats = iree_hal_cpu_slab_provider_query_stats,
        .query_properties = iree_hal_cpu_slab_provider_query_properties,
        .query_import_requirements =
            iree_hal_cpu_slab_provider_query_import_requirements,
};
