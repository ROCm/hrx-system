// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/cpu_slab_provider.h"

typedef struct iree_hal_cpu_slab_provider_t {
  // Base provider header for vtable dispatch and ref counting.
  iree_hal_slab_provider_t base;

  // Host allocator used for provider metadata and slab memory.
  iree_allocator_t host_allocator;
} iree_hal_cpu_slab_provider_t;

static const iree_hal_slab_provider_vtable_t iree_hal_cpu_slab_provider_vtable;

iree_status_t iree_hal_cpu_slab_provider_create(
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;

  iree_hal_cpu_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  iree_hal_slab_provider_initialize(&iree_hal_cpu_slab_provider_vtable,
                                    &provider->base);
  provider->host_allocator = host_allocator;
  *out_provider = &provider->base;
  return iree_ok_status();
}

static void iree_hal_cpu_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  iree_allocator_t allocator = provider->host_allocator;
  iree_allocator_free(allocator, provider);
}

static iree_status_t iree_hal_cpu_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  memset(out_slab, 0, sizeof(*out_slab));
  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      provider->host_allocator, min_length, IREE_HAL_HEAP_BUFFER_ALIGNMENT,
      /*offset=*/0, &ptr));
  out_slab->base_ptr = (uint8_t*)ptr;
  out_slab->length = min_length;
  out_slab->provider_handle = 0;
  return iree_ok_status();
}

static void iree_hal_cpu_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  iree_allocator_free_aligned(provider->host_allocator, slab->base_ptr);
}

static iree_status_t iree_hal_cpu_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_cpu_slab_provider_t* provider =
      (iree_hal_cpu_slab_provider_t*)base_provider;
  iree_byte_span_t data = {
      .data = slab->base_ptr + slab_offset,
      .data_length = (iree_host_size_t)allocation_size,
  };
  return iree_hal_heap_buffer_wrap(iree_hal_buffer_placement_undefined(),
                                   params.type, params.access, params.usage,
                                   allocation_size, data, release_callback,
                                   provider->host_allocator, out_buffer);
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

// The CPU provider releases all resources with their slabs.
static void iree_hal_cpu_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider) {}

// The CPU provider tracks no statistics beyond what the allocator itself
// provides.
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
  out_properties->memory_type = IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE;
  out_properties->supported_usage = IREE_HAL_CPU_SLAB_PROVIDER_BUFFER_USAGE;
  out_properties->queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  out_properties->atomic_operations =
      iree_hal_atomic_operation_capabilities_for_host(
          IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
}

static const iree_hal_slab_provider_vtable_t iree_hal_cpu_slab_provider_vtable =
    {
        .destroy = iree_hal_cpu_slab_provider_destroy,
        .acquire_slab = iree_hal_cpu_slab_provider_acquire_slab,
        .release_slab = iree_hal_cpu_slab_provider_release_slab,
        .wrap_buffer = iree_hal_cpu_slab_provider_wrap_buffer,
        .validate_asan_options =
            iree_hal_cpu_slab_provider_validate_asan_options,
        .advise_asan_range = iree_hal_cpu_slab_provider_advise_asan_range,
        .trim = iree_hal_cpu_slab_provider_trim,
        .query_stats = iree_hal_cpu_slab_provider_query_stats,
        .query_properties = iree_hal_cpu_slab_provider_query_properties,
};
