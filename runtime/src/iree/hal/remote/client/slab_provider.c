// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/slab_provider.h"

#include "iree/hal/remote/client/buffer.h"
#include "iree/hal/remote/client/device.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_slab_provider_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_remote_client_slab_provider_t {
  // Base slab provider resource.
  iree_hal_slab_provider_t base;

  // Remote device used to allocate backing slabs.
  iree_hal_remote_client_device_t* device;

  // Host allocator used for provider storage.
  iree_allocator_t host_allocator;

  // Properties of the preferred server-side backing memory class.
  iree_hal_slab_provider_properties_t properties;

  // Total slabs acquired through this provider.
  iree_atomic_int64_t total_acquired;

  // Total slabs released through this provider.
  iree_atomic_int64_t total_released;
} iree_hal_remote_client_slab_provider_t;

static const iree_hal_slab_provider_vtable_t
    iree_hal_remote_client_slab_provider_vtable;

static iree_hal_remote_client_slab_provider_t*
iree_hal_remote_client_slab_provider_cast(
    iree_hal_slab_provider_t* base_provider) {
  return (iree_hal_remote_client_slab_provider_t*)base_provider;
}

static const iree_hal_remote_client_slab_provider_t*
iree_hal_remote_client_slab_provider_const_cast(
    const iree_hal_slab_provider_t* base_provider) {
  return (const iree_hal_remote_client_slab_provider_t*)base_provider;
}

iree_status_t iree_hal_remote_client_slab_provider_create(
    iree_hal_remote_client_device_t* device, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_client_slab_provider_t* provider = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*provider), (void**)&provider);
  if (iree_status_is_ok(status)) {
    iree_hal_slab_provider_initialize(
        &iree_hal_remote_client_slab_provider_vtable, &provider->base);
    provider->device = device;
    provider->host_allocator = host_allocator;
    memset(&provider->properties, 0, sizeof(provider->properties));
    iree_atomic_store(&provider->total_acquired, 0, iree_memory_order_relaxed);
    iree_atomic_store(&provider->total_released, 0, iree_memory_order_relaxed);
    *out_provider = &provider->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_remote_client_slab_provider_configure(
    iree_hal_slab_provider_t* base_provider,
    const iree_hal_device_spec_t* device_spec) {
  iree_hal_remote_client_slab_provider_t* provider =
      iree_hal_remote_client_slab_provider_cast(base_provider);
  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(device_spec);
  const iree_hal_memory_type_t required_memory_type =
      IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE & ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
  const iree_hal_buffer_usage_t required_usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

  for (iree_host_size_t i = 0; i < memory->memory_type_count; ++i) {
    const iree_hal_memory_type_spec_t* memory_type = &memory->memory_types[i];
    if (!iree_all_bits_set(memory_type->memory_type, required_memory_type) ||
        !iree_all_bits_set(memory_type->allowed_buffer_usage, required_usage)) {
      continue;
    }
    provider->properties = (iree_hal_slab_provider_properties_t){
        .memory_type = memory_type->memory_type,
        .supported_usage = memory_type->allowed_buffer_usage,
        .atomic_operations = memory_type->atomic_operations,
    };
    break;
  }
}

static void iree_hal_remote_client_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_remote_client_slab_provider_t* provider =
      iree_hal_remote_client_slab_provider_cast(base_provider);
  iree_allocator_free(provider->host_allocator, provider);
}

static iree_status_t iree_hal_remote_client_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab) {
  iree_hal_remote_client_slab_provider_t* provider =
      iree_hal_remote_client_slab_provider_cast(base_provider);
  memset(out_slab, 0, sizeof(*out_slab));

  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
  };
  iree_hal_buffer_t* root_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      provider->device->device_allocator, params, min_length, &root_buffer));

  out_slab->base_ptr = NULL;
  out_slab->length = iree_hal_buffer_byte_length(root_buffer);
  out_slab->provider_handle = (uint64_t)(uintptr_t)root_buffer;
  iree_atomic_fetch_add(&provider->total_acquired, 1,
                        iree_memory_order_relaxed);
  return iree_ok_status();
}

static void iree_hal_remote_client_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_remote_client_slab_provider_t* provider =
      iree_hal_remote_client_slab_provider_cast(base_provider);
  iree_hal_buffer_t* root_buffer =
      (iree_hal_buffer_t*)(uintptr_t)slab->provider_handle;
  iree_hal_buffer_release(root_buffer);
  iree_atomic_fetch_add(&provider->total_released, 1,
                        iree_memory_order_relaxed);
}

static iree_status_t iree_hal_remote_client_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_remote_client_slab_provider_t* provider =
      iree_hal_remote_client_slab_provider_cast(base_provider);
  iree_hal_buffer_t* root_buffer =
      (iree_hal_buffer_t*)(uintptr_t)slab->provider_handle;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(root_buffer, slab_offset,
                                                      allocation_size));
  iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(root_buffer);
  params.type = iree_hal_buffer_memory_type(root_buffer);
  params.queue_affinity = placement.queue_affinity;
  return iree_hal_remote_client_buffer_create_view(
      provider->device, root_buffer, slab_offset, allocation_size, &params,
      placement.flags, release_callback, provider->host_allocator, out_buffer);
}

static iree_status_t iree_hal_remote_client_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* base_provider,
    const iree_hal_asan_pool_options_t* options) {
  (void)base_provider;
  (void)options;
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "remote slab provider does not support HAL ASAN range advice");
}

static void iree_hal_remote_client_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  (void)base_provider;
  (void)slab;
  (void)backing_offset;
  (void)advice_flags;
  (void)layout;
  IREE_ASSERT(false, "remote slab provider cannot advise ASAN ranges");
}

static void iree_hal_remote_client_slab_provider_prefault(
    iree_hal_slab_provider_t* provider, iree_hal_slab_t* slab) {
  (void)provider;
  (void)slab;
}

static void iree_hal_remote_client_slab_provider_trim(
    iree_hal_slab_provider_t* provider,
    iree_hal_slab_provider_trim_flags_t flags) {
  (void)provider;
  (void)flags;
}

static void iree_hal_remote_client_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base_provider)) return;
  const iree_hal_remote_client_slab_provider_t* provider =
      iree_hal_remote_client_slab_provider_const_cast(base_provider);
  out_stats->total_acquired += (uint64_t)iree_atomic_load(
      &provider->total_acquired, iree_memory_order_relaxed);
  out_stats->total_released += (uint64_t)iree_atomic_load(
      &provider->total_released, iree_memory_order_relaxed);
}

static void iree_hal_remote_client_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  const iree_hal_remote_client_slab_provider_t* provider =
      iree_hal_remote_client_slab_provider_const_cast(base_provider);
  *out_properties = provider->properties;
}

static const iree_hal_slab_provider_vtable_t
    iree_hal_remote_client_slab_provider_vtable = {
        .destroy = iree_hal_remote_client_slab_provider_destroy,
        .acquire_slab = iree_hal_remote_client_slab_provider_acquire_slab,
        .release_slab = iree_hal_remote_client_slab_provider_release_slab,
        .wrap_buffer = iree_hal_remote_client_slab_provider_wrap_buffer,
        .validate_asan_options =
            iree_hal_remote_client_slab_provider_validate_asan_options,
        .advise_asan_range =
            iree_hal_remote_client_slab_provider_advise_asan_range,
        .prefault = iree_hal_remote_client_slab_provider_prefault,
        .trim = iree_hal_remote_client_slab_provider_trim,
        .query_stats = iree_hal_remote_client_slab_provider_query_stats,
        .query_properties =
            iree_hal_remote_client_slab_provider_query_properties,
};
