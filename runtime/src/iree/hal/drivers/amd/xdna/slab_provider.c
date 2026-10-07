// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/slab_provider.h"

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amd/status.h"

typedef struct iree_hal_amd_xdna_slab_provider_t {
  // Base slab-provider interface header.
  iree_hal_slab_provider_t base;
  // Host allocator owning this provider and slab metadata.
  iree_allocator_t host_allocator;
  // Borrowed HAL device used for materialized buffer placement.
  iree_hal_device_t* device;
  // Borrowed native owner and diagnostic sink.
  iree_hal_amd_xdna_context_t* context;
  // Borrowed system scope used for allocation.
  amdf_memory_scope_t* scope;
  // Qualified CREATE and HOST_MAP profile.
  amdf_memory_profile_t profile;
  // Number of entries in |accesses|.
  uint32_t access_count;
  // Provider-owned copy of caller-ordered native accesses.
  amdf_memory_device_access_t* accesses;
  // Immutable HAL capabilities of all slabs.
  iree_hal_slab_provider_properties_t properties;
  // Cumulative successful slab acquisitions.
  iree_atomic_int64_t total_acquired;
  // Cumulative slab release attempts.
  iree_atomic_int64_t total_released;
} iree_hal_amd_xdna_slab_provider_t;

typedef struct iree_hal_amd_xdna_slab_handle_t {
  // Native allocation and its persistent host mapping.
  iree_hal_amd_xdna_memory_t memory;
  // Complete number of entries in |bindings|.
  uint32_t binding_count;
  // Prepared HOST followed by caller-ordered XDNA shim-DMA bindings.
  iree_hal_buffer_native_binding_t* bindings;
} iree_hal_amd_xdna_slab_handle_t;

static const iree_hal_slab_provider_vtable_t
    iree_hal_amd_xdna_slab_provider_vtable;

static iree_hal_amd_xdna_slab_provider_t* iree_hal_amd_xdna_slab_provider_cast(
    iree_hal_slab_provider_t* base_provider) {
  return (iree_hal_amd_xdna_slab_provider_t*)base_provider;
}

static const iree_hal_amd_xdna_slab_provider_t*
iree_hal_amd_xdna_slab_provider_const_cast(
    const iree_hal_slab_provider_t* base_provider) {
  return (const iree_hal_amd_xdna_slab_provider_t*)base_provider;
}

iree_status_t iree_hal_amd_xdna_slab_provider_create(
    iree_hal_device_t* device, iree_hal_amd_xdna_context_t* context,
    const iree_hal_amd_xdna_slab_provider_options_t* options,
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider) {
  *out_provider = NULL;
  if (!device || !context || !options || !options->access_count ||
      !options->accesses || !options->scope ||
      !iree_all_bits_set(options->profile.roles,
                         AMDF_MEMORY_PROFILE_ROLE_CREATE |
                             AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) ||
      !iree_all_bits_set(options->profile.supported_flags,
                         AMDF_MEMORY_FLAG_HOST_VISIBLE) ||
      !options->profile.allocation.byte_length_granularity ||
      !iree_device_size_is_power_of_two(
          options->profile.allocation.minimum_alignment) ||
      !iree_device_size_is_power_of_two(options->maintenance_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid XDNA slab construction contract");
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t accesses_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_slab_provider_t), &total_size,
      IREE_STRUCT_FIELD(options->access_count, amdf_memory_device_access_t,
                        &accesses_offset)));
  iree_hal_amd_xdna_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&provider));
  memset(provider, 0, total_size);
  iree_hal_slab_provider_initialize(&iree_hal_amd_xdna_slab_provider_vtable,
                                    &provider->base);
  provider->host_allocator = host_allocator;
  provider->device = device;
  provider->context = context;
  provider->scope = options->scope;
  provider->profile = options->profile;
  provider->access_count = options->access_count;
  provider->accesses =
      (amdf_memory_device_access_t*)((uint8_t*)provider + accesses_offset);
  memcpy(provider->accesses, options->accesses,
         options->access_count * sizeof(*provider->accesses));
  provider->properties = (iree_hal_slab_provider_properties_t){
      .memory_type = options->memory_type,
      .supported_usage = options->supported_usage,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      .allocation_alignment = options->profile.allocation.minimum_alignment,
      .maintenance_alignment = options->maintenance_alignment,
  };
  *out_provider = &provider->base;
  return iree_ok_status();
}

static void iree_hal_amd_xdna_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_amd_xdna_slab_provider_t* provider =
      iree_hal_amd_xdna_slab_provider_cast(base_provider);
  iree_allocator_free(provider->host_allocator, provider);
}

static iree_status_t iree_hal_amd_xdna_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab) {
  iree_hal_amd_xdna_slab_provider_t* provider =
      iree_hal_amd_xdna_slab_provider_cast(base_provider);
  memset(out_slab, 0, sizeof(*out_slab));
  if (!min_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA slab allocations must be non-empty");
  }
  const uint64_t granularity =
      provider->profile.allocation.byte_length_granularity;
  uint64_t allocation_length = 0;
  if (!iree_checked_add_u64(min_length, granularity - 1, &allocation_length) ||
      allocation_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA slab allocation extent overflows host size");
  }
  allocation_length = allocation_length / granularity * granularity;
  if (allocation_length > provider->profile.allocation.maximum_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA slab allocation exceeds the native profile");
  }

  const uint32_t binding_count = provider->access_count + 1;
  iree_host_size_t total_size = 0;
  iree_host_size_t bindings_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_slab_handle_t), &total_size,
      IREE_STRUCT_FIELD(binding_count, iree_hal_buffer_native_binding_t,
                        &bindings_offset)));
  iree_hal_amd_xdna_slab_handle_t* handle = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(provider->host_allocator,
                                             total_size, (void**)&handle));
  memset(handle, 0, total_size);
  handle->binding_count = binding_count;
  handle->bindings =
      (iree_hal_buffer_native_binding_t*)((uint8_t*)handle + bindings_offset);

  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = provider->profile.ordinal,
      .access_count = provider->access_count,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = allocation_length,
      .minimum_alignment = provider->profile.allocation.minimum_alignment,
      .accesses = provider->accesses,
  };
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      provider->context->api->memory_create(provider->scope, &create_info,
                                            &handle->memory.handle),
      "memory_create");
  const amdf_memory_map_info_t map_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map_info),
      .byte_length = allocation_length,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        provider->context->api->memory_map(handle->memory.handle, &map_info,
                                           &handle->memory.mapping),
        "memory_map");
  }
  amdf_host_mapping_info_t mapping_info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(mapping_info),
  };
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        provider->context->api->host_mapping_query_info(handle->memory.mapping,
                                                        &mapping_info),
        "host_mapping_query_info");
  }
  if (iree_status_is_ok(status)) {
    handle->bindings[0].host_pointer = mapping_info.pointer;
    for (uint32_t i = 0;
         i < provider->access_count && iree_status_is_ok(status); ++i) {
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          provider->context->api->memory_query_address(
              handle->memory.handle, i, AMDF_MEMORY_ADDRESS_XDNA_DMA,
              &handle->bindings[i + 1].device_address),
          "memory_query_address");
    }
  }
  if (iree_status_is_ok(status)) {
    handle->memory.contents =
        iree_make_byte_span(mapping_info.pointer, (iree_host_size_t)min_length);
    handle->memory.device_address = handle->bindings[1].device_address;
    handle->memory.host_cacheability = mapping_info.cacheability;
    out_slab->base_ptr = mapping_info.pointer;
    out_slab->length = min_length;
    out_slab->provider_handle = (uint64_t)(uintptr_t)handle;
    iree_atomic_fetch_add(&provider->total_acquired, 1,
                          iree_memory_order_relaxed);
  } else {
    iree_hal_amd_xdna_memory_deinitialize(provider->context, &handle->memory);
    iree_allocator_free(provider->host_allocator, handle);
  }
  return status;
}

static void iree_hal_amd_xdna_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_amd_xdna_slab_provider_t* provider =
      iree_hal_amd_xdna_slab_provider_cast(base_provider);
  iree_hal_amd_xdna_slab_handle_t* handle =
      (iree_hal_amd_xdna_slab_handle_t*)(uintptr_t)slab->provider_handle;
  if (!handle) {
    return;
  }
  iree_hal_amd_xdna_memory_deinitialize(provider->context, &handle->memory);
  iree_allocator_free(provider->host_allocator, handle);
  iree_atomic_fetch_add(&provider->total_released, 1,
                        iree_memory_order_relaxed);
}

static iree_status_t iree_hal_amd_xdna_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_amd_xdna_slab_provider_t* provider =
      iree_hal_amd_xdna_slab_provider_cast(base_provider);
  iree_hal_memory_type_t resolved_type = params.type;
  if (iree_any_bit_set(resolved_type, IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
    resolved_type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
    resolved_type |= provider->properties.memory_type;
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      provider->properties.memory_type, resolved_type));
  if (!iree_all_bits_set(provider->properties.supported_usage, params.usage)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA slab does not support requested buffer usage");
  }
  params.type = resolved_type;
  iree_hal_amd_xdna_slab_handle_t* handle =
      (iree_hal_amd_xdna_slab_handle_t*)(uintptr_t)slab->provider_handle;
  return iree_hal_amd_xdna_buffer_wrap(provider->device, provider->context,
                                       (iree_hal_amd_xdna_buffer_storage_t){
                                           .memory = &handle->memory,
                                           .bindings = handle->bindings,
                                           .host_binding_index = 0,
                                           .offset = slab_offset,
                                       },
                                       allocation_size, params,
                                       release_callback,
                                       provider->host_allocator, out_buffer);
}

static iree_status_t iree_hal_amd_xdna_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* base_provider,
    const iree_hal_asan_pool_options_t* options) {
  (void)base_provider;
  (void)options;
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "XDNA AMDF slab provider does not support HAL ASAN range advice");
}

static void iree_hal_amd_xdna_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  (void)base_provider;
  (void)slab;
  (void)backing_offset;
  (void)advice_flags;
  (void)layout;
  IREE_ASSERT(false, "XDNA AMDF slab provider cannot advise ASAN ranges");
}

static void iree_hal_amd_xdna_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t offset, iree_device_size_t length) {
  (void)base_provider;
  (void)slab;
  (void)offset;
  (void)length;
}

static void iree_hal_amd_xdna_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider, iree_hal_pool_trim_flags_t flags) {
  (void)base_provider;
  (void)flags;
}

static void iree_hal_amd_xdna_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base_provider)) {
    return;
  }
  const iree_hal_amd_xdna_slab_provider_t* provider =
      iree_hal_amd_xdna_slab_provider_const_cast(base_provider);
  out_stats->total_acquired += (uint64_t)iree_atomic_load(
      &provider->total_acquired, iree_memory_order_relaxed);
  out_stats->total_released += (uint64_t)iree_atomic_load(
      &provider->total_released, iree_memory_order_relaxed);
}

static void iree_hal_amd_xdna_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  const iree_hal_amd_xdna_slab_provider_t* provider =
      iree_hal_amd_xdna_slab_provider_const_cast(base_provider);
  *out_properties = provider->properties;
}

static const iree_hal_slab_provider_vtable_t
    iree_hal_amd_xdna_slab_provider_vtable = {
        .destroy = iree_hal_amd_xdna_slab_provider_destroy,
        .acquire_slab = iree_hal_amd_xdna_slab_provider_acquire_slab,
        .release_slab = iree_hal_amd_xdna_slab_provider_release_slab,
        .wrap_buffer = iree_hal_amd_xdna_slab_provider_wrap_buffer,
        .validate_asan_options =
            iree_hal_amd_xdna_slab_provider_validate_asan_options,
        .advise_asan_range = iree_hal_amd_xdna_slab_provider_advise_asan_range,
        .prefault = iree_hal_amd_xdna_slab_provider_prefault,
        .trim = iree_hal_amd_xdna_slab_provider_trim,
        .query_stats = iree_hal_amd_xdna_slab_provider_query_stats,
        .query_properties = iree_hal_amd_xdna_slab_provider_query_properties,
};
