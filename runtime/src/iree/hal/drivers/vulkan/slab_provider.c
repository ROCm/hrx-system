// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/slab_provider.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/memory/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_slab_provider_t
//===----------------------------------------------------------------------===//

static const char* IREE_HAL_VULKAN_SLAB_PROVIDER_TRACE_ID =
    "iree-hal-vulkan-slab-provider";

typedef struct iree_hal_vulkan_slab_provider_t {
  // Base slab provider for vtable dispatch and reference counting.
  iree_hal_slab_provider_t base;

  // Host allocator used for provider-owned host allocations.
  iree_allocator_t host_allocator;

  // Parent Vulkan allocator used to materialize whole slabs.
  iree_hal_vulkan_allocator_t* allocator;

  // Parent HAL device used for buffer placement metadata.
  iree_hal_device_t* parent_device;

  // Device-level Vulkan dispatch table copied from the logical device.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device that owns slabs from this provider.
  VkDevice logical_device;

  // Vulkan memory type index used for whole-slab allocations.
  uint32_t memory_type_index;

  // Prequalified Vulkan memory types usable for external host attachments.
  uint32_t import_memory_type_bits;

  // Immutable HAL memory properties exposed by this provider.
  iree_hal_slab_provider_properties_t properties;

  // Queue families that may access buffers materialized from this provider.
  iree_hal_queue_family_affinity_t queue_family_affinity_mask;

  // Minimum alignment used by whole-slab allocation requests.
  iree_device_size_t min_alignment;

  // Physical-device nonCoherentAtomSize used for mapped-memory ranges.
  VkDeviceSize non_coherent_atom_size;

  // Named-memory trace stream for whole-slab backing allocations.
  iree_hal_memory_trace_t trace;

  // Cumulative slabs acquired from Vulkan.
  iree_atomic_int64_t total_acquired;

  // Cumulative slabs released back to Vulkan.
  iree_atomic_int64_t total_released;
} iree_hal_vulkan_slab_provider_t;

static const iree_hal_slab_provider_vtable_t
    iree_hal_vulkan_slab_provider_vtable;

static iree_hal_vulkan_slab_provider_t* iree_hal_vulkan_slab_provider_cast(
    iree_hal_slab_provider_t* base_provider) {
  return (iree_hal_vulkan_slab_provider_t*)base_provider;
}

static const iree_hal_vulkan_slab_provider_t*
iree_hal_vulkan_slab_provider_const_cast(
    const iree_hal_slab_provider_t* base_provider) {
  return (const iree_hal_vulkan_slab_provider_t*)base_provider;
}

static void iree_hal_vulkan_slab_provider_populate_properties(
    const iree_hal_vulkan_slab_provider_t* provider,
    iree_hal_memory_type_t memory_type,
    iree_hal_slab_provider_properties_t* out_properties) {
  *out_properties = provider->properties;
  out_properties->memory_type = memory_type;
  out_properties->allocation_alignment =
      provider->min_alignment ? provider->min_alignment : 1;
  out_properties->max_allocation_alignment =
      out_properties->allocation_alignment;
  out_properties->maintenance_alignment =
      iree_any_bit_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
              !iree_any_bit_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT)
          ? provider->non_coherent_atom_size
          : 1;
}

iree_status_t iree_hal_vulkan_slab_provider_create(
    iree_hal_vulkan_allocator_t* allocator,
    iree_hal_vulkan_slab_provider_options_t options,
    iree_string_view_t trace_name, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(options.parent_device);
  IREE_ASSERT_ARGUMENT(options.syms);
  IREE_ASSERT_ARGUMENT(options.logical_device);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (options.memory_type == IREE_HAL_MEMORY_TYPE_NONE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan slab provider requires a non-empty HAL memory type");
  }
  if (options.supported_usage == IREE_HAL_BUFFER_USAGE_NONE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan slab provider requires non-empty HAL buffer usage bits");
  }
  if (!iree_device_size_is_valid_alignment(options.min_alignment)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan slab provider alignment %" PRIu64
                            " is not a power-of-two",
                            (uint64_t)options.min_alignment);
  }

  iree_hal_vulkan_slab_provider_t* provider = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*provider),
                                (void**)&provider));
  memset(provider, 0, sizeof(*provider));
  iree_hal_slab_provider_initialize(&iree_hal_vulkan_slab_provider_vtable,
                                    &provider->base);
  provider->host_allocator = host_allocator;
  provider->allocator = allocator;
  provider->parent_device = options.parent_device;
  provider->syms = *options.syms;
  provider->logical_device = options.logical_device;
  provider->memory_type_index = options.memory_type_index;
  provider->import_memory_type_bits = options.import_memory_type_bits;
  provider->properties.memory_type = options.memory_type;
  provider->properties.supported_usage = options.supported_usage;
  provider->properties.queue_family_affinity =
      options.queue_family_affinity_mask;
  provider->properties.atomic_operations = options.atomic_operations;
  provider->queue_family_affinity_mask = options.queue_family_affinity_mask;
  provider->min_alignment = options.min_alignment;
  provider->non_coherent_atom_size =
      options.non_coherent_atom_size ? options.non_coherent_atom_size : 1;
  provider->total_acquired = IREE_ATOMIC_VAR_INIT(0);
  provider->total_released = IREE_ATOMIC_VAR_INIT(0);

  iree_status_t status = iree_hal_memory_trace_initialize(
      trace_name, IREE_HAL_VULKAN_SLAB_PROVIDER_TRACE_ID, host_allocator,
      &provider->trace);
  if (iree_status_is_ok(status)) {
    *out_provider = &provider->base;
  } else {
    iree_hal_slab_provider_release(&provider->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_cast(base_provider);
  iree_allocator_t host_allocator = provider->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_memory_trace_deinitialize(&provider->trace);
  iree_allocator_free(host_allocator, provider);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_device_size_t alignment, iree_hal_slab_t* out_slab) {
  IREE_ASSERT_ARGUMENT(out_slab);
  iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_slab, 0, sizeof(*out_slab));

  if (IREE_UNLIKELY(min_length == 0)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "Vulkan slab allocations must be non-empty"));
  }

  iree_device_size_t allocation_size = min_length;
  const iree_device_size_t native_alignment =
      provider->min_alignment ? provider->min_alignment : 1;
  if (IREE_UNLIKELY(!iree_device_size_checked_align(
          allocation_size, native_alignment, &allocation_size))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "Vulkan slab allocation size overflow aligning %" PRIu64
                " bytes to a %" PRIu64 "-byte allocation boundary",
                (uint64_t)min_length, (uint64_t)native_alignment));
  }

  // Vulkan guarantees the native alignment captured by the provider. The
  // direct allocator only needs to create an aligned interior view when this
  // acquisition asks for a stronger address alignment.
  const iree_device_size_t min_address_alignment =
      alignment > native_alignment ? alignment : 0;

  const iree_hal_vulkan_buffer_spec_t buffer_spec = {
      .type = provider->properties.memory_type,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = provider->properties.supported_usage,
      .queue_family_affinity = provider->queue_family_affinity_mask,
      .min_alignment = min_address_alignment,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_hal_vulkan_allocator_allocate_direct_buffer_from_type(
          provider->allocator, provider->memory_type_index, &buffer_spec,
          allocation_size, &buffer);

  VkDeviceAddress device_address = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_buffer_device_address(buffer, &device_address);
  }

  if (iree_status_is_ok(status)) {
    out_slab->base_ptr = (uint8_t*)(uintptr_t)device_address;
    out_slab->length = iree_hal_buffer_allocation_size(buffer);
    out_slab->provider_handle = (uint64_t)(uintptr_t)buffer;
    iree_hal_memory_trace_alloc(&provider->trace, out_slab->base_ptr,
                                out_slab->length);
    iree_atomic_fetch_add(&provider->total_acquired, 1,
                          iree_memory_order_relaxed);
  } else {
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_slab_provider_import_slab(
    iree_hal_slab_provider_t* base_provider,
    const iree_hal_external_storage_t* storage,
    iree_hal_imported_slab_t* out_import) {
  iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_vulkan_buffer_spec_t buffer_spec = {
      .type = provider->properties.memory_type,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = provider->properties.supported_usage,
      .queue_family_affinity = provider->queue_family_affinity_mask,
      .min_alignment = provider->min_alignment,
  };
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_vulkan_allocator_import_direct_buffer(
      provider->allocator, provider->import_memory_type_bits, &buffer_spec,
      storage, &buffer);

  VkDeviceAddress device_address = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_buffer_device_address(buffer, &device_address);
  }

  if (iree_status_is_ok(status)) {
    out_import->slab.base_ptr = (uint8_t*)(uintptr_t)device_address;
    out_import->slab.length = storage->byte_length;
    out_import->slab.provider_handle = (uint64_t)(uintptr_t)buffer;
    iree_hal_vulkan_slab_provider_populate_properties(
        provider, iree_hal_buffer_memory_type(buffer), &out_import->properties);
    out_import->properties.supported_usage =
        iree_hal_buffer_allowed_usage(buffer);
    iree_hal_memory_trace_alloc(&provider->trace, out_import->slab.base_ptr,
                                out_import->slab.length);
    iree_atomic_fetch_add(&provider->total_acquired, 1,
                          iree_memory_order_relaxed);
  } else {
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_vulkan_slab_provider_query_import_requirements(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_import_requirements_t* out_requirements) {
  const iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_const_cast(base_provider);
  if (storage_class.type != IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER ||
      storage_class.provenance[0] || storage_class.provenance[1] ||
      !provider->import_memory_type_bits) {
    return false;
  }
  *out_requirements = (iree_hal_pool_import_requirements_t){
      .min_byte_length = provider->min_alignment,
      .max_byte_length = 0,
      .base_alignment = provider->min_alignment,
      .byte_length_alignment = provider->min_alignment,
  };
  return true;
}

static void iree_hal_vulkan_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  IREE_ASSERT_ARGUMENT(slab);
  iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);
  if (slab->provider_handle) {
    iree_hal_memory_trace_free(&provider->trace, slab->base_ptr);
    iree_hal_buffer_t* buffer =
        (iree_hal_buffer_t*)(uintptr_t)slab->provider_handle;
    iree_hal_buffer_release(buffer);
    iree_atomic_fetch_add(&provider->total_released, 1,
                          iree_memory_order_relaxed);
  }
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_cast(base_provider);
  iree_hal_buffer_t* slab_buffer =
      (iree_hal_buffer_t*)(uintptr_t)slab->provider_handle;

  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_handle(slab_buffer, &device_memory, &handle));
  VkDeviceAddress device_address = 0;
  const iree_hal_vulkan_buffer_native_t* native =
      iree_hal_vulkan_buffer_native(slab_buffer);
  device_address =
      native->bindings[IREE_HAL_VULKAN_BUFFER_BINDING_DEVICE_ADDRESS]
          .device_address;
  iree_hal_vulkan_buffer_mapping_state_t* mapping_state =
      iree_hal_vulkan_buffer_mapping_state(slab_buffer);
  const iree_hal_memory_type_t memory_type =
      iree_hal_buffer_memory_type(slab_buffer);
  const VkMemoryPropertyFlags memory_property_flags =
      mapping_state ? iree_hal_vulkan_buffer_memory_property_flags(slab_buffer)
                    : 0;
  const iree_device_size_t handle_offset =
      slab_buffer->memory.binding_offset + slab_offset;

  const iree_hal_buffer_placement_t placement = {
      .device = provider->parent_device,
      .queue_family_affinity = provider->queue_family_affinity_mask,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
  };
  return iree_hal_vulkan_buffer_create_borrowed(
      &provider->syms, provider->logical_device, placement, memory_type,
      allowed_access, allowed_usage, native->handle_length, handle_offset,
      allocation_size, memory_property_flags, provider->non_coherent_atom_size,
      device_memory, mapping_state, handle, device_address, release_callback,
      provider->host_allocator, out_buffer);
}

static iree_status_t iree_hal_vulkan_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* base_provider,
    const iree_hal_asan_pool_options_t* options) {
  (void)base_provider;
  (void)options;
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "Vulkan slab provider does not support HAL ASAN range advice");
}

static void iree_hal_vulkan_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  (void)base_provider;
  (void)slab;
  (void)backing_offset;
  (void)advice_flags;
  (void)layout;
  IREE_ASSERT(false, "Vulkan slab provider cannot advise ASAN ranges");
}

static void iree_hal_vulkan_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t offset, iree_device_size_t length) {
  (void)base_provider;
  (void)slab;
}

static void iree_hal_vulkan_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider, iree_hal_pool_trim_flags_t flags) {
  (void)base_provider;
  (void)flags;
}

static void iree_hal_vulkan_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  if (iree_hal_slab_provider_visited(visited, base_provider)) {
    return;
  }
  const iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_const_cast(base_provider);
  out_stats->total_acquired += (uint64_t)iree_atomic_load(
      &provider->total_acquired, iree_memory_order_relaxed);
  out_stats->total_released += (uint64_t)iree_atomic_load(
      &provider->total_released, iree_memory_order_relaxed);
}

static void iree_hal_vulkan_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  const iree_hal_vulkan_slab_provider_t* provider =
      iree_hal_vulkan_slab_provider_const_cast(base_provider);
  iree_hal_vulkan_slab_provider_populate_properties(
      provider, provider->properties.memory_type, out_properties);
  out_properties->max_allocation_alignment = IREE_DEVICE_SIZE_MAX / 2 + 1;
}

static const iree_hal_slab_provider_vtable_t
    iree_hal_vulkan_slab_provider_vtable = {
        .destroy = iree_hal_vulkan_slab_provider_destroy,
        .acquire_slab = iree_hal_vulkan_slab_provider_acquire_slab,
        .release_slab = iree_hal_vulkan_slab_provider_release_slab,
        .wrap_buffer = iree_hal_vulkan_slab_provider_wrap_buffer,
        .validate_asan_options =
            iree_hal_vulkan_slab_provider_validate_asan_options,
        .advise_asan_range = iree_hal_vulkan_slab_provider_advise_asan_range,
        .prefault = iree_hal_vulkan_slab_provider_prefault,
        .trim = iree_hal_vulkan_slab_provider_trim,
        .query_stats = iree_hal_vulkan_slab_provider_query_stats,
        .query_properties = iree_hal_vulkan_slab_provider_query_properties,
        .import_slab = iree_hal_vulkan_slab_provider_import_slab,
        .query_import_requirements =
            iree_hal_vulkan_slab_provider_query_import_requirements,
};
