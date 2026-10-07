// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/memory.h"

#include "iree/hal/drivers/amd/status.h"

void iree_hal_amd_xdna_memory_deinitialize(iree_hal_amd_xdna_context_t* context,
                                           iree_hal_amd_xdna_memory_t* memory) {
  if (memory->mapping) {
    amdf_status_t status = context->api->host_mapping_destroy(memory->mapping);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_xdna_context_report(context, status, "host_mapping_destroy");
      return;
    }
  }
  if (memory->handle) {
    // memory_destroy consumes its handle even on native cleanup failure; the
    // provider preserves any backing required by failed detach as a leak.
    amdf_status_t status = context->api->memory_destroy(memory->handle);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_xdna_context_report(context, status, "memory_destroy");
    }
  }
  *memory = (iree_hal_amd_xdna_memory_t){0};
}

iree_status_t iree_hal_amd_xdna_memory_allocate(
    iree_hal_amd_xdna_context_t* context,
    const iree_hal_amd_xdna_memory_source_t* source, uint64_t byte_length,
    uint64_t alignment, iree_hal_amd_xdna_memory_t* out_memory) {
  *out_memory = (iree_hal_amd_xdna_memory_t){0};
  const uint64_t granularity =
      source->profile.allocation.byte_length_granularity;
  uint64_t allocation_length = 0;
  if (!iree_checked_add_u64(iree_max(byte_length, 1), granularity - 1,
                            &allocation_length) ||
      allocation_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA allocation extent overflows host size");
  }
  allocation_length = allocation_length / granularity * granularity;
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = source->profile.ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = allocation_length,
      .minimum_alignment =
          iree_max(alignment, source->profile.allocation.minimum_alignment),
      .accesses = &source->access,
  };
  iree_hal_amd_xdna_memory_t memory = {0};
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_create(source->scope, &create_info, &memory.handle),
      "memory_create"));
  const amdf_memory_map_info_t map_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map_info),
      .byte_length = allocation_length,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_map(memory.handle, &map_info, &memory.mapping),
      "memory_map");
  amdf_host_mapping_info_t mapping_info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(mapping_info),
  };
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        context->api->host_mapping_query_info(memory.mapping, &mapping_info),
        "host_mapping_query_info");
  }
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        context->api->memory_query_address(
            memory.handle, 0, source->address_kind, &memory.device_address),
        "memory_query_address");
  }
  if (iree_status_is_ok(status)) {
    memory.contents = iree_make_byte_span(mapping_info.pointer, byte_length);
    memory.host_cacheability = mapping_info.cacheability;
    *out_memory = memory;
  } else {
    iree_hal_amd_xdna_memory_deinitialize(context, &memory);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Native buffers
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amd_xdna_buffer_t {
  // Logical HAL allocation and view metadata.
  iree_hal_buffer_t base;
  // Prepared execution addresses published through base.memory.bindings.
  iree_hal_buffer_native_binding_t
      bindings[IREE_HAL_AMD_XDNA_BUFFER_BINDING_COUNT];
  // Borrowed native device owner.
  iree_hal_amd_xdna_context_t* context;
  // Allocator owning the HAL wrapper.
  iree_allocator_t host_allocator;
  // Native allocation, map, and device address.
  iree_hal_amd_xdna_memory_t memory;
} iree_hal_amd_xdna_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_amd_xdna_buffer_vtable;

const iree_hal_buffer_binding_layout_t* iree_hal_amd_xdna_buffer_binding_layout(
    void) {
  static const uint16_t types[] = {
      IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
      IREE_HAL_BUFFER_INTERFACE_HOST,
  };
  static const iree_hal_buffer_binding_layout_t layout = {
      .byte_length = sizeof(iree_hal_buffer_native_binding_t) *
                     IREE_HAL_AMD_XDNA_BUFFER_BINDING_COUNT,
      .binding_count = IREE_ARRAYSIZE(types),
      .host_binding_index = IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST,
      .types = types,
  };
  return &layout;
}

static void iree_hal_amd_xdna_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_amd_xdna_buffer_t* buffer = (iree_hal_amd_xdna_buffer_t*)base_buffer;
  iree_hal_amd_xdna_memory_deinitialize(buffer->context, &buffer->memory);
  iree_allocator_free(buffer->host_allocator, buffer);
}

static iree_status_t iree_hal_amd_xdna_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mode,
    iree_hal_memory_access_t access, iree_hal_buffer_map_flags_t flags,
    iree_device_size_t offset, iree_device_size_t length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_amd_xdna_buffer_t* buffer = (iree_hal_amd_xdna_buffer_t*)base_buffer;
  mapping->contents =
      iree_make_byte_span(buffer->memory.contents.data + offset, length);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_buffer_unmap_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_buffer_mapping_t* mapping) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  iree_hal_amd_xdna_buffer_t* buffer = (iree_hal_amd_xdna_buffer_t*)base_buffer;
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      buffer->context->api->host_mapping_cache_control(
          buffer->memory.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, offset,
          length),
      "host_mapping_cache_control(INVALIDATE)");
}

static iree_status_t iree_hal_amd_xdna_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  iree_hal_amd_xdna_buffer_t* buffer = (iree_hal_amd_xdna_buffer_t*)base_buffer;
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      buffer->context->api->host_mapping_cache_control(
          buffer->memory.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, offset,
          length),
      "host_mapping_cache_control(FLUSH)");
}

static iree_status_t iree_hal_amd_xdna_buffer_export_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA buffer export is not supported");
}

iree_status_t iree_hal_amd_xdna_buffer_resolve_binding_slot(
    iree_hal_amd_xdna_context_t* context, const iree_hal_queue_family_t* family,
    const iree_hal_buffer_t* buffer,
    iree_hal_buffer_native_binding_slot_t* out_slot) {
  *out_slot = (iree_hal_buffer_native_binding_slot_t){
      .index = IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE,
  };
  const iree_hal_memory_contract_t* contract = buffer->memory.contract;
  if (contract) {
    const uint64_t program_scope_id =
        (uint64_t)family->memory.queue_scope_id + 1;
    if (!family->memory.domain || family->memory.domain != contract->domain ||
        program_scope_id >= contract->scope_count) {
      return iree_make_status(
          IREE_STATUS_PERMISSION_DENIED,
          "XDNA program family is outside the buffer memory scope");
    }
    const iree_hal_memory_scope_access_t* access =
        &contract->scopes[program_scope_id];
    const uint32_t interface_bit = UINT32_C(1)
                                   << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA;
    const uint16_t index =
        access->bindings[IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA];
    if (!iree_any_bit_set(access->interfaces, interface_bit) ||
        index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
      return iree_make_status(
          IREE_STATUS_PERMISSION_DENIED,
          "XDNA shim DMA access is not prepared for this program family");
    }
    *out_slot = (iree_hal_buffer_native_binding_slot_t){
        .index = index,
        .type = IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
    };
    return iree_ok_status();
  }

  iree_hal_buffer_t* allocation = iree_hal_buffer_allocated_buffer(buffer);
  if (!iree_hal_resource_is((iree_hal_resource_t*)allocation,
                            &iree_hal_amd_xdna_buffer_vtable)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA dispatch requires a prepared shim DMA buffer binding");
  }
  const iree_hal_amd_xdna_buffer_t* native_buffer =
      (const iree_hal_amd_xdna_buffer_t*)allocation;
  if (native_buffer->context != context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer belongs to another XDNA device");
  }
  *out_slot = (iree_hal_buffer_native_binding_slot_t){
      .index = IREE_HAL_AMD_XDNA_BUFFER_BINDING_SHIM_DMA,
      .type = IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA,
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_buffer_load_device_address(
    iree_hal_buffer_ref_t buffer_ref,
    iree_hal_buffer_native_binding_slot_t slot, uint64_t* out_device_address) {
  *out_device_address = 0;
  const uint64_t base_address =
      iree_hal_buffer_native_binding(buffer_ref.buffer, slot).device_address;
  if (!base_address) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA binding storage is not committed after its queue prerequisites");
  }
  if (!iree_checked_add_u64(base_address, buffer_ref.offset,
                            out_device_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA binding address overflows");
  }
  return iree_ok_status();
}

static const iree_hal_buffer_vtable_t iree_hal_amd_xdna_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_amd_xdna_buffer_destroy,
    .export_range = iree_hal_amd_xdna_buffer_export_range,
    .map_range = iree_hal_amd_xdna_buffer_map_range,
    .unmap_range = iree_hal_amd_xdna_buffer_unmap_range,
    .invalidate_range = iree_hal_amd_xdna_buffer_invalidate_range,
    .flush_range = iree_hal_amd_xdna_buffer_flush_range,
};

//===----------------------------------------------------------------------===//
// Allocation facade
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amd_xdna_allocator_t {
  // HAL allocator resource header.
  iree_hal_resource_t resource;
  // Borrowed device identifying the origin of allocated buffers.
  iree_hal_device_t* device;
  // Borrowed context dominating all allocation lifetimes.
  iree_hal_amd_xdna_context_t* context;
  // Allocator owning the facade and each buffer wrapper.
  iree_allocator_t host_allocator;
  // Guaranteed memory properties of the selected UMA source.
  iree_hal_memory_type_t memory_type;
} iree_hal_amd_xdna_allocator_t;

static void iree_hal_amd_xdna_allocator_destroy(iree_hal_allocator_t* base) {
  iree_hal_amd_xdna_allocator_t* allocator =
      (iree_hal_amd_xdna_allocator_t*)base;
  iree_allocator_free(allocator->host_allocator, allocator);
}

static iree_allocator_t iree_hal_amd_xdna_allocator_host_allocator(
    const iree_hal_allocator_t* base) {
  return ((const iree_hal_amd_xdna_allocator_t*)base)->host_allocator;
}

static iree_status_t iree_hal_amd_xdna_allocator_trim(
    iree_hal_allocator_t* base) {
  return iree_ok_status();
}

static void iree_hal_amd_xdna_allocator_query_statistics(
    iree_hal_allocator_t* base,
    iree_hal_allocator_statistics_t* out_statistics) {
  *out_statistics = (iree_hal_allocator_statistics_t){0};
}

static iree_status_t iree_hal_amd_xdna_allocator_query_memory_heaps(
    iree_hal_allocator_t* base, iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* heaps, iree_host_size_t* out_count) {
  iree_hal_amd_xdna_allocator_t* allocator =
      (iree_hal_amd_xdna_allocator_t*)base;
  if (out_count) {
    *out_count = 1;
  }
  if (!heaps) {
    return iree_ok_status();
  }
  if (capacity < 1) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
  }
  *heaps = (iree_hal_allocator_memory_heap_t){
      .type = allocator->memory_type,
      .allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                       IREE_HAL_BUFFER_USAGE_STORAGE |
                       IREE_HAL_BUFFER_USAGE_MAPPING,
      .max_allocation_size = allocator->context->data_source.profile.allocation
                                 .maximum_byte_length,
      .min_alignment =
          allocator->context->data_source.profile.allocation.minimum_alignment,
  };
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_amd_xdna_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base, iree_hal_buffer_params_t* params,
    iree_device_size_t* allocation_size) {
  iree_hal_amd_xdna_allocator_t* allocator =
      (iree_hal_amd_xdna_allocator_t*)base;
  if (!iree_all_bits_set(allocator->memory_type,
                         params->type & ~IREE_HAL_MEMORY_TYPE_OPTIMAL) ||
      iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_SHARING_EXPORT) ||
      params->min_alignment > allocator->context->data_source.profile.allocation
                                  .maximum_alignment ||
      *allocation_size > allocator->context->data_source.profile.allocation
                             .maximum_byte_length) {
    return IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }
  params->type = allocator->memory_type;
  params->usage |= IREE_HAL_BUFFER_USAGE_MAPPING;
  return IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
         IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER |
         IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
}

static iree_status_t iree_hal_amd_xdna_allocator_allocate_buffer(
    iree_hal_allocator_t* base, const iree_hal_buffer_params_t* params,
    iree_device_size_t allocation_size, iree_hal_buffer_t** out_buffer) {
  iree_hal_amd_xdna_allocator_t* allocator =
      (iree_hal_amd_xdna_allocator_t*)base;
  iree_hal_buffer_params_t actual_params = *params;
  if (!iree_any_bit_set(iree_hal_amd_xdna_allocator_query_buffer_compatibility(
                            base, &actual_params, &allocation_size),
                        IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "allocation requirements exceed the XDNA memory profile");
  }
  iree_hal_amd_xdna_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator->host_allocator,
                                             sizeof(*buffer), (void**)&buffer));
  buffer->context = allocator->context;
  buffer->host_allocator = allocator->host_allocator;
  iree_status_t status = iree_hal_amd_xdna_memory_allocate(
      buffer->context, &buffer->context->data_source, allocation_size,
      params->min_alignment, &buffer->memory);
  if (iree_status_is_ok(status)) {
    if (buffer->memory.host_cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK) {
      actual_params.type |= IREE_HAL_MEMORY_TYPE_HOST_CACHED;
    }
    iree_hal_buffer_initialize(
        (iree_hal_buffer_placement_t){
            .device = allocator->device,
            .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        },
        &buffer->base, allocation_size, 0, allocation_size, actual_params.type,
        actual_params.access, actual_params.usage,
        &iree_hal_amd_xdna_buffer_vtable, &buffer->base);
    buffer->bindings[IREE_HAL_AMD_XDNA_BUFFER_BINDING_SHIM_DMA].device_address =
        buffer->memory.device_address;
    buffer->bindings[IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST].host_pointer =
        buffer->memory.contents.data;
    buffer->base.memory.bindings = buffer->bindings;
    buffer->base.host_binding_index = IREE_HAL_AMD_XDNA_BUFFER_BINDING_HOST;
    *out_buffer = &buffer->base;
  } else {
    iree_allocator_free(allocator->host_allocator, buffer);
  }
  return status;
}

static void iree_hal_amd_xdna_allocator_deallocate_buffer(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* buffer) {
  iree_hal_buffer_destroy(buffer);
}

static iree_status_t iree_hal_amd_xdna_allocator_import_buffer(
    iree_hal_allocator_t* allocator, const iree_hal_buffer_params_t* params,
    iree_hal_external_buffer_t* external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA buffer import requires a prepared native view");
}

static bool iree_hal_amd_xdna_allocator_supports_virtual_memory(
    iree_hal_allocator_t* allocator) {
  return false;
}

static iree_status_t
iree_hal_amd_xdna_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params,
    iree_device_size_t* IREE_RESTRICT out_minimum_page_size,
    iree_device_size_t* IREE_RESTRICT out_recommended_page_size) {
  *out_minimum_page_size = 0;
  *out_recommended_page_size = 0;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA allocator does not support virtual memory");
}

static iree_status_t iree_hal_amd_xdna_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_device_size_t size,
    iree_hal_buffer_t** IREE_RESTRICT out_virtual_buffer) {
  *out_virtual_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA allocator does not support virtual memory");
}

static iree_status_t iree_hal_amd_xdna_allocator_virtual_memory_release(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA allocator does not support virtual memory");
}

static iree_status_t iree_hal_amd_xdna_allocator_physical_memory_allocate(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params, iree_device_size_t size,
    iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** IREE_RESTRICT out_physical_memory) {
  *out_physical_memory = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA allocator does not support virtual memory");
}

static iree_status_t iree_hal_amd_xdna_allocator_physical_memory_free(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA allocator does not support virtual memory");
}

static iree_status_t iree_hal_amd_xdna_allocator_virtual_memory_map(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA allocator does not support virtual memory");
}

static iree_status_t iree_hal_amd_xdna_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA allocator does not support virtual memory");
}

static iree_status_t iree_hal_amd_xdna_allocator_virtual_memory_protect(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA allocator does not support virtual memory");
}

static iree_status_t iree_hal_amd_xdna_allocator_virtual_memory_advise(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_advice_t advice) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA allocator does not support virtual memory");
}

static const iree_hal_allocator_vtable_t iree_hal_amd_xdna_allocator_vtable = {
    .destroy = iree_hal_amd_xdna_allocator_destroy,
    .host_allocator = iree_hal_amd_xdna_allocator_host_allocator,
    .trim = iree_hal_amd_xdna_allocator_trim,
    .query_statistics = iree_hal_amd_xdna_allocator_query_statistics,
    .query_memory_heaps = iree_hal_amd_xdna_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_amd_xdna_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_amd_xdna_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_amd_xdna_allocator_deallocate_buffer,
    .import_buffer = iree_hal_amd_xdna_allocator_import_buffer,
    .supports_virtual_memory =
        iree_hal_amd_xdna_allocator_supports_virtual_memory,
    .virtual_memory_query_granularity =
        iree_hal_amd_xdna_allocator_virtual_memory_query_granularity,
    .virtual_memory_reserve =
        iree_hal_amd_xdna_allocator_virtual_memory_reserve,
    .virtual_memory_release =
        iree_hal_amd_xdna_allocator_virtual_memory_release,
    .physical_memory_allocate =
        iree_hal_amd_xdna_allocator_physical_memory_allocate,
    .physical_memory_free = iree_hal_amd_xdna_allocator_physical_memory_free,
    .virtual_memory_map = iree_hal_amd_xdna_allocator_virtual_memory_map,
    .virtual_memory_unmap = iree_hal_amd_xdna_allocator_virtual_memory_unmap,
    .virtual_memory_protect =
        iree_hal_amd_xdna_allocator_virtual_memory_protect,
    .virtual_memory_advise = iree_hal_amd_xdna_allocator_virtual_memory_advise,
};

iree_status_t iree_hal_amd_xdna_allocator_create(
    iree_hal_device_t* device, iree_hal_amd_xdna_context_t* context,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  *out_allocator = NULL;
  iree_hal_amd_xdna_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                             (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_amd_xdna_allocator_vtable,
                               &allocator->resource);
  allocator->device = device;
  allocator->context = context;
  allocator->host_allocator = host_allocator;
  // XDNA uses system DRAM as its ordinary storage; both processors access the
  // same backing rather than a separate device-local heap.
  allocator->memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  if (iree_any_bit_set(context->data_source.capabilities.guaranteed_flags,
                       AMDF_MEMORY_FLAG_HOST_COHERENT)) {
    allocator->memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  }
  *out_allocator = (iree_hal_allocator_t*)allocator;
  return iree_ok_status();
}
