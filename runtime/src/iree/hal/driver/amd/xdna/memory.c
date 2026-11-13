// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/memory.h"

#include "iree/hal/driver/amd/status.h"

void iree_hal_amd_xdna_memory_deinitialize(iree_hal_amd_xdna_context_t* context,
                                           iree_hal_amd_xdna_memory_t* memory) {
  iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                          &memory->storage);
  memory->device_address = 0;
}

iree_status_t iree_hal_amd_xdna_memory_allocate(
    iree_hal_amd_xdna_context_t* context,
    const iree_hal_amd_xdna_memory_source_t* source, uint64_t byte_length,
    uint64_t alignment, iree_hal_amd_xdna_memory_t* out_memory) {
  *out_memory = (iree_hal_amd_xdna_memory_t){0};
  const iree_hal_amd_fabric_memory_options_t options = {
      .scope = source->scope,
      .profile = source->profile,
      .access_count = 1,
      .accesses = &source->access,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  iree_hal_amd_xdna_memory_t memory = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, &options, byte_length, alignment, context->event_sink,
      &memory.storage, NULL));
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(memory.storage.handle, 0,
                                         source->address_kind,
                                         &memory.device_address),
      "memory_query_address");
  if (iree_status_is_ok(status)) {
    *out_memory = memory;
  } else {
    iree_hal_amd_xdna_memory_deinitialize(context, &memory);
  }
  return status;
}

iree_status_t iree_hal_amd_xdna_buffer_resolve_binding_slot(
    const iree_hal_queue_family_t* family, const iree_hal_buffer_t* buffer,
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

  return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                          "XDNA dispatch requires a prepared memory contract");
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

//===----------------------------------------------------------------------===//
// Allocation facade
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amd_xdna_allocator_t {
  // HAL allocator resource header.
  iree_hal_resource_t resource;
  // Borrowed context dominating all allocation lifetimes.
  iree_hal_amd_xdna_context_t* context;
  // Allocator owning the facade and each buffer wrapper.
  iree_allocator_t host_allocator;
  // Guaranteed memory properties of the selected UMA source.
  iree_hal_memory_type_t memory_type;
} iree_hal_amd_xdna_allocator_t;

static void iree_hal_amd_xdna_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base) {
  iree_hal_amd_xdna_allocator_t* allocator =
      (iree_hal_amd_xdna_allocator_t*)base;
  iree_allocator_free(allocator->host_allocator, allocator);
}

static iree_allocator_t iree_hal_amd_xdna_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base) {
  return ((const iree_hal_amd_xdna_allocator_t*)base)->host_allocator;
}

static void iree_hal_amd_xdna_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  *out_statistics = (iree_hal_allocator_statistics_t){0};
}

static iree_status_t iree_hal_amd_xdna_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base, iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
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

static const iree_hal_allocator_vtable_t iree_hal_amd_xdna_allocator_vtable = {
    .destroy = iree_hal_amd_xdna_allocator_destroy,
    .host_allocator = iree_hal_amd_xdna_allocator_host_allocator,
    .query_statistics = iree_hal_amd_xdna_allocator_query_statistics,
    .query_memory_heaps = iree_hal_amd_xdna_allocator_query_memory_heaps,
};

iree_status_t iree_hal_amd_xdna_allocator_create(
    iree_hal_amd_xdna_context_t* context, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  *out_allocator = NULL;
  iree_hal_amd_xdna_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                             (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_amd_xdna_allocator_vtable,
                               &allocator->resource);
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
