// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/host_mapping.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/memory_resource.h"
#include "libamdf/src/structure.h"

amdf_status_t amdf_host_mapping_initialize(
    amdf_host_mapping_t* mapping, const amdf_host_mapping_vtable_t* vtable,
    amdf_memory_t* memory) {
  const amdf_status_t status = amdf_memory_register_child(memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  mapping->host_allocator = amdf_memory_host_allocator(memory);
  mapping->vtable = vtable;
  mapping->memory = memory;
  return AMDF_STATUS_OK;
}

void amdf_host_mapping_deinitialize(amdf_host_mapping_t* mapping) {
  amdf_memory_unregister_child(mapping->memory);
  mapping->memory = NULL;
}

amdf_status_t AMDF_CALL amdf_host_mapping_query_info(
    amdf_host_mapping_t* mapping, amdf_host_mapping_info_t* out_info) {
  if (mapping == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      (uint32_t)sizeof(amdf_host_mapping_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = mapping->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_host_mapping_cache_control(
    amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t byte_offset, uint64_t byte_length) {
  if (mapping == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (operation != AMDF_HOST_CACHE_OPERATION_FLUSH &&
      operation != AMDF_HOST_CACHE_OPERATION_INVALIDATE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (byte_offset > mapping->info.byte_length ||
      byte_length > mapping->info.byte_length - byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (byte_length == 0) {
    return AMDF_STATUS_OK;
  }
  if ((operation == AMDF_HOST_CACHE_OPERATION_FLUSH &&
       (mapping->info.flags & AMDF_MEMORY_MAP_FLAG_WRITE) == 0) ||
      (operation == AMDF_HOST_CACHE_OPERATION_INVALIDATE &&
       (mapping->info.flags & AMDF_MEMORY_MAP_FLAG_READ) == 0)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  const amdf_cache_transition_t* transition =
      operation == AMDF_HOST_CACHE_OPERATION_FLUSH ? &mapping->info.flush
                                                   : &mapping->info.invalidate;
  if (transition->kind == AMDF_CACHE_TRANSITION_KIND_UNKNOWN) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (transition->kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
    return AMDF_STATUS_OK;
  }
  return mapping->vtable->cache_control(mapping, operation, byte_offset,
                                        byte_length);
}

amdf_status_t AMDF_CALL
amdf_host_mapping_destroy(amdf_host_mapping_t* mapping) {
  if (mapping == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = mapping->vtable->destroy_native(mapping);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = mapping->host_allocator;
    amdf_host_mapping_deinitialize(mapping);
    amdf_free(host_allocator, mapping);
  }
  return status;
}
