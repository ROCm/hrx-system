// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/host_memory.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/host_mapping.h"
#include "libamdf/src/memory_resource.h"
#include "libamdf/src/platform/host_memory.h"

static amdf_status_t amdf_host_memory_mapping_cache_control(
    amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t byte_offset, uint64_t byte_length) {
  (void)operation;
  return amdf_platform_host_memory_cache_control(
      (uint8_t*)mapping->info.pointer + byte_offset, byte_length,
      mapping->info.cache_line_size);
}

static amdf_status_t amdf_host_memory_mapping_destroy_native(
    amdf_host_mapping_t* mapping) {
  // CPU views borrow the persistent backing range without another native map.
  (void)mapping;
  return AMDF_STATUS_OK;
}

static const amdf_host_mapping_vtable_t amdf_host_memory_mapping_vtable = {
    .cache_control = amdf_host_memory_mapping_cache_control,
    .destroy_native = amdf_host_memory_mapping_destroy_native,
};

static amdf_status_t amdf_host_memory_map(
    amdf_memory_t* memory, uint32_t access_ordinal,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info, amdf_host_mapping_t** out_mapping) {
  uint32_t line_size = 0;
  amdf_status_t status =
      amdf_platform_host_memory_query_cache_line_size(&line_size);
  if (!amdf_status_is_ok(status)) return status;
  amdf_host_mapping_t* mapping = NULL;
  status = amdf_calloc(memory->host_allocator, sizeof(*mapping),
                       amdf_alignof(amdf_host_mapping_t), (void**)&mapping);
  if (!amdf_status_is_ok(status)) return status;
  status = amdf_host_mapping_initialize(
      mapping, &amdf_host_memory_mapping_vtable, memory);
  if (amdf_status_is_ok(status)) {
    const amdf_cache_transition_t flush = {
        .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
        .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
        .host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH,
        .host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
        .host_fence_before = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
        .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
        .range_granularity = line_size,
    };
    mapping->info = (amdf_host_mapping_info_t){
        .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
        .structure_size = sizeof(mapping->info),
        .flags = capabilities->supported_access,
        .cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK,
        .pointer = (uint8_t*)memory->accesses[access_ordinal].native +
                   map_info->byte_offset,
        .memory_byte_offset = map_info->byte_offset,
        .byte_length = map_info->byte_length,
        .byte_offset_granularity = capabilities->byte_offset_granularity,
        .byte_length_granularity = capabilities->byte_length_granularity,
        .cache_line_size = line_size,
        .flush = flush,
        .invalidate = flush,
    };
    mapping->info.invalidate.host_operation =
        AMDF_HOST_CACHE_OPERATION_INVALIDATE;
    *out_mapping = mapping;
  } else {
    amdf_free(memory->host_allocator, mapping);
  }
  return status;
}

static amdf_status_t amdf_host_memory_destroy_native(amdf_memory_t* memory,
                                                     uint32_t access_ordinal) {
  const amdf_status_t status = amdf_platform_host_memory_free(
      memory->accesses[access_ordinal].native,
      memory->info.native_allocation_byte_length);
  if (amdf_status_is_ok(status)) memory->accesses[access_ordinal].native = NULL;
  return status;
}

static amdf_status_t amdf_host_memory_unregister(amdf_memory_t* memory,
                                                 uint32_t access_ordinal) {
  memory->accesses[access_ordinal].native = NULL;
  return AMDF_STATUS_OK;
}

static void amdf_host_memory_abandon_native(amdf_memory_t* memory,
                                            uint32_t access_ordinal) {
  // The native slot is the backing address itself, not separately owned
  // metadata.
  memory->accesses[access_ordinal].native = NULL;
}

static const amdf_memory_vtable_t amdf_host_memory_vtable = {
    .map = amdf_host_memory_map,
    .destroy_native = amdf_host_memory_destroy_native,
    .abandon_native = amdf_host_memory_abandon_native,
};

static const amdf_memory_vtable_t amdf_registered_host_memory_vtable = {
    .map = amdf_host_memory_map,
    .destroy_native = amdf_host_memory_unregister,
    .abandon_native = amdf_host_memory_abandon_native,
};

amdf_status_t amdf_host_memory_prepare(
    amdf_memory_t* memory, const amdf_memory_native_profile_t* profile,
    const amdf_memory_create_info_t* create_info,
    amdf_memory_info_t* out_info) {
  const bool registration =
      (profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0;
  const uint64_t granularity =
      registration ? profile->registration.native_byte_length_granularity
                   : profile->allocation.native_byte_length_granularity;
  const uint64_t native_length =
      (create_info->byte_length + granularity - 1) & ~(granularity - 1);
  memory->accesses[0].vtable = registration
                                   ? &amdf_registered_host_memory_vtable
                                   : &amdf_host_memory_vtable;
  amdf_status_t status = AMDF_STATUS_OK;
  if (registration) {
    memory->accesses[0].native = create_info->registered_host_pointer;
  } else {
    status = amdf_platform_host_memory_allocate(native_length,
                                                &memory->accesses[0].native);
  }
  if (amdf_status_is_ok(status)) {
    const uint64_t address = (uintptr_t)memory->accesses[0].native;
    *out_info = (amdf_memory_info_t){
        .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
        .structure_size = sizeof(*out_info),
        .memory_profile_ordinal = profile->ordinal,
        .memory_class = AMDF_MEMORY_CLASS_SYSTEM,
        .flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .byte_length = create_info->byte_length,
        .alignment = registration ? address & (~address + 1) : granularity,
        .native_allocation_byte_length = native_length,
        .native_allocation_granularity = granularity,
    };
  }
  return status;
}
