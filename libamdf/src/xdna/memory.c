// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/memory.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/host_mapping.h"
#include "libamdf/src/memory.h"
#include "libamdf/src/xdna/device.h"
#include "libamdf/src/xdna/umd/memory.h"

typedef struct amdf_xdna_host_mapping_t {
  // Generic host mapping state shared by every engine implementation.
  amdf_host_mapping_t base;
  // Exact native host mapping state.
  amdf_xdna_umd_host_mapping_t* umd;
} amdf_xdna_host_mapping_t;

_Static_assert(offsetof(amdf_xdna_host_mapping_t, base) == 0,
               "XDNA mapping base must be the first field");

static amdf_status_t amdf_xdna_memory_export(
    amdf_memory_t* memory, uint32_t access_ordinal,
    const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  return amdf_xdna_umd_memory_export(memory->accesses[access_ordinal].native,
                                     export_info, out_value);
}

static amdf_status_t amdf_xdna_memory_describe_site(
    amdf_memory_t* memory, uint32_t access_ordinal,
    uint32_t queue_family_ordinal,
    amdf_memory_site_description_t* out_description) {
  amdf_queue_family_info_t queue_family_info = {
      .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
      .structure_size = sizeof(queue_family_info),
  };
  const amdf_status_t status = amdf_endpoint_query_queue_family_info(
      memory->accesses[access_ordinal].device->endpoint, queue_family_ordinal,
      &queue_family_info);
  if (!amdf_status_is_ok(status)) return status;
  const amdf_memory_site_query_t query = {
      .access_info = &memory->accesses[access_ordinal].info,
      .queue_family_info = &queue_family_info,
  };
  return amdf_xdna_umd_memory_describe_site(
      memory->accesses[access_ordinal].native, &query, out_description);
}

static amdf_status_t amdf_xdna_host_mapping_cache_control(
    amdf_host_mapping_t* base_mapping, amdf_host_cache_operation_t operation,
    uint64_t byte_offset, uint64_t byte_length) {
  amdf_xdna_host_mapping_t* mapping = (amdf_xdna_host_mapping_t*)base_mapping;
  return amdf_xdna_umd_host_mapping_cache_control(mapping->umd, operation,
                                                  byte_offset, byte_length);
}

static amdf_status_t amdf_xdna_host_mapping_destroy_native(
    amdf_host_mapping_t* base_mapping) {
  amdf_xdna_host_mapping_t* mapping = (amdf_xdna_host_mapping_t*)base_mapping;
  const amdf_status_t status = amdf_xdna_umd_host_mapping_destroy(mapping->umd);
  if (amdf_status_is_ok(status)) {
    mapping->umd = NULL;
  }
  return status;
}

static const amdf_host_mapping_vtable_t amdf_xdna_host_mapping_vtable = {
    .cache_control = amdf_xdna_host_mapping_cache_control,
    .destroy_native = amdf_xdna_host_mapping_destroy_native,
};

static amdf_status_t amdf_xdna_memory_map(
    amdf_memory_t* memory, uint32_t access_ordinal,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info, amdf_host_mapping_t** out_mapping) {
  const amdf_allocator_t host_allocator = amdf_memory_host_allocator(memory);
  amdf_xdna_host_mapping_t* mapping = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*mapping),
                  amdf_alignof(amdf_xdna_host_mapping_t), (void**)&mapping);
  if (!amdf_status_is_ok(status)) return status;
  status = amdf_host_mapping_initialize(&mapping->base,
                                        &amdf_xdna_host_mapping_vtable, memory);

  amdf_xdna_umd_host_mapping_result_t result = {0};
  if (amdf_status_is_ok(status)) {
    status = amdf_xdna_umd_memory_map(memory->accesses[access_ordinal].native,
                                      capabilities, map_info, &mapping->umd,
                                      &result);
  }
  if (amdf_status_is_ok(status)) {
    amdf_assert((result.flags & map_info->flags) == map_info->flags &&
                (result.flags & ~capabilities->supported_access) == 0 &&
                result.pointer != NULL &&
                result.byte_length == map_info->byte_length &&
                "XDNA host mapping must achieve the selected profile request");
    mapping->base.info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    mapping->base.info.structure_size = sizeof(mapping->base.info);
    mapping->base.info.flags = result.flags;
    mapping->base.info.cacheability = result.cacheability;
    mapping->base.info.pointer = result.pointer;
    mapping->base.info.memory_byte_offset = map_info->byte_offset;
    mapping->base.info.byte_length = result.byte_length;
    mapping->base.info.byte_offset_granularity =
        capabilities->byte_offset_granularity;
    mapping->base.info.byte_length_granularity =
        capabilities->byte_length_granularity;
    mapping->base.info.cache_line_size = result.cache_line_size;
    mapping->base.info.flush = result.flush;
    mapping->base.info.invalidate = result.invalidate;
    *out_mapping = &mapping->base;
  } else {
    if (mapping->base.memory != NULL) {
      amdf_host_mapping_deinitialize(&mapping->base);
    }
    amdf_free(host_allocator, mapping);
  }
  return status;
}

static amdf_status_t amdf_xdna_memory_destroy_native(amdf_memory_t* memory,
                                                     uint32_t access_ordinal) {
  const amdf_status_t status =
      amdf_xdna_umd_memory_destroy(memory->accesses[access_ordinal].native);
  if (amdf_status_is_ok(status)) {
    memory->accesses[access_ordinal].native = NULL;
  }
  return status;
}

static void amdf_xdna_memory_abandon_native(amdf_memory_t* memory,
                                            uint32_t access_ordinal) {
  amdf_xdna_umd_memory_abandon(memory->accesses[access_ordinal].native);
  memory->accesses[access_ordinal].native = NULL;
}

static const amdf_memory_vtable_t amdf_xdna_memory_vtable = {
    .export_external = amdf_xdna_memory_export,
    .describe_site = amdf_xdna_memory_describe_site,
    .map = amdf_xdna_memory_map,
    .destroy_native = amdf_xdna_memory_destroy_native,
    .abandon_native = amdf_xdna_memory_abandon_native,
};

amdf_status_t amdf_xdna_device_query_memory_profile(
    amdf_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  return amdf_xdna_umd_device_query_memory_profile(
      amdf_xdna_device_get_umd(device), memory_profile_ordinal, out_profile);
}

static void amdf_xdna_memory_set_info(
    amdf_memory_t* memory, uint32_t access_ordinal, amdf_device_t* device,
    const amdf_memory_native_profile_t* profile,
    amdf_memory_access_t device_access, amdf_xdna_umd_memory_result_t result,
    amdf_memory_info_t* out_info) {
  out_info->type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  out_info->structure_size = sizeof(*out_info);
  out_info->memory_profile_ordinal = profile->ordinal;
  out_info->memory_class = profile->memory_class;
  memory->accesses[access_ordinal].info.access = device_access;
  memory->accesses[access_ordinal].info.atomic_operations_32 =
      result.atomic_operations_32;
  memory->accesses[access_ordinal].info.atomic_operations_64 =
      result.atomic_operations_64;
  memory->accesses[access_ordinal].info.address_domain_ordinal =
      profile->device_address.address_domain_ordinal;
  memory->accesses[access_ordinal].info.device_id =
      amdf_xdna_device_get_info(device)->id;
  out_info->flags = result.flags & AMDF_MEMORY_BACKING_FLAGS;
  memory->accesses[access_ordinal].info.flags =
      result.flags & AMDF_MEMORY_ACCESS_FLAGS;
  out_info->source_byte_offset = result.source_byte_offset;
  out_info->byte_length = result.byte_length;
  out_info->alignment = result.alignment;
  out_info->native_allocation_byte_length =
      result.native_allocation_byte_length;
  out_info->native_allocation_granularity =
      result.native_allocation_granularity;
  out_info->physical_backing_id = result.physical_backing_id;
  memory->accesses[access_ordinal]
      .addresses[AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE] = result.device_address;
  memory->accesses[access_ordinal].info.address_kinds = result.address_kinds;
  memory->accesses[access_ordinal].addresses[AMDF_MEMORY_ADDRESS_XDNA_DMA] =
      result.dma_address;
  memory->accesses[access_ordinal].info.reset_epoch =
      amdf_xdna_device_query_reset_epoch(device);
}

amdf_status_t amdf_xdna_memory_prepare(
    amdf_memory_t* memory, const amdf_memory_native_group_t* group,
    const amdf_memory_native_create_info_t* create_info,
    amdf_memory_info_t* out_info) {
  const uint32_t access_ordinal = group->access_ordinals[0];
  const amdf_memory_native_profile_t* profile =
      &group->profiles[access_ordinal];
  memory->accesses[access_ordinal].vtable = &amdf_xdna_memory_vtable;
  amdf_xdna_umd_memory_t* native = NULL;
  amdf_xdna_umd_memory_result_t result = {0};
  const amdf_status_t status = amdf_xdna_umd_memory_prepare(
      amdf_xdna_device_get_umd(memory->accesses[access_ordinal].device),
      profile, create_info, &native, &result);
  memory->accesses[access_ordinal].native = native;
  if (amdf_status_is_ok(status)) {
    amdf_xdna_memory_set_info(memory, access_ordinal,
                              memory->accesses[access_ordinal].device, profile,
                              create_info->device_access, result, out_info);
  }
  return status;
}

amdf_status_t amdf_xdna_memory_prepare_import(
    amdf_memory_t* memory, uint32_t access_ordinal,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_memory_info_t* out_info) {
  memory->accesses[access_ordinal].vtable = &amdf_xdna_memory_vtable;
  amdf_xdna_umd_memory_t* native = NULL;
  amdf_xdna_umd_memory_result_t result = {0};
  const amdf_status_t status = amdf_xdna_umd_memory_prepare_import(
      amdf_xdna_device_get_umd(memory->accesses[access_ordinal].device),
      profile, import_info, external_memory, &native, &result);
  memory->accesses[access_ordinal].native = native;
  if (amdf_status_is_ok(status)) {
    amdf_xdna_memory_set_info(memory, access_ordinal,
                              memory->accesses[access_ordinal].device, profile,
                              import_info->device_access, result, out_info);
  }
  return status;
}

static void amdf_xdna_memory_scope_query_profile(
    amdf_memory_scope_t* scope, amdf_memory_native_profile_t* out_profile) {
  amdf_xdna_umd_context_query_memory_profile(scope->owner.private_storage.owner,
                                             out_profile);
}

static amdf_status_t amdf_xdna_memory_scope_prepare(
    amdf_memory_scope_t* scope, amdf_memory_t* memory,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_memory_info_t* out_info) {
  memory->accesses[0].vtable = &amdf_xdna_memory_vtable;
  amdf_xdna_umd_memory_t* native = NULL;
  amdf_xdna_umd_memory_result_t result = {0};
  const amdf_status_t status = amdf_xdna_umd_memory_prepare_private(
      scope->owner.private_storage.owner, profile, create_info, &native,
      &result);
  memory->accesses[0].native = native;
  if (amdf_status_is_ok(status)) {
    amdf_xdna_memory_set_info(memory, 0, scope->owner.private_storage.device,
                              profile, create_info->device_access, result,
                              out_info);
  }
  return status;
}

static const amdf_memory_scope_vtable_t amdf_xdna_memory_scope_vtable = {
    .query_profile = amdf_xdna_memory_scope_query_profile,
    .prepare = amdf_xdna_memory_scope_prepare,
};

void amdf_xdna_memory_scope_initialize(amdf_device_t* device,
                                       amdf_xdna_umd_context_t* context,
                                       amdf_memory_scope_t* scope) {
  *scope = (amdf_memory_scope_t){
      .kind = AMDF_MEMORY_SCOPE_KIND_PRIVATE,
      .owner.private_storage =
          {
              .device = device,
              .vtable = &amdf_xdna_memory_scope_vtable,
              .owner = context,
          },
  };
}
