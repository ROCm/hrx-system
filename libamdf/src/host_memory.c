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

amdf_status_t amdf_host_memory_query_description(
    amdf_memory_host_description_t* out_description) {
  amdf_cache_transition_t transition;
  const amdf_status_t status =
      amdf_platform_host_memory_query_cache_transition(&transition);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  *out_description = (amdf_memory_host_description_t){
      .cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK,
      .cache_line_size = (uint32_t)transition.range_granularity,
      .flush = transition,
      .invalidate = transition,
  };
  out_description->invalidate.host_operation =
      AMDF_HOST_CACHE_OPERATION_INVALIDATE;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_host_memory_mapping_cache_control(
    amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t byte_offset, uint64_t byte_length) {
  (void)operation;
  return amdf_platform_host_memory_cache_control(
      (uint8_t*)mapping->info.pointer + byte_offset, byte_length,
      mapping->info.cache_line_size);
}

static void amdf_host_memory_mapping_destroy_native(
    amdf_host_mapping_t* mapping) {
  // CPU views borrow the persistent backing range without another native map.
  (void)mapping;
}

static const amdf_host_mapping_vtable_t amdf_host_memory_mapping_vtable = {
    .cache_control = amdf_host_memory_mapping_cache_control,
    .destroy_native = amdf_host_memory_mapping_destroy_native,
};

amdf_status_t amdf_host_memory_create_view(
    amdf_allocator_t host_allocator, amdf_memory_t* memory, void* base_pointer,
    const amdf_memory_map_info_t* map_info, amdf_host_mapping_t** out_mapping) {
  amdf_memory_host_description_t description;
  amdf_status_t status = amdf_host_memory_query_description(&description);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_host_mapping_t* mapping = NULL;
  status = amdf_calloc(host_allocator, sizeof(*mapping),
                       amdf_alignof(amdf_host_mapping_t), (void**)&mapping);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  mapping->host_allocator = host_allocator;
  mapping->vtable = &amdf_host_memory_mapping_vtable;
  mapping->memory = memory;
  mapping->info = (amdf_host_mapping_info_t){
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(mapping->info),
      .flags = map_info->flags,
      .cacheability = description.cacheability,
      .pointer = (uint8_t*)base_pointer + map_info->byte_offset,
      .memory_byte_offset = map_info->byte_offset,
      .byte_length = map_info->byte_length,
      .byte_offset_granularity = 1,
      .byte_length_granularity = 1,
      .cache_line_size = description.cache_line_size,
      .flush = description.flush,
      .invalidate = description.invalidate,
  };
  *out_mapping = mapping;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_host_memory_map(
    amdf_memory_t* memory, uint32_t access_ordinal,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info, amdf_host_mapping_t** out_mapping) {
  void* pointer = memory->accesses[access_ordinal].native;
  if ((memory->info.flags & AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE) != 0) {
    pointer = ((amdf_platform_host_backing_t*)pointer)->pointer;
  }
  amdf_memory_map_info_t view_info = *map_info;
  view_info.flags = capabilities->supported_access;
  return amdf_host_memory_create_view(memory->host_allocator, memory, pointer,
                                      &view_info, out_mapping);
}

static uintptr_t amdf_host_memory_mapping_source(amdf_memory_t* memory,
                                                 uint32_t access_ordinal) {
  const amdf_platform_host_backing_t* backing =
      memory->accesses[access_ordinal].native;
  return backing->handle;
}

static amdf_status_t amdf_host_backing_destroy_native(amdf_memory_t* memory,
                                                      uint32_t ordinal) {
  amdf_platform_host_backing_t* backing = memory->accesses[ordinal].native;
  const amdf_status_t status = amdf_platform_host_backing_destroy(
      backing, memory->info.native_allocation_byte_length);
  amdf_free(memory->host_allocator, backing);
  memory->accesses[ordinal].native = NULL;
  return status;
}

static void amdf_host_backing_abandon_native(amdf_memory_t* memory,
                                             uint32_t ordinal) {
  amdf_free(memory->host_allocator, memory->accesses[ordinal].native);
  memory->accesses[ordinal].native = NULL;
}

static const amdf_memory_vtable_t amdf_host_backing_vtable = {
    .map = amdf_host_memory_map,
    .host_mapping_source = amdf_host_memory_mapping_source,
    .destroy_native = amdf_host_backing_destroy_native,
    .abandon_native = amdf_host_backing_abandon_native,
};

static amdf_status_t amdf_host_memory_destroy_native(amdf_memory_t* memory,
                                                     uint32_t access_ordinal) {
  const amdf_status_t status = amdf_platform_host_memory_free(
      memory->accesses[access_ordinal].native,
      memory->info.native_allocation_byte_length);
  if (amdf_status_is_ok(status)) {
    memory->accesses[access_ordinal].native = NULL;
  }
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

void* amdf_host_memory_pointer(const amdf_memory_t* memory) {
  void* native = memory->accesses[memory->backing_access_ordinal].native;
  return (memory->info.flags & AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE) != 0
             ? ((amdf_platform_host_backing_t*)native)->pointer
             : native;
}

amdf_status_t amdf_host_memory_export(
    amdf_memory_t* memory, const amdf_platform_host_export_t* service,
    bool first_export, amdf_external_memory_t* out_memory) {
  const amdf_platform_host_backing_t* backing =
      memory->accesses[memory->backing_access_ordinal].native;
  amdf_external_memory_t exported = {0};
  amdf_status_t status = amdf_platform_host_export_memory(
      service, backing, memory->info.byte_length, &exported);
  if (amdf_status_is_ok(status) && first_export) {
    // Export has pinned/populated every page. Retire CPU initialization once
    // before any attachment can install a device-writable alias.
    amdf_memory_host_description_t description;
    status = amdf_host_memory_query_description(&description);
    if (amdf_status_is_ok(status)) {
      status = amdf_platform_host_memory_cache_control(
          backing->pointer, memory->info.byte_length,
          description.cache_line_size);
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_memory = exported;
  } else if (exported.release != NULL) {
    exported.release(exported.release_user_data, exported.type,
                     exported.payload);
  }
  return status;
}

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
  const bool mapping_source =
      ((create_info->required_flags | profile->guaranteed_flags) &
       AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE) != 0;
  memory->accesses[memory->backing_access_ordinal].vtable =
      registration ? &amdf_registered_host_memory_vtable
                   : &amdf_host_memory_vtable;
  amdf_status_t status = AMDF_STATUS_OK;
  if (registration) {
    memory->accesses[memory->backing_access_ordinal].native =
        create_info->registered_host_pointer;
  } else if (mapping_source) {
    amdf_platform_host_backing_t* backing = NULL;
    status = amdf_calloc(memory->host_allocator, sizeof(*backing),
                         amdf_alignof(amdf_platform_host_backing_t),
                         (void**)&backing);
    if (amdf_status_is_ok(status)) {
      status = amdf_platform_host_backing_create(native_length, backing);
      if (amdf_status_is_ok(status)) {
        memory->accesses[memory->backing_access_ordinal].vtable =
            &amdf_host_backing_vtable;
        memory->accesses[memory->backing_access_ordinal].native = backing;
      } else {
        amdf_free(memory->host_allocator, backing);
      }
    }
  } else {
    status = amdf_platform_host_memory_allocate(
        native_length,
        &memory->accesses[memory->backing_access_ordinal].native);
  }
  if (amdf_status_is_ok(status)) {
    const uint64_t address =
        (uintptr_t)memory->accesses[memory->backing_access_ordinal].native;
    *out_info = (amdf_memory_info_t){
        .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
        .structure_size = sizeof(*out_info),
        .memory_profile_ordinal = profile->ordinal,
        .memory_class = AMDF_MEMORY_CLASS_SYSTEM,
        .access_count = memory->info.access_count,
        .flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                 (registration ? 0 : AMDF_MEMORY_FLAG_HOST_FILE_IO) |
                 (mapping_source ? AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE : 0),
        .byte_length = create_info->byte_length,
        .alignment = registration ? address & (~address + 1) : granularity,
        .native_allocation_byte_length = native_length,
        .native_allocation_granularity = granularity,
        .placement = create_info->placement,
    };
    if (create_info->placement.kind != AMDF_MEMORY_PLACEMENT_KIND_NONE) {
      status = amdf_platform_host_memory_place(
          amdf_host_memory_pointer(memory), native_length,
          create_info->placement, memory->host_allocator);
    }
  }
  return status;
}
