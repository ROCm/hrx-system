// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/memory.h"

#include "iree/async/proactor.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amd/status.h"

static iree_hal_atomic_operation_flags_t
iree_hal_amd_fabric_memory_translate_atomic_flags(
    amdf_atomic_operations_t operations) {
  return ((operations & AMDF_ATOMIC_OPERATION_WAIT)
              ? IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT
              : 0) |
         ((operations & AMDF_ATOMIC_OPERATION_STORE)
              ? IREE_HAL_ATOMIC_OPERATION_FLAG_STORE
              : 0) |
         ((operations & AMDF_ATOMIC_OPERATION_ADD)
              ? IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD
              : 0) |
         // HAL modular subtraction uses ADD of a negated operand. Native
         // raw SUB support alone does not qualify that implementation.
         ((operations & AMDF_ATOMIC_OPERATION_ADD)
              ? IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT
              : 0) |
         ((operations & AMDF_ATOMIC_OPERATION_AND)
              ? IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_AND
              : 0) |
         ((operations & AMDF_ATOMIC_OPERATION_OR)
              ? IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_OR
              : 0) |
         ((operations & AMDF_ATOMIC_OPERATION_XOR)
              ? IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR
              : 0);
}

iree_hal_atomic_operation_capabilities_t
iree_hal_amd_fabric_memory_translate_atomics(
    amdf_atomic_operations_t operations_32,
    amdf_atomic_operations_t operations_64, amdf_atomic_reach_t reach) {
  const iree_hal_atomic_operation_flags_t flags_32 =
      iree_hal_amd_fabric_memory_translate_atomic_flags(operations_32);
  const iree_hal_atomic_operation_flags_t flags_64 =
      iree_hal_amd_fabric_memory_translate_atomic_flags(operations_64);
  return (iree_hal_atomic_operation_capabilities_t){
      .device_scope_32 =
          reach.scope_32 >= AMDF_ATOMIC_SCOPE_DEVICE ? flags_32 : 0,
      .device_scope_64 =
          reach.scope_64 >= AMDF_ATOMIC_SCOPE_DEVICE ? flags_64 : 0,
      .system_scope_32 =
          reach.scope_32 == AMDF_ATOMIC_SCOPE_SYSTEM ? flags_32 : 0,
      .system_scope_64 =
          reach.scope_64 == AMDF_ATOMIC_SCOPE_SYSTEM ? flags_64 : 0,
  };
}

amdf_external_memory_type_t iree_hal_amd_fabric_memory_external_type(
    iree_hal_external_storage_type_t type) {
  switch (type) {
    case IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER:
      return AMDF_EXTERNAL_MEMORY_TYPE_HOST_POINTER;
    case IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS:
      return AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS;
    case IREE_HAL_EXTERNAL_STORAGE_TYPE_DMA_BUF_FD:
      return AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
    case IREE_HAL_EXTERNAL_STORAGE_TYPE_OPAQUE_FD:
      return AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD;
    case IREE_HAL_EXTERNAL_STORAGE_TYPE_D3D12_RESOURCE:
      return AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE;
    default:
      return AMDF_EXTERNAL_MEMORY_TYPE_NONE;
  }
}

const amdf_external_memory_support_t*
iree_hal_amd_fabric_memory_external_support(
    const amdf_memory_profile_t* profile,
    iree_hal_external_storage_class_t storage_class,
    amdf_external_memory_support_flags_t required_flags) {
  const amdf_external_memory_type_t type =
      iree_hal_amd_fabric_memory_external_type(storage_class.type);
  for (uint32_t i = 0; i < profile->external_memory_support_count; ++i) {
    const amdf_external_memory_support_t* support =
        &profile->external_memory_support[i];
    if (type != AMDF_EXTERNAL_MEMORY_TYPE_NONE && support->type == type &&
        iree_all_bits_set(support->flags, required_flags) &&
        support->provenance.words[0] == storage_class.provenance[0] &&
        support->provenance.words[1] == storage_class.provenance[1]) {
      return support;
    }
  }
  return NULL;
}

static bool iree_hal_amd_fabric_memory_constrain_extent(
    const amdf_external_memory_support_t* support,
    amdf_memory_construction_capabilities_t* capabilities) {
  if (support->maximum_byte_length) {
    capabilities->maximum_byte_length = iree_min(
        capabilities->maximum_byte_length, support->maximum_byte_length);
  }
  return iree_math_checked_lcm_u64(capabilities->byte_length_granularity,
                                   support->byte_length_alignment,
                                   &capabilities->byte_length_granularity) &&
         capabilities->byte_length_granularity <=
             capabilities->maximum_byte_length;
}

bool iree_hal_amd_fabric_memory_constrain_profile(
    iree_hal_external_storage_class_t source_class,
    iree_hal_pool_export_classes_t exports, amdf_memory_profile_t* profile) {
  amdf_memory_construction_capabilities_t* capabilities = &profile->allocation;
  if (source_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
    capabilities = &profile->registration;
  } else if (source_class.type != IREE_HAL_EXTERNAL_STORAGE_TYPE_NONE) {
    capabilities = &profile->import;
    if (!iree_hal_amd_fabric_memory_constrain_extent(
            iree_hal_amd_fabric_memory_external_support(
                profile, source_class,
                AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT),
            capabilities)) {
      return false;
    }
  }
  for (iree_host_size_t i = 0; i < exports.class_count; ++i) {
    if (exports.classes[i].type ==
        IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
      continue;
    }
    const amdf_external_memory_support_t* support =
        iree_hal_amd_fabric_memory_external_support(
            profile, exports.classes[i],
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT);
    if (!iree_hal_amd_fabric_memory_constrain_extent(support, capabilities)) {
      return false;
    }
    const uint64_t prefix = capabilities->native_byte_length_prefix;
    if (prefix &&
        (!iree_all_bits_set(support->flags,
                            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET) ||
         !support->source_offset_alignment ||
         prefix % support->source_offset_alignment)) {
      return false;
    }
  }
  return true;
}

iree_status_t iree_hal_amd_fabric_memory_query_host_publication(
    const amdf_api_t* api, const iree_hal_amd_fabric_memory_options_t* options,
    uint32_t access_ordinal, uint32_t queue_family_ordinal,
    bool* out_supported) {
  *out_supported = false;
  const amdf_memory_profile_pair_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
      .structure_size = sizeof(query),
      .memory_profile_ordinal = options->profile.ordinal,
      .access_count = options->access_count,
      .required_flags = options->required_flags,
      .accesses = options->accesses,
      .producer = {.kind = AMDF_MEMORY_SITE_KIND_HOST,
                   .value.host_access = AMDF_MEMORY_MAP_FLAG_WRITE},
      .consumer = {.kind = AMDF_MEMORY_SITE_KIND_DEVICE,
                   .value.device = {.access_ordinal = access_ordinal,
                                    .queue_family_ordinal =
                                        queue_family_ordinal}},
  };
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  const amdf_status_t status =
      api->memory_scope_query_pair_info(options->scope, &query, &pair);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(status, "qualify host publication"));
  *out_supported = true;
  return iree_ok_status();
}

amdf_memory_release_info_t iree_hal_amd_fabric_memory_deinitialize(
    const amdf_api_t* api, iree_hal_device_event_sink_t event_sink,
    iree_hal_amd_fabric_memory_t* memory) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, memory->info.byte_length);
  iree_async_buffer_registration_state_deinitialize(&memory->registrations);
  if (memory->mapping) {
    // A live host view has metadata-only, infallible destruction. Its native
    // backing is released by memory_destroy after all views have retired.
    const amdf_status_t status = api->host_mapping_destroy(memory->mapping);
    IREE_ASSERT(amdf_status_is_ok(status));
    (void)status;
  }
  amdf_memory_release_info_t release_info = {0};
  if (memory->handle) {
    // memory_destroy consumes its handle even on native cleanup failure; the
    // provider preserves any backing required by failed detach as a leak.
    amdf_status_t status = api->memory_destroy(memory->handle, &release_info);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_report_failure(event_sink, IREE_SV("amd"), status,
                                  "memory_destroy");
    }
  }
  *memory = (iree_hal_amd_fabric_memory_t){0};
  IREE_TRACE_ZONE_END(z0);
  return release_info;
}

static iree_status_t iree_hal_amd_fabric_memory_validate_atomic_access(
    const amdf_api_t* api, amdf_memory_t* memory,
    const iree_hal_amd_fabric_memory_atomic_access_t* required) {
  amdf_memory_access_info_t access = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
      .structure_size = sizeof(access),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_query_access_info(memory, required->access_ordinal, &access),
      "memory_query_access_info"));
  const amdf_memory_site_t site = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SITE,
      .structure_size = sizeof(site),
      .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
      .value.device =
          {
              .memory = memory,
              .access_ordinal = required->access_ordinal,
              .queue_family_ordinal = required->queue_family_ordinal,
          },
  };
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_query_pair_info(&site, &site, &pair),
      "memory_query_pair_info"));
  if (!iree_hal_atomic_operation_capabilities_support(
          iree_hal_amd_fabric_memory_translate_atomics(
              access.atomic_operations_32, access.atomic_operations_64,
              pair.atomic_reach),
          required->operations)) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "native attachment cannot supply pool atomics");
  }
  return iree_ok_status();
}

// Completes an attached owner before it can be published. All failures leave
// cleanup with the constructing caller; external transport ownership is absent.
static iree_status_t iree_hal_amd_fabric_memory_prepare(
    const amdf_api_t* api, const iree_hal_amd_fabric_memory_options_t* options,
    iree_hal_amd_fabric_memory_t* memory) {
  memory->info = (amdf_memory_info_t){
      .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
      .structure_size = sizeof(memory->info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_query_info(memory->handle, &memory->info),
      "memory_query_info"));
  for (iree_host_size_t i = 0; i < options->exports.class_count; ++i) {
    const iree_hal_external_storage_class_t storage_class =
        options->exports.classes[i];
    if (storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
      continue;
    }
    const amdf_external_memory_support_t* support =
        iree_hal_amd_fabric_memory_external_support(
            &options->profile, storage_class,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT);
    if (memory->info.source_byte_offset &&
        (!iree_all_bits_set(support->flags,
                            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET) ||
         !support->source_offset_alignment ||
         memory->info.source_byte_offset % support->source_offset_alignment)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native attachment origin cannot be exported");
    }
  }
  memory->exports = options->exports;
  memory->maintenance_range =
      iree_hal_make_buffer_range(0, memory->info.byte_length);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < options->atomic_access_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amd_fabric_memory_validate_atomic_access(
        api, memory->handle, &options->atomic_accesses[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  if (!options->host_access) {
    return iree_ok_status();
  }
  if (memory->info.byte_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMD host mapping exceeds host address space");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, memory->info.byte_length);
  const amdf_memory_map_info_t map_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map_info),
      .byte_length = memory->info.byte_length,
      .flags = options->host_access,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
              api->memory_map(memory->handle, &map_info, &memory->mapping),
              "memory_map"));
  amdf_host_mapping_info_t mapping_info = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(mapping_info),
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
              api->host_mapping_query_info(memory->mapping, &mapping_info),
              "host_mapping_query_info"));
  memory->contents =
      iree_make_byte_span(mapping_info.pointer, memory->info.byte_length);
  memory->host_access = map_info.flags;
  memory->host_features = mapping_info.features;
  if (options->host_atomic_access &&
      !iree_any_bit_set(mapping_info.features,
                        AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "native host mapping cannot supply pool atomics");
  }
  switch (mapping_info.cacheability) {
    case AMDF_HOST_CACHEABILITY_WRITE_BACK:
      memory->host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
      break;
    case AMDF_HOST_CACHEABILITY_WRITE_COMBINED:
      memory->host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_COMBINED;
      break;
    case AMDF_HOST_CACHEABILITY_UNCACHED:
      memory->host_cacheability = IREE_HAL_HOST_CACHEABILITY_UNCACHED;
      break;
    default:
      memory->host_cacheability = IREE_HAL_HOST_CACHEABILITY_UNKNOWN;
      break;
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_fabric_memory_allocate(
    const amdf_api_t* api, const iree_hal_amd_fabric_memory_options_t* options,
    uint64_t byte_length, uint64_t alignment,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amd_fabric_memory_t* out_memory,
    amdf_memory_release_info_t* out_release_info) {
  *out_memory = (iree_hal_amd_fabric_memory_t){0};
  if (out_release_info) {
    *out_release_info = (amdf_memory_release_info_t){0};
  }
  const uint64_t granularity =
      options->profile.allocation.byte_length_granularity;
  uint64_t allocation_length = 0;
  if (!iree_checked_add_u64(iree_max(byte_length, 1), granularity - 1,
                            &allocation_length) ||
      (options->host_access && allocation_length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMD allocation extent overflows host size");
  }
  allocation_length = allocation_length / granularity * granularity;
  if (allocation_length > options->profile.allocation.maximum_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMD allocation exceeds the native profile");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation_length);
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = options->profile.ordinal,
      .access_count = options->access_count,
      .required_flags = options->required_flags,
      .byte_length = allocation_length,
      .minimum_alignment =
          iree_max(alignment, options->profile.allocation.minimum_alignment),
      .accesses = options->accesses,
      .placement = options->placement,
  };
  iree_hal_amd_fabric_memory_t memory = {0};
  amdf_memory_release_info_t release_info = {0};
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_create(options->scope, &create_info, &memory.handle,
                         &release_info),
      "memory_create");
  if (!amdf_status_is_ok(release_info.status)) {
    iree_hal_amd_report_failure(event_sink, IREE_SV("amd"), release_info.status,
                                "memory_create cleanup");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_fabric_memory_prepare(api, options, &memory);
  }
  for (iree_host_size_t i = 0;
       i < options->io.proactor_count && iree_status_is_ok(status); ++i) {
    iree_async_buffer_registration_entry_t* entry = NULL;
    status = iree_async_proactor_prepare_buffer(
        options->io.proactors[i], &memory.registrations, memory.contents,
        IREE_ASYNC_BUFFER_ACCESS_FLAG_READ |
            IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
        &entry);
  }
  if (iree_status_is_ok(status)) {
    *out_memory = memory;
  } else if (memory.handle) {
    release_info =
        iree_hal_amd_fabric_memory_deinitialize(api, event_sink, &memory);
  }
  if (out_release_info) {
    *out_release_info = release_info;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amd_fabric_memory_register(
    const amdf_api_t* api, const iree_hal_amd_fabric_memory_options_t* options,
    const iree_hal_external_storage_t* storage,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amd_fabric_memory_t* out_memory) {
  *out_memory = (iree_hal_amd_fabric_memory_t){0};
  if (!storage->payload.host_pointer || !storage->byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host registration requires a nonempty "
                            "source-owned maintenance envelope");
  }
  if (storage->byte_length >
          options->profile.registration.maximum_byte_length ||
      storage->byte_length %
          options->profile.registration.byte_length_granularity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "host storage extent violates the import/export contract");
  }
  amdf_host_cacheability_t cacheability = AMDF_HOST_CACHEABILITY_UNKNOWN;
  switch (storage->host_cacheability) {
    case IREE_HAL_HOST_CACHEABILITY_WRITE_BACK:
      cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
      break;
    case IREE_HAL_HOST_CACHEABILITY_WRITE_COMBINED:
      cacheability = AMDF_HOST_CACHEABILITY_WRITE_COMBINED;
      break;
    case IREE_HAL_HOST_CACHEABILITY_UNCACHED:
      cacheability = AMDF_HOST_CACHEABILITY_UNCACHED;
      break;
    default:
      break;
  }
  if (cacheability == AMDF_HOST_CACHEABILITY_UNKNOWN ||
      cacheability !=
          options->profile.registration.registered_host_cacheability) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "host storage cacheability does not match the "
                            "qualified AMD registration profile");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, storage->byte_length);
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = options->profile.ordinal,
      .access_count = options->access_count,
      .required_flags = options->required_flags,
      .byte_length = storage->byte_length,
      .minimum_alignment = options->profile.registration.minimum_alignment,
      .registered_host_pointer = storage->payload.host_pointer,
      .accesses = options->accesses,
      .registered_host_cacheability = cacheability,
  };
  iree_hal_amd_fabric_memory_t memory = {0};
  amdf_memory_release_info_t release_info = {0};
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_create(options->scope, &create_info, &memory.handle,
                         &release_info),
      "memory_create(REGISTER)");
  if (!amdf_status_is_ok(release_info.status)) {
    iree_hal_amd_report_failure(event_sink, IREE_SV("amd"), release_info.status,
                                "memory_create(REGISTER) cleanup");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_fabric_memory_prepare(api, options, &memory);
  }
  if (iree_status_is_ok(status)) {
    memory.host_cacheability = storage->host_cacheability;
    memory.maintenance_range = storage->maintenance_range;
    if (memory.maintenance_range.byte_length == IREE_HAL_WHOLE_BUFFER) {
      memory.maintenance_range.byte_length =
          storage->byte_length - memory.maintenance_range.byte_offset;
    }
    *out_memory = memory;
  } else {
    iree_hal_amd_fabric_memory_deinitialize(api, event_sink, &memory);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amd_fabric_memory_import(
    const amdf_api_t* api, const iree_hal_amd_fabric_memory_options_t* options,
    const iree_hal_external_storage_t* storage,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_amd_fabric_memory_t* out_memory) {
  *out_memory = (iree_hal_amd_fabric_memory_t){0};
  if (storage->byte_length > options->profile.import.maximum_byte_length ||
      storage->byte_length % options->profile.import.byte_length_granularity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "external storage extent violates the import/export contract");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, storage->byte_length);
  amdf_external_memory_t external = {
      .type =
          iree_hal_amd_fabric_memory_external_type(storage->storage_class.type),
      .provenance = {{storage->storage_class.provenance[0],
                      storage->storage_class.provenance[1]}},
      .byte_length = storage->byte_length,
      .physical_backing_id = {{storage->physical_backing_id.words[0],
                               storage->physical_backing_id.words[1]}},
  };
  switch (external.type) {
    case AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD:
    case AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD:
      external.payload.file_descriptor = storage->payload.file_descriptor;
      break;
    case AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE:
      external.payload.native_handle = storage->payload.win32_handle;
      break;
    case AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS:
      external.payload.device_address = storage->payload.device_address;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("qualified native import has no HAL payload");
      break;
  }
  const amdf_memory_import_info_t import_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO,
      .structure_size = sizeof(import_info),
      .memory_profile_ordinal = options->profile.ordinal,
      .access_count = options->access_count,
      .required_flags = options->required_flags,
      .minimum_alignment = options->profile.import.minimum_alignment,
      .accesses = options->accesses,
  };
  iree_hal_amd_fabric_memory_t memory = {0};
  amdf_memory_release_info_t release_info = {0};
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->memory_import(options->scope, &import_info, &external,
                         &memory.handle, &release_info),
      "memory_import");
  if (!amdf_status_is_ok(release_info.status)) {
    iree_hal_amd_report_failure(event_sink, IREE_SV("amd"), release_info.status,
                                "memory_import cleanup");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_fabric_memory_prepare(api, options, &memory);
  }
  if (iree_status_is_ok(status)) {
    memory.maintenance_range = storage->maintenance_range;
    if (memory.maintenance_range.byte_length == IREE_HAL_WHOLE_BUFFER) {
      memory.maintenance_range.byte_length =
          storage->byte_length - memory.maintenance_range.byte_offset;
    }
    *out_memory = memory;
  } else {
    iree_hal_amd_fabric_memory_deinitialize(api, event_sink, &memory);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
