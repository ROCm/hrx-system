// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/fabric/memory_spec.h"

#include "iree/base/internal/math.h"
#include "iree/hal/driver/amd/fabric/memory.h"
#include "iree/hal/driver/amd/status.h"

static amdf_memory_scope_t* iree_hal_amd_fabric_memory_spec_scope(
    const iree_hal_amd_fabric_memory_spec_params_t* params,
    iree_host_size_t index) {
  return index < params->local_scope_count ? params->local_scopes[index]
                                           : params->system_scope;
}

static iree_status_t iree_hal_amd_fabric_memory_spec_query_type(
    const iree_hal_amd_fabric_memory_spec_params_t* params,
    amdf_memory_scope_t* scope, const amdf_memory_profile_t* profile,
    const amdf_memory_access_capabilities_t* access,
    iree_hal_memory_type_spec_t* out_type) {
  out_type->memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  if (iree_all_bits_set(profile->supported_flags,
                        AMDF_MEMORY_FLAG_DEVICE_LOCAL)) {
    out_type->memory_type |= IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  }
  if (iree_all_bits_set(profile->roles, AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
      iree_all_bits_set(profile->supported_flags,
                        AMDF_MEMORY_FLAG_HOST_VISIBLE)) {
    out_type->memory_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    if (profile->memory_class == AMDF_MEMORY_CLASS_SYSTEM) {
      out_type->memory_type |= IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    }
    out_type->allowed_buffer_usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                                      IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
    if (profile->host_mapping.supported_access & AMDF_MEMORY_MAP_FLAG_READ) {
      out_type->allowed_memory_access |= IREE_HAL_MEMORY_ACCESS_READ;
    }
    if (profile->host_mapping.supported_access & AMDF_MEMORY_MAP_FLAG_WRITE) {
      out_type->allowed_memory_access |= IREE_HAL_MEMORY_ACCESS_WRITE;
    }
  }
  if (iree_all_bits_set(access->supported_flags,
                        AMDF_MEMORY_FLAG_HOST_COHERENT)) {
    out_type->memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  }
  out_type->allowed_buffer_usage |= params->supported_usage;
  out_type->minimum_alignment = profile->allocation.minimum_alignment;
  out_type->optimal_transfer_granularity = 1;
  const amdf_memory_profile_site_t site = {
      .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
      .value.device.queue_family_ordinal = params->queue_family_ordinal,
  };
  const amdf_memory_profile_pair_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
      .structure_size = sizeof(query),
      .memory_profile_ordinal = profile->ordinal,
      .access_count = 1,
      .accesses = &params->access,
      .producer = site,
      .consumer = site,
  };
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  const amdf_status_t status =
      params->api->memory_scope_query_pair_info(scope, &query, &pair);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(status, "query memory type atomic reach"));
  out_type->atomic_operations = iree_hal_amd_fabric_memory_translate_atomics(
      access->atomic_operations_32, access->atomic_operations_64,
      pair.atomic_reach);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_fabric_memory_spec_populate(
    const iree_hal_amd_fabric_memory_spec_params_t* params,
    iree_hal_device_spec_builder_t* builder) {
  const iree_host_size_t scope_count = params->local_scope_count + 1;
  iree_host_size_t info_offset = 0;
  iree_host_size_t heaps_offset = 0;
  iree_host_size_t storage_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD(scope_count, amdf_memory_scope_info_t, &info_offset),
      IREE_STRUCT_FIELD(scope_count, iree_hal_memory_heap_spec_t,
                        &heaps_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(builder->host_allocator,
                                             storage_size, (void**)&storage));
  amdf_memory_scope_info_t* infos = (void*)(storage + info_offset);
  iree_hal_memory_heap_spec_t* heaps = (void*)(storage + heaps_offset);
  iree_host_size_t type_capacity = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < scope_count && iree_status_is_ok(status);
       ++i) {
    infos[i] = (amdf_memory_scope_info_t){
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(infos[i]),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        params->api->memory_scope_query_info(
            iree_hal_amd_fabric_memory_spec_scope(params, i), &infos[i]),
        "query physical memory scope");
    if (iree_status_is_ok(status) &&
        !iree_host_size_checked_add(
            type_capacity, infos[i].memory_profile_count, &type_capacity)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "memory profile table exceeds host size");
    }
  }
  iree_hal_memory_type_spec_t* types = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(builder->host_allocator, type_capacity,
                                         sizeof(*types), (void**)&types);
  }
  uint32_t heap_count = 0;
  iree_host_size_t type_count = 0;
  uint32_t system_heap = UINT32_MAX;
  for (iree_host_size_t i = 0; i < scope_count && iree_status_is_ok(status);
       ++i) {
    amdf_memory_scope_t* scope =
        iree_hal_amd_fabric_memory_spec_scope(params, i);
    const bool system =
        iree_any_bit_set(infos[i].flags, AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED);
    const uint32_t heap_index =
        system && system_heap != UINT32_MAX ? system_heap : heap_count++;
    iree_hal_memory_heap_spec_t* heap = &heaps[heap_index];
    if (system && system_heap == UINT32_MAX) {
      system_heap = heap_index;
    }
    if (!heap->name.size) {
      *heap = (iree_hal_memory_heap_spec_t){
          .name = system ? IREE_SV("system") : IREE_SV("local"),
          .capacity_bytes = infos[i].capacity_byte_length,
          .physical_device_affinity = 1,
          .flags =
              (infos[i].capacity_byte_length
                   ? 0
                   : IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN) |
              IREE_HAL_MEMORY_HEAP_SPEC_FLAG_MAXIMUM_ALLOCATION_SIZE_UNKNOWN,
      };
    }
    for (uint32_t p = 0;
         p < infos[i].memory_profile_count && iree_status_is_ok(status); ++p) {
      amdf_memory_profile_t profile = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
          .structure_size = sizeof(profile),
      };
      amdf_memory_access_capabilities_t access = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(access),
      };
      const amdf_status_t native_status =
          params->api->memory_scope_query_device_profile(
              scope, p, 1, &params->access, &profile, &access);
      if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(native_status,
                                             "query memory type profile");
      if (!iree_status_is_ok(status) ||
          !amdf_status_is_ok(profile.availability) ||
          !iree_all_bits_set(profile.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE)) {
        continue;
      }
      types[type_count] = (iree_hal_memory_type_spec_t){
          .heap_index = heap_index,
      };
      status = iree_hal_amd_fabric_memory_spec_query_type(
          params, scope, &profile, &access, &types[type_count]);
      if (iree_status_is_ok(status)) {
        ++type_count;
        const amdf_memory_construction_capabilities_t* allocation =
            &profile.allocation;
        heap->allocation_granularity =
            !heap->allocation_granularity
                ? allocation->byte_length_granularity
                : iree_min(heap->allocation_granularity,
                           allocation->byte_length_granularity);
        heap->allocation_alignment =
            !heap->allocation_alignment
                ? allocation->minimum_alignment
                : iree_min(heap->allocation_alignment,
                           allocation->minimum_alignment);
        heap->maximum_allocation_size = iree_max(
            heap->maximum_allocation_size, allocation->maximum_byte_length);
        heap->flags &=
            ~IREE_HAL_MEMORY_HEAP_SPEC_FLAG_MAXIMUM_ALLOCATION_SIZE_UNKNOWN;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_device_memory_spec_t memory = {
        .heap_count = heap_count,
        .heaps = heaps,
        .memory_type_count = type_count,
        .memory_types = types,
    };
    status = iree_hal_device_spec_builder_set_memory(builder, &memory);
  }
  iree_allocator_free(builder->host_allocator, types);
  iree_allocator_free(builder->host_allocator, storage);
  return status;
}

iree_status_t iree_hal_amd_fabric_memory_spec_sample_observation(
    const iree_hal_amd_fabric_memory_spec_params_t* params,
    iree_hal_device_observation_t* out_observation) {
  const iree_host_size_t scope_count =
      params->local_scope_count ? params->local_scope_count : 1;
  iree_device_size_t total = 0;
  iree_device_size_t available = 0;
  bool capacity_known = true;
  bool availability_known = true;
  bool system_counted = false;
  for (iree_host_size_t i = 0; i < scope_count; ++i) {
    amdf_memory_scope_t* scope = params->local_scope_count
                                     ? params->local_scopes[i]
                                     : params->system_scope;
    amdf_memory_scope_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(info),
    };
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        params->api->memory_scope_query_info(scope, &info),
        "query memory observation scope"));
    if (iree_any_bit_set(info.flags, AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED)) {
      if (system_counted) {
        continue;
      }
      system_counted = true;
    }
    uint64_t scope_total = info.capacity_byte_length;
    if (iree_any_bit_set(info.flags, AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES)) {
      amdf_memory_scope_observation_t observation = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_OBSERVATION,
          .structure_size = sizeof(observation),
      };
      IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
          params->api->memory_scope_sample_observation(scope, &observation),
          "sample native memory observation"));
      scope_total = observation.capacity_byte_length;
      if (!iree_device_size_checked_add(
              available, observation.available_byte_length, &available)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "memory availability exceeds device size");
      }
    } else {
      availability_known = false;
    }
    capacity_known &= scope_total != 0;
    if (!iree_device_size_checked_add(total, scope_total, &total)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "memory capacity exceeds device size");
    }
  }
  if (capacity_known) {
    iree_hal_device_observation_set_memory_total(total, out_observation);
  }
  if (availability_known) {
    iree_hal_device_observation_set_memory_available(available,
                                                     out_observation);
  }
  return iree_ok_status();
}
