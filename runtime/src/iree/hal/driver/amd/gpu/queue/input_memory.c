// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/input_memory.h"

#include "iree/hal/driver/amd/status.h"

iree_status_t iree_hal_amd_gpu_input_memory_initialize(
    const iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    amdf_memory_access_t device_access,
    iree_hal_amd_gpu_input_memory_t* out_memory) {
  const amdf_memory_device_access_t access = {
      .device = context->device,
      .requirements = {
          .access = device_access,
          .flags =
              AMDF_MEMORY_FLAG_DEVICE_ADDRESS | AMDF_MEMORY_FLAG_HOST_COHERENT,
          .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU}};
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info)};
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(context->api->memory_scope_query_info(
                                        context->fabric->system_scope, &info),
                                    "memory_scope_query_info(shader inputs)"));
  iree_hal_amd_fabric_memory_options_t options = {
      .scope = context->fabric->system_scope,
      .access_count = 1,
      .accesses = &access,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .host_access = AMDF_MEMORY_MAP_FLAG_WRITE};
  iree_status_t status = iree_ok_status();
  bool found = false;
  for (uint32_t i = 0;
       i < info.memory_profile_count && !found && iree_status_is_ok(status);
       ++i) {
    amdf_memory_profile_t profile = {.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
                                     .structure_size = sizeof(profile)};
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities)};
    amdf_status_t native_status =
        context->api->memory_scope_query_device_profile(
            options.scope, i, 1, &access, &profile, &capabilities);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        native_status, "memory_scope_query_device_profile(shader inputs)");
    if (!iree_status_is_ok(status) ||
        !amdf_status_is_ok(profile.availability) ||
        !iree_all_bits_set(profile.roles,
                           AMDF_MEMORY_PROFILE_ROLE_CREATE |
                               AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) ||
        !iree_any_bit_set(profile.supported_flags,
                          AMDF_MEMORY_FLAG_HOST_VISIBLE) ||
        !iree_any_bit_set(profile.host_mapping.supported_access,
                          AMDF_MEMORY_MAP_FLAG_WRITE) ||
        !iree_any_bit_set(profile.host_mapping.properties,
                          AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK) ||
        iree_any_bit_set(capabilities.guaranteed_flags,
                         AMDF_MEMORY_FLAG_DEVICE_UNCACHED)) {
      continue;
    }
    const amdf_memory_profile_pair_query_t query = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
        .structure_size = sizeof(query),
        .memory_profile_ordinal = profile.ordinal,
        .access_count = 1,
        .required_flags = options.required_flags,
        .accesses = &access,
        .producer = {.kind = AMDF_MEMORY_SITE_KIND_HOST,
                     .value.host_access = AMDF_MEMORY_MAP_FLAG_WRITE},
        .consumer = {.kind = AMDF_MEMORY_SITE_KIND_DEVICE,
                     .value.device = {.access_ordinal = 0,
                                      .queue_family_ordinal = family_ordinal}}};
    amdf_memory_pair_info_t pair = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(pair)};
    native_status = context->api->memory_scope_query_pair_info(options.scope,
                                                               &query, &pair);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        native_status, "memory_scope_query_pair_info(shader inputs)");
    const amdf_queue_family_info_t* family = &context->families[family_ordinal];
    if (iree_status_is_ok(status) &&
        iree_any_bit_set(pair.flags,
                         AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE) &&
        pair.release.kind == AMDF_CACHE_TRANSITION_KIND_NONE &&
        (pair.acquire.kind == AMDF_CACHE_TRANSITION_KIND_NONE ||
         (pair.acquire.kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL &&
          pair.acquire.executor == AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE &&
          pair.acquire.operation == AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM &&
          iree_any_bit_set(family->cache_operations,
                           UINT64_C(1)
                               << AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM) &&
          iree_any_bit_set(family->cache_transition_kinds,
                           AMDF_CACHE_TRANSITION_KINDS_GLOBAL)))) {
      options.profile = profile;
      found = true;
    }
  }
  if (iree_status_is_ok(status)) {
    if (found) {
      out_memory->access = access;
      out_memory->options = options;
      out_memory->options.accesses = &out_memory->access;
    } else {
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "GPU queue family %u has no cached coherent shader-input backing "
          "with host write-back mapping and supported system acquisition",
          family_ordinal);
    }
  }
  return status;
}
