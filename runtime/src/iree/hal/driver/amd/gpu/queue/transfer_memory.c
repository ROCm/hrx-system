// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/transfer_memory.h"

#include "iree/hal/driver/amd/status.h"

static bool iree_hal_amd_gpu_transfer_memory_pair_supported(
    const amdf_queue_family_info_t* family, const amdf_memory_pair_info_t* pair,
    bool upload) {
  const amdf_cache_transition_t* host =
      upload ? &pair->release : &pair->acquire;
  const amdf_cache_transition_t* queue =
      upload ? &pair->acquire : &pair->release;
  const amdf_cache_operation_t operation =
      upload ? AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
             : AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM;
  return iree_any_bit_set(pair->flags,
                          AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE) &&
         host->kind == AMDF_CACHE_TRANSITION_KIND_NONE &&
         (queue->kind == AMDF_CACHE_TRANSITION_KIND_NONE ||
          (queue->kind == AMDF_CACHE_TRANSITION_KIND_GLOBAL &&
           queue->executor == AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE &&
           queue->operation == operation &&
           iree_any_bit_set(family->cache_operations, UINT64_C(1)
                                                          << operation) &&
           iree_any_bit_set(family->cache_transition_kinds,
                            AMDF_CACHE_TRANSITION_KINDS_GLOBAL)));
}

iree_status_t iree_hal_amd_gpu_transfer_memory_initialize(
    const iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    iree_hal_amd_gpu_transfer_memory_t* out_memory) {
  const amdf_memory_device_access_t access = {
      .device = context->device,
      .requirements = {
          .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
          .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                   AMDF_MEMORY_FLAG_HOST_COHERENT |
                   AMDF_MEMORY_FLAG_DEVICE_UNCACHED,
          .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU}};
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info)};
  IREE_RETURN_IF_ERROR(
      IREE_HAL_AMD_STATUS_FROM_AMDF(context->api->memory_scope_query_info(
                                        context->fabric->system_scope, &info),
                                    "memory_scope_query_info(transfer)"));
  iree_hal_amd_fabric_memory_options_t options = {
      .scope = context->fabric->system_scope,
      .access_count = 1,
      .accesses = &access,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE};
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
        native_status, "memory_scope_query_device_profile(transfer)");
    if (!iree_status_is_ok(status) ||
        !amdf_status_is_ok(profile.availability) ||
        !iree_all_bits_set(profile.roles,
                           AMDF_MEMORY_PROFILE_ROLE_CREATE |
                               AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) ||
        !iree_any_bit_set(profile.supported_flags,
                          AMDF_MEMORY_FLAG_HOST_VISIBLE) ||
        !iree_all_bits_set(profile.host_mapping.supported_access,
                           options.host_access) ||
        !iree_any_bit_set(profile.host_mapping.properties,
                          AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK)) {
      continue;
    }
    bool supported = true;
    for (uint32_t direction = 0;
         direction < 2 && supported && iree_status_is_ok(status); ++direction) {
      const bool upload = direction == 0;
      const amdf_memory_profile_site_t host = {
          .kind = AMDF_MEMORY_SITE_KIND_HOST,
          .value.host_access =
              upload ? AMDF_MEMORY_MAP_FLAG_WRITE : AMDF_MEMORY_MAP_FLAG_READ};
      const amdf_memory_profile_site_t device = {
          .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
          .value.device = {.access_ordinal = 0,
                           .queue_family_ordinal = family_ordinal}};
      const amdf_memory_profile_pair_query_t query = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
          .structure_size = sizeof(query),
          .memory_profile_ordinal = profile.ordinal,
          .access_count = 1,
          .required_flags = options.required_flags,
          .accesses = &access,
          .producer = upload ? host : device,
          .consumer = upload ? device : host};
      amdf_memory_pair_info_t pair = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
          .structure_size = sizeof(pair)};
      native_status = context->api->memory_scope_query_pair_info(options.scope,
                                                                 &query, &pair);
      if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        supported = false;
      } else {
        status = IREE_HAL_AMD_STATUS_FROM_AMDF(
            native_status, "memory_scope_query_pair_info(transfer)");
        supported = iree_status_is_ok(status) &&
                    iree_hal_amd_gpu_transfer_memory_pair_supported(
                        &context->families[family_ordinal], &pair, upload);
      }
    }
    if (iree_status_is_ok(status) && supported) {
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
          "GPU queue family %u has no coherent host write-back transfer "
          "backing with supported upload and download visibility",
          family_ordinal);
    }
  }
  return status;
}
