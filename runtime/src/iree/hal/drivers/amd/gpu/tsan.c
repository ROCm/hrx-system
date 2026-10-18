// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/tsan.h"

#include <string.h>

#include "iree/hal/drivers/amd/status.h"

void iree_hal_amd_gpu_tsan_options_initialize(
    iree_hal_amd_gpu_tsan_options_t* out_options) {
  *out_options = (iree_hal_amd_gpu_tsan_options_t){
      .shadow_size = UINT64_C(512) * 1024 * 1024,
  };
}

iree_status_t iree_hal_amd_gpu_tsan_calculate_config(
    const iree_hal_amd_gpu_tsan_options_t* options,
    iree_hal_amd_gpu_tsan_config_t* out_config) {
  *out_config = (iree_hal_amd_gpu_tsan_config_t){0};
  if (!options->shadow_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GPU TSAN shadow size must be nonzero");
  }
  if (options->shadow_size > (UINT64_C(1) << 32)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "GPU TSAN shadow exceeds the 4 GiB address range");
  }
  *out_config = (iree_hal_amd_gpu_tsan_config_t){
      .record_length = sizeof(*out_config),
      .flags = 3,
      .shadow_size = options->shadow_size,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_tsan_select_memory(
    const iree_hal_amd_gpu_context_t* context,
    amdf_memory_device_access_t* access,
    iree_hal_amd_fabric_memory_options_t* out_options) {
  *access = (amdf_memory_device_access_t){
      .device = context->device,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
              .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
          },
  };
  // Unified-memory devices may expose system storage without a local heap.
  amdf_memory_scope_t* scope = context->memory_scope_count
                                   ? context->memory_scopes[0]
                                   : context->fabric->system_scope;
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_scope_query_info(scope, &info),
      "TSAN memory scope"));
  iree_status_t status = iree_ok_status();
  bool found = false;
  for (uint32_t i = 0;
       i < info.memory_profile_count && !found && iree_status_is_ok(status);
       ++i) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const amdf_status_t native_status =
        context->api->memory_scope_query_device_profile(
            scope, i, 1, access, &profile, &capabilities);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    status =
        IREE_HAL_AMD_STATUS_FROM_AMDF(native_status, "TSAN memory profile");
    if (iree_status_is_ok(status) &&
        iree_all_bits_set(profile.roles,
                          AMDF_MEMORY_PROFILE_ROLE_CREATE |
                              AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
        (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) &&
        iree_all_bits_set(
            profile.host_mapping.supported_access,
            AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE)) {
      *out_options = (iree_hal_amd_fabric_memory_options_t){
          .scope = scope,
          .profile = profile,
          .access_count = 1,
          .accesses = access,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
          .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
      };
      found = true;
    }
  }
  if (iree_status_is_ok(status) && !found) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "GPU TSAN shadow has no host-visible profile");
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_tsan_initialize(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_tsan_options_t* options,
    iree_hal_amd_gpu_tsan_t* out_tsan) {
  *out_tsan = (iree_hal_amd_gpu_tsan_t){0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_tsan_calculate_config(options, &out_tsan->config));
  amdf_memory_device_access_t access;
  iree_hal_amd_fabric_memory_options_t memory_options;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_tsan_select_memory(context, &access, &memory_options));
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, &memory_options, out_tsan->config.shadow_size, 64,
      context->event_sink, &out_tsan->memory));
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(out_tsan->memory.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address),
      "TSAN shadow address"));
  out_tsan->config.shadow_base = address;
  memset(out_tsan->memory.contents.data, 0,
         out_tsan->memory.contents.data_length);
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->host_mapping_cache_control(
          out_tsan->memory.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
          out_tsan->memory.contents.data_length),
      "TSAN shadow publication");
}

void iree_hal_amd_gpu_tsan_deinitialize(
    const iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_tsan_t* tsan) {
  iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                          &tsan->memory);
  *tsan = (iree_hal_amd_gpu_tsan_t){0};
}
