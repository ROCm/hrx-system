// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/tsan.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amd/status.h"

// Queue-owned TSAN state read by instrumented kernels.
//
// One state record exists for each provisioned GPU queue when queue-scoped
// TSAN support is enabled. The record owns the stable queue shadow geometry
// used by payload kernels.
typedef struct iree_hal_amd_gpu_tsan_queue_state_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Reserved queue-state flags, initialized to zero.
  uint32_t flags;
  // Flattened logical queue ordinal in the owning HAL device.
  uint32_t queue_ordinal;
  // Physical GPU device ordinal owning this queue.
  uint32_t physical_device_ordinal;
  // Queue ordinal relative to |physical_device_ordinal|.
  uint32_t physical_queue_ordinal;
  // Reserved for future queue state. Must be zero.
  uint32_t reserved0;
  // Number of queue-local dispatch shadow slots available.
  uint32_t shadow_slot_count;
  // Host-observed base address of the HSA AQL packet ring. This is metadata,
  // not a device-addressing contract.
  uint64_t aql_ring_base;
  // Power-of-two packet-ring slot mask for host packet IDs.
  uint64_t aql_ring_mask;
  // Reserved for future queue state. Must be zero.
  uint64_t reserved1;
  // Device-visible base of this queue's shadow allocation.
  uint64_t shadow_base;
  // Total bytes in this queue's shadow allocation.
  uint64_t shadow_size;
  // Bytes reserved for one dispatch shadow slot.
  uint64_t dispatch_shadow_stride;
  // Bytes reserved for one workgroup inside a dispatch shadow slot.
  uint64_t workgroup_shadow_stride;
  // Maximum workgroup ordinals represented by one dispatch shadow slot.
  uint32_t workgroup_capacity;
  // Bytes in each shadow entry.
  uint32_t shadow_entry_size;
  // Log2 application memory bytes represented by one shadow entry.
  uint32_t memory_granule_shift;
  // Reserved for future queue state. Must be zero.
  uint32_t reserved2;
  // Reserved for future queue state. Must be zero.
  uint64_t reserved3;
} iree_hal_amd_gpu_tsan_queue_state_t;
static_assert(sizeof(iree_hal_amd_gpu_tsan_queue_state_t) == 112,
              "TSAN queue state size is part of the device ABI");
static_assert(offsetof(iree_hal_amd_gpu_tsan_queue_state_t, aql_ring_base) ==
                  32,
              "TSAN queue state AQL fields must follow the fixed header");
static_assert(offsetof(iree_hal_amd_gpu_tsan_queue_state_t, shadow_base) == 56,
              "TSAN queue state shadow fields must follow AQL geometry fields");

void iree_hal_amd_gpu_tsan_options_initialize(
    iree_hal_amd_gpu_tsan_options_t* out_options) {
  *out_options = (iree_hal_amd_gpu_tsan_options_t){
      .memory_granule_shift = 2,
      .workgroup_capacity = 256,
      .shadow_slot_count = 16,
  };
}

iree_status_t iree_hal_amd_gpu_tsan_calculate_config(
    const iree_hal_amd_gpu_tsan_options_t* options,
    uint32_t workgroup_local_memory_size,
    iree_hal_amd_gpu_tsan_config_t* out_config) {
  *out_config = (iree_hal_amd_gpu_tsan_config_t){0};
  if (options->memory_granule_shift > 16 || !options->workgroup_capacity ||
      !iree_device_size_is_power_of_two(options->shadow_slot_count) ||
      (options->workgroup_local_memory_size &&
       !iree_device_size_is_power_of_two(
           options->workgroup_local_memory_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid GPU TSAN shadow geometry");
  }
  const uint64_t local_size = options->workgroup_local_memory_size
                                  ? options->workgroup_local_memory_size
                                  : workgroup_local_memory_size;
  if (!local_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GPU TSAN requires nonzero LDS coverage");
  }
  const uint64_t granule_size = UINT64_C(1) << options->memory_granule_shift;
  const uint64_t workgroup_stride =
      8 + iree_device_size_ceil_div(local_size, granule_size) * 8;
  uint64_t dispatch_stride = 0, shadow_size = 0;
  if (!iree_checked_mul_u64(workgroup_stride, options->workgroup_capacity,
                            &dispatch_stride) ||
      !iree_checked_mul_u64(dispatch_stride, options->shadow_slot_count,
                            &shadow_size) ||
      shadow_size > SIZE_MAX - 128) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "GPU TSAN shadow allocation overflows");
  }
  *out_config = (iree_hal_amd_gpu_tsan_config_t){
      .record_length = sizeof(*out_config),
      .flags = 3,
      .memory_granule_shift = options->memory_granule_shift,
      .shadow_size = shadow_size,
      .dispatch_shadow_stride = dispatch_stride,
      .workgroup_shadow_stride = workgroup_stride,
      .workgroup_capacity = options->workgroup_capacity,
      .shadow_entry_size = 8,
      .shadow_slot_count = options->shadow_slot_count,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_tsan_select_memory(
    const iree_hal_amd_gpu_context_t* context,
    amdf_memory_device_access_t* access,
    iree_hal_amd_fabric_memory_options_t* out_options) {
  if (!context->memory_scope_count) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "GPU TSAN requires device-local shadow storage");
  }
  *access = (amdf_memory_device_access_t){
      .device = context->device,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
              .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
          },
  };
  amdf_memory_scope_t* scope = context->memory_scopes[0];
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
    status =
        iree_make_status(IREE_STATUS_UNAVAILABLE,
                         "GPU TSAN local shadow has no host-visible profile");
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_tsan_initialize(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_tsan_options_t* options, uint32_t queue_ordinal,
    iree_hal_amd_gpu_tsan_t* out_tsan) {
  *out_tsan = (iree_hal_amd_gpu_tsan_t){0};
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_tsan_calculate_config(
      options,
      (uint32_t)context->target_info.compute.local_data_share_byte_length,
      &out_tsan->config));
  amdf_memory_device_access_t access;
  iree_hal_amd_fabric_memory_options_t memory_options;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_tsan_select_memory(context, &access, &memory_options));
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, &memory_options, 128 + out_tsan->config.shadow_size, 64,
      context->event_sink, &out_tsan->memory));
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(out_tsan->memory.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address),
      "TSAN shadow address"));
  out_tsan->config.queue_state_base = address;
  out_tsan->config.shadow_base = address + 128;
  const iree_hal_amd_gpu_tsan_queue_state_t header = {
      .record_length = sizeof(header),
      .queue_ordinal = queue_ordinal,
      .physical_queue_ordinal = queue_ordinal,
      .shadow_slot_count = out_tsan->config.shadow_slot_count,
      .shadow_base = out_tsan->config.shadow_base,
      .shadow_size = out_tsan->config.shadow_size,
      .dispatch_shadow_stride = out_tsan->config.dispatch_shadow_stride,
      .workgroup_shadow_stride = out_tsan->config.workgroup_shadow_stride,
      .workgroup_capacity = out_tsan->config.workgroup_capacity,
      .shadow_entry_size = out_tsan->config.shadow_entry_size,
      .memory_granule_shift = out_tsan->config.memory_granule_shift,
  };
  memset(out_tsan->memory.contents.data, 0,
         out_tsan->memory.contents.data_length);
  memcpy(out_tsan->memory.contents.data, &header, sizeof(header));
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
