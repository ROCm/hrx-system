// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/tsan_state.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/physical_device.h"

static iree_status_t iree_hal_amdgpu_tsan_state_select_memory_policy(
    const iree_hal_amdgpu_physical_device_t* physical_device,
    iree_hal_amdgpu_tsan_memory_policy_t* out_memory_policy) {
  memset(out_memory_policy, 0, sizeof(*out_memory_policy));
  if (IREE_UNLIKELY(
          !physical_device->coarse_block_pools.large.is_initialized)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU TSAN requires device-coarse memory for device-owned "
        "sanitizer state");
  }
  *out_memory_policy = (iree_hal_amdgpu_tsan_memory_policy_t){
      .memory_pool = physical_device->coarse_block_pools.large.memory_pool,
      .access_agents = &physical_device->device_agent,
      .access_agent_count = 1,
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_tsan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_tsan_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(out_state);
  memset(out_state, 0, sizeof(*out_state));
  if (!options->tsan.enabled) {
    return iree_ok_status();
  }
  out_state->is_enabled = true;
  out_state->shadow_size = options->tsan.shadow_size;
  return iree_ok_status();
}

void iree_hal_amdgpu_tsan_state_deinitialize(
    iree_hal_amdgpu_tsan_state_t* state) {
  if (state) {
    memset(state, 0, sizeof(*state));
  }
}

bool iree_hal_amdgpu_tsan_state_is_enabled(
    const iree_hal_amdgpu_tsan_state_t* state) {
  return state && state->is_enabled;
}

iree_status_t iree_hal_amdgpu_tsan_state_assign_queues(
    iree_hal_amdgpu_tsan_state_t* state, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices) {
  if (!iree_hal_amdgpu_tsan_state_is_enabled(state)) {
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < physical_device_count && iree_status_is_ok(status); ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device = physical_devices[i];
    iree_hal_amdgpu_tsan_memory_policy_t memory_policy;
    status = iree_hal_amdgpu_tsan_state_select_memory_policy(physical_device,
                                                             &memory_policy);
    for (iree_host_size_t j = 0;
         j < physical_device->host_queue_count && iree_status_is_ok(status);
         ++j) {
      status = iree_hal_amdgpu_host_queue_initialize_tsan_state(
          &physical_device->host_queues[j], &memory_policy, state->shadow_size);
    }
  }
  return status;
}

void iree_hal_amdgpu_tsan_state_populate_queue_config(
    const iree_hal_amdgpu_tsan_state_t* state,
    const iree_hal_amdgpu_queue_scope_t* queue_scope,
    iree_hal_amdgpu_tsan_config_t* out_config) {
  IREE_ASSERT(iree_hal_amdgpu_tsan_state_is_enabled(state));
  IREE_ASSERT(queue_scope->tsan.shadow_base != 0);
  IREE_ASSERT(queue_scope->tsan.shadow_size == state->shadow_size);
  *out_config = (iree_hal_amdgpu_tsan_config_t){
      .record_length = sizeof(*out_config),
      .abi_version = IREE_HAL_AMDGPU_TSAN_CONFIG_ABI_VERSION_0,
      .flags = IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_ENABLED |
               IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_QUEUE_SHADOW,
      .reserved0 = 0,
      .shadow_base = queue_scope->tsan.shadow_base,
      .shadow_size = queue_scope->tsan.shadow_size,
  };
}
