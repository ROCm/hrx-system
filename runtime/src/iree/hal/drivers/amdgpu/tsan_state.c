// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/tsan_state.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"

static iree_status_t iree_hal_amdgpu_tsan_state_calculate_layout(
    const iree_hal_amdgpu_logical_device_options_t* options,
    uint32_t workgroup_local_memory_size,
    iree_device_size_t* out_workgroup_shadow_stride,
    iree_device_size_t* out_dispatch_shadow_stride,
    iree_device_size_t* out_shadow_size) {
  *out_workgroup_shadow_stride = 0;
  *out_dispatch_shadow_stride = 0;
  *out_shadow_size = 0;

  const iree_device_size_t granule_size = (iree_device_size_t)1ull
                                          << options->tsan.memory_granule_shift;
  const iree_device_size_t workgroup_entry_count =
      iree_device_size_ceil_div(workgroup_local_memory_size, granule_size);
  iree_device_size_t workgroup_shadow_data_size = 0;
  if (!iree_device_size_checked_mul(workgroup_entry_count,
                                    IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SIZE,
                                    &workgroup_shadow_data_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU TSAN workgroup shadow data size overflow: entry_count=%" PRIu64,
        (uint64_t)workgroup_entry_count);
  }
  iree_device_size_t workgroup_shadow_stride = 0;
  if (!iree_device_size_checked_add(
          IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_HEADER_SIZE,
          workgroup_shadow_data_size, &workgroup_shadow_stride)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU TSAN workgroup shadow size overflow: "
                            "header_size=%u, data_size=%" PRIu64,
                            IREE_HAL_AMDGPU_TSAN_WORKGROUP_SHADOW_HEADER_SIZE,
                            (uint64_t)workgroup_shadow_data_size);
  }
  iree_device_size_t dispatch_shadow_stride = 0;
  if (!iree_device_size_checked_mul(workgroup_shadow_stride,
                                    options->tsan.workgroup_capacity,
                                    &dispatch_shadow_stride)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU TSAN dispatch shadow size overflow: "
        "workgroup_shadow_stride=%" PRIu64 ", workgroup_capacity=%u",
        (uint64_t)workgroup_shadow_stride, options->tsan.workgroup_capacity);
  }
  iree_device_size_t shadow_size = 0;
  if (!iree_device_size_checked_mul(dispatch_shadow_stride,
                                    options->tsan.shadow_slot_count,
                                    &shadow_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU TSAN queue shadow size overflow: "
        "dispatch_shadow_stride=%" PRIu64 ", shadow_slot_count=%u",
        (uint64_t)dispatch_shadow_stride, options->tsan.shadow_slot_count);
  }

  *out_workgroup_shadow_stride = workgroup_shadow_stride;
  *out_dispatch_shadow_stride = dispatch_shadow_stride;
  *out_shadow_size = shadow_size;
  return iree_ok_status();
}

static uint32_t iree_hal_amdgpu_tsan_state_workgroup_local_memory_size(
    const iree_hal_amdgpu_logical_device_options_t* options,
    const iree_hal_amdgpu_physical_device_t* physical_device) {
  return options->tsan.workgroup_local_memory_size != 0
             ? options->tsan.workgroup_local_memory_size
             : physical_device->group_segment_max_size;
}

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

static iree_status_t iree_hal_amdgpu_tsan_state_initialize_device(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_device_size_t workgroup_shadow_stride,
    iree_device_size_t dispatch_shadow_stride, iree_device_size_t shadow_size,
    iree_hal_amdgpu_tsan_device_state_t* out_device_state) {
  memset(out_device_state, 0, sizeof(*out_device_state));

  if (IREE_UNLIKELY(shadow_size == 0 ||
                    (iree_device_size_t)(size_t)shadow_size != shadow_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU TSAN shadow allocation size is invalid: %" PRIu64,
        (uint64_t)shadow_size);
  }

  out_device_state->physical_device_ordinal = physical_device->device_ordinal;
  out_device_state->config = (iree_hal_amdgpu_tsan_config_t){
      .record_length = sizeof(out_device_state->config),
      .abi_version = IREE_HAL_AMDGPU_TSAN_CONFIG_ABI_VERSION_0,
      .flags = IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_ENABLED,
      .memory_granule_shift = options->tsan.memory_granule_shift,
      .shadow_base = 0,
      .shadow_size = shadow_size,
      .dispatch_shadow_stride = dispatch_shadow_stride,
      .workgroup_shadow_stride = workgroup_shadow_stride,
      .workgroup_capacity = options->tsan.workgroup_capacity,
      .shadow_entry_size = IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SIZE,
      .queue_aql_base = 0,
      .queue_aql_slot_mask = 0,
      .queue_state_base = 0,
      .shadow_slot_count = options->tsan.shadow_slot_count,
      .reserved0 = 0,
      .reserved1 = 0,
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_tsan_state_assign_queues(
    iree_hal_amdgpu_tsan_state_t* state, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices) {
  if (!iree_hal_amdgpu_tsan_state_is_enabled(state)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(physical_device_count > state->device_state_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU TSAN queue assignment device count %" PRIhsz
                            " exceeds TSAN device state count %" PRIhsz,
                            physical_device_count, state->device_state_count);
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < physical_device_count && iree_status_is_ok(status); ++i) {
    iree_hal_amdgpu_physical_device_t* physical_device = physical_devices[i];
    const iree_hal_amdgpu_tsan_config_t* config =
        &state->device_states[i].config;
    iree_hal_amdgpu_tsan_memory_policy_t memory_policy;
    status = iree_hal_amdgpu_tsan_state_select_memory_policy(physical_device,
                                                             &memory_policy);
    for (iree_host_size_t j = 0;
         j < physical_device->host_queue_count && iree_status_is_ok(status);
         ++j) {
      const iree_host_size_t queue_ordinal =
          physical_device->device_ordinal *
              physical_device->host_queue_capacity +
          j;
      status = iree_hal_amdgpu_host_queue_initialize_tsan_state(
          &physical_device->host_queues[j], &memory_policy, queue_ordinal, j,
          config->workgroup_shadow_stride, config->dispatch_shadow_stride,
          config->workgroup_capacity, config->shadow_entry_size,
          config->memory_granule_shift, config->shadow_slot_count);
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_tsan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_allocator_t host_allocator, iree_hal_amdgpu_tsan_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(out_state);
  memset(out_state, 0, sizeof(*out_state));

  if (!options->tsan.enabled) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(physical_device_count == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU TSAN requires at least one initialized physical device");
  }

  out_state->libhsa = &system->libhsa;
  out_state->host_allocator = host_allocator;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, physical_device_count,
      sizeof(out_state->device_states[0]), (void**)&out_state->device_states);
  for (iree_host_size_t i = 0;
       i < physical_device_count && iree_status_is_ok(status); ++i) {
    const uint32_t workgroup_local_memory_size =
        iree_hal_amdgpu_tsan_state_workgroup_local_memory_size(
            options, physical_devices[i]);
    iree_device_size_t workgroup_shadow_stride = 0;
    iree_device_size_t dispatch_shadow_stride = 0;
    iree_device_size_t shadow_size = 0;
    status = iree_hal_amdgpu_tsan_state_calculate_layout(
        options, workgroup_local_memory_size, &workgroup_shadow_stride,
        &dispatch_shadow_stride, &shadow_size);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_tsan_state_initialize_device(
          options, physical_devices[i], workgroup_shadow_stride,
          dispatch_shadow_stride, shadow_size, &out_state->device_states[i]);
    }
    if (iree_status_is_ok(status)) {
      ++out_state->device_state_count;
    }
  }

  if (iree_status_is_ok(status)) {
    out_state->is_enabled = true;
  } else {
    iree_hal_amdgpu_tsan_state_deinitialize(out_state);
  }
  return status;
}

void iree_hal_amdgpu_tsan_state_deinitialize(
    iree_hal_amdgpu_tsan_state_t* state) {
  if (!state || !state->libhsa) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(state->host_allocator, state->device_states);
  memset(state, 0, sizeof(*state));

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_tsan_state_is_enabled(
    const iree_hal_amdgpu_tsan_state_t* state) {
  return state && state->is_enabled;
}

iree_status_t iree_hal_amdgpu_tsan_state_populate_config(
    const iree_hal_amdgpu_tsan_state_t* state,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_tsan_config_t* out_config) {
  IREE_ASSERT_ARGUMENT(out_config);
  memset(out_config, 0, sizeof(*out_config));

  if (IREE_UNLIKELY(!iree_hal_amdgpu_tsan_state_is_enabled(state))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU TSAN state is not enabled");
  }
  if (IREE_UNLIKELY(physical_device_ordinal >= state->device_state_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU TSAN physical device ordinal %" PRIhsz
                            " exceeds device count %" PRIhsz,
                            physical_device_ordinal, state->device_state_count);
  }

  *out_config = state->device_states[physical_device_ordinal].config;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_tsan_state_populate_queue_config(
    const iree_hal_amdgpu_tsan_state_t* state,
    const iree_hal_amdgpu_queue_scope_t* queue_scope,
    iree_hal_amdgpu_tsan_config_t* out_config) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_tsan_state_populate_config(
      state, queue_scope->physical_device_ordinal, out_config));
  out_config->queue_aql_base = queue_scope->aql_ring_base;
  out_config->queue_aql_slot_mask = queue_scope->aql_ring_mask;
  if (queue_scope->tsan.queue_state_base) {
    out_config->flags |= IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_QUEUE_STATE;
    out_config->shadow_base = queue_scope->tsan.shadow_base;
    out_config->shadow_size = queue_scope->tsan.shadow_size;
    out_config->dispatch_shadow_stride =
        queue_scope->tsan.dispatch_shadow_stride;
    out_config->workgroup_shadow_stride =
        queue_scope->tsan.workgroup_shadow_stride;
    out_config->workgroup_capacity = queue_scope->tsan.workgroup_capacity;
    out_config->shadow_entry_size = queue_scope->tsan.shadow_entry_size;
    out_config->memory_granule_shift = queue_scope->tsan.memory_granule_shift;
    out_config->queue_state_base = queue_scope->tsan.queue_state_base;
    out_config->shadow_slot_count = queue_scope->tsan.shadow_slot_count;
    out_config->reserved1 = 0;
  }
  return iree_ok_status();
}
