// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/pm4_command_atomic.h"

#include <inttypes.h>

#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"

typedef struct iree_hal_amdgpu_pm4_atomic_fallback_layout_t {
  // Immutable PM4 launch metadata for the selected fallback kernel.
  const iree_hal_amdgpu_device_atomic_pm4_launch_t* launch;
  // Required kernarg template alignment in bytes.
  iree_host_size_t kernarg_alignment;
  // Required kernarg template length in bytes.
  iree_host_size_t kernarg_length;
} iree_hal_amdgpu_pm4_atomic_fallback_layout_t;

static iree_status_t iree_hal_amdgpu_pm4_atomic_record_fallback_layout(
    const iree_hal_amdgpu_pm4_atomic_record_t* record,
    const iree_hal_amdgpu_device_atomic_pm4_context_t* atomic_context,
    iree_hal_amdgpu_pm4_atomic_fallback_layout_t* out_layout) {
  memset(out_layout, 0, sizeof(*out_layout));
  switch (record->header.opcode) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT:
      out_layout->launch =
          iree_hal_amdgpu_device_atomic_pm4_context_select_wait(
              atomic_context, record->params.wait.width);
      out_layout->kernarg_alignment =
          IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_KERNARG_ALIGNMENT;
      out_layout->kernarg_length =
          IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_KERNARG_SIZE;
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE:
      out_layout->launch =
          iree_hal_amdgpu_device_atomic_pm4_context_select_store(
              atomic_context, record->params.store.width);
      out_layout->kernarg_alignment =
          IREE_HAL_AMDGPU_DEVICE_ATOMIC_STORE_KERNARG_ALIGNMENT;
      out_layout->kernarg_length =
          IREE_HAL_AMDGPU_DEVICE_ATOMIC_STORE_KERNARG_SIZE;
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW:
      out_layout->launch = iree_hal_amdgpu_device_atomic_pm4_context_select_rmw(
          atomic_context, record->params.rmw.width);
      out_layout->kernarg_alignment =
          IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_KERNARG_ALIGNMENT;
      out_layout->kernarg_length =
          IREE_HAL_AMDGPU_DEVICE_ATOMIC_RMW_KERNARG_SIZE;
      break;
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 atomic record has unknown opcode %u",
                              record->header.opcode);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_atomic_measure_barrier(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    iree_hal_amdgpu_pm4_barrier_flags_t barrier_flags,
    iree_hsa_fence_scope_t acquire_scope, iree_hsa_fence_scope_t release_scope,
    uint32_t* inout_dword_count) {
  const uint32_t barrier_dword_count = iree_hal_amdgpu_pm4_barrier_dword_count(
      capabilities, barrier_flags, acquire_scope, release_scope);
  if (IREE_UNLIKELY(barrier_dword_count == 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 atomic barrier cannot be emitted with capabilities 0x%08" PRIx32,
        capabilities);
  }
  if (IREE_UNLIKELY(*inout_dword_count > UINT32_MAX - barrier_dword_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 atomic dword count overflows uint32_t");
  }
  *inout_dword_count += barrier_dword_count;
  return iree_ok_status();
}

static void iree_hal_amdgpu_pm4_atomic_record_initialize(
    iree_hal_amdgpu_pm4_command_record_opcode_t opcode,
    iree_hal_amdgpu_pm4_atomic_target_record_t target, uint32_t command_index,
    iree_hal_amdgpu_pm4_atomic_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_atomic_record_t* out_record) {
  memset(out_record, 0, sizeof(*out_record));
  out_record->header.length = sizeof(*out_record);
  out_record->header.opcode = opcode;
  out_record->target = target;
  out_record->command_index = command_index;
  out_record->flags = flags;
  out_record->barrier_acquire_scope = barrier_acquire_scope;
  out_record->barrier_release_scope = barrier_release_scope;
}

void iree_hal_amdgpu_pm4_atomic_record_initialize_wait(
    iree_hal_amdgpu_pm4_atomic_target_record_t target,
    iree_hal_atomic_wait_params_t params, uint32_t command_index,
    iree_hal_amdgpu_pm4_atomic_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_atomic_record_t* out_record) {
  iree_hal_amdgpu_pm4_atomic_record_initialize(
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT, target,
      command_index, flags, barrier_acquire_scope, barrier_release_scope,
      out_record);
  out_record->params.wait = params;
}

void iree_hal_amdgpu_pm4_atomic_record_initialize_store(
    iree_hal_amdgpu_pm4_atomic_target_record_t target,
    iree_hal_atomic_store_params_t params, uint32_t command_index,
    iree_hal_amdgpu_pm4_atomic_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_atomic_record_t* out_record) {
  iree_hal_amdgpu_pm4_atomic_record_initialize(
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE, target,
      command_index, flags, barrier_acquire_scope, barrier_release_scope,
      out_record);
  out_record->params.store = params;
}

void iree_hal_amdgpu_pm4_atomic_record_initialize_rmw(
    iree_hal_amdgpu_pm4_atomic_target_record_t target,
    iree_hal_atomic_rmw_params_t params, uint32_t command_index,
    iree_hal_amdgpu_pm4_atomic_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_atomic_record_t* out_record) {
  iree_hal_amdgpu_pm4_atomic_record_initialize(
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW, target,
      command_index, flags, barrier_acquire_scope, barrier_release_scope,
      out_record);
  out_record->params.rmw = params;
}

const iree_hal_amdgpu_device_atomic_pm4_launch_t*
iree_hal_amdgpu_pm4_atomic_record_launch(
    const iree_hal_amdgpu_pm4_atomic_record_t* record,
    const iree_hal_amdgpu_device_atomic_pm4_context_t* atomic_context) {
  switch (record->header.opcode) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT:
      return iree_hal_amdgpu_device_atomic_pm4_context_select_wait(
          atomic_context, record->params.wait.width);
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE:
      return iree_hal_amdgpu_device_atomic_pm4_context_select_store(
          atomic_context, record->params.store.width);
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW:
      return iree_hal_amdgpu_device_atomic_pm4_context_select_rmw(
          atomic_context, record->params.rmw.width);
    default:
      return NULL;
  }
}

iree_status_t iree_hal_amdgpu_pm4_atomic_record_measure(
    iree_hal_amdgpu_pm4_atomic_record_t* record,
    const iree_hal_amdgpu_device_atomic_pm4_context_t* atomic_context,
    iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities,
    iree_host_size_t current_template_byte_length,
    bool has_previous_launch_state,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* previous_launch_state,
    iree_hal_amdgpu_pm4_atomic_record_measurement_t* out_measurement) {
  memset(out_measurement, 0, sizeof(*out_measurement));
  iree_hal_amdgpu_pm4_atomic_fallback_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_record_fallback_layout(
      record, atomic_context, &layout));
  if (IREE_UNLIKELY(layout.launch->launch_state.user_data_dword_count == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 atomic fallback has no kernarg user-data dwords");
  }

  if (IREE_UNLIKELY(current_template_byte_length >
                    UINT32_MAX - (layout.kernarg_alignment - 1))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 atomic template alignment overflows");
  }
  const iree_host_size_t template_offset =
      iree_host_align(current_template_byte_length, layout.kernarg_alignment);
  if (IREE_UNLIKELY(template_offset > UINT32_MAX ||
                    layout.kernarg_length > UINT32_MAX - template_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 atomic template storage exceeds uint32_t fixup offsets");
  }
  record->template_offset = (uint32_t)template_offset;
  out_measurement->template_byte_length =
      template_offset + layout.kernarg_length;

  uint32_t operation_dword_count =
      IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT;
  if (!has_previous_launch_state ||
      memcmp(previous_launch_state, &layout.launch->launch_state,
             sizeof(*previous_launch_state)) != 0) {
    operation_dword_count += IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT;
  }
  operation_dword_count +=
      2u + layout.launch->launch_state.user_data_dword_count;
  out_measurement->program_dword_count = operation_dword_count;
  out_measurement->profile_program_dword_count = operation_dword_count;

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_EXECUTION_BARRIER)) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_measure_barrier(
        vendor_packet_capabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        record->barrier_acquire_scope, record->barrier_release_scope,
        &out_measurement->program_dword_count));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_measure_barrier(
        vendor_packet_capabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        record->barrier_acquire_scope, record->barrier_release_scope,
        &out_measurement->profile_program_dword_count));
  }
  if (iree_any_bit_set(record->flags,
                       IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_FIXUP_BARRIER)) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_measure_barrier(
        vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB, IREE_HSA_FENCE_SCOPE_NONE,
        IREE_HSA_FENCE_SCOPE_NONE, &out_measurement->program_dword_count));
  }
  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_PROFILE_FIXUP_BARRIER)) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_measure_barrier(
        vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB, IREE_HSA_FENCE_SCOPE_NONE,
        IREE_HSA_FENCE_SCOPE_NONE,
        &out_measurement->profile_program_dword_count));
  }

  if (iree_any_bit_set(record->flags,
                       IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_DYNAMIC_TARGET)) {
    out_measurement->fixup_entry_count = 1;
    out_measurement->profile_fixup_entry_count = 1;
    bool is_preloaded = false;
    uint32_t preload_dword_offset = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_dispatch_kernarg_range_preload_offset(
            &layout.launch->launch_state, /*kernarg_byte_offset=*/0,
            sizeof(uint64_t), &is_preloaded, &preload_dword_offset));
    if (is_preloaded) {
      ++out_measurement->fixup_entry_count;
      ++out_measurement->profile_fixup_entry_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_atomic_append_fixup(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* fixup_builder,
    iree_host_size_t target_offset,
    const iree_hal_amdgpu_pm4_atomic_target_record_t* target) {
  if (IREE_UNLIKELY(target_offset > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 atomic fixup target offset overflows");
  }
  return iree_hal_amdgpu_pm4_fixup_entry_builder_append(
      fixup_builder, (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t){
                         .target_offset = (uint32_t)target_offset,
                         .binding_slot = target->binding_slot,
                         .binding_offset = target->value,
                     });
}

static iree_status_t iree_hal_amdgpu_pm4_atomic_initialize_template(
    const iree_hal_amdgpu_pm4_atomic_record_t* record, void* target_pointer,
    uint8_t* template_bytes) {
  switch (record->header.opcode) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT:
      iree_hal_amdgpu_device_atomic_wait_initialize_kernargs(
          target_pointer, record->params.wait,
          (iree_hal_amdgpu_device_atomic_wait_kernargs_t*)template_bytes);
      return iree_ok_status();
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE:
      iree_hal_amdgpu_device_atomic_store_initialize_kernargs(
          target_pointer, record->params.store,
          (iree_hal_amdgpu_device_atomic_store_kernargs_t*)template_bytes);
      return iree_ok_status();
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW:
      iree_hal_amdgpu_device_atomic_rmw_initialize_kernargs(
          target_pointer, record->params.rmw,
          (iree_hal_amdgpu_device_atomic_rmw_kernargs_t*)template_bytes);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 atomic record has unknown opcode %u",
                              record->header.opcode);
  }
}

iree_status_t iree_hal_amdgpu_pm4_atomic_record_materialize(
    const iree_hal_amdgpu_pm4_atomic_record_t* record,
    iree_hal_amdgpu_pm4_atomic_materialization_state_t* state,
    iree_hal_amdgpu_pm4_atomic_materialization_stats_t* out_stats) {
  iree_hal_amdgpu_pm4_atomic_materialization_stats_t stats = {0};
  const bool is_profile = iree_any_bit_set(
      state->flags, IREE_HAL_AMDGPU_PM4_ATOMIC_MATERIALIZATION_FLAG_PROFILE);
  iree_hal_amdgpu_pm4_atomic_fallback_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_record_fallback_layout(
      record, state->atomic_context, &layout));

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_EXECUTION_BARRIER)) {
    const uint32_t dword_count_before = state->dword_builder->dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        state->dword_builder, state->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        record->barrier_acquire_scope, record->barrier_release_scope));
    stats.execution_barrier_dwords =
        state->dword_builder->dword_count - dword_count_before;
  }

  uint8_t* template_bytes = NULL;
  if (is_profile) {
    if (IREE_UNLIKELY(record->template_offset >
                          state->template_builder->length ||
                      layout.kernarg_length > state->template_builder->length -
                                                  record->template_offset)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 profile atomic template lies outside resident storage");
    }
    template_bytes = state->template_builder->bytes + record->template_offset;
  } else {
    uint32_t template_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_byte_builder_append_aligned(
        state->template_builder, layout.kernarg_alignment,
        layout.kernarg_length, &template_offset, &template_bytes));
    if (IREE_UNLIKELY(template_offset != record->template_offset)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 atomic template offset changed");
    }
    void* target_pointer =
        iree_any_bit_set(record->flags,
                         IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_DYNAMIC_TARGET)
            ? NULL
            : (void*)(uintptr_t)record->target.value;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_initialize_template(
        record, target_pointer, template_bytes));
  }

  const bool has_dynamic_target = iree_any_bit_set(
      record->flags, IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_DYNAMIC_TARGET);
  if (has_dynamic_target) {
    iree_host_size_t target_offset = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            state->resident_template_offset, record->template_offset,
            &target_offset))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "PM4 atomic template fixup offset overflows");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_append_fixup(
        state->fixup_builder, target_offset, &record->target));
  }

  const iree_hal_amdgpu_pm4_atomic_record_flags_t fixup_barrier_flag =
      is_profile ? IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_PROFILE_FIXUP_BARRIER
                 : IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_FIXUP_BARRIER;
  if (iree_any_bit_set(record->flags, fixup_barrier_flag)) {
    const uint32_t dword_count_before = state->dword_builder->dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        state->dword_builder, state->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB, IREE_HSA_FENCE_SCOPE_NONE,
        IREE_HSA_FENCE_SCOPE_NONE));
    stats.fixup_barrier_dwords =
        state->dword_builder->dword_count - dword_count_before;
  }

  const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state =
      &layout.launch->launch_state;
  if (!state->has_previous_launch_state ||
      memcmp(&state->previous_launch_state, launch_state,
             sizeof(*launch_state)) != 0) {
    const uint32_t dword_count_before = state->dword_builder->dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_setup(
        state->dword_builder, layout.launch->setup_dwords,
        IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT));
    stats.dispatch_setup_dwords =
        state->dword_builder->dword_count - dword_count_before;
    state->previous_launch_state = *launch_state;
    state->has_previous_launch_state = true;
  }

  if (IREE_UNLIKELY(!state->template_base)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 atomic fallback requires resident kernarg storage");
  }
  const uintptr_t kernarg_address =
      (uintptr_t)state->template_base + record->template_offset;
  const uint32_t user_data_program_dword_offset =
      state->dword_builder->dword_count;
  const uint8_t* kernarg_preload_data =
      launch_state->kernarg_preload_dword_count != 0
          ? template_bytes +
                launch_state->kernarg_preload_dword_offset * sizeof(uint32_t)
          : NULL;
  uint32_t dword_count_before = state->dword_builder->dword_count;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_user_data(
      state->dword_builder, launch_state, kernarg_address,
      kernarg_preload_data));
  stats.dispatch_user_data_dwords =
      state->dword_builder->dword_count - dword_count_before;

  if (has_dynamic_target) {
    bool is_preloaded = false;
    uint32_t preload_dword_offset = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_dispatch_kernarg_range_preload_offset(
            launch_state, /*kernarg_byte_offset=*/0, sizeof(uint64_t),
            &is_preloaded, &preload_dword_offset));
    if (is_preloaded) {
      const uint32_t user_data_payload_dword_offset =
          launch_state->kernarg_preload_user_data_offset + preload_dword_offset;
      const iree_host_size_t target_dword_offset =
          (iree_host_size_t)user_data_program_dword_offset + 2u +
          user_data_payload_dword_offset;
      iree_host_size_t target_offset = 0;
      if (IREE_UNLIKELY(target_dword_offset > UINT32_MAX / sizeof(uint32_t) ||
                        !iree_host_size_checked_add(
                            state->program_offset,
                            target_dword_offset * sizeof(uint32_t),
                            &target_offset))) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "PM4 atomic user-data fixup target offset overflows");
      }
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_atomic_append_fixup(
          state->fixup_builder, target_offset, &record->target));
    }
  }

  const uint32_t dispatch_thread_count[3] = {1, 1, 1};
  dword_count_before = state->dword_builder->dword_count;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_direct(
      state->dword_builder, dispatch_thread_count,
      launch_state->dispatch_initiator));
  stats.dispatch_direct_dwords =
      state->dword_builder->dword_count - dword_count_before;
  if (out_stats) *out_stats = stats;
  return iree_ok_status();
}

void iree_hal_amdgpu_pm4_atomic_record_initialize_profile_operation(
    uint64_t command_buffer_id,
    const iree_hal_amdgpu_pm4_atomic_record_t* atomic_record,
    iree_hal_profile_command_operation_record_t* out_record) {
  iree_hal_profile_command_operation_record_t record =
      iree_hal_profile_command_operation_record_default();
  switch (atomic_record->header.opcode) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT:
      record.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_WAIT;
      record.length =
          iree_hal_atomic_width_byte_count(atomic_record->params.wait.width);
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE:
      record.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_STORE;
      record.length =
          iree_hal_atomic_width_byte_count(atomic_record->params.store.width);
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW:
      record.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_RMW;
      record.length =
          iree_hal_atomic_width_byte_count(atomic_record->params.rmw.width);
      break;
    default:
      break;
  }
  record.command_buffer_id = command_buffer_id;
  record.command_index = atomic_record->command_index;
  record.target_offset = atomic_record->target.profile_offset;
  if (iree_any_bit_set(atomic_record->flags,
                       IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_DYNAMIC_TARGET)) {
    record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS;
    record.target_ordinal = atomic_record->target.binding_slot;
  } else {
    record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS;
  }
  if (iree_any_bit_set(
          atomic_record->flags,
          IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_EXECUTION_BARRIER)) {
    record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_EXECUTION_BARRIER;
  }
  *out_record = record;
}
