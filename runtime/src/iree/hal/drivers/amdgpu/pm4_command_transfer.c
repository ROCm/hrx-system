// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/pm4_command_transfer.h"

#include <inttypes.h>

#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"

typedef struct iree_hal_amdgpu_pm4_transfer_binding_layout_t {
  // Captured static or dynamic buffer reference.
  const iree_hal_amdgpu_pm4_buffer_ref_record_t* buffer_ref;
  // Byte offset of its pointer in the operation kernargs.
  uint32_t kernarg_offset;
} iree_hal_amdgpu_pm4_transfer_binding_layout_t;

typedef struct iree_hal_amdgpu_pm4_transfer_layout_t {
  // Immutable PM4 launch metadata for the selected builtin kernel.
  const iree_hal_amdgpu_device_kernel_pm4_launch_t* launch;
  // Dispatch grid dimensions in work-items.
  uint32_t grid_size[3];
  // Required resident template alignment in bytes.
  iree_host_size_t template_alignment;
  // Required resident template length in bytes.
  iree_host_size_t template_length;
  // Buffer pointer fields patched through static values or dynamic fixups.
  iree_hal_amdgpu_pm4_transfer_binding_layout_t bindings[2];
  // Number of populated entries in |bindings|.
  uint32_t binding_count;
} iree_hal_amdgpu_pm4_transfer_layout_t;

static iree_status_t iree_hal_amdgpu_pm4_transfer_record_layout(
    const iree_hal_amdgpu_pm4_transfer_record_t* record,
    const iree_hal_amdgpu_device_buffer_transfer_pm4_context_t*
        transfer_context,
    iree_hal_amdgpu_pm4_transfer_layout_t* out_layout) {
  iree_hal_amdgpu_pm4_transfer_layout_t layout = {0};
  switch (record->header.opcode) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL:
      layout.launch = iree_hal_amdgpu_device_buffer_transfer_pm4_context_select(
          transfer_context, record->plan.fill.kernel);
      memcpy(layout.grid_size, record->plan.fill.grid_size,
             sizeof(layout.grid_size));
      layout.template_alignment =
          IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_ALIGNMENT;
      layout.template_length = IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE;
      layout.bindings[0].buffer_ref = &record->target;
      layout.bindings[0].kernarg_offset =
          offsetof(iree_hal_amdgpu_device_buffer_fill_kernargs_t, target_ptr);
      layout.binding_count = 1;
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE:
      layout.launch = iree_hal_amdgpu_device_buffer_transfer_pm4_context_select(
          transfer_context, record->plan.copy.kernel);
      memcpy(layout.grid_size, record->plan.copy.grid_size,
             sizeof(layout.grid_size));
      layout.template_alignment =
          IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT;
      if (IREE_UNLIKELY(!iree_host_size_checked_add(
              IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET,
              (iree_host_size_t)record->length, &layout.template_length))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "PM4 update template length overflows");
      }
      layout.bindings[0].buffer_ref = &record->target;
      layout.bindings[0].kernarg_offset =
          offsetof(iree_hal_amdgpu_device_buffer_copy_kernargs_t, target_ptr);
      layout.binding_count = 1;
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY:
      layout.launch = iree_hal_amdgpu_device_buffer_transfer_pm4_context_select(
          transfer_context, record->plan.copy.kernel);
      memcpy(layout.grid_size, record->plan.copy.grid_size,
             sizeof(layout.grid_size));
      layout.template_alignment =
          IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_ALIGNMENT;
      layout.template_length = IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE;
      layout.bindings[0].buffer_ref = &record->source;
      layout.bindings[0].kernarg_offset =
          offsetof(iree_hal_amdgpu_device_buffer_copy_kernargs_t, source_ptr);
      layout.bindings[1].buffer_ref = &record->target;
      layout.bindings[1].kernarg_offset =
          offsetof(iree_hal_amdgpu_device_buffer_copy_kernargs_t, target_ptr);
      layout.binding_count = 2;
      break;
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 transfer record has unknown opcode %u",
                              record->header.opcode);
  }
  *out_layout = layout;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_transfer_measure_barrier(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    iree_hal_amdgpu_pm4_barrier_flags_t barrier_flags,
    iree_hsa_fence_scope_t acquire_scope, iree_hsa_fence_scope_t release_scope,
    uint32_t* inout_dword_count) {
  const uint32_t barrier_dword_count = iree_hal_amdgpu_pm4_barrier_dword_count(
      capabilities, barrier_flags, acquire_scope, release_scope);
  if (IREE_UNLIKELY(barrier_dword_count == 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 transfer barrier cannot be emitted with capabilities 0x%08" PRIx32,
        capabilities);
  }
  if (IREE_UNLIKELY(*inout_dword_count > UINT32_MAX - barrier_dword_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 transfer dword count overflows uint32_t");
  }
  *inout_dword_count += barrier_dword_count;
  return iree_ok_status();
}

static void iree_hal_amdgpu_pm4_transfer_record_initialize(
    iree_hal_amdgpu_pm4_command_record_opcode_t opcode,
    iree_hal_amdgpu_pm4_buffer_ref_record_t source,
    iree_hal_amdgpu_pm4_buffer_ref_record_t target, uint64_t length,
    uint32_t command_index, iree_hal_amdgpu_pm4_transfer_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_transfer_record_t* out_record) {
  *out_record = (iree_hal_amdgpu_pm4_transfer_record_t){
      .header =
          {
              .length = sizeof(*out_record),
              .opcode = opcode,
          },
      .source = source,
      .target = target,
      .length = length,
      .command_index = command_index,
      .flags = flags,
      .barrier_acquire_scope = barrier_acquire_scope,
      .barrier_release_scope = barrier_release_scope,
  };
}

static void iree_hal_amdgpu_pm4_transfer_record_initialize_fill(
    iree_hal_amdgpu_pm4_buffer_ref_record_t target,
    const iree_hal_amdgpu_device_buffer_fill_plan_t* plan, uint64_t length,
    uint32_t command_index, iree_hal_amdgpu_pm4_transfer_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_transfer_record_t* out_record) {
  iree_hal_amdgpu_pm4_buffer_ref_record_t source = {0};
  source.binding_slot = UINT32_MAX;
  iree_hal_amdgpu_pm4_transfer_record_initialize(
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL, source, target, length,
      command_index, flags, barrier_acquire_scope, barrier_release_scope,
      out_record);
  out_record->plan.fill = *plan;
}

static void iree_hal_amdgpu_pm4_transfer_record_initialize_copy(
    iree_hal_amdgpu_pm4_buffer_ref_record_t source,
    iree_hal_amdgpu_pm4_buffer_ref_record_t target,
    const iree_hal_amdgpu_device_buffer_copy_plan_t* plan, uint64_t length,
    uint32_t command_index, iree_hal_amdgpu_pm4_transfer_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_transfer_record_t* out_record) {
  iree_hal_amdgpu_pm4_transfer_record_initialize(
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY, source, target, length,
      command_index, flags, barrier_acquire_scope, barrier_release_scope,
      out_record);
  out_record->plan.copy = *plan;
}

static iree_status_t iree_hal_amdgpu_pm4_transfer_record_update_length(
    iree_host_size_t source_length, iree_host_size_t* out_record_length) {
  *out_record_length = 0;
  iree_host_size_t unaligned_record_length = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(
              sizeof(iree_hal_amdgpu_pm4_transfer_record_t), source_length,
              &unaligned_record_length) ||
          unaligned_record_length >
              UINT32_MAX -
                  (iree_alignof(iree_hal_amdgpu_pm4_transfer_record_t) - 1))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 update record length overflows uint32_t");
  }
  *out_record_length =
      iree_host_align(unaligned_record_length,
                      iree_alignof(iree_hal_amdgpu_pm4_transfer_record_t));
  return iree_ok_status();
}

static void iree_hal_amdgpu_pm4_transfer_record_initialize_update(
    iree_hal_amdgpu_pm4_buffer_ref_record_t target,
    const iree_hal_amdgpu_device_buffer_copy_plan_t* plan,
    iree_const_byte_span_t source, uint32_t command_index,
    iree_hal_amdgpu_pm4_transfer_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_host_size_t record_length, void* record_storage) {
  memset(record_storage, 0, record_length);
  iree_hal_amdgpu_pm4_transfer_record_t* record =
      (iree_hal_amdgpu_pm4_transfer_record_t*)record_storage;
  iree_hal_amdgpu_pm4_buffer_ref_record_t unused_source = {0};
  unused_source.binding_slot = UINT32_MAX;
  iree_hal_amdgpu_pm4_transfer_record_initialize(
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE, unused_source, target,
      source.data_length, command_index, flags, barrier_acquire_scope,
      barrier_release_scope, record);
  record->header.length = (uint32_t)record_length;
  record->plan.copy = *plan;
  memcpy((uint8_t*)record + sizeof(*record), source.data, source.data_length);
}

static const iree_hal_amdgpu_device_kernel_pm4_launch_t*
iree_hal_amdgpu_pm4_transfer_record_launch(
    const iree_hal_amdgpu_pm4_transfer_record_t* record,
    const iree_hal_amdgpu_device_buffer_transfer_pm4_context_t*
        transfer_context) {
  switch (record->header.opcode) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL:
      return iree_hal_amdgpu_device_buffer_transfer_pm4_context_select(
          transfer_context, record->plan.fill.kernel);
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE:
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY:
      return iree_hal_amdgpu_device_buffer_transfer_pm4_context_select(
          transfer_context, record->plan.copy.kernel);
    default:
      return NULL;
  }
}

static iree_status_t iree_hal_amdgpu_pm4_transfer_record_measure(
    iree_hal_amdgpu_pm4_transfer_record_t* record,
    const iree_hal_amdgpu_device_buffer_transfer_pm4_context_t*
        transfer_context,
    iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities,
    iree_host_size_t current_template_byte_length,
    bool has_previous_launch_state,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* previous_launch_state,
    iree_hal_amdgpu_pm4_command_record_measurement_t* out_measurement) {
  memset(out_measurement, 0, sizeof(*out_measurement));
  iree_hal_amdgpu_pm4_transfer_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_record_layout(
      record, transfer_context, &layout));
  if (IREE_UNLIKELY(layout.launch->launch_state.user_data_dword_count == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 transfer kernel has no kernarg user-data dwords");
  }

  if (IREE_UNLIKELY(current_template_byte_length >
                    UINT32_MAX - (layout.template_alignment - 1))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 transfer template alignment overflows");
  }
  const iree_host_size_t template_offset =
      iree_host_align(current_template_byte_length, layout.template_alignment);
  if (IREE_UNLIKELY(template_offset > UINT32_MAX ||
                    layout.template_length > UINT32_MAX - template_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 transfer template storage exceeds uint32_t fixup offsets");
  }
  record->template_offset = (uint32_t)template_offset;
  out_measurement->template_byte_length =
      template_offset + layout.template_length;

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
          IREE_HAL_AMDGPU_PM4_TRANSFER_RECORD_FLAG_EXECUTION_BARRIER)) {
    const iree_hal_amdgpu_pm4_barrier_flags_t barrier_flags =
        iree_hal_amdgpu_pm4_command_record_barrier_flags(record->flags);
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_measure_barrier(
        vendor_packet_capabilities, barrier_flags,
        record->barrier_acquire_scope, record->barrier_release_scope,
        &out_measurement->program_dword_count));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_measure_barrier(
        vendor_packet_capabilities, barrier_flags,
        record->barrier_acquire_scope, record->barrier_release_scope,
        &out_measurement->profile_program_dword_count));
  }
  for (uint32_t i = 0; i < layout.binding_count; ++i) {
    const iree_hal_amdgpu_pm4_transfer_binding_layout_t* binding =
        &layout.bindings[i];
    if (binding->buffer_ref->binding_slot == UINT32_MAX) continue;
    ++out_measurement->fixup_entry_count;
    ++out_measurement->profile_fixup_entry_count;
    bool is_preloaded = false;
    uint32_t preload_dword_offset = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_dispatch_kernarg_range_preload_offset(
            &layout.launch->launch_state, binding->kernarg_offset,
            sizeof(uint64_t), &is_preloaded, &preload_dword_offset));
    if (is_preloaded) {
      ++out_measurement->fixup_entry_count;
      ++out_measurement->profile_fixup_entry_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_transfer_recorder_append_storage(
    iree_hal_amdgpu_pm4_transfer_recorder_t* recorder,
    iree_host_size_t record_length, iree_host_size_t* out_base_length,
    iree_hal_amdgpu_pm4_transfer_record_t** out_record) {
  *out_base_length = recorder->recording_state->record_builder.length;
  *out_record = NULL;
  if (IREE_UNLIKELY(recorder->recording_state->record_command_count ==
                    UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer command index exceeds uint32_t storage");
  }
  uint8_t* record_bytes = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_byte_builder_append_record(
      &recorder->recording_state->record_builder, record_length,
      &record_bytes));
  *out_record = (iree_hal_amdgpu_pm4_transfer_record_t*)record_bytes;
  return iree_ok_status();
}

static iree_hal_amdgpu_pm4_transfer_record_flags_t
iree_hal_amdgpu_pm4_transfer_recorder_record_flags(
    const iree_hal_amdgpu_pm4_command_recording_state_t* recording_state) {
  return recording_state->barrier_state.flags;
}

static iree_status_t iree_hal_amdgpu_pm4_transfer_recorder_commit(
    iree_hal_amdgpu_pm4_transfer_recorder_t* recorder,
    iree_host_size_t record_builder_base_length,
    iree_hal_amdgpu_pm4_transfer_record_t* record) {
  iree_hal_amdgpu_pm4_command_recording_state_t* recording_state =
      recorder->recording_state;
  iree_hal_amdgpu_pm4_command_record_measurement_t measurement;
  iree_status_t status = iree_hal_amdgpu_pm4_transfer_record_measure(
      record, recorder->transfer_pm4_context,
      recorder->vendor_packet_capabilities,
      recording_state->record_template_byte_length,
      recording_state->has_previous_launch_state,
      &recording_state->previous_launch_state, &measurement);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pm4_command_recording_state_validate_measurement(
        recording_state, recorder->materializes_profile, &measurement);
  }
  if (!iree_status_is_ok(status)) {
    recording_state->record_builder.length = record_builder_base_length;
    return status;
  }

  iree_hal_amdgpu_pm4_command_recording_state_commit_measurement(
      recording_state, recorder->materializes_profile, &measurement);
  const iree_hal_amdgpu_device_kernel_pm4_launch_t* launch =
      iree_hal_amdgpu_pm4_transfer_record_launch(
          record, recorder->transfer_pm4_context);
  recording_state->previous_launch_state = launch->launch_state;
  recording_state->has_previous_launch_state = true;
  if (record->source.binding_slot != UINT32_MAX) {
    *recorder->binding_count =
        iree_max(*recorder->binding_count, record->source.binding_slot + 1u);
  }
  if (record->target.binding_slot != UINT32_MAX) {
    *recorder->binding_count =
        iree_max(*recorder->binding_count, record->target.binding_slot + 1u);
  }
  recording_state->barrier_state =
      (iree_hal_amdgpu_pm4_command_barrier_state_t){0};
  ++recording_state->record_command_count;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_transfer_recorder_fill(
    iree_hal_amdgpu_pm4_transfer_recorder_t* recorder,
    iree_hal_amdgpu_pm4_buffer_ref_record_t target, uint64_t target_alignment,
    uint64_t length, const void* pattern, iree_host_size_t pattern_length) {
  uint64_t pattern_bits = 0;
  memcpy(&pattern_bits, pattern, pattern_length);
  iree_hal_amdgpu_device_buffer_fill_plan_t plan;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_fill_plan(
          recorder->transfer_context, target_alignment, length, pattern_bits,
          (uint8_t)pattern_length, &plan))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 fill parameters cannot be represented");
  }

  iree_host_size_t record_builder_base_length = 0;
  iree_hal_amdgpu_pm4_transfer_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_recorder_append_storage(
      recorder, sizeof(*record), &record_builder_base_length, &record));
  const iree_hal_amdgpu_pm4_transfer_record_flags_t record_flags =
      iree_hal_amdgpu_pm4_transfer_recorder_record_flags(
          recorder->recording_state);
  iree_hal_amdgpu_pm4_transfer_record_initialize_fill(
      target, &plan, length, recorder->recording_state->record_command_count,
      record_flags, recorder->recording_state->barrier_state.acquire_scope,
      recorder->recording_state->barrier_state.release_scope, record);
  return iree_hal_amdgpu_pm4_transfer_recorder_commit(
      recorder, record_builder_base_length, record);
}

iree_status_t iree_hal_amdgpu_pm4_transfer_recorder_update(
    iree_hal_amdgpu_pm4_transfer_recorder_t* recorder,
    iree_hal_amdgpu_pm4_buffer_ref_record_t target, uint64_t target_alignment,
    iree_const_byte_span_t source) {
  iree_hal_amdgpu_device_buffer_copy_plan_t plan;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_plan(
          recorder->transfer_context,
          IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT,
          target_alignment, source.data_length, &plan))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 update parameters cannot be represented");
  }
  iree_host_size_t record_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_record_update_length(
      source.data_length, &record_length));

  iree_host_size_t record_builder_base_length = 0;
  iree_hal_amdgpu_pm4_transfer_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_recorder_append_storage(
      recorder, record_length, &record_builder_base_length, &record));
  const iree_hal_amdgpu_pm4_transfer_record_flags_t record_flags =
      iree_hal_amdgpu_pm4_transfer_recorder_record_flags(
          recorder->recording_state);
  iree_hal_amdgpu_pm4_transfer_record_initialize_update(
      target, &plan, source, recorder->recording_state->record_command_count,
      record_flags, recorder->recording_state->barrier_state.acquire_scope,
      recorder->recording_state->barrier_state.release_scope, record_length,
      record);
  return iree_hal_amdgpu_pm4_transfer_recorder_commit(
      recorder, record_builder_base_length, record);
}

iree_status_t iree_hal_amdgpu_pm4_transfer_recorder_copy(
    iree_hal_amdgpu_pm4_transfer_recorder_t* recorder,
    iree_hal_amdgpu_pm4_buffer_ref_record_t source, uint64_t source_alignment,
    iree_hal_amdgpu_pm4_buffer_ref_record_t target, uint64_t target_alignment,
    uint64_t length) {
  iree_hal_amdgpu_device_buffer_copy_plan_t plan;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_plan(
          recorder->transfer_context, source_alignment, target_alignment,
          length, &plan))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 copy parameters cannot be represented");
  }

  iree_host_size_t record_builder_base_length = 0;
  iree_hal_amdgpu_pm4_transfer_record_t* record = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_recorder_append_storage(
      recorder, sizeof(*record), &record_builder_base_length, &record));
  const iree_hal_amdgpu_pm4_transfer_record_flags_t record_flags =
      iree_hal_amdgpu_pm4_transfer_recorder_record_flags(
          recorder->recording_state);
  iree_hal_amdgpu_pm4_transfer_record_initialize_copy(
      source, target, &plan, length,
      recorder->recording_state->record_command_count, record_flags,
      recorder->recording_state->barrier_state.acquire_scope,
      recorder->recording_state->barrier_state.release_scope, record);
  return iree_hal_amdgpu_pm4_transfer_recorder_commit(
      recorder, record_builder_base_length, record);
}

static iree_status_t iree_hal_amdgpu_pm4_transfer_append_fixup(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* fixup_builder,
    iree_host_size_t target_offset,
    const iree_hal_amdgpu_pm4_buffer_ref_record_t* buffer_ref) {
  if (IREE_UNLIKELY(target_offset > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 transfer fixup target offset overflows");
  }
  return iree_hal_amdgpu_pm4_fixup_entry_builder_append(
      fixup_builder, (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t){
                         .target_offset = (uint32_t)target_offset,
                         .binding_slot = buffer_ref->binding_slot,
                         .binding_offset = buffer_ref->value,
                     });
}

static iree_status_t iree_hal_amdgpu_pm4_transfer_initialize_template(
    const iree_hal_amdgpu_pm4_transfer_record_t* record,
    IREE_AMDGPU_DEVICE_PTR uint8_t* template_base, uint8_t* template_bytes) {
  void* target_pointer = record->target.binding_slot == UINT32_MAX
                             ? (void*)(uintptr_t)record->target.value
                             : NULL;
  switch (record->header.opcode) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL:
      iree_hal_amdgpu_device_buffer_fill_plan_initialize_kernargs(
          &record->plan.fill, target_pointer,
          (iree_hal_amdgpu_device_buffer_fill_kernargs_t*)template_bytes);
      return iree_ok_status();
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE: {
      uint8_t* source_pointer =
          template_base + record->template_offset +
          IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET;
      iree_hal_amdgpu_device_buffer_copy_plan_initialize_kernargs(
          &record->plan.copy, source_pointer, target_pointer,
          (iree_hal_amdgpu_device_buffer_copy_kernargs_t*)template_bytes);
      memcpy(template_bytes +
                 IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET,
             (const uint8_t*)record + sizeof(*record),
             (iree_host_size_t)record->length);
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY: {
      const void* source_pointer = record->source.binding_slot == UINT32_MAX
                                       ? (void*)(uintptr_t)record->source.value
                                       : NULL;
      iree_hal_amdgpu_device_buffer_copy_plan_initialize_kernargs(
          &record->plan.copy, source_pointer, target_pointer,
          (iree_hal_amdgpu_device_buffer_copy_kernargs_t*)template_bytes);
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 transfer record has unknown opcode %u",
                              record->header.opcode);
  }
}

iree_status_t iree_hal_amdgpu_pm4_transfer_record_materialize(
    const iree_hal_amdgpu_pm4_transfer_record_t* record,
    const iree_hal_amdgpu_device_buffer_transfer_pm4_context_t*
        transfer_context,
    iree_hal_amdgpu_pm4_command_materialization_state_t* state,
    iree_hal_amdgpu_pm4_command_materialization_stats_t* out_stats) {
  iree_hal_amdgpu_pm4_command_materialization_stats_t stats = {0};
  const bool is_profile = iree_any_bit_set(
      state->flags, IREE_HAL_AMDGPU_PM4_COMMAND_MATERIALIZATION_FLAG_PROFILE);
  iree_hal_amdgpu_pm4_transfer_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_record_layout(
      record, transfer_context, &layout));

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_TRANSFER_RECORD_FLAG_EXECUTION_BARRIER)) {
    const uint32_t dword_count_before = state->dword_builder->dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        state->dword_builder, state->vendor_packet_capabilities,
        iree_hal_amdgpu_pm4_command_record_barrier_flags(record->flags),
        record->barrier_acquire_scope, record->barrier_release_scope));
    stats.execution_barrier_dwords =
        state->dword_builder->dword_count - dword_count_before;
  }

  if (IREE_UNLIKELY(!state->template_base)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 transfer kernel requires resident kernarg storage");
  }
  uint8_t* template_bytes = NULL;
  if (is_profile) {
    if (IREE_UNLIKELY(record->template_offset >
                          state->template_builder->length ||
                      layout.template_length > state->template_builder->length -
                                                   record->template_offset)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 profile transfer template lies outside resident storage");
    }
    template_bytes = state->template_builder->bytes + record->template_offset;
  } else {
    uint32_t template_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_byte_builder_append_aligned(
        state->template_builder, layout.template_alignment,
        layout.template_length, &template_offset, &template_bytes));
    if (IREE_UNLIKELY(template_offset != record->template_offset)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 transfer template offset changed");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_initialize_template(
        record, state->template_base, template_bytes));
  }

  for (uint32_t i = 0; i < layout.binding_count; ++i) {
    const iree_hal_amdgpu_pm4_transfer_binding_layout_t* binding =
        &layout.bindings[i];
    if (binding->buffer_ref->binding_slot == UINT32_MAX) continue;
    iree_host_size_t target_offset = 0;
    if (IREE_UNLIKELY(
            !iree_host_size_checked_add(state->resident_template_offset,
                                        record->template_offset,
                                        &target_offset) ||
            !iree_host_size_checked_add(target_offset, binding->kernarg_offset,
                                        &target_offset))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 transfer template fixup target offset overflows");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_append_fixup(
        state->fixup_builder, target_offset, binding->buffer_ref));
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

  for (uint32_t i = 0; i < layout.binding_count; ++i) {
    const iree_hal_amdgpu_pm4_transfer_binding_layout_t* binding =
        &layout.bindings[i];
    if (binding->buffer_ref->binding_slot == UINT32_MAX) continue;
    bool is_preloaded = false;
    uint32_t preload_dword_offset = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_dispatch_kernarg_range_preload_offset(
            launch_state, binding->kernarg_offset, sizeof(uint64_t),
            &is_preloaded, &preload_dword_offset));
    if (!is_preloaded) continue;

    const uint32_t user_data_payload_dword_offset =
        launch_state->kernarg_preload_user_data_offset + preload_dword_offset;
    const iree_host_size_t target_dword_offset =
        (iree_host_size_t)user_data_program_dword_offset + 2u +
        user_data_payload_dword_offset;
    iree_host_size_t target_offset = 0;
    if (IREE_UNLIKELY(
            target_dword_offset > UINT32_MAX / sizeof(uint32_t) ||
            !iree_host_size_checked_add(state->program_offset,
                                        target_dword_offset * sizeof(uint32_t),
                                        &target_offset))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 transfer user-data fixup target offset overflows");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_transfer_append_fixup(
        state->fixup_builder, target_offset, binding->buffer_ref));
  }

  dword_count_before = state->dword_builder->dword_count;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_direct(
      state->dword_builder, layout.grid_size,
      launch_state->dispatch_initiator));
  stats.dispatch_dwords =
      state->dword_builder->dword_count - dword_count_before;
  if (out_stats) *out_stats = stats;
  return iree_ok_status();
}

static void iree_hal_amdgpu_pm4_transfer_profile_buffer_ref(
    const iree_hal_amdgpu_pm4_buffer_ref_record_t* buffer_ref, bool is_source,
    iree_hal_profile_command_operation_record_t* record) {
  if (buffer_ref->binding_slot == UINT32_MAX) {
    record->flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS;
  } else {
    record->flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS;
  }
  if (is_source) {
    record->source_offset = buffer_ref->profile_offset;
    if (buffer_ref->binding_slot != UINT32_MAX) {
      record->source_ordinal = buffer_ref->binding_slot;
    }
  } else {
    record->target_offset = buffer_ref->profile_offset;
    if (buffer_ref->binding_slot != UINT32_MAX) {
      record->target_ordinal = buffer_ref->binding_slot;
    }
  }
}

void iree_hal_amdgpu_pm4_transfer_record_initialize_profile_operation(
    uint64_t command_buffer_id,
    const iree_hal_amdgpu_pm4_transfer_record_t* transfer_record,
    iree_hal_profile_command_operation_record_t* out_record) {
  iree_hal_profile_command_operation_record_t record =
      iree_hal_profile_command_operation_record_default();
  switch (transfer_record->header.opcode) {
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL:
      record.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL;
      iree_hal_amdgpu_pm4_transfer_profile_buffer_ref(
          &transfer_record->target, /*is_source=*/false, &record);
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE:
      record.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE;
      iree_hal_amdgpu_pm4_transfer_profile_buffer_ref(
          &transfer_record->target, /*is_source=*/false, &record);
      break;
    case IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY:
      record.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY;
      iree_hal_amdgpu_pm4_transfer_profile_buffer_ref(
          &transfer_record->source, /*is_source=*/true, &record);
      iree_hal_amdgpu_pm4_transfer_profile_buffer_ref(
          &transfer_record->target, /*is_source=*/false, &record);
      break;
    default:
      break;
  }
  record.command_buffer_id = command_buffer_id;
  record.command_index = transfer_record->command_index;
  record.length = transfer_record->length;
  if (iree_any_bit_set(
          transfer_record->flags,
          IREE_HAL_AMDGPU_PM4_TRANSFER_RECORD_FLAG_EXECUTION_BARRIER)) {
    record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_EXECUTION_BARRIER;
  }
  *out_record = record;
}
