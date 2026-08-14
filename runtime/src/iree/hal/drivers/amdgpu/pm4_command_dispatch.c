// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/pm4_command_dispatch.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

static iree_status_t iree_hal_amdgpu_pm4_dispatch_align_host_size(
    iree_host_size_t value, iree_host_size_t alignment,
    iree_host_size_t* out_aligned_value) {
  *out_aligned_value = 0;
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 alignment must be a power-of-two");
  }
  if (IREE_UNLIKELY(value > UINTPTR_MAX - (alignment - 1))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 alignment overflows host size");
  }
  *out_aligned_value = iree_host_align(value, alignment);
  return iree_ok_status();
}

static const uint8_t* iree_hal_amdgpu_pm4_dispatch_record_constants(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  return (const uint8_t*)record + sizeof(*record);
}

static const iree_hal_amdgpu_pm4_buffer_ref_record_t*
iree_hal_amdgpu_pm4_dispatch_record_bindings(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record) {
  const uintptr_t constants_end =
      (uintptr_t)iree_hal_amdgpu_pm4_dispatch_record_constants(record) +
      record->constant_length;
  return (const iree_hal_amdgpu_pm4_buffer_ref_record_t*)iree_host_align(
      constants_end, iree_alignof(iree_hal_amdgpu_pm4_buffer_ref_record_t));
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_resolve_buffer_ref(
    iree_hal_buffer_ref_t buffer_ref, uint32_t binding_capacity,
    iree_hal_amdgpu_pm4_buffer_ref_record_t* out_record) {
  memset(out_record, 0, sizeof(*out_record));
  out_record->profile_offset = buffer_ref.offset;
  out_record->binding_slot = UINT32_MAX;
  if (!buffer_ref.buffer) {
    if (IREE_UNLIKELY(buffer_ref.buffer_slot >= binding_capacity)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 dispatch binding slot %u exceeds binding capacity %u",
          buffer_ref.buffer_slot, binding_capacity);
    }
    out_record->value = buffer_ref.offset;
    out_record->binding_slot = buffer_ref.buffer_slot;
    return iree_ok_status();
  }

  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer_ref.buffer);
  void* device_pointer =
      iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_pointer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 dispatch binding must be backed by an AMDGPU allocation");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(buffer_ref.buffer), buffer_ref.offset,
          &device_offset))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dispatch binding offset overflows");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 dispatch binding offset exceeds host pointer size");
  }
  out_record->value =
      (uint64_t)((uintptr_t)device_pointer + (uintptr_t)device_offset);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_resolve_thread_count(
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    iree_hal_dispatch_config_t config, uint32_t out_thread_count[3]) {
  for (iree_host_size_t i = 0; i < 3; ++i) {
    const uint64_t thread_count = (uint64_t)config.workgroup_count[i] *
                                  descriptor->kernel_args.workgroup_size[i];
    if (IREE_UNLIKELY(thread_count > UINT32_MAX)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 DISPATCH_DIRECT dimension %" PRIhsz
          " overflows uint32_t (workgroup_count=%u, workgroup_size=%u)",
          i, config.workgroup_count[i],
          descriptor->kernel_args.workgroup_size[i]);
    }
    out_thread_count[i] = (uint32_t)thread_count;
  }
  return iree_ok_status();
}

static iree_hal_amdgpu_pm4_dispatch_record_flags_t
iree_hal_amdgpu_pm4_dispatch_recorder_record_flags(
    const iree_hal_amdgpu_pm4_dispatch_recorder_t* recorder,
    const iree_hal_amdgpu_pm4_buffer_ref_record_t* bindings,
    uint32_t binding_count) {
  const iree_hal_amdgpu_pm4_command_recording_state_t* recording_state =
      recorder->recording_state;
  iree_hal_amdgpu_pm4_dispatch_record_flags_t flags =
      IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_NONE;
  if (recording_state->barrier_state.pending) {
    flags |= IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER;
  }
  bool has_dynamic_binding = false;
  for (uint32_t i = 0; i < binding_count; ++i) {
    has_dynamic_binding |= bindings[i].binding_slot != UINT32_MAX;
  }
  if (has_dynamic_binding && !recording_state->has_planned_fixup_barrier) {
    flags |= IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER;
  }
  if (recorder->materializes_profile &&
      !recording_state->profile.has_planned_fixup_barrier) {
    flags |= IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_PROFILE_FIXUP_BARRIER;
  }
  return flags;
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_measure_barrier(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    iree_hal_amdgpu_pm4_barrier_flags_t flags,
    iree_hsa_fence_scope_t acquire_scope, iree_hsa_fence_scope_t release_scope,
    uint32_t* inout_dword_count) {
  const uint32_t barrier_dword_count = iree_hal_amdgpu_pm4_barrier_dword_count(
      capabilities, flags, acquire_scope, release_scope);
  if (IREE_UNLIKELY(barrier_dword_count == 0)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 dispatch requires unsupported barrier flags");
  }
  if (IREE_UNLIKELY(*inout_dword_count > UINT32_MAX - barrier_dword_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dispatch dword count overflows");
  }
  *inout_dword_count += barrier_dword_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_record_measure(
    iree_hal_amdgpu_pm4_dispatch_record_t* record,
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    iree_host_size_t current_template_byte_length,
    bool has_previous_launch_state,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* previous_launch_state,
    bool materializes_profile,
    iree_hal_amdgpu_pm4_command_record_measurement_t* out_measurement) {
  memset(out_measurement, 0, sizeof(*out_measurement));
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor =
      record->descriptor;
  const iree_hal_amdgpu_kernarg_layout_t* layout = descriptor->kernarg_layout;
  const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state =
      &descriptor->pm4_launch_state;

  iree_host_size_t template_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_align_host_size(
      current_template_byte_length, layout->kernarg_alignment,
      &template_offset));
  if (IREE_UNLIKELY(template_offset > UINT32_MAX ||
                    layout->kernarg_byte_length >
                        UINT32_MAX - template_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 dispatch template storage exceeds uint32_t fixup offsets");
  }
  record->template_offset = (uint32_t)template_offset;
  out_measurement->template_byte_length =
      template_offset + layout->kernarg_byte_length;

  uint32_t operation_dword_count =
      IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT;
  if (!has_previous_launch_state ||
      memcmp(previous_launch_state, launch_state,
             sizeof(*previous_launch_state)) != 0) {
    operation_dword_count += descriptor->pm4_setup_dword_count;
  }
  if (IREE_UNLIKELY(launch_state->user_data_dword_count == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "PM4 dispatch has no kernarg user-data dwords");
  }
  operation_dword_count += 2u + launch_state->user_data_dword_count;
  out_measurement->program_dword_count = operation_dword_count;
  out_measurement->profile_program_dword_count = operation_dword_count;

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER)) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_measure_barrier(
        capabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        record->barrier_acquire_scope, record->barrier_release_scope,
        &out_measurement->program_dword_count));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_measure_barrier(
        capabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        record->barrier_acquire_scope, record->barrier_release_scope,
        &out_measurement->profile_program_dword_count));
  }
  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER)) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_measure_barrier(
        capabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB,
        IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
        &out_measurement->program_dword_count));
  }
  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_PROFILE_FIXUP_BARRIER)) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_measure_barrier(
        capabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB,
        IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
        &out_measurement->profile_program_dword_count));
  }

  const iree_hal_amdgpu_pm4_buffer_ref_record_t* bindings =
      iree_hal_amdgpu_pm4_dispatch_record_bindings(record);
  const iree_hal_amdgpu_kernarg_binding_slot_t* binding_slots =
      iree_hal_amdgpu_kernarg_layout_binding_slots(layout);
  for (uint32_t i = 0; i < record->binding_count; ++i) {
    if (bindings[i].binding_slot == UINT32_MAX) continue;
    ++out_measurement->fixup_entry_count;
    ++out_measurement->profile_fixup_entry_count;
    const uint32_t kernarg_offset =
        (uint32_t)binding_slots[i].target_qword_index * sizeof(uint64_t);
    bool is_preloaded = false;
    uint32_t preload_dword_offset = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_dispatch_kernarg_range_preload_offset(
            launch_state, kernarg_offset, sizeof(uint64_t), &is_preloaded,
            &preload_dword_offset));
    if (is_preloaded) {
      ++out_measurement->fixup_entry_count;
      ++out_measurement->profile_fixup_entry_count;
    }
  }

  if (materializes_profile) {
    const uint32_t timestamp_dword_count =
        2u * IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT;
    const uint32_t timestamp_alignment_dword_count = 6u;
    if (IREE_UNLIKELY(out_measurement->profile_program_dword_count >
                      UINT32_MAX - timestamp_dword_count -
                          timestamp_alignment_dword_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "PM4 profile dispatch dword count overflows");
    }
    out_measurement->profile_program_dword_count +=
        timestamp_dword_count + timestamp_alignment_dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_measure_barrier(
        capabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
        &out_measurement->profile_program_dword_count));
    if (IREE_UNLIKELY(out_measurement->profile_fixup_entry_count >
                      UINT32_MAX - 2u)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 profile dispatch fixup entry count overflows");
    }
    out_measurement->profile_fixup_entry_count += 2u;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_pm4_dispatch_recorder_record_direct(
    iree_hal_amdgpu_pm4_dispatch_recorder_t* recorder,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    uint64_t executable_id, uint32_t export_ordinal,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings) {
  const iree_hal_amdgpu_kernarg_layout_t* layout = descriptor->kernarg_layout;
  if (IREE_UNLIKELY(constants.data_length > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer dispatch constants exceed uint32_t storage");
  }
  if (IREE_UNLIKELY(bindings.count != layout->binding_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 dispatch binding count %" PRIhsz
                            " does not match executable binding count %u",
                            bindings.count, (uint32_t)layout->binding_count);
  }
  if (IREE_UNLIKELY(bindings.count > 0 && !bindings.values)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PM4 dispatch bindings must be non-null when count is non-zero");
  }
  if (IREE_UNLIKELY(constants.data_length != layout->constant_byte_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 dispatch constant byte length mismatch; "
                            "expected %u but got %" PRIhsz,
                            (uint32_t)layout->constant_byte_length,
                            constants.data_length);
  }
  iree_hal_amdgpu_pm4_command_recording_state_t* recording_state =
      recorder->recording_state;
  if (IREE_UNLIKELY(recording_state->record_command_count == UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer command index exceeds uint32_t storage");
  }

  iree_host_size_t binding_bytes = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          bindings.count, sizeof(iree_hal_amdgpu_pm4_buffer_ref_record_t),
          &binding_bytes))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dispatch binding records overflow");
  }
  iree_host_size_t constants_end = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          sizeof(iree_hal_amdgpu_pm4_dispatch_record_t), constants.data_length,
          &constants_end))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dispatch record constants overflow");
  }
  const iree_host_size_t bindings_offset = iree_host_align(
      constants_end, iree_alignof(iree_hal_amdgpu_pm4_buffer_ref_record_t));
  iree_host_size_t record_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(bindings_offset, binding_bytes,
                                                &record_length) ||
                    record_length > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dispatch record length overflows");
  }

  const iree_host_size_t record_builder_base_length =
      recording_state->record_builder.length;
  uint8_t* record_bytes = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_byte_builder_append_record(
      &recording_state->record_builder, record_length, &record_bytes));
  memset(record_bytes, 0, record_length);
  iree_hal_amdgpu_pm4_dispatch_record_t* record =
      (iree_hal_amdgpu_pm4_dispatch_record_t*)record_bytes;
  record->header.length = (uint32_t)record_length;
  record->header.opcode = IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH;
  record->descriptor = descriptor;
  record->executable_id = executable_id;
  memcpy(record->workgroup_count, config.workgroup_count,
         sizeof(record->workgroup_count));
  record->command_index = recording_state->record_command_count;
  record->export_ordinal = export_ordinal;
  record->constant_length = (uint32_t)constants.data_length;
  record->binding_count = (uint32_t)bindings.count;
  record->barrier_acquire_scope = recording_state->barrier_state.acquire_scope;
  record->barrier_release_scope = recording_state->barrier_state.release_scope;
  if (constants.data_length != 0) {
    memcpy(record_bytes + sizeof(*record), constants.data,
           constants.data_length);
  }

  iree_status_t status = iree_hal_amdgpu_pm4_dispatch_resolve_thread_count(
      descriptor, config, record->dispatch_thread_count);
  iree_hal_amdgpu_pm4_buffer_ref_record_t* binding_records =
      (iree_hal_amdgpu_pm4_buffer_ref_record_t*)(record_bytes +
                                                 bindings_offset);
  for (iree_host_size_t i = 0; i < bindings.count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amdgpu_pm4_dispatch_resolve_buffer_ref(
        bindings.values[i], recorder->binding_capacity, &binding_records[i]);
  }
  if (!iree_status_is_ok(status)) {
    recording_state->record_builder.length = record_builder_base_length;
    return status;
  }
  record->flags = iree_hal_amdgpu_pm4_dispatch_recorder_record_flags(
      recorder, binding_records, record->binding_count);

  iree_hal_amdgpu_pm4_command_record_measurement_t measurement;
  status = iree_hal_amdgpu_pm4_dispatch_record_measure(
      record, recorder->vendor_packet_capabilities,
      recording_state->record_template_byte_length,
      recording_state->has_previous_launch_state,
      &recording_state->previous_launch_state, recorder->materializes_profile,
      &measurement);
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
  recording_state->previous_launch_state = descriptor->pm4_launch_state;
  recording_state->has_previous_launch_state = true;
  recording_state->has_planned_fixup_barrier |= iree_any_bit_set(
      record->flags, IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER);
  recording_state->profile.has_planned_fixup_barrier |= iree_any_bit_set(
      record->flags,
      IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_PROFILE_FIXUP_BARRIER);
  for (uint32_t i = 0; i < record->binding_count; ++i) {
    if (binding_records[i].binding_slot != UINT32_MAX) {
      *recorder->binding_count = iree_max(*recorder->binding_count,
                                          binding_records[i].binding_slot + 1u);
    }
  }
  recording_state->barrier_state =
      (iree_hal_amdgpu_pm4_command_barrier_state_t){0};
  ++recording_state->record_command_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_append_fixup(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* fixup_builder,
    iree_host_size_t target_offset,
    const iree_hal_amdgpu_pm4_buffer_ref_record_t* buffer_ref) {
  if (IREE_UNLIKELY(target_offset > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 dispatch fixup target offset overflows");
  }
  return iree_hal_amdgpu_pm4_fixup_entry_builder_append(
      fixup_builder, (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t){
                         .target_offset = (uint32_t)target_offset,
                         .binding_slot = buffer_ref->binding_slot,
                         .binding_offset = buffer_ref->value,
                     });
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_write_template(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record, void* hostcall_buffer,
    iree_hal_amdgpu_pm4_command_materialization_state_t* state,
    uint8_t* template_bytes) {
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor =
      record->descriptor;
  const iree_hal_amdgpu_kernarg_layout_t* layout = descriptor->kernarg_layout;
  const iree_hal_amdgpu_kernarg_binding_slot_t* binding_slots =
      iree_hal_amdgpu_kernarg_layout_binding_slots(layout);
  const iree_hal_amdgpu_pm4_buffer_ref_record_t* bindings =
      iree_hal_amdgpu_pm4_dispatch_record_bindings(record);
  uint64_t* binding_pointers =
      record->binding_count
          ? (uint64_t*)iree_alloca(record->binding_count *
                                   sizeof(binding_pointers[0]))
          : NULL;
  for (uint32_t i = 0; i < record->binding_count; ++i) {
    if (bindings[i].binding_slot == UINT32_MAX) {
      binding_pointers[i] = bindings[i].value;
      continue;
    }
    binding_pointers[i] = 0;
    iree_host_size_t target_offset = 0;
    const iree_host_size_t kernarg_offset =
        (iree_host_size_t)binding_slots[i].target_qword_index *
        sizeof(uint64_t);
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
                          state->resident_template_offset,
                          record->template_offset, &target_offset) ||
                      !iree_host_size_checked_add(target_offset, kernarg_offset,
                                                  &target_offset))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 dispatch template fixup target offset overflows");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_append_fixup(
        state->fixup_builder, target_offset, &bindings[i]));
  }

  const iree_const_byte_span_t constants = iree_make_const_byte_span(
      iree_hal_amdgpu_pm4_dispatch_record_constants(record),
      record->constant_length);
  iree_hal_amdgpu_kernarg_layout_emplace_explicit_args(
      layout, binding_pointers, constants, template_bytes);
  if (iree_any_bit_set(layout->flags,
                       IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_IMPLICIT_ARGS)) {
    iree_amdgpu_kernel_implicit_args_t* implicit_args =
        (iree_amdgpu_kernel_implicit_args_t*)(template_bytes +
                                              layout
                                                  ->implicit_args_byte_offset);
    const iree_hal_dispatch_config_t config = {
        .workgroup_count = {record->workgroup_count[0],
                            record->workgroup_count[1],
                            record->workgroup_count[2]},
    };
    iree_hal_amdgpu_device_dispatch_initialize_implicit_args(
        &descriptor->kernel_args, config.workgroup_count,
        config.dynamic_workgroup_local_memory, hostcall_buffer, implicit_args);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_append_template_fixups(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record,
    iree_hal_amdgpu_pm4_command_materialization_state_t* state) {
  const iree_hal_amdgpu_kernarg_layout_t* layout =
      record->descriptor->kernarg_layout;
  const iree_hal_amdgpu_kernarg_binding_slot_t* binding_slots =
      iree_hal_amdgpu_kernarg_layout_binding_slots(layout);
  const iree_hal_amdgpu_pm4_buffer_ref_record_t* bindings =
      iree_hal_amdgpu_pm4_dispatch_record_bindings(record);
  for (uint32_t i = 0; i < record->binding_count; ++i) {
    if (bindings[i].binding_slot == UINT32_MAX) continue;
    iree_host_size_t target_offset = 0;
    const iree_host_size_t kernarg_offset =
        (iree_host_size_t)binding_slots[i].target_qword_index *
        sizeof(uint64_t);
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
                          state->resident_template_offset,
                          record->template_offset, &target_offset) ||
                      !iree_host_size_checked_add(target_offset, kernarg_offset,
                                                  &target_offset))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 profile dispatch template fixup target offset overflows");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_append_fixup(
        state->fixup_builder, target_offset, &bindings[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_append_preload_fixups(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record,
    uint32_t user_data_program_dword_offset,
    iree_hal_amdgpu_pm4_command_materialization_state_t* state) {
  const iree_hal_amdgpu_kernarg_layout_t* layout =
      record->descriptor->kernarg_layout;
  const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state =
      &record->descriptor->pm4_launch_state;
  if (launch_state->kernarg_preload_dword_count == 0) return iree_ok_status();
  const iree_hal_amdgpu_kernarg_binding_slot_t* binding_slots =
      iree_hal_amdgpu_kernarg_layout_binding_slots(layout);
  const iree_hal_amdgpu_pm4_buffer_ref_record_t* bindings =
      iree_hal_amdgpu_pm4_dispatch_record_bindings(record);
  for (uint32_t i = 0; i < record->binding_count; ++i) {
    if (bindings[i].binding_slot == UINT32_MAX) continue;
    const uint32_t kernarg_offset =
        (uint32_t)binding_slots[i].target_qword_index * sizeof(uint64_t);
    bool is_preloaded = false;
    uint32_t preload_dword_offset = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_dispatch_kernarg_range_preload_offset(
            launch_state, kernarg_offset, sizeof(uint64_t), &is_preloaded,
            &preload_dword_offset));
    if (!is_preloaded) continue;

    const iree_host_size_t target_dword_offset =
        (iree_host_size_t)user_data_program_dword_offset + 2u +
        launch_state->kernarg_preload_user_data_offset + preload_dword_offset;
    iree_host_size_t target_offset = 0;
    if (IREE_UNLIKELY(
            target_dword_offset > UINT32_MAX / sizeof(uint32_t) ||
            !iree_host_size_checked_add(state->program_offset,
                                        target_dword_offset * sizeof(uint32_t),
                                        &target_offset))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 dispatch user-data fixup target offset overflows");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_append_fixup(
        state->fixup_builder, target_offset, &bindings[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_emit_nop(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t dword_count) {
  if (IREE_UNLIKELY(dword_count < 2)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 NOP packet requires at least 2 dwords");
  }
  uint32_t* nop_dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, dword_count, &nop_dwords));
  nop_dwords[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_NOP, dword_count);
  memset(nop_dwords + 1, 0, (dword_count - 1) * sizeof(nop_dwords[0]));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_align_timestamp_target(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    iree_host_size_t program_offset) {
  if (IREE_UNLIKELY(program_offset % sizeof(uint64_t) != 0)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "PM4 profile program offset is not aligned for timestamp fixup");
  }
  const iree_host_size_t program_dword_offset =
      program_offset / sizeof(uint32_t);
  const bool target_is_aligned =
      ((program_dword_offset + builder->dword_count + 4u) & 1u) == 0;
  return target_is_aligned
             ? iree_ok_status()
             : iree_hal_amdgpu_pm4_dispatch_emit_nop(builder,
                                                     /*dword_count=*/3);
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_emit_timestamp(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy, void* target,
    uint32_t* out_target_dword_offset) {
  *out_target_dword_offset = 0;
  const uint32_t control = iree_hal_amdgpu_pm4_copy_timestamp_control(strategy);
  if (IREE_UNLIKELY(control == 0 || !iree_host_ptr_has_alignment(target, 8))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 profile timestamp strategy cannot write aligned timestamp ranges");
  }
  const uint32_t packet_dword_offset = builder->dword_count;
  uint32_t* dwords = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_append(
      builder, IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT, &dwords));
  const uintptr_t target_address = (uintptr_t)target;
  dwords[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
      IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT);
  dwords[1] = control;
  dwords[2] = 0;
  dwords[3] = 0;
  dwords[4] = iree_hal_amdgpu_pm4_addr_lo_8(target_address);
  dwords[5] = iree_hal_amdgpu_pm4_addr_hi(target_address);
  *out_target_dword_offset = packet_dword_offset + 4u;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pm4_dispatch_append_timestamp_fixup(
    iree_hal_amdgpu_pm4_command_materialization_state_t* state,
    uint32_t target_dword_offset, uint32_t binding_slot) {
  iree_host_size_t target_offset = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(
              state->program_offset,
              (iree_host_size_t)target_dword_offset * sizeof(uint32_t),
              &target_offset) ||
          target_offset > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 profile timestamp fixup target offset exceeds uint32_t storage");
  }
  return iree_hal_amdgpu_pm4_fixup_entry_builder_append(
      state->fixup_builder, (iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t){
                                .target_offset = (uint32_t)target_offset,
                                .binding_slot = binding_slot,
                                .binding_offset = 0,
                            });
}

iree_status_t iree_hal_amdgpu_pm4_dispatch_record_materialize(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record, void* hostcall_buffer,
    const iree_hal_amdgpu_pm4_dispatch_profile_context_t* profile_context,
    iree_hal_amdgpu_pm4_command_materialization_state_t* state,
    iree_hal_amdgpu_pm4_command_materialization_stats_t* out_stats) {
  iree_hal_amdgpu_pm4_command_materialization_stats_t stats = {0};
  const bool is_profile = iree_any_bit_set(
      state->flags, IREE_HAL_AMDGPU_PM4_COMMAND_MATERIALIZATION_FLAG_PROFILE);
  if (IREE_UNLIKELY(is_profile != (profile_context != NULL))) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "PM4 dispatch profile context does not match materialization mode");
  }
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor =
      record->descriptor;
  const iree_hal_amdgpu_kernarg_layout_t* layout = descriptor->kernarg_layout;
  const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state =
      &descriptor->pm4_launch_state;

  if (iree_any_bit_set(
          record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER)) {
    const uint32_t dword_count_before = state->dword_builder->dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        state->dword_builder, state->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
        record->barrier_acquire_scope, record->barrier_release_scope));
    stats.execution_barrier_dwords =
        state->dword_builder->dword_count - dword_count_before;
  }

  if (IREE_UNLIKELY(!state->template_base)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "PM4 command-buffer dispatch requires resident kernarg storage");
  }
  uint8_t* template_bytes = NULL;
  if (is_profile) {
    if (IREE_UNLIKELY(
            record->template_offset > state->template_builder->length ||
            layout->kernarg_byte_length >
                state->template_builder->length - record->template_offset)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 profile dispatch template lies outside resident storage");
    }
    template_bytes = state->template_builder->bytes + record->template_offset;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_pm4_dispatch_append_template_fixups(record, state));
  } else {
    uint32_t template_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_byte_builder_append_aligned(
        state->template_builder, layout->kernarg_alignment,
        layout->kernarg_byte_length, &template_offset, &template_bytes));
    if (IREE_UNLIKELY(template_offset != record->template_offset)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "PM4 dispatch template offset changed");
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_write_template(
        record, hostcall_buffer, state, template_bytes));
  }

  const iree_hal_amdgpu_pm4_dispatch_record_flags_t fixup_barrier_flag =
      is_profile
          ? IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_PROFILE_FIXUP_BARRIER
          : IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_FIXUP_BARRIER;
  if (iree_any_bit_set(record->flags, fixup_barrier_flag)) {
    const uint32_t dword_count_before = state->dword_builder->dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        state->dword_builder, state->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB, IREE_HSA_FENCE_SCOPE_NONE,
        IREE_HSA_FENCE_SCOPE_NONE));
    stats.fixup_barrier_dwords =
        state->dword_builder->dword_count - dword_count_before;
  }

  if (!state->has_previous_launch_state ||
      memcmp(&state->previous_launch_state, launch_state,
             sizeof(*launch_state)) != 0) {
    const uint32_t dword_count_before = state->dword_builder->dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_setup(
        state->dword_builder, descriptor->pm4_setup_dwords,
        descriptor->pm4_setup_dword_count));
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
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_append_preload_fixups(
      record, user_data_program_dword_offset, state));

  if (is_profile) {
    if (IREE_UNLIKELY(record->command_index >=
                      profile_context->operation_count)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 profile command index exceeds retained operation count");
    }
    const uint32_t timestamp_binding_slot =
        profile_context->timestamp_binding_base + record->command_index * 2u;
    uint32_t timestamp_target_dword_offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_align_timestamp_target(
        state->dword_builder, state->program_offset));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_emit_timestamp(
        state->dword_builder, profile_context->timestamp_strategy,
        &profile_context->dummy_ticks->start_tick,
        &timestamp_target_dword_offset));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_append_timestamp_fixup(
        state, timestamp_target_dword_offset, timestamp_binding_slot));

    dword_count_before = state->dword_builder->dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_direct(
        state->dword_builder, record->dispatch_thread_count,
        launch_state->dispatch_initiator));
    stats.dispatch_direct_dwords =
        state->dword_builder->dword_count - dword_count_before;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
        state->dword_builder, state->vendor_packet_capabilities,
        IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION, IREE_HSA_FENCE_SCOPE_NONE,
        IREE_HSA_FENCE_SCOPE_NONE));

    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_align_timestamp_target(
        state->dword_builder, state->program_offset));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_emit_timestamp(
        state->dword_builder, profile_context->timestamp_strategy,
        &profile_context->dummy_ticks->end_tick,
        &timestamp_target_dword_offset));
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_append_timestamp_fixup(
        state, timestamp_target_dword_offset, timestamp_binding_slot + 1u));
  } else {
    dword_count_before = state->dword_builder->dword_count;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_direct(
        state->dword_builder, record->dispatch_thread_count,
        launch_state->dispatch_initiator));
    stats.dispatch_direct_dwords =
        state->dword_builder->dword_count - dword_count_before;
  }
  if (out_stats) *out_stats = stats;
  return iree_ok_status();
}

void iree_hal_amdgpu_pm4_dispatch_record_initialize_profile_operation(
    uint64_t command_buffer_id,
    const iree_hal_amdgpu_pm4_dispatch_record_t* dispatch_record,
    iree_hal_profile_command_operation_record_t* out_record) {
  iree_hal_profile_command_operation_record_t record =
      iree_hal_profile_command_operation_record_default();
  record.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  record.command_buffer_id = command_buffer_id;
  record.command_index = dispatch_record->command_index;
  if (iree_any_bit_set(
          dispatch_record->flags,
          IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER)) {
    record.flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_EXECUTION_BARRIER;
  }
  const iree_hal_amdgpu_pm4_buffer_ref_record_t* bindings =
      iree_hal_amdgpu_pm4_dispatch_record_bindings(dispatch_record);
  for (uint32_t i = 0; i < dispatch_record->binding_count; ++i) {
    record.flags |=
        bindings[i].binding_slot == UINT32_MAX
            ? IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS
            : IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS;
  }
  if (!iree_any_bit_set(
          record.flags,
          IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS)) {
    record.flags |=
        IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_PREPUBLISHED_ARGUMENTS;
  }
  record.executable_id = dispatch_record->executable_id;
  record.function_ordinal = dispatch_record->export_ordinal;
  record.binding_count = dispatch_record->binding_count;
  memcpy(record.workgroup_count, dispatch_record->workgroup_count,
         sizeof(record.workgroup_count));
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args =
      &dispatch_record->descriptor->kernel_args;
  record.workgroup_size[0] = kernel_args->workgroup_size[0];
  record.workgroup_size[1] = kernel_args->workgroup_size[1];
  record.workgroup_size[2] = kernel_args->workgroup_size[2];
  *out_record = record;
}
