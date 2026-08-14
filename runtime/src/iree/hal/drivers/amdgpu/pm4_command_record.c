// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/pm4_command_record.h"

iree_status_t iree_hal_amdgpu_pm4_command_recording_state_validate_measurement(
    const iree_hal_amdgpu_pm4_command_recording_state_t* recording_state,
    bool materializes_profile,
    const iree_hal_amdgpu_pm4_command_record_measurement_t* measurement) {
  if (IREE_UNLIKELY(measurement->program_dword_count >
                    IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT -
                        recording_state->record_ib_dword_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command buffer requires more than the PM4-IB maximum %u dwords",
        IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  }
  if (IREE_UNLIKELY(measurement->fixup_entry_count >
                    UINT32_MAX - recording_state->record_fixup_entry_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 command-buffer fixup entry count exceeds uint32_t storage");
  }
  if (materializes_profile) {
    if (IREE_UNLIKELY(
            measurement->profile_program_dword_count >
            IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT -
                recording_state->profile.record_program_dword_count)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 profile command buffer requires more than the PM4-IB maximum "
          "%u dwords",
          IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
    }
    if (IREE_UNLIKELY(measurement->profile_fixup_entry_count >
                      UINT32_MAX -
                          recording_state->profile.record_fixup_entry_count)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "PM4 profile command-buffer fixup entry count exceeds uint32_t "
          "storage");
    }
  }
  return iree_ok_status();
}

void iree_hal_amdgpu_pm4_command_recording_state_commit_measurement(
    iree_hal_amdgpu_pm4_command_recording_state_t* recording_state,
    bool materializes_profile,
    const iree_hal_amdgpu_pm4_command_record_measurement_t* measurement) {
  recording_state->record_ib_dword_count += measurement->program_dword_count;
  recording_state->record_template_byte_length =
      measurement->template_byte_length;
  recording_state->record_fixup_entry_count += measurement->fixup_entry_count;
  if (materializes_profile) {
    recording_state->profile.record_program_dword_count +=
        measurement->profile_program_dword_count;
    recording_state->profile.record_fixup_entry_count +=
        measurement->profile_fixup_entry_count;
  }
}
