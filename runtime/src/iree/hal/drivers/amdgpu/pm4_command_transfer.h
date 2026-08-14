// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_TRANSFER_H_
#define IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_TRANSFER_H_

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/blit_pm4.h"
#include "iree/hal/drivers/amdgpu/pm4_command_record.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Properties controlling materialization of one transfer record.
typedef uint32_t iree_hal_amdgpu_pm4_transfer_record_flags_t;
typedef enum iree_hal_amdgpu_pm4_transfer_record_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_TRANSFER_RECORD_FLAG_NONE =
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_NONE,
  IREE_HAL_AMDGPU_PM4_TRANSFER_RECORD_FLAG_EXECUTION_BARRIER =
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_EXECUTION_BARRIER,
  IREE_HAL_AMDGPU_PM4_TRANSFER_RECORD_FLAG_SOURCE_BYPASSES_GL2 =
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_SOURCE_BYPASSES_GL2,
  IREE_HAL_AMDGPU_PM4_TRANSFER_RECORD_FLAG_TARGET_BYPASSES_GL2 =
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_TARGET_BYPASSES_GL2,
} iree_hal_amdgpu_pm4_transfer_record_flag_bits_t;

// Compact fill, update, or copy record replayed into a resident PM4 program.
typedef struct iree_hal_amdgpu_pm4_transfer_record_t {
  // Common command-record header.
  iree_hal_amdgpu_pm4_command_record_header_t header;
  // Static or dynamic copy source; unused by fill and update.
  iree_hal_amdgpu_pm4_buffer_ref_record_t source;
  // Static or dynamic transfer target.
  iree_hal_amdgpu_pm4_buffer_ref_record_t target;
  // Launch plan selected by |header.opcode|.
  union {
    // Fill launch plan.
    iree_hal_amdgpu_device_buffer_fill_plan_t fill;
    // Copy launch plan used by copy and update.
    iree_hal_amdgpu_device_buffer_copy_plan_t copy;
  } plan;
  // Original transfer byte length.
  uint64_t length;
  // HAL command ordinal within this command buffer.
  uint32_t command_index;
  // Byte offset of kernargs in resident template storage.
  uint32_t template_offset;
  // iree_hal_amdgpu_pm4_transfer_record_flag_bits_t mask.
  iree_hal_amdgpu_pm4_transfer_record_flags_t flags;
  // Acquire fence scope for a pending execution barrier.
  iree_hsa_fence_scope_t barrier_acquire_scope;
  // Release fence scope for a pending execution barrier.
  iree_hsa_fence_scope_t barrier_release_scope;
  // Reserved padding; must be zero.
  uint32_t reserved0;
} iree_hal_amdgpu_pm4_transfer_record_t;

// Borrowed command-buffer state used to append transfer records.
typedef struct iree_hal_amdgpu_pm4_transfer_recorder_t {
  // Shared compact command recording state to mutate.
  iree_hal_amdgpu_pm4_command_recording_state_t* recording_state;
  // HAL command-buffer binding count to expand for dynamic references.
  uint32_t* binding_count;
  // Immutable host-side transfer planning context.
  const iree_hal_amdgpu_device_buffer_transfer_context_t* transfer_context;
  // Immutable PM4 launch metadata for builtin transfer kernels.
  const iree_hal_amdgpu_device_buffer_transfer_pm4_context_t*
      transfer_pm4_context;
  // PM4 packet-family capabilities for barrier measurement.
  iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities;
  // Whether records contribute to a profile PM4 program.
  bool materializes_profile;
} iree_hal_amdgpu_pm4_transfer_recorder_t;

// Appends a fill operation for a validated target and pattern.
iree_status_t iree_hal_amdgpu_pm4_transfer_recorder_fill(
    iree_hal_amdgpu_pm4_transfer_recorder_t* recorder,
    iree_hal_amdgpu_pm4_buffer_ref_record_t target, uint64_t target_alignment,
    uint64_t length, const void* pattern, iree_host_size_t pattern_length);

// Appends an update operation and captures |source| inline.
iree_status_t iree_hal_amdgpu_pm4_transfer_recorder_update(
    iree_hal_amdgpu_pm4_transfer_recorder_t* recorder,
    iree_hal_amdgpu_pm4_buffer_ref_record_t target, uint64_t target_alignment,
    iree_const_byte_span_t source);

// Appends a copy operation for validated equal-length source and target refs.
iree_status_t iree_hal_amdgpu_pm4_transfer_recorder_copy(
    iree_hal_amdgpu_pm4_transfer_recorder_t* recorder,
    iree_hal_amdgpu_pm4_buffer_ref_record_t source, uint64_t source_alignment,
    iree_hal_amdgpu_pm4_buffer_ref_record_t target, uint64_t target_alignment,
    uint64_t length);

// Materializes |record| into the selected PM4 program and fixup builders.
iree_status_t iree_hal_amdgpu_pm4_transfer_record_materialize(
    const iree_hal_amdgpu_pm4_transfer_record_t* record,
    const iree_hal_amdgpu_device_buffer_transfer_pm4_context_t*
        transfer_context,
    iree_hal_amdgpu_pm4_command_materialization_state_t* state,
    iree_hal_amdgpu_pm4_command_materialization_stats_t* out_stats);

// Initializes one profile-visible transfer operation record.
void iree_hal_amdgpu_pm4_transfer_record_initialize_profile_operation(
    uint64_t command_buffer_id,
    const iree_hal_amdgpu_pm4_transfer_record_t* transfer_record,
    iree_hal_profile_command_operation_record_t* out_record);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_TRANSFER_H_
