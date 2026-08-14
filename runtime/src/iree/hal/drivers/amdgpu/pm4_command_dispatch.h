// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_DISPATCH_H_
#define IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_DISPATCH_H_

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/abi/timestamp.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/pm4_command_record.h"
#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Properties controlling materialization of one dispatch record.
typedef uint32_t iree_hal_amdgpu_pm4_dispatch_record_flags_t;
typedef enum iree_hal_amdgpu_pm4_dispatch_record_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_NONE =
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_NONE,
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_EXECUTION_BARRIER =
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_EXECUTION_BARRIER,
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_SOURCE_BYPASSES_GL2 =
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_SOURCE_BYPASSES_GL2,
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_TARGET_BYPASSES_GL2 =
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_TARGET_BYPASSES_GL2,
  // Workgroup counts are fetched from device memory at execution time.
  IREE_HAL_AMDGPU_PM4_DISPATCH_RECORD_FLAG_INDIRECT_PARAMETERS = 1u << 3,
} iree_hal_amdgpu_pm4_dispatch_record_flag_bits_t;

// Compact dispatch record replayed into resident normal and profile programs.
typedef struct iree_hal_amdgpu_pm4_dispatch_record_t {
  // Common command-record header.
  iree_hal_amdgpu_pm4_command_record_header_t header;
  // Host descriptor with executable-load PM4 metadata for this dispatch.
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor;
  // Session-local executable identifier, or 0 when unavailable.
  uint64_t executable_id;
  // Static dispatch workgroup counts, or zeros for indirect dispatch.
  uint32_t workgroup_count[3];
  // Terminal dispatch packet parameters.
  union {
    // Parameters for a direct dispatch packet.
    struct {
      // DISPATCH_DIRECT thread dimensions.
      uint32_t thread_count[3];
    } direct;
    // Static address or dynamic binding reference for DISPATCH_INDIRECT.
    iree_hal_amdgpu_pm4_buffer_ref_record_t indirect;
  } parameters;
  // HAL command ordinal within this command buffer.
  uint32_t command_index;
  // Executable export ordinal dispatched by this command.
  uint32_t export_ordinal;
  // Byte offset of this dispatch's kernarg template in resident storage.
  uint32_t template_offset;
  // Byte length of inline constant data following the record.
  uint32_t constant_length;
  // Number of inline buffer references following constants.
  uint32_t binding_count;
  // iree_hal_amdgpu_pm4_dispatch_record_flag_bits_t mask.
  iree_hal_amdgpu_pm4_dispatch_record_flags_t flags;
  // Acquire fence scope for a pending execution barrier.
  iree_hsa_fence_scope_t barrier_acquire_scope;
  // Release fence scope for a pending execution barrier.
  iree_hsa_fence_scope_t barrier_release_scope;
} iree_hal_amdgpu_pm4_dispatch_record_t;

// Borrowed command-buffer state used to append dispatch records.
typedef struct iree_hal_amdgpu_pm4_dispatch_recorder_t {
  // Shared compact command recording state to mutate.
  iree_hal_amdgpu_pm4_command_recording_state_t* recording_state;
  // HAL command-buffer binding count to expand for dynamic references.
  uint32_t* binding_count;
  // Maximum number of dynamic binding-table slots accepted by the command
  // buffer.
  uint32_t binding_capacity;
  // PM4 packet-family capabilities for barrier measurement.
  iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities;
  // Whether records contribute to a timestamped profile PM4 program.
  bool materializes_profile;
} iree_hal_amdgpu_pm4_dispatch_recorder_t;

// Dispatch-attributed profiling state borrowed during profile materialization.
typedef struct iree_hal_amdgpu_pm4_dispatch_profile_context_t {
  // PM4 packet strategy used to capture device timestamps.
  iree_hal_amdgpu_pm4_timestamp_strategy_t timestamp_strategy;
  // Device-visible fallback range targeted before submission-time fixup.
  iree_hal_amdgpu_timestamp_range_t* dummy_ticks;
  // First synthetic binding slot used for timestamp destinations.
  uint32_t timestamp_binding_base;
  // Number of profile-visible operations addressable by timestamp bindings.
  uint32_t operation_count;
} iree_hal_amdgpu_pm4_dispatch_profile_context_t;

// Appends one direct or indirect dispatch after executable/resource validation.
iree_status_t iree_hal_amdgpu_pm4_dispatch_recorder_record(
    iree_hal_amdgpu_pm4_dispatch_recorder_t* recorder,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    uint64_t executable_id, uint32_t export_ordinal,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags);

// Materializes |record| into the selected PM4 program and fixup builders.
// |profile_context| is required exactly when |state| selects profile output.
iree_status_t iree_hal_amdgpu_pm4_dispatch_record_materialize(
    const iree_hal_amdgpu_pm4_dispatch_record_t* record, void* hostcall_buffer,
    const iree_hal_amdgpu_pm4_dispatch_profile_context_t* profile_context,
    iree_hal_amdgpu_pm4_command_materialization_state_t* state,
    iree_hal_amdgpu_pm4_command_materialization_stats_t* out_stats);

// Initializes one profile-visible dispatch operation record.
void iree_hal_amdgpu_pm4_dispatch_record_initialize_profile_operation(
    uint64_t command_buffer_id,
    const iree_hal_amdgpu_pm4_dispatch_record_t* dispatch_record,
    iree_hal_profile_command_operation_record_t* out_record);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_DISPATCH_H_
