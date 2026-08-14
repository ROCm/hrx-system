// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_RECORD_H_
#define IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_RECORD_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/pm4_command_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opcode identifying one compact PM4 command record.
typedef uint16_t iree_hal_amdgpu_pm4_command_record_opcode_t;
typedef enum iree_hal_amdgpu_pm4_command_record_opcode_e {
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_DISPATCH = 1,
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_WAIT = 2,
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_STORE = 3,
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_ATOMIC_RMW = 4,
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_FILL = 5,
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_UPDATE = 6,
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_OPCODE_COPY = 7,
} iree_hal_amdgpu_pm4_command_record_opcode_e;

// Prefix shared by every compact PM4 command record.
typedef struct iree_hal_amdgpu_pm4_command_record_header_t {
  // Total byte length of this record including inline payload.
  uint32_t length;
  // iree_hal_amdgpu_pm4_command_record_opcode_t value.
  iree_hal_amdgpu_pm4_command_record_opcode_t opcode;
  // Reserved padding; must be zero.
  uint16_t reserved0;
} iree_hal_amdgpu_pm4_command_record_header_t;

// Barrier and fixup properties shared by compact command records.
typedef uint32_t iree_hal_amdgpu_pm4_command_record_flags_t;
typedef enum iree_hal_amdgpu_pm4_command_record_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_NONE = 0u,
  // An execution/visibility barrier precedes the operation.
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_EXECUTION_BARRIER = 1u << 0,
  // The normal program emits its first fixup-to-IB barrier before this op.
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_FIXUP_BARRIER = 1u << 1,
  // The profile program emits its first fixup-to-IB barrier before this op.
  IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_PROFILE_FIXUP_BARRIER = 1u << 2,
} iree_hal_amdgpu_pm4_command_record_flag_bits_t;

// Static or dynamic buffer reference captured by a compact PM4 record.
typedef struct iree_hal_amdgpu_pm4_buffer_ref_record_t {
  // Static device address, or dynamic binding-relative byte offset.
  uint64_t value;
  // Original byte offset reported in profile metadata.
  uint64_t profile_offset;
  // Dynamic binding slot, or UINT32_MAX for a static reference.
  uint32_t binding_slot;
  // Reserved padding; must be zero.
  uint32_t reserved0;
} iree_hal_amdgpu_pm4_buffer_ref_record_t;

// Exact resident storage contribution of one compact PM4 command record.
typedef struct iree_hal_amdgpu_pm4_command_record_measurement_t {
  // Dwords appended to the normal execution program.
  uint32_t program_dword_count;
  // Dwords appended to the profile execution program.
  uint32_t profile_program_dword_count;
  // Total template byte length after appending this operation's storage.
  iree_host_size_t template_byte_length;
  // Dynamic fixup entries appended to the normal program.
  uint32_t fixup_entry_count;
  // Dynamic fixup entries appended to the profile program.
  uint32_t profile_fixup_entry_count;
} iree_hal_amdgpu_pm4_command_record_measurement_t;

// Selects the materialization destination for a compact PM4 command record.
typedef uint32_t iree_hal_amdgpu_pm4_command_materialization_flags_t;
typedef enum iree_hal_amdgpu_pm4_command_materialization_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_COMMAND_MATERIALIZATION_FLAG_NONE = 0u,
  // Reuses normal-program templates while emitting the profile program.
  IREE_HAL_AMDGPU_PM4_COMMAND_MATERIALIZATION_FLAG_PROFILE = 1u << 0,
} iree_hal_amdgpu_pm4_command_materialization_flag_bits_t;

// Mutable PM4 program state shared by compact command materializers.
typedef struct iree_hal_amdgpu_pm4_command_materialization_state_t {
  // Destination PM4 program builder.
  iree_hal_amdgpu_pm4_dword_builder_t* dword_builder;
  // Shared resident kernarg-template builder.
  iree_hal_amdgpu_pm4_byte_builder_t* template_builder;
  // Destination dynamic binding fixup builder.
  iree_hal_amdgpu_pm4_fixup_entry_builder_t* fixup_builder;
  // Device-visible base of shared resident kernarg templates.
  uint8_t* template_base;
  // Byte offset of |template_base| from the resident allocation base.
  iree_host_size_t resident_template_offset;
  // Byte offset of the destination program from the resident allocation base.
  iree_host_size_t program_offset;
  // PM4 packet-family capabilities for barrier emission.
  iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities;
  // Most recent shader launch state emitted into this program.
  iree_hal_amdgpu_pm4_dispatch_launch_state_t previous_launch_state;
  // Whether |previous_launch_state| contains a valid value.
  bool has_previous_launch_state;
  // iree_hal_amdgpu_pm4_command_materialization_flag_bits_t mask.
  iree_hal_amdgpu_pm4_command_materialization_flags_t flags;
} iree_hal_amdgpu_pm4_command_materialization_state_t;

// Materialized dword classes accumulated for publication diagnostics.
typedef struct iree_hal_amdgpu_pm4_command_materialization_stats_t {
  // Dwords emitted for execution/visibility barriers.
  uint32_t execution_barrier_dwords;
  // Dwords emitted for fixup-to-IB barriers.
  uint32_t fixup_barrier_dwords;
  // Dwords emitted for shader setup.
  uint32_t dispatch_setup_dwords;
  // Dwords emitted for user data.
  uint32_t dispatch_user_data_dwords;
  // Dwords emitted for terminal dispatch packets and their alignment.
  uint32_t dispatch_dwords;
  // Dwords emitted for native atomic packets and their alignment.
  uint32_t atomic_dwords;
} iree_hal_amdgpu_pm4_command_materialization_stats_t;

// Pending execution/visibility barrier accumulated while recording.
typedef struct iree_hal_amdgpu_pm4_command_barrier_state_t {
  // True when the next executable operation must emit a barrier.
  bool pending;
  // Maximum pending acquire fence scope.
  iree_hsa_fence_scope_t acquire_scope;
  // Maximum pending release fence scope.
  iree_hsa_fence_scope_t release_scope;
} iree_hal_amdgpu_pm4_command_barrier_state_t;

// Mutable state shared by compact PM4 command recorders.
typedef struct iree_hal_amdgpu_pm4_command_recording_state_t {
  // Compact host command records accumulated while recording.
  iree_hal_amdgpu_pm4_byte_builder_t record_builder;
  // Expected resident PM4 IB dword count computed while appending records.
  uint32_t record_ib_dword_count;
  // Expected resident kernarg-template byte length computed while appending
  // records.
  iree_host_size_t record_template_byte_length;
  // Expected resident fixup entry count computed while appending records.
  uint32_t record_fixup_entry_count;
  // Next HAL command ordinal assigned while recording.
  uint32_t record_command_count;
  // Pending execution/visibility barrier debt.
  iree_hal_amdgpu_pm4_command_barrier_state_t barrier_state;
  // Previous dispatch launch state emitted into the current IB.
  iree_hal_amdgpu_pm4_dispatch_launch_state_t previous_launch_state;
  // True once |previous_launch_state| contains a valid value.
  bool has_previous_launch_state;
  // True once a normal-program fixup barrier has been assigned to a record.
  bool has_planned_fixup_barrier;
  // Profile-program planning state.
  struct {
    // Expected profile PM4 IB dword count computed while appending records.
    uint32_t record_program_dword_count;
    // Expected profile fixup entry count computed while appending records.
    uint32_t record_fixup_entry_count;
    // True once a profile fixup barrier has been assigned to a record.
    bool has_planned_fixup_barrier;
  } profile;
} iree_hal_amdgpu_pm4_command_recording_state_t;

// Validates that |measurement| can be accumulated into |recording_state|.
iree_status_t iree_hal_amdgpu_pm4_command_recording_state_validate_measurement(
    const iree_hal_amdgpu_pm4_command_recording_state_t* recording_state,
    bool materializes_profile,
    const iree_hal_amdgpu_pm4_command_record_measurement_t* measurement);

// Accumulates a previously validated |measurement|.
void iree_hal_amdgpu_pm4_command_recording_state_commit_measurement(
    iree_hal_amdgpu_pm4_command_recording_state_t* recording_state,
    bool materializes_profile,
    const iree_hal_amdgpu_pm4_command_record_measurement_t* measurement);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_RECORD_H_
