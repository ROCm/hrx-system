// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_ATOMIC_H_
#define IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_ATOMIC_H_

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/atomic_pm4.h"
#include "iree/hal/drivers/amdgpu/pm4_command_builder.h"
#include "iree/hal/drivers/amdgpu/pm4_command_record.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Properties controlling materialization of one atomic record.
typedef uint32_t iree_hal_amdgpu_pm4_atomic_record_flags_t;
typedef enum iree_hal_amdgpu_pm4_atomic_record_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_NONE = 0u,
  // An execution/visibility barrier precedes the operation.
  IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_EXECUTION_BARRIER = 1u << 0,
  // The normal program emits its first fixup-to-IB barrier before this op.
  IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_FIXUP_BARRIER = 1u << 1,
  // The profile program emits its first fixup-to-IB barrier before this op.
  IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_PROFILE_FIXUP_BARRIER = 1u << 2,
  // The target is resolved through a queue_execute binding table.
  IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_DYNAMIC_TARGET = 1u << 3,
} iree_hal_amdgpu_pm4_atomic_record_flag_bits_t;

// Static or dynamic target captured by an atomic record.
typedef struct iree_hal_amdgpu_pm4_atomic_target_record_t {
  // Static device address, or dynamic binding-relative byte offset.
  uint64_t value;
  // Original target byte offset reported in profile metadata.
  uint64_t profile_offset;
  // Dynamic binding slot, or UINT32_MAX for a static target.
  uint32_t binding_slot;
  // Reserved padding; must be zero.
  uint32_t reserved0;
} iree_hal_amdgpu_pm4_atomic_target_record_t;

// Compact atomic record replayed into the resident PM4 program.
typedef struct iree_hal_amdgpu_pm4_atomic_record_t {
  // Common command-record header.
  iree_hal_amdgpu_pm4_command_record_header_t header;
  // Static or dynamic atomic target.
  iree_hal_amdgpu_pm4_atomic_target_record_t target;
  // Parameters selected by |header.opcode|.
  union {
    // Atomic wait parameters.
    iree_hal_atomic_wait_params_t wait;
    // Atomic store parameters.
    iree_hal_atomic_store_params_t store;
    // Atomic read-modify-write parameters.
    iree_hal_atomic_rmw_params_t rmw;
  } params;
  // HAL command ordinal within this command buffer.
  uint32_t command_index;
  // Byte offset of fallback kernargs in resident template storage.
  uint32_t template_offset;
  // iree_hal_amdgpu_pm4_atomic_record_flag_bits_t mask.
  iree_hal_amdgpu_pm4_atomic_record_flags_t flags;
  // Acquire fence scope for a pending execution barrier.
  iree_hsa_fence_scope_t barrier_acquire_scope;
  // Release fence scope for a pending execution barrier.
  iree_hsa_fence_scope_t barrier_release_scope;
  // Reserved padding; must be zero.
  uint32_t reserved0;
} iree_hal_amdgpu_pm4_atomic_record_t;

// Exact resident storage contribution of one atomic record.
typedef struct iree_hal_amdgpu_pm4_atomic_record_measurement_t {
  // Dwords appended to the normal execution program.
  uint32_t program_dword_count;
  // Dwords appended to the profile execution program.
  uint32_t profile_program_dword_count;
  // Total template byte length after appending this operation's kernargs.
  iree_host_size_t template_byte_length;
  // Dynamic fixup entries appended to the normal program.
  uint32_t fixup_entry_count;
  // Dynamic fixup entries appended to the profile program.
  uint32_t profile_fixup_entry_count;
} iree_hal_amdgpu_pm4_atomic_record_measurement_t;

// Selects the materialization destination for an atomic record.
typedef uint32_t iree_hal_amdgpu_pm4_atomic_materialization_flags_t;
typedef enum iree_hal_amdgpu_pm4_atomic_materialization_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_ATOMIC_MATERIALIZATION_FLAG_NONE = 0u,
  // Reuses normal-program templates while emitting the profile program.
  IREE_HAL_AMDGPU_PM4_ATOMIC_MATERIALIZATION_FLAG_PROFILE = 1u << 0,
} iree_hal_amdgpu_pm4_atomic_materialization_flag_bits_t;

// Mutable PM4 program state used while materializing atomic records.
typedef struct iree_hal_amdgpu_pm4_atomic_materialization_state_t {
  // Destination PM4 program builder.
  iree_hal_amdgpu_pm4_dword_builder_t* dword_builder;
  // Shared resident kernarg-template builder.
  iree_hal_amdgpu_pm4_byte_builder_t* template_builder;
  // Destination dynamic binding fixup builder.
  iree_hal_amdgpu_pm4_fixup_entry_builder_t* fixup_builder;
  // Immutable launch metadata for fallback atomic kernels.
  const iree_hal_amdgpu_device_atomic_pm4_context_t* atomic_context;
  // Device-visible base of shared resident kernarg templates.
  IREE_AMDGPU_DEVICE_PTR uint8_t* template_base;
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
  // iree_hal_amdgpu_pm4_atomic_materialization_flag_bits_t mask.
  iree_hal_amdgpu_pm4_atomic_materialization_flags_t flags;
} iree_hal_amdgpu_pm4_atomic_materialization_state_t;

// Materialized dword classes accumulated for publication diagnostics.
typedef struct iree_hal_amdgpu_pm4_atomic_materialization_stats_t {
  // Dwords emitted for execution/visibility barriers.
  uint32_t execution_barrier_dwords;
  // Dwords emitted for fixup-to-IB barriers.
  uint32_t fixup_barrier_dwords;
  // Dwords emitted for shader setup.
  uint32_t dispatch_setup_dwords;
  // Dwords emitted for user data.
  uint32_t dispatch_user_data_dwords;
  // Dwords emitted for direct dispatch.
  uint32_t dispatch_direct_dwords;
} iree_hal_amdgpu_pm4_atomic_materialization_stats_t;

// Initializes a compact fallback atomic wait record.
void iree_hal_amdgpu_pm4_atomic_record_initialize_wait(
    iree_hal_amdgpu_pm4_atomic_target_record_t target,
    iree_hal_atomic_wait_params_t params, uint32_t command_index,
    iree_hal_amdgpu_pm4_atomic_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_atomic_record_t* out_record);

// Initializes a compact fallback atomic store record.
void iree_hal_amdgpu_pm4_atomic_record_initialize_store(
    iree_hal_amdgpu_pm4_atomic_target_record_t target,
    iree_hal_atomic_store_params_t params, uint32_t command_index,
    iree_hal_amdgpu_pm4_atomic_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_atomic_record_t* out_record);

// Initializes a compact fallback atomic RMW record.
void iree_hal_amdgpu_pm4_atomic_record_initialize_rmw(
    iree_hal_amdgpu_pm4_atomic_target_record_t target,
    iree_hal_atomic_rmw_params_t params, uint32_t command_index,
    iree_hal_amdgpu_pm4_atomic_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_atomic_record_t* out_record);

// Measures |record| and assigns its resident template offset.
iree_status_t iree_hal_amdgpu_pm4_atomic_record_measure(
    iree_hal_amdgpu_pm4_atomic_record_t* record,
    const iree_hal_amdgpu_device_atomic_pm4_context_t* atomic_context,
    iree_hal_amdgpu_vendor_packet_capability_flags_t vendor_packet_capabilities,
    iree_host_size_t current_template_byte_length,
    bool has_previous_launch_state,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* previous_launch_state,
    iree_hal_amdgpu_pm4_atomic_record_measurement_t* out_measurement);

// Returns the fallback launch selected by |record|.
const iree_hal_amdgpu_device_atomic_pm4_launch_t*
iree_hal_amdgpu_pm4_atomic_record_launch(
    const iree_hal_amdgpu_pm4_atomic_record_t* record,
    const iree_hal_amdgpu_device_atomic_pm4_context_t* atomic_context);

// Materializes |record| into the selected PM4 program and fixup builders.
iree_status_t iree_hal_amdgpu_pm4_atomic_record_materialize(
    const iree_hal_amdgpu_pm4_atomic_record_t* record,
    iree_hal_amdgpu_pm4_atomic_materialization_state_t* state,
    iree_hal_amdgpu_pm4_atomic_materialization_stats_t* out_stats);

// Initializes one profile-visible atomic operation record.
void iree_hal_amdgpu_pm4_atomic_record_initialize_profile_operation(
    uint64_t command_buffer_id,
    const iree_hal_amdgpu_pm4_atomic_record_t* atomic_record,
    iree_hal_profile_command_operation_record_t* out_record);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_ATOMIC_H_
