// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_ATOMIC_H_
#define IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_ATOMIC_H_

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/device/atomic_pm4.h"
#include "iree/hal/drivers/amdgpu/pm4_command_record.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Properties controlling materialization of one atomic record.
typedef uint32_t iree_hal_amdgpu_pm4_atomic_record_flags_t;
typedef enum iree_hal_amdgpu_pm4_atomic_record_flag_bits_e {
  IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_NONE =
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_NONE,
  IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_EXECUTION_BARRIER =
      IREE_HAL_AMDGPU_PM4_COMMAND_RECORD_FLAG_EXECUTION_BARRIER,
  // The target is resolved through a queue_execute binding table.
  IREE_HAL_AMDGPU_PM4_ATOMIC_RECORD_FLAG_DYNAMIC_TARGET = 1u << 1,
} iree_hal_amdgpu_pm4_atomic_record_flag_bits_t;

// Encoding selected for one PM4 atomic record.
typedef enum iree_hal_amdgpu_pm4_atomic_lowering_e {
  // One-workitem dispatch of the target-compiled fallback kernel.
  IREE_HAL_AMDGPU_PM4_ATOMIC_LOWERING_FALLBACK = 0,
  // Native command-processor packet in the resident PM4 program.
  IREE_HAL_AMDGPU_PM4_ATOMIC_LOWERING_NATIVE = 1,
} iree_hal_amdgpu_pm4_atomic_lowering_t;

// Compact atomic record replayed into the resident PM4 program.
typedef struct iree_hal_amdgpu_pm4_atomic_record_t {
  // Common command-record header.
  iree_hal_amdgpu_pm4_command_record_header_t header;
  // Static or dynamic atomic target.
  iree_hal_amdgpu_pm4_buffer_ref_record_t target;
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
  // Record-time lowering selected from the physical device capabilities.
  iree_hal_amdgpu_pm4_atomic_lowering_t lowering;
} iree_hal_amdgpu_pm4_atomic_record_t;

// Initializes a compact atomic wait record.
void iree_hal_amdgpu_pm4_atomic_record_initialize_wait(
    iree_hal_amdgpu_pm4_buffer_ref_record_t target,
    iree_hal_atomic_wait_params_t params, uint32_t command_index,
    iree_hal_amdgpu_pm4_atomic_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_atomic_record_t* out_record);

// Initializes a compact atomic store record.
void iree_hal_amdgpu_pm4_atomic_record_initialize_store(
    iree_hal_amdgpu_pm4_buffer_ref_record_t target,
    iree_hal_atomic_store_params_t params, uint32_t command_index,
    iree_hal_amdgpu_pm4_atomic_record_flags_t flags,
    iree_hsa_fence_scope_t barrier_acquire_scope,
    iree_hsa_fence_scope_t barrier_release_scope,
    iree_hal_amdgpu_pm4_atomic_record_t* out_record);

// Initializes a compact atomic RMW record.
void iree_hal_amdgpu_pm4_atomic_record_initialize_rmw(
    iree_hal_amdgpu_pm4_buffer_ref_record_t target,
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
    uint32_t current_program_dword_count,
    iree_host_size_t current_template_byte_length,
    bool has_previous_launch_state,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* previous_launch_state,
    iree_hal_amdgpu_pm4_command_record_measurement_t* out_measurement);

// Returns the fallback launch selected by |record|, or NULL for native records.
const iree_hal_amdgpu_device_kernel_pm4_launch_t*
iree_hal_amdgpu_pm4_atomic_record_launch(
    const iree_hal_amdgpu_pm4_atomic_record_t* record,
    const iree_hal_amdgpu_device_atomic_pm4_context_t* atomic_context);

// Materializes |record| into the selected PM4 program and fixup builders.
iree_status_t iree_hal_amdgpu_pm4_atomic_record_materialize(
    const iree_hal_amdgpu_pm4_atomic_record_t* record,
    const iree_hal_amdgpu_device_atomic_pm4_context_t* atomic_context,
    iree_hal_amdgpu_pm4_command_materialization_state_t* state,
    iree_hal_amdgpu_pm4_command_materialization_stats_t* out_stats);

// Initializes one profile-visible atomic operation record.
void iree_hal_amdgpu_pm4_atomic_record_initialize_profile_operation(
    uint64_t command_buffer_id,
    const iree_hal_amdgpu_pm4_atomic_record_t* atomic_record,
    iree_hal_profile_command_operation_record_t* out_record);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_ATOMIC_H_
