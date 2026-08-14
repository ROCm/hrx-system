// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_AQL_ATOMIC_H_
#define IREE_HAL_DRIVERS_AMDGPU_AQL_ATOMIC_H_

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/device/atomic.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Records an atomic wait into |builder| using a captured target descriptor.
iree_status_t iree_hal_amdgpu_aql_atomic_record_wait(
    iree_hal_amdgpu_aql_program_builder_t* builder,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_amdgpu_command_buffer_atomic_target_t target,
    iree_hal_atomic_wait_params_t params);

// Records an atomic store into |builder| using a captured target descriptor.
iree_status_t iree_hal_amdgpu_aql_atomic_record_store(
    iree_hal_amdgpu_aql_program_builder_t* builder,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_amdgpu_command_buffer_atomic_target_t target,
    iree_hal_atomic_store_params_t params);

// Records an atomic RMW into |builder| using a captured target descriptor.
iree_status_t iree_hal_amdgpu_aql_atomic_record_rmw(
    iree_hal_amdgpu_aql_program_builder_t* builder,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_amdgpu_command_buffer_atomic_target_t target,
    iree_hal_atomic_rmw_params_t params);

// Resolves and validates the target of an atomic command and populates one
// already-reserved dispatch packet and kernarg block. The caller owns packet
// header commit, completion-signal assignment, and queue publication.
iree_status_t iree_hal_amdgpu_aql_atomic_emplace_command(
    const iree_hal_amdgpu_device_kernels_t* kernels,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_command_header_t* command,
    iree_hsa_kernel_dispatch_packet_t* dispatch_packet, void* kernarg_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_ATOMIC_H_
