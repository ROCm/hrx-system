// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_AQL_BUFFER_REF_H_
#define IREE_HAL_DRIVERS_AMDGPU_AQL_BUFFER_REF_H_

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates |buffer_ref| and resolves it to an AMDGPU device pointer.
iree_status_t iree_hal_amdgpu_aql_resolve_buffer_ref_device_pointer(
    iree_hal_buffer_ref_t buffer_ref, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access, uint8_t** out_device_pointer);

// Resolves a recorded static or dynamic command-buffer reference and validates
// its range, usage, and access before returning its AMDGPU device pointer.
iree_status_t iree_hal_amdgpu_aql_resolve_command_buffer_ref(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_amdgpu_command_buffer_binding_kind_t kind, uint32_t ordinal,
    uint64_t offset, uint64_t length, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access,
    iree_hal_buffer_ref_t* out_buffer_ref, uint8_t** out_device_pointer);

// Resolves a retained static dispatch binding source to an AMDGPU device
// pointer after staged transient-buffer materialization has completed.
iree_status_t iree_hal_amdgpu_aql_resolve_static_binding_source_pointer(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source,
    uint64_t* out_binding_pointer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_BUFFER_REF_H_
