// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL-kernel ABI materialization pass.
//
// This pass is the pass-pipeline surface over
// loom_amdgpu_hal_binding_materialize. It resolves each low function's target
// binding from IR facts, verifies the AMDGPU HAL ABI with structured
// diagnostics, and materializes the HAL-facing kernarg loads for selected
// AMDGPU low functions.

#ifndef LOOM_TARGET_ARCH_AMDGPU_HAL_BINDING_MATERIALIZATION_PASS_H_
#define LOOM_TARGET_ARCH_AMDGPU_HAL_BINDING_MATERIALIZATION_PASS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns static pass metadata for amdgpu-materialize-hal-kernel-abi.
const loom_pass_info_t* loom_amdgpu_materialize_hal_kernel_abi_pass_info(void);

// Returns static pass metadata for amdgpu-materialize-hal-buffer-descriptors.
const loom_pass_info_t*
loom_amdgpu_materialize_hal_buffer_descriptors_pass_info(void);

// Runs AMDGPU HAL-kernel ABI materialization on one target-low function.
iree_status_t loom_amdgpu_materialize_hal_kernel_abi_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function);

// Runs AMDGPU HAL buffer descriptor pseudo expansion on one target-low
// function.
iree_status_t loom_amdgpu_materialize_hal_buffer_descriptors_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_HAL_BINDING_MATERIALIZATION_PASS_H_
