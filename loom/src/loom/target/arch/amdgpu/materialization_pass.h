// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low materialization pass.

#ifndef LOOM_TARGET_ARCH_AMDGPU_MATERIALIZATION_PASS_H_
#define LOOM_TARGET_ARCH_AMDGPU_MATERIALIZATION_PASS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns static pass metadata for amdgpu-materialize-target-low.
const loom_pass_info_t* loom_amdgpu_materialize_target_low_pass_info(void);

// Returns static pass metadata for amdgpu-materialize-source-low-artifacts.
const loom_pass_info_t* loom_amdgpu_materialize_source_low_artifacts_pass_info(
    void);

// Materializes target-low structural operations and the selected function ABI.
iree_status_t loom_amdgpu_materialize_target_low_run(loom_pass_t* pass,
                                                     loom_module_t* module,
                                                     loom_func_like_t function);

// Materializes target-low structural operations before printing artifacts.
iree_status_t loom_amdgpu_materialize_source_low_artifacts_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_MATERIALIZATION_PASS_H_
