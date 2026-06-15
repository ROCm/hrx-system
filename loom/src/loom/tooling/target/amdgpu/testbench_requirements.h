// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL requirement providers for Loom testbench execution.

#ifndef LOOM_TOOLING_TARGET_AMDGPU_TESTBENCH_REQUIREMENTS_H_
#define LOOM_TOOLING_TARGET_AMDGPU_TESTBENCH_REQUIREMENTS_H_

#include "loom/tooling/testbench/requirements.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_testbench_context_t
    loom_run_hal_testbench_context_t;

// Initializes the hal.amdgpu.descriptor_set requirement provider.
void loom_amdgpu_hal_testbench_requirement_provider_initialize(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_t* out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_AMDGPU_TESTBENCH_REQUIREMENTS_H_
