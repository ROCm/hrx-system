// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V/Vulkan HAL requirement providers for Loom testbench execution.

#ifndef LOOM_TOOLING_EXECUTION_HAL_SPIRV_TESTBENCH_REQUIREMENTS_H_
#define LOOM_TOOLING_EXECUTION_HAL_SPIRV_TESTBENCH_REQUIREMENTS_H_

#include "loom/tooling/testbench/requirements.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_testbench_context_t
    loom_run_hal_testbench_context_t;

// Initializes the hal.spirv.cooperative_matrix row requirement provider.
void loom_spirv_vulkan_hal_testbench_requirement_provider_initialize(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_t* out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_SPIRV_TESTBENCH_REQUIREMENTS_H_
