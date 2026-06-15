// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/execution/provider.h"

#include "loom/target/arch/amdgpu/provider.h"
#include "loom/tooling/execution/hal/execution_backend.h"
#include "loom/tooling/target/amdgpu/artifact_provider.h"

static const loom_run_hal_execution_backend_t kLoomAmdgpuHalExecutionBackend = {
    .base =
        {
            .name = IREE_SVL("amdgpu-hal"),
            .flags = LOOM_RUN_EXECUTION_BACKEND_FLAG_HAL_OPTIONS,
            .probe = loom_run_hal_execution_backend_probe,
            .run_one_shot = loom_run_hal_execution_backend_run_one_shot,
        },
    .artifact_provider = &loom_amdgpu_hal_artifact_provider,
};

static const loom_run_execution_backend_t* const
    kLoomAmdgpuExecutionBackends[] = {
        &kLoomAmdgpuHalExecutionBackend.base,
};

const loom_run_execution_provider_t loom_amdgpu_hal_execution_provider = {
    .name = IREE_SVL("amdgpu"),
    .target_provider = &loom_amdgpu_target_provider,
    .execution_backends = kLoomAmdgpuExecutionBackends,
    .execution_backend_count = IREE_ARRAYSIZE(kLoomAmdgpuExecutionBackends),
};
