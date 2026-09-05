// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/execution_provider.h"

#include "loom/target/arch/amdgpu/provider.h"
#include "loom/tooling/target/amdgpu/device_provider.h"

const loom_run_execution_provider_t loom_amdgpu_execution_provider = {
    .name = IREE_SVL("amdgpu"),
    .target_provider = &loom_amdgpu_target_provider,
    .device_provider = &loom_amdgpu_device_provider,
};
