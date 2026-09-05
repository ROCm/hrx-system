// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/execution_provider.h"

#include "loom/target/arch/spirv/provider.h"
#include "loom/tooling/target/spirv/device_provider.h"

const loom_run_execution_provider_t loom_spirv_vulkan_execution_provider = {
    .name = IREE_SVL("spirv"),
    .target_provider = &loom_spirv_target_provider,
    .device_provider = &loom_spirv_vulkan_device_provider,
};
