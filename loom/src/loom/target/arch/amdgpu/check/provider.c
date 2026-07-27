// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/check/provider.h"

#include "loom/target/arch/amdgpu/check/occupancy.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/emit/native/amdgpu/check/loom_check.h"

static const loom_check_emit_provider_t* const kLoomAmdgpuCheckEmitProviders[] =
    {
        &loom_amdgpu_occupancy_loom_check_emit_provider,
        &loom_amdgpu_native_loom_check_emit_provider,
};

const loom_check_provider_t loom_amdgpu_check_provider = {
    .name = IREE_SVL("amdgpu"),
    .target_provider = &loom_amdgpu_target_provider,
    .emit_providers = kLoomAmdgpuCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomAmdgpuCheckEmitProviders),
};
