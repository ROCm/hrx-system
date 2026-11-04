// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/check/test_provider.h"

#include "loom/target/arch/amdgpu/check/occupancy.h"

static const loom_check_emit_provider_t* const
    kLoomAmdgpuCheckTestEmitProviders[] = {
        &loom_amdgpu_occupancy_loom_check_emit_provider.base,
};

const loom_check_provider_t loom_amdgpu_arch_check_test_provider = {
    .name = IREE_SVL("amdgpu-arch-test"),
    .emit_providers = kLoomAmdgpuCheckTestEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomAmdgpuCheckTestEmitProviders),
};
