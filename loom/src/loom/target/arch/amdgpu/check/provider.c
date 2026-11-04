// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/check/provider.h"

#include "loom/target/arch/amdgpu/check/bank_service.h"
static const loom_check_emit_provider_t* const kLoomAmdgpuCheckEmitProviders[] =
    {
        &loom_amdgpu_bank_service_loom_check_emit_provider,
};

const loom_check_provider_t loom_amdgpu_arch_check_provider = {
    .name = IREE_SVL("amdgpu-arch"),
    .emit_providers = kLoomAmdgpuCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomAmdgpuCheckEmitProviders),
};
