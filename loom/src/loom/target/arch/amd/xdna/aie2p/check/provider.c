// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/check/provider.h"

#include "loom/target/arch/amd/xdna/aie2p/check/array_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/provider.h"

static const loom_check_emit_provider_t* const kAie2pCheckEmitProviders[] = {
    &loom_aie2p_array_plan_check_emit_provider,
};

const loom_check_provider_t loom_aie2p_check_provider = {
    .name = IREE_SVL("amd-xdna-aie2p"),
    .target_provider = &loom_aie2p_target_provider,
    .emit_providers = kAie2pCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kAie2pCheckEmitProviders),
};
