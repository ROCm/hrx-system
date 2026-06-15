// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/check/provider.h"

#include "loom/target/emit/spirv/check/loom_check.h"

static const loom_check_emit_provider_t* const kLoomSpirvCheckEmitProviders[] =
    {
        &loom_spirv_loom_check_emit_provider,
};

static const loom_check_requirement_provider_t* const
    kLoomSpirvCheckRequirementProviders[] = {
        &loom_spirv_loom_check_requirement_provider,
};

const loom_check_provider_t loom_spirv_emit_check_provider = {
    .name = IREE_SVL("spirv-emit"),
    .emit_providers = kLoomSpirvCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomSpirvCheckEmitProviders),
    .requirement_providers = kLoomSpirvCheckRequirementProviders,
    .requirement_provider_count =
        IREE_ARRAYSIZE(kLoomSpirvCheckRequirementProviders),
};
