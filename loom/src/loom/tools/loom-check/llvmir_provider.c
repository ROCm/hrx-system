// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/llvmir_provider.h"

#include "loom/target/emit/llvmir/check/loom_check.h"

static const loom_check_emit_provider_t* const kLoomLlvmirCheckEmitProviders[] =
    {
        &loom_llvmir_loom_check_emit_provider,
};

static const loom_check_requirement_provider_t* const
    kLoomLlvmirCheckRequirementProviders[] = {
        &loom_llvmir_loom_check_requirement_provider,
};

const loom_check_provider_t loom_llvmir_check_provider = {
    .name = IREE_SVL("llvmir"),
    .emit_providers = kLoomLlvmirCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomLlvmirCheckEmitProviders),
    .requirement_providers = kLoomLlvmirCheckRequirementProviders,
    .requirement_provider_count =
        IREE_ARRAYSIZE(kLoomLlvmirCheckRequirementProviders),
};
