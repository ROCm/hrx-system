// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/check/provider.h"

#include "loom/target/emit/wasm/check/loom_check.h"

static const loom_check_emit_provider_t* const kLoomWasmCheckEmitProviders[] = {
    &loom_wasm_loom_check_emit_provider,
};

static const loom_check_requirement_provider_t* const
    kLoomWasmCheckRequirementProviders[] = {
        &loom_wasm_loom_check_requirement_provider,
};

const loom_check_provider_t loom_wasm_emit_check_provider = {
    .name = IREE_SVL("wasm-emit"),
    .emit_providers = kLoomWasmCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomWasmCheckEmitProviders),
    .requirement_providers = kLoomWasmCheckRequirementProviders,
    .requirement_provider_count =
        IREE_ARRAYSIZE(kLoomWasmCheckRequirementProviders),
};
