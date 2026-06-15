// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/check/provider.h"

#include "loom/target/arch/x86/provider.h"
#include "loom/target/emit/native/x86/check/loom_check.h"

static const loom_check_emit_provider_t* const kLoomX86CheckEmitProviders[] = {
    &loom_x86_native_loom_check_emit_provider,
};

const loom_check_provider_t loom_x86_check_provider = {
    .name = IREE_SVL("x86"),
    .target_provider = &loom_x86_target_provider,
    .emit_providers = kLoomX86CheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomX86CheckEmitProviders),
};
