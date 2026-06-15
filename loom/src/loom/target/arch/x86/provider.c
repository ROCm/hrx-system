// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/provider.h"

#include "loom/target/arch/x86/descriptors/low_registry.h"
#include "loom/target/arch/x86/legalization.h"
#include "loom/target/arch/x86/lower/lower.h"
#include "loom/target/arch/x86/math_policy.h"
#include "loom/target/arch/x86/ops/registry.h"

static const loom_target_legalizer_provider_t* kLoomX86LegalizerProviders[] = {
    &loom_x86_target_legalizer_provider_storage,
};

const loom_target_provider_t loom_x86_target_provider = {
    .register_context = loom_x86_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_x86_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_x86_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry = loom_x86_math_policy_registry_initialize,
    .legalizer_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomX86LegalizerProviders),
            .values = kLoomX86LegalizerProviders,
        },
};

static const loom_target_provider_t* const kLoomX86TargetProviders[] = {
    &loom_x86_target_provider,
};

const loom_target_provider_set_t loom_x86_target_provider_set = {
    .providers = kLoomX86TargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomX86TargetProviders),
};
