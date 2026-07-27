// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/ireevm/provider.h"

#include "loom/target/arch/ireevm/descriptors/low_registry.h"
#include "loom/target/arch/ireevm/lower/lower.h"
#include "loom/target/arch/ireevm/math_policy.h"
#include "loom/target/arch/ireevm/ops/registry.h"
#include "loom/target/arch/ireevm/pass_registry.h"

const loom_target_provider_t loom_ireevm_target_provider = {
    .register_context = loom_ireevm_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_ireevm_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_ireevm_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_ireevm_math_policy_registry_initialize,
    .pass_registry = &loom_ireevm_pass_registry,
};

static const loom_target_provider_t* const kLoomIreeVmTargetProviders[] = {
    &loom_ireevm_target_provider,
};

const loom_target_provider_set_t loom_ireevm_target_provider_set = {
    .providers = kLoomIreeVmTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomIreeVmTargetProviders),
};
