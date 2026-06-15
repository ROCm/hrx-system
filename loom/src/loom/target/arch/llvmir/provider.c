// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/llvmir/provider.h"

#include "loom/ops/llvmir/registry.h"
#include "loom/target/arch/llvmir/descriptors/low_registry.h"
#include "loom/target/arch/llvmir/lower/lower.h"
#include "loom/target/arch/llvmir/math_policy.h"

const loom_target_provider_t loom_llvmir_target_provider = {
    .register_context = loom_llvmir_ops_register_dialect,
    .initialize_math_policy_registry =
        loom_llvmir_math_policy_registry_initialize,
    .initialize_low_descriptor_registry =
        loom_llvmir_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_llvmir_low_lower_policy_registry_initialize,
};

static const loom_target_provider_t* const kLoomLlvmirTargetProviders[] = {
    &loom_llvmir_target_provider,
};

const loom_target_provider_set_t loom_llvmir_target_provider_set = {
    .providers = kLoomLlvmirTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomLlvmirTargetProviders),
};
