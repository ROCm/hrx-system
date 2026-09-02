// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/provider.h"

#include "loom/target/arch/spirv/descriptors/low_registry.h"
#include "loom/target/arch/spirv/legalization.h"
#include "loom/target/arch/spirv/low_verify.h"
#include "loom/target/arch/spirv/lower/lower.h"
#include "loom/target/arch/spirv/math_policy.h"
#include "loom/target/arch/spirv/ops/registry.h"
#include "loom/target/arch/spirv/profile.h"

static const loom_low_verify_provider_t* const kLoomSpirvLowVerifyProviders[] =
    {
        &loom_spirv_low_verify_provider,
};

static const loom_target_legalizer_provider_t* const
    kLoomSpirvLegalizerProviders[] = {
        &loom_spirv_target_legalizer_provider_storage,
};

const loom_target_provider_t loom_spirv_target_provider = {
    .profile_type = &loom_spirv_target_profile_type,
    .register_context = loom_spirv_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_spirv_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_spirv_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_spirv_math_policy_registry_initialize,
    .legalizer_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomSpirvLegalizerProviders),
            .values = kLoomSpirvLegalizerProviders,
        },
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomSpirvLowVerifyProviders),
            .values = kLoomSpirvLowVerifyProviders,
        },
};

static const loom_target_provider_t* const kLoomSpirvTargetProviders[] = {
    &loom_spirv_target_provider,
};

const loom_target_provider_set_t loom_spirv_target_provider_set = {
    .providers = kLoomSpirvTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomSpirvTargetProviders),
};
