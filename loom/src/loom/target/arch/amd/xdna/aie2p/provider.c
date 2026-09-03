// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/provider.h"

#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/artifact_emitter.h"
#include "loom/target/arch/amd/xdna/aie2p/low_verify.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/registry.h"

static const loom_target_emitter_t* const kAie2pTargetEmitters[] = {
    &loom_aie2p_xdna_emitter,
};

static const loom_low_verify_provider_t* const kAie2pLowVerifyProviders[] = {
    &loom_aie2p_low_verify_provider,
};

const loom_target_provider_t loom_aie2p_target_provider = {
    .register_context = loom_aie2p_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_aie2p_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_aie2p_low_lower_policy_registry_initialize,
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kAie2pLowVerifyProviders),
            .values = kAie2pLowVerifyProviders,
        },
    .emitter_list =
        {
            .values = kAie2pTargetEmitters,
            .count = IREE_ARRAYSIZE(kAie2pTargetEmitters),
        },
};

static const loom_target_provider_t* const kAie2pTargetProviders[] = {
    &loom_aie2p_target_provider,
};

const loom_target_provider_set_t loom_aie2p_target_provider_set = {
    .providers = kAie2pTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kAie2pTargetProviders),
};
