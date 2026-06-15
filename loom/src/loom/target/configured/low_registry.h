// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Configured target-low registries selected by //loom/config/target.

#ifndef LOOM_TARGET_CONFIGURED_LOW_REGISTRY_H_
#define LOOM_TARGET_CONFIGURED_LOW_REGISTRY_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/legalization.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

void loom_configured_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry);

void loom_configured_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

// Returns the configured target-specific source-to-low legality providers.
loom_target_low_legality_provider_list_t
loom_configured_low_legality_provider_list(void);

// Returns the configured target-specific legalization providers.
loom_target_legalizer_provider_list_t loom_configured_legalizer_provider_list(
    void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_CONFIGURED_LOW_REGISTRY_H_
