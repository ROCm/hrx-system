// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Developer convenience registry that links every current target-low descriptor
// package. Core tools should not depend on this package.

#ifndef LOOM_TARGET_ALL_LOW_REGISTRY_H_
#define LOOM_TARGET_ALL_LOW_REGISTRY_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/legalization.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

void loom_all_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry);

void loom_all_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

// Returns every target-specific source-to-low legality provider linked into the
// all-target developer registry.
loom_target_low_legality_provider_list_t loom_all_low_legality_provider_list(
    void);

// Returns every target-specific legalization provider linked into the
// all-target developer registry.
loom_target_legalizer_provider_list_t loom_all_legalizer_provider_list(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ALL_LOW_REGISTRY_H_
