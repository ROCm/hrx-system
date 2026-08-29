// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMD XDNA AIE2P source-to-target-Low lowering policy.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_LOWER_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_LOWER_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the AIE2P compute-tile core lowering policy.
//
// Logical i1 and scalar i32 values occupy one eR register. Source operations
// are selected through the generated AIE2P core descriptor rules.
const loom_low_lower_policy_t* loom_aie2p_core_low_lower_policy(void);

// Initializes the AIE2P target-contract to lowering-policy registry.
void loom_aie2p_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_LOWER_H_
