// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P target math legalization policy.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_MATH_POLICY_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_MATH_POLICY_H_

#include "loom/target/math_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the target-owned math policy registry for AIE2P contracts.
void loom_aie2p_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_MATH_POLICY_H_
