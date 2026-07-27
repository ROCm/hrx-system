// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target math legalization policy.

#ifndef LOOM_TARGET_ARCH_AMDGPU_MATH_POLICY_H_
#define LOOM_TARGET_ARCH_AMDGPU_MATH_POLICY_H_

#include "loom/target/math_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes a target-owned registry mapping AMDGPU target-contract keys to
// their math legalization policies.
void loom_amdgpu_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_MATH_POLICY_H_
