// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured AMDGPU target identity.

#ifndef LOOM_TARGET_ARCH_AMDGPU_TARGET_IDENTITY_H_
#define LOOM_TARGET_ARCH_AMDGPU_TARGET_IDENTITY_H_

#include "loom/target/arch/amdgpu/target_info_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Complete structured identity shared by profiles, compiler facts, and
// target-native artifact construction.
typedef struct loom_amdgpu_target_identity_t {
  // Exact or generic target selected for this identity.
  const loom_amdgpu_target_info_t* target;

  // Normalized AMDHSA target-ID feature states.
  loom_amdgpu_amdhsa_feature_states_t amdhsa_features;
} loom_amdgpu_target_identity_t;

// Initializes the normalized default identity for |target|.
//
// Supported AMDHSA modes remain unconstrained and unsupported modes are
// explicit.
void loom_amdgpu_target_identity_initialize(
    const loom_amdgpu_target_info_t* target,
    loom_amdgpu_target_identity_t* out_identity);

// Returns whether two identities select the same canonical target and every
// known target-ID feature state. Catalog records from separate linked images
// of the same compiler build compare by target kind, not storage address.
bool loom_amdgpu_target_identity_equal(
    const loom_amdgpu_target_identity_t* lhs,
    const loom_amdgpu_target_identity_t* rhs);

// Returns whether |effective| refines every processor and target-ID feature
// requirement carried by |requirement|.
bool loom_amdgpu_target_identity_satisfies_requirement(
    const loom_amdgpu_target_identity_t* effective,
    const loom_amdgpu_target_identity_t* requirement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_TARGET_IDENTITY_H_
