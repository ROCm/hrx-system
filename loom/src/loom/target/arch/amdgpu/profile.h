// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured AMDGPU target profile.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PROFILE_H_
#define LOOM_TARGET_ARCH_AMDGPU_PROFILE_H_

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// AMDGPU family profile identity.
extern const loom_target_profile_type_t loom_amdgpu_target_profile_type;

// Structured external AMDGPU target refinement profile.
typedef struct loom_amdgpu_target_profile_t {
  // Target-neutral family identity and bundle projection.
  loom_target_profile_t base;

  // Structured target and AMDHSA feature identity.
  loom_amdgpu_target_identity_t identity;
} loom_amdgpu_target_profile_t;

// Initializes an AMDGPU profile and its target-neutral bundle projection.
iree_status_t loom_amdgpu_target_profile_initialize(
    const loom_amdgpu_target_identity_t* identity,
    loom_amdgpu_target_profile_t* out_profile);

// Selects the immutable profile matching one AMDGPU target selector.
//
// The returned profile has process lifetime. Target names and target-ID feature
// suffixes use the same syntax as AMDGPU artifact keys.
iree_status_t loom_amdgpu_target_profile_select(
    iree_string_view_t selector,
    const loom_amdgpu_target_profile_t** out_profile);

// Returns |profile| as an AMDGPU profile, or NULL for another family.
static inline const loom_amdgpu_target_profile_t*
loom_amdgpu_target_profile_cast(const loom_target_profile_t* profile) {
  return loom_target_profile_has_type(profile, &loom_amdgpu_target_profile_type)
             ? (const loom_amdgpu_target_profile_t*)profile
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PROFILE_H_
