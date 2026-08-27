// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured SPIR-V target profile.

#ifndef LOOM_TARGET_ARCH_SPIRV_PROFILE_H_
#define LOOM_TARGET_ARCH_SPIRV_PROFILE_H_

#include "iree/base/api.h"
#include "loom/target/arch/spirv/facts.h"
#include "loom/target/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// SPIR-V family profile identity.
extern const loom_target_profile_type_t loom_spirv_target_profile_type;

typedef struct loom_spirv_target_profile_t {
  // Target-neutral family identity and bundle projection.
  loom_target_profile_t base;

  // Cooperative operation rows selected for the active target, or NULL to use
  // the static modeled rows implied by the target feature set.
  const loom_spirv_cooperative_property_set_t* cooperative_properties;
} loom_spirv_target_profile_t;

// Initializes an immutable SPIR-V target profile.
void loom_spirv_target_profile_initialize(
    const loom_target_bundle_t* target_bundle,
    const loom_spirv_cooperative_property_set_t* cooperative_properties,
    loom_spirv_target_profile_t* out_profile);

// Returns |profile| as a SPIR-V profile, or NULL for another family.
static inline const loom_spirv_target_profile_t* loom_spirv_target_profile_cast(
    const loom_target_profile_t* profile) {
  return loom_target_profile_has_type(profile, &loom_spirv_target_profile_type)
             ? (const loom_spirv_target_profile_t*)profile
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_PROFILE_H_
