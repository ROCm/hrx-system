// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured AMD XDNA AIE2P target profile.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROFILE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROFILE_H_

#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// AIE2P target profile identity.
extern const loom_target_profile_type_t loom_aie2p_target_profile_type;

typedef struct loom_aie2p_target_profile_t {
  // Target-neutral family identity and array-program bundle projection.
  loom_target_profile_t base;

  // Exact physical deployment profile.
  const loom_xdna_device_profile_t* device_profile;
} loom_aie2p_target_profile_t;

// Initializes an AIE2P profile for one exact physical deployment profile.
iree_status_t loom_aie2p_target_profile_initialize(
    const loom_xdna_device_profile_t* device_profile,
    loom_aie2p_target_profile_t* out_profile);

// Returns |profile| as an AIE2P profile, or NULL for another family.
static inline const loom_aie2p_target_profile_t* loom_aie2p_target_profile_cast(
    const loom_target_profile_t* profile) {
  return loom_target_profile_has_type(profile, &loom_aie2p_target_profile_type)
             ? (const loom_aie2p_target_profile_t*)profile
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROFILE_H_
