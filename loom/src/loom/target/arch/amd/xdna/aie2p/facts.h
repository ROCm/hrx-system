// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable AMD XDNA AIE2P target facts.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_FACTS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_FACTS_H_

#include "loom/target/arch/amd/xdna/device/profile.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_aie2p_target_facts_t {
  // Target-neutral facts shared by all target families.
  loom_target_facts_t base;

  // Exact physical deployment profile, or NULL for a generic authored target.
  const loom_xdna_device_profile_t* device_profile;
} loom_aie2p_target_facts_t;

// Static fact type shared by AIE2P target records and structured profiles.
extern const loom_target_fact_type_t loom_aie2p_target_fact_type;

// Returns |facts| as AIE2P facts, or NULL for another target family.
static inline const loom_aie2p_target_facts_t* loom_aie2p_target_facts_cast(
    const loom_target_facts_t* facts) {
  return facts != NULL && facts->fact_type == &loom_aie2p_target_fact_type
             ? (const loom_aie2p_target_facts_t*)facts
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_FACTS_H_
