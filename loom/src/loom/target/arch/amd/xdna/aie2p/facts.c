// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/facts.h"

static bool loom_aie2p_target_facts_satisfy_device_requirement(
    const loom_aie2p_target_facts_t* effective,
    const loom_aie2p_target_facts_t* requirement) {
  return requirement->device_profile == NULL ||
         effective->device_profile == requirement->device_profile;
}

static bool loom_aie2p_target_facts_satisfy_identity_requirement(
    const loom_target_facts_t* base_effective,
    const loom_target_facts_t* base_requirement) {
  const loom_aie2p_target_facts_t* effective =
      (const loom_aie2p_target_facts_t*)base_effective;
  const loom_aie2p_target_facts_t* requirement =
      (const loom_aie2p_target_facts_t*)base_requirement;
  return effective->base.selector == requirement->base.selector &&
         loom_aie2p_target_facts_satisfy_device_requirement(effective,
                                                            requirement);
}

static bool loom_aie2p_target_facts_satisfy_specialization_requirement(
    const loom_target_facts_t* base_effective,
    const loom_target_facts_t* base_requirement) {
  const loom_aie2p_target_facts_t* effective =
      (const loom_aie2p_target_facts_t*)base_effective;
  const loom_aie2p_target_facts_t* requirement =
      (const loom_aie2p_target_facts_t*)base_requirement;
  return loom_target_facts_structural_satisfy_specialization_requirement(
             base_effective, base_requirement) &&
         loom_aie2p_target_facts_satisfy_device_requirement(effective,
                                                            requirement);
}

static iree_string_view_t loom_aie2p_target_facts_identity_name(
    const loom_target_facts_t* base_facts) {
  const loom_aie2p_target_facts_t* facts =
      (const loom_aie2p_target_facts_t*)base_facts;
  return facts->device_profile != NULL
             ? iree_make_cstring_view(facts->device_profile->key)
             : facts->base.storage.bundle.name;
}

const loom_target_fact_type_t loom_aie2p_target_fact_type = {
    .name = IREE_SVL("amd.xdna.aie2p"),
    .storage_size = sizeof(loom_aie2p_target_facts_t),
    .satisfies_identity_requirement =
        loom_aie2p_target_facts_satisfy_identity_requirement,
    .satisfies_specialization_requirement =
        loom_aie2p_target_facts_satisfy_specialization_requirement,
    .identity_name = loom_aie2p_target_facts_identity_name,
};
