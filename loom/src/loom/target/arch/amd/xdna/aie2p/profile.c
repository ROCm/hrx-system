// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/profile.h"

#include "loom/target/arch/amd/xdna/aie2p/ops/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/records/target_records.h"

static iree_status_t loom_aie2p_target_profile_project_facts(
    const loom_target_profile_t* base_profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* base_facts) {
  (void)arena;
  const loom_aie2p_target_profile_t* profile =
      loom_aie2p_target_profile_cast(base_profile);
  IREE_ASSERT(profile != NULL);
  loom_aie2p_target_facts_t* facts = (loom_aie2p_target_facts_t*)base_facts;
  facts->base.selector = LOOM_AIE2P_TARGET_KIND_ARRAY;
  facts->device_profile = profile->device_profile;
  return iree_ok_status();
}

const loom_target_profile_type_t loom_aie2p_target_profile_type = {
    .name = IREE_SVL("amd.xdna.aie2p"),
    .fact_type = &loom_aie2p_target_fact_type,
    .project_facts = loom_aie2p_target_profile_project_facts,
};

iree_status_t loom_aie2p_target_profile_initialize(
    const loom_xdna_device_profile_t* device_profile,
    loom_aie2p_target_profile_t* out_profile) {
  IREE_ASSERT_ARGUMENT(device_profile);
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = (loom_aie2p_target_profile_t){0};

  const loom_xdna_array_family_t* array_family =
      loom_xdna_device_profile_array_family(device_profile);
  if (array_family->architecture != LOOM_XDNA_ARCHITECTURE_AIE2P) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA device profile '%s' does not select the AIE2P architecture",
        device_profile->key);
  }

  const loom_target_bundle_t* array_bundle = loom_aie2p_array_target_bundle();

  *out_profile = (loom_aie2p_target_profile_t){
      .base =
          {
              .type = &loom_aie2p_target_profile_type,
              .target_bundle = array_bundle,
          },
      .device_profile = device_profile,
  };
  return iree_ok_status();
}
