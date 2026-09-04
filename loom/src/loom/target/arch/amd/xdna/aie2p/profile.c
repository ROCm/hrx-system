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

#include "loom/target/arch/amd/xdna/device/aie2p_profile_tables.inl"

iree_status_t loom_aie2p_target_profile_select(
    iree_string_view_t selector,
    const loom_aie2p_target_profile_t** out_profile) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = NULL;
  selector = iree_string_view_trim(selector);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kLoomAie2pTargetProfiles);
       ++i) {
    const loom_aie2p_target_profile_t* profile = &kLoomAie2pTargetProfiles[i];
    if (iree_string_view_equal(
            selector, iree_make_cstring_view(profile->device_profile->key))) {
      *out_profile = profile;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown AIE2P target selector '%.*s'",
                          (int)selector.size, selector.data);
}
