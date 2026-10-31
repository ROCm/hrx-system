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
  facts->base.selector = profile->kind;
  facts->device_profile = profile->device_profile;
  facts->array_contract_unselected =
      profile->kind == LOOM_AIE2P_TARGET_KIND_ARRAY;
  return iree_ok_status();
}

static bool loom_aie2p_target_profile_equal(const loom_target_profile_t* lhs,
                                            const loom_target_profile_t* rhs) {
  const loom_aie2p_target_profile_t* lhs_profile =
      (const loom_aie2p_target_profile_t*)lhs;
  const loom_aie2p_target_profile_t* rhs_profile =
      (const loom_aie2p_target_profile_t*)rhs;
  // Device profiles are canonical immutable rows in the deployment catalog.
  return loom_target_profile_equal_common(lhs, rhs) &&
         lhs_profile->kind == rhs_profile->kind &&
         lhs_profile->device_profile == rhs_profile->device_profile;
}

const loom_target_profile_type_t loom_aie2p_target_profile_type = {
    .name = IREE_SVL("amd.xdna.aie2p"),
    .fact_type = &loom_aie2p_target_fact_type,
    .project_facts = loom_aie2p_target_profile_project_facts,
    .equal = loom_aie2p_target_profile_equal,
};

#include "loom/target/arch/amd/xdna/aie2p_profile_tables.inl"

static const loom_aie2p_target_profile_t kCoreProfile = {
    .base =
        {
            .type = &loom_aie2p_target_profile_type,
            .target_bundle = &loom_aie2p_core_target_bundle,
        },
    .kind = LOOM_AIE2P_TARGET_KIND_CORE,
};

iree_status_t loom_aie2p_target_profile_select(
    iree_string_view_t selector,
    const loom_aie2p_target_profile_t** out_profile) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = NULL;
  selector = iree_string_view_trim(selector);
  if (iree_string_view_equal(selector, IREE_SV("core"))) {
    *out_profile = &kCoreProfile;
    return iree_ok_status();
  }
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
