// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/profile.h"

#include "loom/target/arch/spirv/records/target_records.h"

static iree_status_t loom_spirv_target_profile_project_facts(
    const loom_target_profile_t* base_profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* base_facts) {
  const loom_spirv_target_profile_t* profile =
      loom_spirv_target_profile_cast(base_profile);
  IREE_ASSERT(profile != NULL);
  (void)profile;
  (void)arena;
  loom_spirv_target_facts_t* facts = (loom_spirv_target_facts_t*)base_facts;
  facts->base.selector = LOOM_SPIRV_TARGET_KIND_VULKAN1_3;
  loom_spirv_feature_set_t feature_set = {
      .atom_bits = facts->base.storage.config.contract_feature_bits,
  };
  loom_spirv_cooperative_property_set_prepare(&feature_set,
                                              &facts->cooperative_properties);
  return iree_ok_status();
}

const loom_target_profile_type_t loom_spirv_target_profile_type = {
    .name = IREE_SVL("spirv"),
    .fact_type = &loom_spirv_target_fact_type,
    .project_facts = loom_spirv_target_profile_project_facts,
};

static const loom_spirv_target_profile_t kSpirvVulkan13BdaTargetProfile = {
    .base =
        {
            .type = &loom_spirv_target_profile_type,
            .target_bundle = &loom_spirv_low_target_bundle_vulkan1_3,
        },
};

void loom_spirv_target_profile_initialize(
    const loom_target_bundle_t* target_bundle,
    loom_spirv_target_profile_t* out_profile) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = (loom_spirv_target_profile_t){
      .base =
          {
              .type = &loom_spirv_target_profile_type,
              .target_bundle = target_bundle,
          },
  };
}

iree_status_t loom_spirv_target_profile_select(
    iree_string_view_t selector,
    const loom_spirv_target_profile_t** out_profile) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = NULL;
  if (!iree_string_view_equal(selector, IREE_SV("vulkan1.3+bda"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unknown SPIR-V target selector '%.*s'; expected vulkan1.3+bda",
        (int)selector.size, selector.data);
  }
  *out_profile = &kSpirvVulkan13BdaTargetProfile;
  return iree_ok_status();
}
