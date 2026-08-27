// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/facts.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/target.h"

static void loom_amdgpu_target_facts_project(
    const loom_target_record_view_t* record, loom_target_facts_t* base_facts) {
  loom_amdgpu_target_facts_t* facts = (loom_amdgpu_target_facts_t*)base_facts;
  const loom_amdgpu_target_info_t* target =
      loom_amdgpu_target_info_find_target_by_kind(record->selector);
  IREE_ASSERT(target != NULL);
  const loom_attribute_t features_attr = loom_target_record_view_attribute(
      record, loom_amdgpu_target_features_ATTR_INDEX);
  const loom_signed_enum_set_t features =
      loom_attr_is_absent(features_attr)
          ? loom_signed_enum_set_empty()
          : loom_attr_as_signed_enum_set(features_attr);
  loom_amdgpu_target_identity_initialize_with_features(
      target, features.words, features.word_count, &facts->identity);
  loom_amdgpu_target_properties_resolve(
      &facts->identity, &facts->base.storage.bundle, &facts->properties);
  facts->subgroup_size_explicit = loom_target_facts_field_is_explicit(
      &facts->base, LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE);
  facts->contract_set_key_explicit = loom_target_facts_field_is_explicit(
      &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY);
}

const loom_target_fact_projector_t loom_amdgpu_target_fact_projector = {
    .project = loom_amdgpu_target_facts_project,
};
