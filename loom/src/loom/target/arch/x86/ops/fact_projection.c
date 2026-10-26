// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/facts.h"
#include "loom/target/arch/x86/facts.h"
#include "loom/target/arch/x86/ops/ops.h"
#include "loom/target/arch/x86/records/target_records.h"

static void loom_x86_target_facts_project(
    const loom_target_record_view_t* record, loom_target_facts_t* base_facts) {
  loom_x86_target_facts_t* facts = (loom_x86_target_facts_t*)base_facts;
  facts->executable_target_kinds = LOOM_X86_TARGET_KIND_BIT(record->selector);
  const loom_attribute_t tune_attr = loom_x86_target_tune_from_record(record);
  if (!loom_attr_is_absent(tune_attr)) {
    facts->tune = (loom_x86_target_tune_t)loom_attr_as_enum(tune_attr);
    facts->tune_is_explicit = true;
  }

  const loom_attribute_t features_attr =
      loom_x86_target_features_from_record(record);
  if (loom_attr_is_absent(features_attr)) {
    return;
  }

  const loom_signed_enum_set_t features =
      loom_attr_as_signed_enum_set(features_attr);
  if (features.word_count > 0) {
    facts->positive_feature_assertions = features.words[0];
    facts->negative_feature_assertions = features.words[features.word_count];
  }

  const loom_x86_target_policy_t* policy =
      loom_x86_target_policy_lookup(record->selector);
  facts->base.storage.config.contract_set = policy->feature_contract_set;
  facts->base.storage.config.contract_feature_bits =
      (facts->base.storage.config.contract_feature_bits |
       facts->positive_feature_assertions) &
      ~facts->negative_feature_assertions;
  loom_target_fact_field_set_insert(&facts->base.explicit_fields,
                                    LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY);
}

const loom_target_fact_projector_t loom_x86_target_fact_projector = {
    .project = loom_x86_target_facts_project,
};
