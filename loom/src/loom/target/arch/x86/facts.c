// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/facts.h"

#include <string.h>

#include "loom/target/arch/x86/records/target_records.h"

static bool loom_x86_contract_set_satisfies(
    const loom_x86_target_facts_t* effective,
    const loom_x86_target_facts_t* requirement) {
  if (!loom_target_facts_field_is_explicit(
          &requirement->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY)) {
    return true;
  }
  if (iree_string_view_equal(
          effective->base.storage.config.contract_set_key,
          requirement->base.storage.config.contract_set_key)) {
    return true;
  }

  const loom_target_bundle_t* core_bundle = loom_target_bundle_table_lookup(
      &loom_x86_target_bundles, effective->base.selector);
  const loom_x86_target_policy_t* policy =
      loom_x86_target_policy_lookup(effective->base.selector);
  return iree_string_view_equal(effective->base.storage.config.contract_set_key,
                                core_bundle->config->contract_set_key) &&
         iree_string_view_equal(
             requirement->base.storage.config.contract_set_key,
             policy->feature_contract_set_key);
}

static bool loom_x86_cpu_data_equal(const iree_cpu_data_t* lhs,
                                    const iree_cpu_data_t* rhs) {
  return lhs->architecture == rhs->architecture &&
         memcmp(lhs->fields, rhs->fields, sizeof(lhs->fields)) == 0;
}

static bool loom_x86_target_facts_satisfy_specialization_requirement(
    const loom_target_facts_t* effective_base,
    const loom_target_facts_t* requirement_base) {
  const loom_x86_target_facts_t* effective =
      (const loom_x86_target_facts_t*)effective_base;
  const loom_x86_target_facts_t* requirement =
      (const loom_x86_target_facts_t*)requirement_base;
  if (effective->base.selector != requirement->base.selector ||
      !loom_target_snapshot_satisfies_specialization_requirement(
          &effective->base.storage.snapshot,
          &requirement->base.storage.snapshot)) {
    return false;
  }
  if (!loom_x86_contract_set_satisfies(effective, requirement)) {
    return false;
  }
  if (!iree_all_bits_set(
          effective->base.storage.config.contract_feature_bits,
          requirement->base.storage.config.contract_feature_bits)) {
    return false;
  }
  return requirement->cpu_data.architecture == IREE_CPU_ARCHITECTURE_UNKNOWN ||
         loom_x86_cpu_data_equal(&effective->cpu_data, &requirement->cpu_data);
}

static bool loom_x86_target_facts_family_state_equivalent(
    const loom_target_facts_t* lhs_base, const loom_target_facts_t* rhs_base) {
  const loom_x86_target_facts_t* lhs = (const loom_x86_target_facts_t*)lhs_base;
  const loom_x86_target_facts_t* rhs = (const loom_x86_target_facts_t*)rhs_base;
  return lhs->positive_feature_assertions == rhs->positive_feature_assertions &&
         lhs->negative_feature_assertions == rhs->negative_feature_assertions;
}

static void loom_x86_target_facts_apply_specialization_requirement(
    const loom_target_facts_t* requirement_base,
    loom_target_facts_t* effective_base) {
  const loom_x86_target_facts_t* requirement =
      (const loom_x86_target_facts_t*)requirement_base;
  loom_x86_target_facts_t* effective = (loom_x86_target_facts_t*)effective_base;
  effective->positive_feature_assertions =
      (effective->positive_feature_assertions &
       ~requirement->negative_feature_assertions) |
      requirement->positive_feature_assertions;
  effective->negative_feature_assertions =
      (effective->negative_feature_assertions &
       ~requirement->positive_feature_assertions) |
      requirement->negative_feature_assertions;
  effective->base.storage.config.contract_feature_bits =
      (effective->base.storage.config.contract_feature_bits &
       ~requirement->negative_feature_assertions) |
      requirement->positive_feature_assertions;
}

const loom_target_fact_type_t loom_x86_target_fact_type = {
    .name = IREE_SVL("x86"),
    .storage_size = sizeof(loom_x86_target_facts_t),
    .satisfies_identity_requirement =
        loom_target_facts_selector_satisfies_identity_requirement,
    .satisfies_specialization_requirement =
        loom_x86_target_facts_satisfy_specialization_requirement,
    .family_state_equivalent = loom_x86_target_facts_family_state_equivalent,
    .apply_specialization_requirement =
        loom_x86_target_facts_apply_specialization_requirement,
};
