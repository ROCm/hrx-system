// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/facts.h"

#include <string.h>

#include "loom/target/arch/x86/records/target_records.h"
#include "loom/target/facts_builder.h"

static void loom_x86_target_facts_project_execution_target(
    const loom_x86_target_facts_t* available, uint8_t selector,
    loom_x86_target_facts_t* out_facts) {
  *out_facts = *available;
  const loom_target_bundle_t* bundle =
      loom_target_bundle_table_lookup(&loom_x86_target_bundles, selector);
  out_facts->base.selector = selector;
  out_facts->base.storage = (loom_target_bundle_storage_t){
      .snapshot = *bundle->snapshot,
      .export_plan = *bundle->export_plan,
      .config = *bundle->config,
      .bundle = *bundle,
  };
  loom_target_bundle_storage_rebind(&out_facts->base.storage);

  const loom_x86_target_policy_t* policy =
      loom_x86_target_policy_lookup(selector);
  out_facts->base.storage.config.name = policy->feature_contract_set.spelling;
  out_facts->base.storage.config.contract_set = policy->feature_contract_set;
  out_facts->base.storage.config.contract_feature_bits =
      loom_x86_target_policy_project_cpu_features(
          policy, available->cpu_data.fields[0]);
  out_facts->positive_feature_assertions =
      out_facts->base.storage.config.contract_feature_bits;
  out_facts->negative_feature_assertions =
      policy->supported_feature_bits & ~out_facts->positive_feature_assertions;
  out_facts->executable_target_kinds = LOOM_X86_TARGET_KIND_BIT(selector);
}

static bool loom_x86_contract_set_satisfies(
    const loom_x86_target_facts_t* effective,
    const loom_x86_target_facts_t* requirement) {
  if (!loom_target_facts_field_is_explicit(
          &requirement->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY)) {
    return true;
  }
  if (effective->base.storage.config.contract_set.stable_id ==
      requirement->base.storage.config.contract_set.stable_id) {
    return true;
  }

  const loom_target_bundle_t* core_bundle = loom_target_bundle_table_lookup(
      &loom_x86_target_bundles, effective->base.selector);
  const loom_x86_target_policy_t* policy =
      loom_x86_target_policy_lookup(effective->base.selector);
  return effective->base.storage.config.contract_set.stable_id ==
             core_bundle->config->contract_set.stable_id &&
         requirement->base.storage.config.contract_set.stable_id ==
             policy->feature_contract_set.stable_id;
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
  loom_x86_target_facts_t selected_facts;
  if (effective->base.selector != requirement->base.selector) {
    if (!loom_x86_target_kind_set_contains(effective->executable_target_kinds,
                                           requirement->base.selector)) {
      return false;
    }
    loom_x86_target_facts_project_execution_target(
        effective, requirement->base.selector, &selected_facts);
    effective = &selected_facts;
  }
  if (!loom_target_snapshot_satisfies_specialization_requirement(
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
  if (requirement->tune_is_explicit && effective->tune != requirement->tune) {
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
         lhs->negative_feature_assertions == rhs->negative_feature_assertions &&
         lhs->tune == rhs->tune &&
         lhs->tune_is_explicit == rhs->tune_is_explicit &&
         lhs->executable_target_kinds == rhs->executable_target_kinds;
}

static void loom_x86_target_facts_apply_specialization_requirement(
    const loom_target_facts_t* requirement_base,
    loom_target_facts_t* effective_base) {
  const loom_x86_target_facts_t* requirement =
      (const loom_x86_target_facts_t*)requirement_base;
  loom_x86_target_facts_t* effective = (loom_x86_target_facts_t*)effective_base;
  if (effective->base.selector != requirement->base.selector) {
    loom_x86_target_facts_t selected_facts;
    loom_x86_target_facts_project_execution_target(
        effective, requirement->base.selector, &selected_facts);
    // The exact target row and authored common overrides replace the
    // profile's preferred row. CPU-derived features and tune remain in
    // |selected_facts|.
    selected_facts.base.storage.snapshot = requirement->base.storage.snapshot;
    selected_facts.base.storage.export_plan =
        requirement->base.storage.export_plan;
    selected_facts.base.storage.bundle.name =
        requirement->base.storage.bundle.name;
    loom_target_bundle_storage_rebind(&selected_facts.base.storage);
    *effective = selected_facts;
  }
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
  if (requirement->tune_is_explicit) {
    effective->tune = requirement->tune;
    effective->tune_is_explicit = true;
  }
  effective->executable_target_kinds =
      LOOM_X86_TARGET_KIND_BIT(effective->base.selector);
}

static iree_status_t loom_x86_target_facts_select_execution(
    const loom_target_facts_t* source, iree_arena_allocator_t* arena,
    const loom_target_facts_t** out_facts) {
  *out_facts = source;
  const loom_x86_target_facts_t* source_facts =
      (const loom_x86_target_facts_t*)source;
  const loom_x86_target_kind_set_t selected_target_kind =
      LOOM_X86_TARGET_KIND_BIT(source_facts->base.selector);
  if (source_facts->executable_target_kinds == selected_target_kind) {
    return iree_ok_status();
  }
  loom_target_facts_t* selected = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_facts_builder_clone(source, arena, &selected));
  ((loom_x86_target_facts_t*)selected)->executable_target_kinds =
      selected_target_kind;
  *out_facts = selected;
  return iree_ok_status();
}

static void loom_x86_target_facts_rebind(loom_target_facts_t* base_facts) {
  loom_x86_target_facts_t* facts = (loom_x86_target_facts_t*)base_facts;
  facts->base.storage.config.schedule_model_stable_id =
      loom_x86_target_tune_info_lookup(facts->tune)->schedule_model_stable_id;
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
    .rebind = loom_x86_target_facts_rebind,
    .select_execution = loom_x86_target_facts_select_execution,
};
