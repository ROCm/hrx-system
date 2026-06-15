// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/ireevm/ref_lifetime_pass.h"

#include "loom/analysis/ownership_lifetime.h"
#include "loom/ir/module.h"
#include "loom/target/arch/ireevm/ops/ops.h"

typedef struct loom_ireevm_ref_lifetime_policy_t {
  // Interned type name for ireevm.ref.
  loom_string_id_t ref_type_name_id;
} loom_ireevm_ref_lifetime_policy_t;

static bool loom_ireevm_ref_lifetime_type_matches(loom_type_t type,
                                                  void* user_data) {
  const loom_ireevm_ref_lifetime_policy_t* policy =
      (const loom_ireevm_ref_lifetime_policy_t*)user_data;
  return loom_type_is_dialect(type) &&
         loom_type_dialect_name_id(type) == policy->ref_type_name_id &&
         loom_type_dialect_param_count(type) == 1;
}

static iree_status_t loom_ireevm_ref_lifetime_build_release(
    loom_builder_t* builder, loom_value_id_t value_id,
    loom_location_id_t location, void* user_data, loom_op_t** out_op) {
  (void)user_data;
  return loom_ireevm_ref_release_build(builder, value_id, location, out_op);
}

#define LOOM_IREEVM_REF_LIFETIME_STATISTICS(V, statistics_type) \
  V(statistics_type, blocks_checked, "blocks-checked",          \
    "Number of blocks checked for IREE VM ref lifetimes.")      \
  V(statistics_type, ops_checked, "ops-checked",                \
    "Number of operations visited for IREE VM ref lifetimes.")  \
  V(statistics_type, effects_checked, "effects-checked",        \
    "Number of ownership effects interpreted.")                 \
  V(statistics_type, releases_inserted, "releases-inserted",    \
    "Number of ireevm.ref.release operations inserted.")        \
  V(statistics_type, edges_split, "edges-split",                \
    "Number of CFG edges split for reference cleanup.")

LOOM_PASS_STATISTICS_DEFINE(loom_ireevm_ref_lifetime_statistics,
                            loom_ireevm_ref_lifetime_statistics_t,
                            LOOM_IREEVM_REF_LIFETIME_STATISTICS)

static const loom_pass_info_t loom_ireevm_ref_lifetime_pass_info_storage = {
    .name = IREE_SVL("ireevm-ref-lifetime"),
    .description = IREE_SVL("Materialize explicit IREE VM ref lifetimes."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_ireevm_ref_lifetime_statistics_layout,
};

const loom_pass_info_t* loom_ireevm_ref_lifetime_pass_info(void) {
  return &loom_ireevm_ref_lifetime_pass_info_storage;
}

iree_status_t loom_ireevm_ref_lifetime_run(loom_pass_t* pass,
                                           loom_module_t* module) {
  loom_ireevm_ref_lifetime_policy_t policy_data = {0};
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("ireevm.ref"), &policy_data.ref_type_name_id));
  const loom_ownership_lifetime_materialization_policy_t policy = {
      .family =
          {
              .name = IREE_SVL("ireevm.ref"),
              .type_matches = loom_ireevm_ref_lifetime_type_matches,
              .user_data = &policy_data,
          },
      .flags =
          LOOM_OWNERSHIP_LIFETIME_MATERIALIZATION_POLICY_OWNED_ARGUMENTS |
          LOOM_OWNERSHIP_LIFETIME_MATERIALIZATION_POLICY_OWNED_BODYLESS_RESULTS,
      .build_release = loom_ireevm_ref_lifetime_build_release,
  };
  loom_ownership_lifetime_materialize_options_t options = {
      .arena = pass->arena,
      .emitter = pass->diagnostic_emitter,
      .phase_name = pass->info->name,
      .policies = &policy,
      .policy_count = 1,
  };
  loom_ownership_lifetime_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_ownership_lifetime_materialize_module(module, &options, &result));
  loom_ireevm_ref_lifetime_statistics_t* statistics =
      loom_ireevm_ref_lifetime_statistics(pass);
  statistics->blocks_checked += result.blocks_checked;
  statistics->ops_checked += result.ops_checked;
  statistics->effects_checked += result.effects_checked;
  statistics->releases_inserted += result.releases_inserted;
  statistics->edges_split += result.edges_split;
  if (result.releases_inserted != 0 || result.edges_split != 0) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}
