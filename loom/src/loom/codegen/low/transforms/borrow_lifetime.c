// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/borrow_lifetime.h"

#include "loom/analysis/ownership_lifetime.h"
#include "loom/codegen/low/ownership.h"
#include "loom/ops/low/ops.h"
#include "loom/target/pass_environment.h"
#include "loom/target/provider.h"

static const loom_pass_info_t loom_low_borrow_lifetime_info = {
    .name = IREE_SVL("low-borrow-lifetime"),
    .description =
        IREE_SVL("Retire protected pointer carriers after their final use."),
    .kind = LOOM_PASS_MODULE,
};

const loom_pass_info_t* loom_low_borrow_lifetime_pass_info(void) {
  return &loom_low_borrow_lifetime_info;
}

static bool loom_low_borrow_lifetime_type_matches(loom_type_t type,
                                                  void* user_data) {
  (void)user_data;
  const loom_type_t* value_type = loom_type_register_value_type(type);
  return value_type && loom_type_is_reference(*value_type) &&
         loom_type_reference_ownership(*value_type) ==
             LOOM_REFERENCE_OWNERSHIP_BORROWED;
}

static iree_status_t loom_low_borrow_lifetime_release(
    loom_builder_t* builder, loom_value_id_t value_id,
    loom_location_id_t location, void* user_data, loom_op_t** out_op) {
  (void)user_data;
  return loom_low_release_build(builder, value_id, location, out_op);
}

static iree_status_t loom_low_borrow_lifetime_branch(
    loom_builder_t* builder, loom_block_t* target, loom_location_id_t location,
    loom_op_t** out_op) {
  return loom_low_br_build(builder, target, NULL, 0, location, out_op);
}

iree_status_t loom_low_borrow_lifetime_run(loom_pass_t* pass,
                                           loom_module_t* module) {
  const loom_ownership_lifetime_policy_t policy = {
      .family =
          {
              .name = IREE_SVL("protected pointer"),
              .type_matches = loom_low_borrow_lifetime_type_matches,
          },
      .flags = LOOM_OWNERSHIP_LIFETIME_POLICY_LAST_USE,
      .build_release = loom_low_borrow_lifetime_release,
  };
  const loom_target_low_descriptor_registry_t registry =
      loom_target_environment_low_descriptor_registry(
          loom_target_pass_capability_target_environment(
              loom_target_pass_capability_from_pass(pass)));
  const loom_ownership_lifetime_options_t options = {
      .arena = pass->arena,
      .emitter = pass->diagnostic_emitter,
      .phase_name = pass->info->name,
      .policies = &policy,
      .policy_count = 1,
      .effect_resolver = loom_low_ownership_effect_resolver(&registry.registry),
      .build_branch = loom_low_borrow_lifetime_branch,
  };
  loom_ownership_lifetime_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_ownership_lifetime_materialize_module(module, &options, &result));
  if (result.releases_inserted || result.edges_split) {
    loom_pass_mark_changed(pass);
  }
  return iree_ok_status();
}
