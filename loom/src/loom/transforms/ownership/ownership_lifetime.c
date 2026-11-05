// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/ownership/ownership_lifetime.h"

#include "loom/analysis/ownership_lifetime.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"

#define LOOM_OWNERSHIP_LIFETIME_STATISTICS(V, statistics_type) \
  V(statistics_type, blocks_checked, "blocks-checked",         \
    "Number of blocks checked for ownership lifetimes.")       \
  V(statistics_type, ops_checked, "ops-checked",               \
    "Number of operations visited for ownership lifetimes.")   \
  V(statistics_type, effects_checked, "effects-checked",       \
    "Number of ownership effects interpreted.")

LOOM_PASS_STATISTICS_DEFINE(loom_ownership_lifetime_statistics,
                            loom_ownership_lifetime_statistics_t,
                            LOOM_OWNERSHIP_LIFETIME_STATISTICS)

static const loom_pass_info_t loom_ownership_lifetime_pass_info_storage = {
    .name = IREE_SVL("ownership-lifetime"),
    .description = IREE_SVL("Analyze descriptor-backed owned-resource "
                            "lifetimes across functions and CFG blocks; "
                            "owned-resource functions require flat control."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_ownership_lifetime_statistics_layout,
};

const loom_pass_info_t* loom_ownership_lifetime_pass_info(void) {
  return &loom_ownership_lifetime_pass_info_storage;
}

static bool loom_channel_access_type_matches(loom_type_t type,
                                             void* user_data) {
  (void)user_data;
  return loom_read_type_isa(type) || loom_write_type_isa(type);
}

static bool loom_owned_reference_type_matches(loom_type_t type,
                                              void* user_data) {
  (void)user_data;
  const loom_type_t* register_type = loom_type_register_value_type(type);
  if (register_type) {
    type = *register_type;
  }
  return loom_type_is_reference(type) &&
         loom_type_reference_ownership(type) == LOOM_REFERENCE_OWNERSHIP_OWNED;
}

iree_status_t loom_ownership_lifetime_run(loom_pass_t* pass,
                                          loom_module_t* module) {
  const loom_ownership_lifetime_policy_t policies[] = {
      {
          .family =
              {
                  .name = IREE_SVL("channel access"),
                  .type_matches = loom_channel_access_type_matches,
              },
          .flags = LOOM_OWNERSHIP_LIFETIME_POLICY_OWNED_ARGUMENTS |
                   LOOM_OWNERSHIP_LIFETIME_POLICY_OWNED_BODYLESS_RESULTS,
      },
      {
          .family =
              {
                  .name = IREE_SVL("managed reference"),
                  .type_matches = loom_owned_reference_type_matches,
              },
          .flags = LOOM_OWNERSHIP_LIFETIME_POLICY_OWNED_ARGUMENTS |
                   LOOM_OWNERSHIP_LIFETIME_POLICY_OWNED_BODYLESS_RESULTS,
      },
  };
  loom_ownership_lifetime_options_t options = {
      .arena = pass->arena,
      .emitter = pass->diagnostic_emitter,
      .phase_name = pass->info->name,
      .policies = policies,
      .policy_count = IREE_ARRAYSIZE(policies),
  };
  loom_ownership_lifetime_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_ownership_lifetime_analyze_module(module, &options, &result));

  loom_ownership_lifetime_statistics_t* statistics =
      loom_ownership_lifetime_statistics(pass);
  statistics->blocks_checked += (int64_t)result.blocks_checked;
  statistics->ops_checked += (int64_t)result.ops_checked;
  statistics->effects_checked += (int64_t)result.effects_checked;
  return iree_ok_status();
}
