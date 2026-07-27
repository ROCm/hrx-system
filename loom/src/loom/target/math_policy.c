// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/math_policy.h"

const loom_pass_environment_capability_type_t
    loom_target_math_pass_capability_type = {
        .name = IREE_SVL("target.math"),
};

void loom_target_math_policy_registry_initialize_from_entries(
    loom_target_math_policy_registry_t* out_registry,
    const loom_target_math_policy_registry_entry_t* entries,
    iree_host_size_t entry_count) {
  *out_registry = (loom_target_math_policy_registry_t){
      .entries = entries,
      .entry_count = entry_count,
  };
}

const loom_target_math_policy_t* loom_target_math_policy_registry_lookup(
    const loom_target_math_policy_registry_t* registry,
    iree_string_view_t contract_set_key) {
  if (registry == NULL) {
    return NULL;
  }
  contract_set_key = iree_string_view_trim(contract_set_key);
  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    const loom_target_math_policy_registry_entry_t* entry =
        &registry->entries[i];
    if (iree_string_view_equal(entry->contract_set_key, contract_set_key)) {
      return entry->policy;
    }
  }
  return NULL;
}

const loom_target_math_policy_t*
loom_target_math_policy_registry_lookup_for_bundle(
    const loom_target_math_policy_registry_t* registry,
    const loom_target_bundle_t* bundle) {
  if (bundle == NULL || bundle->config == NULL) {
    return NULL;
  }
  return loom_target_math_policy_registry_lookup(
      registry, bundle->config->contract_set_key);
}

void loom_target_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  *out_decision = (loom_target_math_policy_decision_t){0};
  if (policy == NULL || policy->query == NULL) {
    return;
  }
  policy->query(policy, query, out_decision);
}

iree_string_view_t loom_target_math_op_name(loom_target_math_op_t math_op) {
  switch (math_op) {
    case LOOM_TARGET_MATH_OP_LOGISTICF:
      return IREE_SV("logisticf");
    case LOOM_TARGET_MATH_OP_SILUF:
      return IREE_SV("siluf");
    case LOOM_TARGET_MATH_OP_SOFTPLUSF:
      return IREE_SV("softplusf");
    case LOOM_TARGET_MATH_OP_EXPF:
      return IREE_SV("expf");
    case LOOM_TARGET_MATH_OP_ERFF:
      return IREE_SV("erff");
    case LOOM_TARGET_MATH_OP_GELUF_ERF:
      return IREE_SV("geluf.erf");
    case LOOM_TARGET_MATH_OP_GELUF_TANH:
      return IREE_SV("geluf.tanh");
    case LOOM_TARGET_MATH_OP_GELUF_LOGISTIC:
      return IREE_SV("geluf.logistic");
    case LOOM_TARGET_MATH_OP_ADDF:
      return IREE_SV("addf");
    case LOOM_TARGET_MATH_OP_MULF:
      return IREE_SV("mulf");
    case LOOM_TARGET_MATH_OP_LOGF:
      return IREE_SV("logf");
    case LOOM_TARGET_MATH_OP_LOG2F:
      return IREE_SV("log2f");
    case LOOM_TARGET_MATH_OP_SINF:
      return IREE_SV("sinf");
    case LOOM_TARGET_MATH_OP_COSF:
      return IREE_SV("cosf");
    case LOOM_TARGET_MATH_OP_SINTURNSF:
      return IREE_SV("sinturnsf");
    case LOOM_TARGET_MATH_OP_COSTURNSF:
      return IREE_SV("costurnsf");
    case LOOM_TARGET_MATH_OP_TANHF:
      return IREE_SV("tanhf");
    case LOOM_TARGET_MATH_OP_POWF:
      return IREE_SV("powf");
    case LOOM_TARGET_MATH_OP_CEILF:
      return IREE_SV("ceilf");
    case LOOM_TARGET_MATH_OP_FLOORF:
      return IREE_SV("floorf");
    case LOOM_TARGET_MATH_OP_ROUNDF:
      return IREE_SV("roundf");
    case LOOM_TARGET_MATH_OP_ROUNDEVENF:
      return IREE_SV("roundevenf");
    case LOOM_TARGET_MATH_OP_TRUNCF:
      return IREE_SV("truncf");
    case LOOM_TARGET_MATH_OP_UNKNOWN:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

iree_string_view_t loom_target_math_lane_domain_name(
    loom_target_math_lane_domain_t domain) {
  switch (domain) {
    case LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR:
      return IREE_SV("scalar");
    case LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR:
      return IREE_SV("vector");
    case LOOM_TARGET_MATH_LANE_DOMAIN_UNKNOWN:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

iree_string_view_t loom_target_math_recipe_name(
    loom_target_math_recipe_t recipe) {
  switch (recipe) {
    case LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32:
      return IREE_SV("logistic-exp2-f32");
    case LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32:
      return IREE_SV("silu-logistic-f32");
    case LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32:
      return IREE_SV("softplus-exp2-f32");
    case LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32:
      return IREE_SV("exp-exp2-f32");
    case LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32:
      return IREE_SV("gelu-tanh-f32");
    case LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32:
      return IREE_SV("gelu-logistic-f32");
    case LOOM_TARGET_MATH_RECIPE_ERF_RATIONAL_F32:
      return IREE_SV("erf-rational-f32");
    case LOOM_TARGET_MATH_RECIPE_GELU_ERF_F32:
      return IREE_SV("gelu-erf-f32");
    case LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND_BF16:
      return IREE_SV("widen-f32-round-bf16");
    case LOOM_TARGET_MATH_RECIPE_LOG_LOG2_F32:
      return IREE_SV("log-log2-f32");
    case LOOM_TARGET_MATH_RECIPE_SIN_TURNS_F32:
      return IREE_SV("sin-turns-f32");
    case LOOM_TARGET_MATH_RECIPE_COS_TURNS_F32:
      return IREE_SV("cos-turns-f32");
    case LOOM_TARGET_MATH_RECIPE_TANH_LOGISTIC_F32:
      return IREE_SV("tanh-logistic-f32");
    case LOOM_TARGET_MATH_RECIPE_POW_LOG2_EXP2_F32:
      return IREE_SV("pow-log2-exp2-f32");
    case LOOM_TARGET_MATH_RECIPE_ROUND_AWAY_F32:
      return IREE_SV("round-away-f32");
    case LOOM_TARGET_MATH_RECIPE_UNKNOWN:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

loom_target_math_pass_capability_t loom_target_math_pass_capability_make(
    const loom_target_math_policy_registry_t* policy_registry,
    loom_target_compile_report_t* compile_report) {
  return (loom_target_math_pass_capability_t){
      .base =
          {
              .type = &loom_target_math_pass_capability_type,
          },
      .policy_registry = policy_registry,
      .compile_report = compile_report,
  };
}

const loom_target_math_pass_capability_t*
loom_target_math_pass_capability_from_environment(
    const loom_pass_environment_t* environment) {
  return (const loom_target_math_pass_capability_t*)
      loom_pass_environment_lookup(environment,
                                   &loom_target_math_pass_capability_type);
}

const loom_target_math_pass_capability_t*
loom_target_math_pass_capability_from_pass(const loom_pass_t* pass) {
  return pass && pass->environment
             ? loom_target_math_pass_capability_from_environment(
                   pass->environment)
             : NULL;
}

const loom_target_math_policy_registry_t*
loom_target_math_pass_capability_policy_registry(
    const loom_target_math_pass_capability_t* capability) {
  return capability ? capability->policy_registry : NULL;
}

loom_target_compile_report_t* loom_target_math_pass_capability_compile_report(
    const loom_target_math_pass_capability_t* capability) {
  return capability ? capability->compile_report : NULL;
}
