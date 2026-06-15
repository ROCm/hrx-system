// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/math_policy.h"

static loom_target_math_policy_decision_t loom_amdgpu_math_reject(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_amdgpu_math_keep(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_KEEP,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_amdgpu_math_rewrite(
    loom_target_math_recipe_t recipe, iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,
      .recipe = recipe,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t
loom_amdgpu_math_rewrite_with_recipe_fastmath(
    loom_target_math_recipe_t recipe, iree_string_view_t constraint_key,
    loom_target_math_fastmath_flags_t recipe_fastmath_flags) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,
      .recipe = recipe,
      .recipe_fastmath_flags = recipe_fastmath_flags,
      .constraint_key = constraint_key,
  };
}

static bool loom_amdgpu_math_query_has_afn(
    const loom_target_math_query_t* query) {
  return iree_all_bits_set(query->fastmath_flags,
                           LOOM_TARGET_MATH_FASTMATH_FLAG_AFN);
}

static loom_target_math_policy_decision_t loom_amdgpu_math_rewrite_if_afn(
    const loom_target_math_query_t* query, loom_target_math_recipe_t recipe,
    iree_string_view_t recipe_key, iree_string_view_t reject_key) {
  return loom_amdgpu_math_query_has_afn(query)
             ? loom_amdgpu_math_rewrite(recipe, recipe_key)
             : loom_amdgpu_math_reject(reject_key);
}

static loom_target_math_policy_decision_t loom_amdgpu_math_keep_if_afn(
    const loom_target_math_query_t* query, iree_string_view_t reject_key) {
  return loom_amdgpu_math_query_has_afn(query)
             ? loom_amdgpu_math_keep(IREE_SV("math.op.native_f32"))
             : loom_amdgpu_math_reject(reject_key);
}

static void loom_amdgpu_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  if (query->math_op == LOOM_TARGET_MATH_OP_ADDF ||
      query->math_op == LOOM_TARGET_MATH_OP_MULF) {
    if (query->element_type == LOOM_SCALAR_TYPE_BF16) {
      *out_decision =
          loom_amdgpu_math_rewrite(LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND_BF16,
                                   IREE_SV("math.recipe.widen_f32_round_bf16"));
      return;
    }
    *out_decision = loom_amdgpu_math_keep(IREE_SV("math.basic.keep"));
    return;
  }

  if (query->element_type != LOOM_SCALAR_TYPE_F32) {
    *out_decision = loom_amdgpu_math_reject(IREE_SV("math.element.f32"));
    return;
  }

  switch (query->math_op) {
    case LOOM_TARGET_MATH_OP_LOGISTICF:
      *out_decision = loom_amdgpu_math_rewrite_with_recipe_fastmath(
          LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32,
          IREE_SV("math.recipe.logistic_exp2_f32"),
          LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP);
      return;
    case LOOM_TARGET_MATH_OP_SILUF:
      *out_decision =
          loom_amdgpu_math_rewrite(LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32,
                                   IREE_SV("math.recipe.silu_logistic_f32"));
      return;
    case LOOM_TARGET_MATH_OP_SOFTPLUSF:
      *out_decision =
          loom_amdgpu_math_rewrite(LOOM_TARGET_MATH_RECIPE_SOFTPLUS_EXP2_F32,
                                   IREE_SV("math.recipe.softplus_exp2_f32"));
      return;
    case LOOM_TARGET_MATH_OP_EXPF:
      *out_decision = loom_amdgpu_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32,
          IREE_SV("math.recipe.exp_exp2_f32"), IREE_SV("math.exp.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_LOGF:
      *out_decision = loom_amdgpu_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_LOG_LOG2_F32,
          IREE_SV("math.recipe.log_log2_f32"), IREE_SV("math.log.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_LOG2F:
      *out_decision =
          loom_amdgpu_math_keep_if_afn(query, IREE_SV("math.log2.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_TANHF:
      *out_decision = loom_amdgpu_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_TANH_LOGISTIC_F32,
          IREE_SV("math.recipe.tanh_logistic_f32"),
          IREE_SV("math.tanh.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_POWF:
      *out_decision = loom_amdgpu_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_POW_LOG2_EXP2_F32,
          IREE_SV("math.recipe.pow_log2_exp2_f32"),
          IREE_SV("math.pow.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_SINF:
      *out_decision = loom_amdgpu_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_SIN_TURNS_F32,
          IREE_SV("math.recipe.sin_turns_f32"), IREE_SV("math.trig.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_COSF:
      *out_decision = loom_amdgpu_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_COS_TURNS_F32,
          IREE_SV("math.recipe.cos_turns_f32"), IREE_SV("math.trig.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_SINTURNSF:
    case LOOM_TARGET_MATH_OP_COSTURNSF:
      *out_decision = loom_amdgpu_math_keep_if_afn(
          query, IREE_SV("math.turns_trig.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_ERFF:
      *out_decision = loom_amdgpu_math_rewrite_with_recipe_fastmath(
          LOOM_TARGET_MATH_RECIPE_ERF_RATIONAL_F32,
          IREE_SV("math.recipe.erf_rational_f32"),
          LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP);
      return;
    case LOOM_TARGET_MATH_OP_GELUF_TANH:
      *out_decision =
          loom_amdgpu_math_rewrite(LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32,
                                   IREE_SV("math.recipe.gelu_tanh_f32"));
      return;
    case LOOM_TARGET_MATH_OP_GELUF_LOGISTIC:
      *out_decision =
          loom_amdgpu_math_rewrite(LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32,
                                   IREE_SV("math.recipe.gelu_logistic_f32"));
      return;
    case LOOM_TARGET_MATH_OP_GELUF_ERF:
      *out_decision = loom_amdgpu_math_rewrite_with_recipe_fastmath(
          LOOM_TARGET_MATH_RECIPE_GELU_ERF_F32,
          IREE_SV("math.recipe.gelu_erf_f32"),
          LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP);
      return;
    case LOOM_TARGET_MATH_OP_ADDF:
    case LOOM_TARGET_MATH_OP_MULF:
      *out_decision = loom_amdgpu_math_keep(IREE_SV("math.op.native_f32"));
      return;
    case LOOM_TARGET_MATH_OP_UNKNOWN:
      break;
  }

  *out_decision = loom_amdgpu_math_reject(IREE_SV("math.op.supported"));
}

static const loom_target_math_policy_t kAmdgpuMathPolicy = {
    .name = IREE_SVL("amdgpu-math"),
    .query = loom_amdgpu_math_policy_query,
};

#include "loom/target/arch/amdgpu/math_policy_tables.inl"

void loom_amdgpu_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kAmdgpuMathPolicyEntries,
      IREE_ARRAYSIZE(kAmdgpuMathPolicyEntries));
}
