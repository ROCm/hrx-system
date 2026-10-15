// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/math_policy.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

struct RewriteCase {
  loom_target_math_op_t math_op;
  loom_target_math_recipe_t recipe;
  const char* recipe_constraint;
  const char* exact_constraint;
};

static loom_target_math_policy_decision_t Query(
    const loom_target_math_policy_t* policy,
    iree_string_view_t contract_set_key, loom_target_math_op_t math_op,
    loom_target_math_lane_domain_t lane_domain, loom_scalar_type_t element_type,
    loom_target_math_fastmath_flags_t fastmath_flags) {
  const loom_target_config_t config = {
      .contract_set_key = contract_set_key,
  };
  const loom_target_bundle_t bundle = {
      .config = &config,
  };
  const loom_type_t value_type =
      lane_domain == LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR
          ? loom_type_scalar(element_type)
          : loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type, 8, 0);
  const loom_target_math_query_t query = {
      .target_bundle = &bundle,
      .math_op = math_op,
      .lane_domain = lane_domain,
      .value_type = value_type,
      .element_type = element_type,
      .fastmath_flags = fastmath_flags,
  };
  loom_target_math_policy_decision_t decision = {};
  loom_target_math_policy_query(policy, &query, &decision);
  return decision;
}

static void ExpectConstraint(const loom_target_math_policy_decision_t& decision,
                             const char* expected_constraint) {
  EXPECT_TRUE(iree_string_view_equal(
      decision.constraint_key, iree_make_cstring_view(expected_constraint)));
}

TEST(X86MathPolicyTest, SelectsCompleteApproximateF32Family) {
  const RewriteCase rewrite_cases[] = {
      {LOOM_TARGET_MATH_OP_LOGISTICF, LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32,
       "math.recipe.logistic_exp2_f32", "math.activation.exact_f32"},
      {LOOM_TARGET_MATH_OP_SILUF, LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32,
       "math.recipe.silu_logistic_f32", "math.activation.exact_f32"},
      {LOOM_TARGET_MATH_OP_EXPF, LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32,
       "math.recipe.exp_exp2_f32", "math.exp.exact_f32"},
      {LOOM_TARGET_MATH_OP_ERFF, LOOM_TARGET_MATH_RECIPE_ERF_RATIONAL_F32,
       "math.recipe.erf_rational_f32", "math.erf.exact_f32"},
      {LOOM_TARGET_MATH_OP_GELUF_ERF, LOOM_TARGET_MATH_RECIPE_GELU_ERF_F32,
       "math.recipe.gelu_erf_f32", "math.activation.exact_f32"},
      {LOOM_TARGET_MATH_OP_GELUF_TANH, LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32,
       "math.recipe.gelu_tanh_f32", "math.activation.exact_f32"},
      {LOOM_TARGET_MATH_OP_GELUF_LOGISTIC,
       LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32,
       "math.recipe.gelu_logistic_f32", "math.activation.exact_f32"},
      {LOOM_TARGET_MATH_OP_SINF, LOOM_TARGET_MATH_RECIPE_SIN_TURNS_F32,
       "math.recipe.sin_turns_f32", "math.trig.exact_f32"},
      {LOOM_TARGET_MATH_OP_COSF, LOOM_TARGET_MATH_RECIPE_COS_TURNS_F32,
       "math.recipe.cos_turns_f32", "math.trig.exact_f32"},
      {LOOM_TARGET_MATH_OP_SINTURNSF,
       LOOM_TARGET_MATH_RECIPE_APPROXIMATE_SIN_TURNS_F32,
       "math.recipe.approximate_sin_turns_f32", "math.turns_trig.exact_f32"},
      {LOOM_TARGET_MATH_OP_COSTURNSF,
       LOOM_TARGET_MATH_RECIPE_APPROXIMATE_COS_TURNS_F32,
       "math.recipe.approximate_cos_turns_f32", "math.turns_trig.exact_f32"},
      {LOOM_TARGET_MATH_OP_TANHF, LOOM_TARGET_MATH_RECIPE_TANH_LOGISTIC_F32,
       "math.recipe.tanh_logistic_f32", "math.tanh.exact_f32"},
      {LOOM_TARGET_MATH_OP_EXP2F, LOOM_TARGET_MATH_RECIPE_APPROXIMATE_EXP2_F32,
       "math.recipe.approximate_exp2_f32", "math.exp2.exact_f32"},
  };
  const iree_string_view_t contract_set_keys[] = {
      IREE_SV("x86.avx2.core"),
      IREE_SV("x86.avx512.core"),
  };
  const loom_target_math_lane_domain_t lane_domains[] = {
      LOOM_TARGET_MATH_LANE_DOMAIN_SCALAR,
      LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR,
  };

  loom_target_math_policy_registry_t registry;
  loom_x86_math_policy_registry_initialize(&registry);
  for (iree_string_view_t contract_set_key : contract_set_keys) {
    const loom_target_math_policy_t* policy =
        loom_target_math_policy_registry_lookup(&registry, contract_set_key);
    ASSERT_NE(policy, nullptr);
    for (loom_target_math_lane_domain_t lane_domain : lane_domains) {
      for (const RewriteCase& rewrite_case : rewrite_cases) {
        SCOPED_TRACE(static_cast<int>(rewrite_case.math_op));
        SCOPED_TRACE(static_cast<int>(lane_domain));

        const loom_target_math_policy_decision_t approximate =
            Query(policy, contract_set_key, rewrite_case.math_op, lane_domain,
                  LOOM_SCALAR_TYPE_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN);
        EXPECT_EQ(approximate.action, LOOM_TARGET_MATH_POLICY_ACTION_REWRITE);
        EXPECT_EQ(approximate.recipe, rewrite_case.recipe);
        ExpectConstraint(approximate, rewrite_case.recipe_constraint);

        const loom_target_math_policy_decision_t exact =
            Query(policy, contract_set_key, rewrite_case.math_op, lane_domain,
                  LOOM_SCALAR_TYPE_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_NONE);
        EXPECT_EQ(exact.action, LOOM_TARGET_MATH_POLICY_ACTION_REJECT);
        EXPECT_EQ(exact.recipe, LOOM_TARGET_MATH_RECIPE_UNKNOWN);
        ExpectConstraint(exact, rewrite_case.exact_constraint);

        const loom_target_math_policy_decision_t wrong_element =
            Query(policy, contract_set_key, rewrite_case.math_op, lane_domain,
                  LOOM_SCALAR_TYPE_F64, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN);
        EXPECT_EQ(wrong_element.action, LOOM_TARGET_MATH_POLICY_ACTION_REJECT);
        EXPECT_EQ(wrong_element.recipe, LOOM_TARGET_MATH_RECIPE_UNKNOWN);
        ExpectConstraint(wrong_element, "math.element.f32");
      }
    }
  }
}

TEST(X86MathPolicyTest, NamesMissingBasisAndRoundingFamilies) {
  struct RejectCase {
    loom_target_math_op_t math_op;
    const char* constraint;
  } reject_cases[] = {
      {LOOM_TARGET_MATH_OP_SOFTPLUSF, "math.basis.log2"},
      {LOOM_TARGET_MATH_OP_LOGF, "math.basis.log2"},
      {LOOM_TARGET_MATH_OP_LOG2F, "math.basis.log2"},
      {LOOM_TARGET_MATH_OP_POWF, "math.basis.log2"},
      {LOOM_TARGET_MATH_OP_CEILF, "math.op.rounding"},
      {LOOM_TARGET_MATH_OP_FLOORF, "math.op.rounding"},
      {LOOM_TARGET_MATH_OP_ROUNDF, "math.op.rounding"},
      {LOOM_TARGET_MATH_OP_ROUNDEVENF, "math.op.rounding"},
      {LOOM_TARGET_MATH_OP_TRUNCF, "math.op.rounding"},
  };
  loom_target_math_policy_registry_t registry;
  loom_x86_math_policy_registry_initialize(&registry);
  const iree_string_view_t contract_set_key = IREE_SV("x86.avx2.core");
  const loom_target_math_policy_t* policy =
      loom_target_math_policy_registry_lookup(&registry, contract_set_key);
  ASSERT_NE(policy, nullptr);

  for (const RejectCase& reject_case : reject_cases) {
    SCOPED_TRACE(static_cast<int>(reject_case.math_op));
    const loom_target_math_policy_decision_t decision =
        Query(policy, contract_set_key, reject_case.math_op,
              LOOM_TARGET_MATH_LANE_DOMAIN_VECTOR, LOOM_SCALAR_TYPE_F32,
              LOOM_TARGET_MATH_FASTMATH_FLAG_AFN);
    EXPECT_EQ(decision.action, LOOM_TARGET_MATH_POLICY_ACTION_REJECT);
    EXPECT_EQ(decision.recipe, LOOM_TARGET_MATH_RECIPE_UNKNOWN);
    ExpectConstraint(decision, reject_case.constraint);
  }
}

}  // namespace
}  // namespace loom
