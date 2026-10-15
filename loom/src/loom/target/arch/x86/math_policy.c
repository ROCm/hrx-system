// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/math_policy.h"

#include "loom/target/arch/x86/feature_bits.h"

typedef struct loom_x86_math_profile_t {
  // Element types implemented by exact arithmetic and square root forms.
  loom_scalar_type_set_t native_element_types;
  // Element types implemented by approximate reciprocal square root forms.
  loom_scalar_type_set_t approximate_rsqrt_element_types;
  // Whether AVX-512 FP16 extends both sets when selected by the target.
  bool admits_feature_fp16;
} loom_x86_math_profile_t;

static const loom_x86_math_profile_t kX86Avx2MathProfile = {
    .native_element_types = LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64,
    .approximate_rsqrt_element_types = LOOM_SCALAR_TYPE_SET_F32,
};

static const loom_x86_math_profile_t kX86Avx512MathProfile = {
    .native_element_types = LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64,
    .approximate_rsqrt_element_types =
        LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64,
    .admits_feature_fp16 = true,
};

static loom_target_math_policy_decision_t loom_x86_math_keep(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_KEEP,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_x86_math_reject(
    iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REJECT,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_x86_math_rewrite(
    loom_target_math_recipe_t recipe, iree_string_view_t constraint_key) {
  return (loom_target_math_policy_decision_t){
      .action = LOOM_TARGET_MATH_POLICY_ACTION_REWRITE,
      .recipe = recipe,
      .constraint_key = constraint_key,
  };
}

static loom_target_math_policy_decision_t loom_x86_math_rewrite_if_afn(
    const loom_target_math_query_t* query, loom_target_math_recipe_t recipe,
    iree_string_view_t recipe_constraint_key,
    iree_string_view_t exact_constraint_key) {
  if (iree_all_bits_set(query->fastmath_flags,
                        LOOM_TARGET_MATH_FASTMATH_FLAG_AFN)) {
    return loom_x86_math_rewrite(recipe, recipe_constraint_key);
  }
  return loom_x86_math_reject(exact_constraint_key);
}

static bool loom_x86_math_target_has_fp16(
    const loom_target_bundle_t* target_bundle) {
  return target_bundle != NULL && target_bundle->config != NULL &&
         iree_any_bit_set(target_bundle->config->contract_feature_bits,
                          LOOM_X86_FEATURE_AVX512_FP16);
}

static loom_scalar_type_set_t loom_x86_math_profile_element_types(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query, bool approximate_rsqrt) {
  const loom_x86_math_profile_t* profile =
      (const loom_x86_math_profile_t*)policy->user_data;
  loom_scalar_type_set_t element_types =
      approximate_rsqrt ? profile->approximate_rsqrt_element_types
                        : profile->native_element_types;
  if (profile->admits_feature_fp16 &&
      loom_x86_math_target_has_fp16(query->target_bundle)) {
    element_types |= LOOM_SCALAR_TYPE_SET_F16;
  }
  return element_types;
}

static void loom_x86_math_policy_query(
    const loom_target_math_policy_t* policy,
    const loom_target_math_query_t* query,
    loom_target_math_policy_decision_t* out_decision) {
  switch (query->math_op) {
    case LOOM_TARGET_MATH_OP_ADDF:
    case LOOM_TARGET_MATH_OP_SUBF:
    case LOOM_TARGET_MATH_OP_MULF:
    case LOOM_TARGET_MATH_OP_SQRTF:
    case LOOM_TARGET_MATH_OP_RSQRTF: {
      if (!loom_scalar_type_set_contains(
              loom_x86_math_profile_element_types(policy, query,
                                                  /*approximate_rsqrt=*/false),
              query->element_type)) {
        *out_decision =
            loom_x86_math_reject(IREE_SV("math.element.native_float"));
        return;
      }
      if (query->math_op != LOOM_TARGET_MATH_OP_RSQRTF) {
        *out_decision = loom_x86_math_keep(IREE_SV("math.op.native_float"));
        return;
      }
      const bool has_native_approximate_rsqrt = loom_scalar_type_set_contains(
          loom_x86_math_profile_element_types(policy, query,
                                              /*approximate_rsqrt=*/true),
          query->element_type);
      if (has_native_approximate_rsqrt &&
          iree_all_bits_set(query->fastmath_flags,
                            LOOM_TARGET_MATH_FASTMATH_FLAG_AFN)) {
        *out_decision = loom_x86_math_keep(IREE_SV("math.op.native_rsqrt_afn"));
        return;
      }
      *out_decision =
          loom_x86_math_rewrite(LOOM_TARGET_MATH_RECIPE_RSQRT_SQRT_DIV,
                                IREE_SV("math.recipe.rsqrt_sqrt_div"));
      if (has_native_approximate_rsqrt) {
        out_decision->native_required_fastmath_flags =
            LOOM_TARGET_MATH_FASTMATH_FLAG_AFN;
      }
      return;
    }
    default:
      break;
  }

  if (query->element_type != LOOM_SCALAR_TYPE_F32) {
    *out_decision = loom_x86_math_reject(IREE_SV("math.element.f32"));
    return;
  }

  switch (query->math_op) {
    case LOOM_TARGET_MATH_OP_EXP2F:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_APPROXIMATE_EXP2_F32,
          IREE_SV("math.recipe.approximate_exp2_f32"),
          IREE_SV("math.exp2.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_SINTURNSF:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_APPROXIMATE_SIN_TURNS_F32,
          IREE_SV("math.recipe.approximate_sin_turns_f32"),
          IREE_SV("math.turns_trig.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_COSTURNSF:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_APPROXIMATE_COS_TURNS_F32,
          IREE_SV("math.recipe.approximate_cos_turns_f32"),
          IREE_SV("math.turns_trig.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_EXPF:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_EXP_EXP2_F32,
          IREE_SV("math.recipe.exp_exp2_f32"), IREE_SV("math.exp.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_SINF:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_SIN_TURNS_F32,
          IREE_SV("math.recipe.sin_turns_f32"), IREE_SV("math.trig.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_COSF:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_COS_TURNS_F32,
          IREE_SV("math.recipe.cos_turns_f32"), IREE_SV("math.trig.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_LOGISTICF:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_LOGISTIC_EXP2_F32,
          IREE_SV("math.recipe.logistic_exp2_f32"),
          IREE_SV("math.activation.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_SILUF:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_SILU_LOGISTIC_F32,
          IREE_SV("math.recipe.silu_logistic_f32"),
          IREE_SV("math.activation.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_TANHF:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_TANH_LOGISTIC_F32,
          IREE_SV("math.recipe.tanh_logistic_f32"),
          IREE_SV("math.tanh.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_ERFF:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_ERF_RATIONAL_F32,
          IREE_SV("math.recipe.erf_rational_f32"),
          IREE_SV("math.erf.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_GELUF_TANH:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_GELU_TANH_F32,
          IREE_SV("math.recipe.gelu_tanh_f32"),
          IREE_SV("math.activation.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_GELUF_LOGISTIC:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_GELU_LOGISTIC_F32,
          IREE_SV("math.recipe.gelu_logistic_f32"),
          IREE_SV("math.activation.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_GELUF_ERF:
      *out_decision = loom_x86_math_rewrite_if_afn(
          query, LOOM_TARGET_MATH_RECIPE_GELU_ERF_F32,
          IREE_SV("math.recipe.gelu_erf_f32"),
          IREE_SV("math.activation.exact_f32"));
      return;
    case LOOM_TARGET_MATH_OP_SOFTPLUSF:
    case LOOM_TARGET_MATH_OP_LOGF:
    case LOOM_TARGET_MATH_OP_LOG2F:
    case LOOM_TARGET_MATH_OP_POWF:
      *out_decision = loom_x86_math_reject(IREE_SV("math.basis.log2"));
      return;
    case LOOM_TARGET_MATH_OP_CEILF:
    case LOOM_TARGET_MATH_OP_FLOORF:
    case LOOM_TARGET_MATH_OP_ROUNDF:
    case LOOM_TARGET_MATH_OP_ROUNDEVENF:
    case LOOM_TARGET_MATH_OP_TRUNCF:
      *out_decision = loom_x86_math_reject(IREE_SV("math.op.rounding"));
      return;
    case LOOM_TARGET_MATH_OP_ADDF:
    case LOOM_TARGET_MATH_OP_SUBF:
    case LOOM_TARGET_MATH_OP_MULF:
    case LOOM_TARGET_MATH_OP_SQRTF:
    case LOOM_TARGET_MATH_OP_RSQRTF:
      IREE_ASSERT_UNREACHABLE("native x86 math operation escaped dispatch");
      IREE_BUILTIN_UNREACHABLE();
    case LOOM_TARGET_MATH_OP_UNKNOWN:
      break;
  }

  *out_decision = loom_x86_math_reject(IREE_SV("math.op.supported"));
}

static bool loom_x86_math_prefer_scalar_fma(
    const loom_target_math_policy_t* policy,
    const loom_target_bundle_t* target_bundle, loom_type_t value_type,
    loom_target_math_fastmath_flags_t fastmath_flags) {
  (void)policy;
  (void)target_bundle;
  (void)fastmath_flags;
  if (!loom_type_is_scalar(value_type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(value_type);
  return element_type == LOOM_SCALAR_TYPE_F32 ||
         element_type == LOOM_SCALAR_TYPE_F64;
}

static bool loom_x86_math_prefer_avx512_fma(
    const loom_target_math_policy_t* policy,
    const loom_target_bundle_t* target_bundle, loom_type_t value_type,
    loom_target_math_fastmath_flags_t fastmath_flags) {
  if (loom_x86_math_prefer_scalar_fma(policy, target_bundle, value_type,
                                      fastmath_flags)) {
    return true;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(value_type);
  if (element_type == LOOM_SCALAR_TYPE_F16 &&
      loom_x86_math_target_has_fp16(target_bundle)) {
    if (loom_type_is_scalar(value_type)) {
      return true;
    }
    if (!loom_type_is_vector(value_type) || loom_type_rank(value_type) != 1 ||
        !loom_type_is_all_static(value_type)) {
      return false;
    }
    const int64_t bit_width = loom_type_dim_static_size_at(value_type, 0) * 16;
    return bit_width == 64 || bit_width == 128 || bit_width == 256 ||
           bit_width == 512;
  }
  return loom_type_is_vector(value_type) && loom_type_rank(value_type) == 1 &&
         loom_type_is_all_static(value_type) &&
         loom_type_dim_static_size_at(value_type, 0) == 16 &&
         element_type == LOOM_SCALAR_TYPE_F32;
}

static bool loom_x86_math_prefer_avx2_fma(
    const loom_target_math_policy_t* policy,
    const loom_target_bundle_t* target_bundle, loom_type_t value_type,
    loom_target_math_fastmath_flags_t fastmath_flags) {
  if (loom_x86_math_prefer_scalar_fma(policy, target_bundle, value_type,
                                      fastmath_flags)) {
    return true;
  }
  if (!loom_type_is_vector(value_type) || loom_type_rank(value_type) != 1 ||
      !loom_type_is_all_static(value_type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(value_type);
  if (element_type != LOOM_SCALAR_TYPE_F32 &&
      element_type != LOOM_SCALAR_TYPE_F64) {
    return false;
  }
  const int64_t bit_width = loom_type_dim_static_size_at(value_type, 0) *
                            loom_scalar_type_bitwidth(element_type);
  return bit_width == 128 || bit_width == 256;
}

static const loom_target_math_policy_t kX86MathPolicy = {
    .name = IREE_SVL("x86-math"),
    .query = loom_x86_math_policy_query,
    .prefer_fma = loom_x86_math_prefer_avx2_fma,
    .user_data = &kX86Avx2MathProfile,
};

static const loom_target_math_policy_t kX86Avx512MathPolicy = {
    .name = IREE_SVL("x86-math"),
    .query = loom_x86_math_policy_query,
    .prefer_fma = loom_x86_math_prefer_avx512_fma,
    .user_data = &kX86Avx512MathProfile,
};

static const loom_target_math_policy_t kX86Avx512FeaturesMathPolicy = {
    .name = IREE_SVL("x86-math"),
    .query = loom_x86_math_policy_query,
    .prefer_fma = loom_x86_math_prefer_avx512_fma,
    .user_data = &kX86Avx512MathProfile,
};

static const loom_target_math_policy_registry_entry_t kX86MathPolicyEntries[] =
    {
        {/*.contract_set_key=*/IREE_SVL("x86.avx512.core"),
         /*.policy=*/&kX86Avx512MathPolicy},
        {/*.contract_set_key=*/IREE_SVL("x86.avx2.core"),
         /*.policy=*/&kX86MathPolicy},
        {/*.contract_set_key=*/IREE_SVL("x86.avx2_features.core"),
         /*.policy=*/&kX86MathPolicy},
        {/*.contract_set_key=*/IREE_SVL("x86.packed_dot.core"),
         /*.policy=*/&kX86MathPolicy},
        {/*.contract_set_key=*/IREE_SVL("x86.avx512_features.core"),
         /*.policy=*/&kX86Avx512FeaturesMathPolicy},
};

void loom_x86_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kX86MathPolicyEntries,
      IREE_ARRAYSIZE(kX86MathPolicyEntries));
}
