// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/math_policy.h"

#include <stddef.h>

#include "loom/target/arch/x86/feature_bits.h"

enum loom_x86_math_policy_feature_bits_e {
  LOOM_X86_MATH_POLICY_FEATURE_NONE = 0u,
  LOOM_X86_MATH_POLICY_FEATURE_AVX2 = 1u << 0,
  LOOM_X86_MATH_POLICY_FEATURE_AVX512 = 1u << 1,
  LOOM_X86_MATH_POLICY_FEATURE_TARGET_FP16 = 1u << 2,
};

enum loom_x86_math_feature_class_e {
  LOOM_X86_MATH_FEATURE_CLASS_NONE = 0,
  LOOM_X86_MATH_FEATURE_CLASS_AVX2 = 1,
  LOOM_X86_MATH_FEATURE_CLASS_AVX512 = 2,
  LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16 = 3,
  LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16_VL = 4,
};

static const loom_target_math_policy_feature_bits_t
    kX86MathPolicyFeatureMasks[] = {
        [LOOM_X86_MATH_FEATURE_CLASS_AVX2 - 1] =
            LOOM_X86_MATH_POLICY_FEATURE_AVX2,
        [LOOM_X86_MATH_FEATURE_CLASS_AVX512 - 1] =
            LOOM_X86_MATH_POLICY_FEATURE_AVX512,
        [LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16 - 1] =
            LOOM_X86_MATH_POLICY_FEATURE_TARGET_FP16,
        [LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16_VL - 1] =
            LOOM_X86_MATH_POLICY_FEATURE_TARGET_FP16,
};

static const uint64_t kX86MathTargetFeatureMasks[] = {
    [LOOM_X86_MATH_FEATURE_CLASS_AVX2 - 1] = 0,
    [LOOM_X86_MATH_FEATURE_CLASS_AVX512 - 1] = 0,
    [LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16 - 1] =
        LOOM_X86_FEATURE_AVX512_FP16,
    [LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16_VL - 1] =
        LOOM_X86_FEATURE_AVX512_FP16 | LOOM_X86_FEATURE_AVX512_VL,
};

#define LOOM_X86_MATH_CONSTRAINTS(_)                                         \
  _(ACTIVATION_EXACT_F32, activation_exact_f32, "math.activation.exact_f32") \
  _(BASIS_LOG2, basis_log2, "math.basis.log2")                               \
  _(ELEMENT_F32, element_f32, "math.element.f32")                            \
  _(ELEMENT_NATIVE_FLOAT, element_native_float, "math.element.native_float") \
  _(ERF_EXACT_F32, erf_exact_f32, "math.erf.exact_f32")                      \
  _(EXP_EXACT_F32, exp_exact_f32, "math.exp.exact_f32")                      \
  _(EXP2_EXACT_F32, exp2_exact_f32, "math.exp2.exact_f32")                   \
  _(OP_NATIVE_FLOAT, op_native_float, "math.op.native_float")                \
  _(OP_NATIVE_RSQRT_ESTIMATE, op_native_rsqrt_estimate,                      \
    "math.op.native_rsqrt_estimate")                                         \
  _(OP_ROUNDING, op_rounding, "math.op.rounding")                            \
  _(OP_SUPPORTED, op_supported, "math.op.supported")                         \
  _(RECIPE_APPROXIMATE_COS_TURNS_F32, recipe_approximate_cos_turns_f32,      \
    "math.recipe.approximate_cos_turns_f32")                                 \
  _(RECIPE_APPROXIMATE_EXP2_F32, recipe_approximate_exp2_f32,                \
    "math.recipe.approximate_exp2_f32")                                      \
  _(RECIPE_APPROXIMATE_SIN_TURNS_F32, recipe_approximate_sin_turns_f32,      \
    "math.recipe.approximate_sin_turns_f32")                                 \
  _(RECIPE_COS_TURNS_F32, recipe_cos_turns_f32, "math.recipe.cos_turns_f32") \
  _(RECIPE_ERF_RATIONAL_F32, recipe_erf_rational_f32,                        \
    "math.recipe.erf_rational_f32")                                          \
  _(RECIPE_EXP_EXP2_F32, recipe_exp_exp2_f32, "math.recipe.exp_exp2_f32")    \
  _(RECIPE_GELU_ERF_F32, recipe_gelu_erf_f32, "math.recipe.gelu_erf_f32")    \
  _(RECIPE_GELU_LOGISTIC_F32, recipe_gelu_logistic_f32,                      \
    "math.recipe.gelu_logistic_f32")                                         \
  _(RECIPE_GELU_TANH_F32, recipe_gelu_tanh_f32, "math.recipe.gelu_tanh_f32") \
  _(RECIPE_LOGISTIC_EXP2_F32, recipe_logistic_exp2_f32,                      \
    "math.recipe.logistic_exp2_f32")                                         \
  _(RECIPE_RSQRT_NEWTON_ONE, recipe_rsqrt_newton_one,                        \
    "math.recipe.rsqrt_newton_one")                                          \
  _(RECIPE_RSQRT_NEWTON_TWO, recipe_rsqrt_newton_two,                        \
    "math.recipe.rsqrt_newton_two")                                          \
  _(RECIPE_RSQRT_SQRT_DIV, recipe_rsqrt_sqrt_div,                            \
    "math.recipe.rsqrt_sqrt_div")                                            \
  _(RECIPE_SILU_LOGISTIC_F32, recipe_silu_logistic_f32,                      \
    "math.recipe.silu_logistic_f32")                                         \
  _(RECIPE_SIN_TURNS_F32, recipe_sin_turns_f32, "math.recipe.sin_turns_f32") \
  _(RECIPE_TANH_LOGISTIC_F32, recipe_tanh_logistic_f32,                      \
    "math.recipe.tanh_logistic_f32")                                         \
  _(TANH_EXACT_F32, tanh_exact_f32, "math.tanh.exact_f32")                   \
  _(TRIG_EXACT_F32, trig_exact_f32, "math.trig.exact_f32")                   \
  _(TURNS_TRIG_EXACT_F32, turns_trig_exact_f32, "math.turns_trig.exact_f32")

enum loom_x86_math_constraint_e {
  LOOM_X86_MATH_CONSTRAINT_NONE = 0,
#define LOOM_X86_MATH_CONSTRAINT_ENUM(name, field, value) \
  LOOM_X86_MATH_CONSTRAINT_##name,
  LOOM_X86_MATH_CONSTRAINTS(LOOM_X86_MATH_CONSTRAINT_ENUM)
#undef LOOM_X86_MATH_CONSTRAINT_ENUM
      LOOM_X86_MATH_CONSTRAINT_COUNT_,
};

typedef struct loom_x86_math_constraint_strings_t {
#define LOOM_X86_MATH_CONSTRAINT_FIELD(name, field, value) \
  char field[sizeof(value)];
  LOOM_X86_MATH_CONSTRAINTS(LOOM_X86_MATH_CONSTRAINT_FIELD)
#undef LOOM_X86_MATH_CONSTRAINT_FIELD
} loom_x86_math_constraint_strings_t;

static const loom_x86_math_constraint_strings_t kX86MathConstraintStrings = {
#define LOOM_X86_MATH_CONSTRAINT_VALUE(name, field, value) .field = value,
    LOOM_X86_MATH_CONSTRAINTS(LOOM_X86_MATH_CONSTRAINT_VALUE)
#undef LOOM_X86_MATH_CONSTRAINT_VALUE
};

static const loom_string_ref_t
    kX86MathConstraints[LOOM_X86_MATH_CONSTRAINT_COUNT_ - 1] = {
#define LOOM_X86_MATH_CONSTRAINT_REF(name, field, value)                      \
  [LOOM_X86_MATH_CONSTRAINT_##name - 1] =                                     \
      LOOM_STRING_REF(0, offsetof(loom_x86_math_constraint_strings_t, field), \
                      sizeof(value) - 1),
        LOOM_X86_MATH_CONSTRAINTS(LOOM_X86_MATH_CONSTRAINT_REF)
#undef LOOM_X86_MATH_CONSTRAINT_REF
};

#undef LOOM_X86_MATH_CONSTRAINTS

#define LOOM_X86_MATH_DIRECT_RULE(element_types_, required_fastmath_flags_,   \
                                  recipe_, native_required_fastmath_flags_,   \
                                  constraint_, permission_constraint_)        \
  LOOM_TARGET_MATH_POLICY_DIRECT_RULE(                                        \
      element_types_, LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,                   \
      required_fastmath_flags_, recipe_, LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, \
      native_required_fastmath_flags_, 0, constraint_, permission_constraint_)

#define LOOM_X86_MATH_APPROXIMATE_RULE(op_, recipe_, recipe_constraint_,      \
                                       exact_constraint_)                     \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_X86_MATH_DIRECT_RULE(                    \
      LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,           \
      LOOM_TARGET_MATH_RECIPE_##recipe_, LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, \
      LOOM_X86_MATH_CONSTRAINT_##recipe_constraint_,                          \
      LOOM_X86_MATH_CONSTRAINT_##exact_constraint_)

#define LOOM_X86_MATH_REJECT_F32_RULE(op_, constraint_)              \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_X86_MATH_DIRECT_RULE(           \
      LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, \
      LOOM_TARGET_MATH_POLICY_RECIPE_REJECT,                         \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,                           \
      LOOM_X86_MATH_CONSTRAINT_##constraint_, LOOM_X86_MATH_CONSTRAINT_NONE)

static const loom_target_math_policy_direct_rule_t
    kX86MathDirectRules[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_X86_MATH_NATIVE_RULE(op_)                                      \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_X86_MATH_DIRECT_RULE(                  \
      LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64,                  \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, LOOM_TARGET_MATH_RECIPE_UNKNOWN, \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,                                  \
      LOOM_X86_MATH_CONSTRAINT_OP_NATIVE_FLOAT, LOOM_X86_MATH_CONSTRAINT_NONE)
        LOOM_X86_MATH_NATIVE_RULE(ADDF),
        LOOM_X86_MATH_NATIVE_RULE(SUBF),
        LOOM_X86_MATH_NATIVE_RULE(MULF),
        LOOM_X86_MATH_NATIVE_RULE(SQRTF),
#undef LOOM_X86_MATH_NATIVE_RULE
        [LOOM_TARGET_MATH_OP_RSQRTF] = LOOM_X86_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F32,
            LOOM_TARGET_MATH_FASTMATH_FLAG_AFN |
                LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP,
            LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
            LOOM_X86_MATH_CONSTRAINT_OP_NATIVE_RSQRT_ESTIMATE,
            LOOM_X86_MATH_CONSTRAINT_NONE),
        LOOM_X86_MATH_APPROXIMATE_RULE(LOGISTICF, LOGISTIC_EXP2_F32,
                                       RECIPE_LOGISTIC_EXP2_F32,
                                       ACTIVATION_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(SILUF, SILU_LOGISTIC_F32,
                                       RECIPE_SILU_LOGISTIC_F32,
                                       ACTIVATION_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(EXPF, EXP_EXP2_F32, RECIPE_EXP_EXP2_F32,
                                       EXP_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(ERFF, ERF_RATIONAL_F32,
                                       RECIPE_ERF_RATIONAL_F32, ERF_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(
            GELUF_ERF, GELU_ERF_F32, RECIPE_GELU_ERF_F32, ACTIVATION_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(GELUF_TANH, GELU_TANH_F32,
                                       RECIPE_GELU_TANH_F32,
                                       ACTIVATION_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(GELUF_LOGISTIC, GELU_LOGISTIC_F32,
                                       RECIPE_GELU_LOGISTIC_F32,
                                       ACTIVATION_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(SINF, SIN_TURNS_F32,
                                       RECIPE_SIN_TURNS_F32, TRIG_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(COSF, COS_TURNS_F32,
                                       RECIPE_COS_TURNS_F32, TRIG_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(SINTURNSF, APPROXIMATE_SIN_TURNS_F32,
                                       RECIPE_APPROXIMATE_SIN_TURNS_F32,
                                       TURNS_TRIG_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(COSTURNSF, APPROXIMATE_COS_TURNS_F32,
                                       RECIPE_APPROXIMATE_COS_TURNS_F32,
                                       TURNS_TRIG_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(
            TANHF, TANH_LOGISTIC_F32, RECIPE_TANH_LOGISTIC_F32, TANH_EXACT_F32),
        LOOM_X86_MATH_APPROXIMATE_RULE(EXP2F, APPROXIMATE_EXP2_F32,
                                       RECIPE_APPROXIMATE_EXP2_F32,
                                       EXP2_EXACT_F32),
        LOOM_X86_MATH_REJECT_F32_RULE(SOFTPLUSF, BASIS_LOG2),
        LOOM_X86_MATH_REJECT_F32_RULE(LOGF, BASIS_LOG2),
        LOOM_X86_MATH_REJECT_F32_RULE(LOG2F, BASIS_LOG2),
        LOOM_X86_MATH_REJECT_F32_RULE(POWF, BASIS_LOG2),
        LOOM_X86_MATH_REJECT_F32_RULE(CEILF, OP_ROUNDING),
        LOOM_X86_MATH_REJECT_F32_RULE(FLOORF, OP_ROUNDING),
        LOOM_X86_MATH_REJECT_F32_RULE(ROUNDF, OP_ROUNDING),
        LOOM_X86_MATH_REJECT_F32_RULE(ROUNDEVENF, OP_ROUNDING),
        LOOM_X86_MATH_REJECT_F32_RULE(TRUNCF, OP_ROUNDING),
        LOOM_X86_MATH_REJECT_F32_RULE(ASINHF, OP_SUPPORTED),
        LOOM_X86_MATH_REJECT_F32_RULE(ACOSHF, OP_SUPPORTED),
        LOOM_X86_MATH_REJECT_F32_RULE(ATANHF, OP_SUPPORTED),
};

#undef LOOM_X86_MATH_REJECT_F32_RULE
#undef LOOM_X86_MATH_APPROXIMATE_RULE
#undef LOOM_X86_MATH_DIRECT_RULE

#define LOOM_X86_MATH_MATCH(element_types_, lane_domains_, shape_,    \
                            minimum_lane_count_, maximum_lane_count_, \
                            required_fastmath_flags_, feature_class_) \
  {                                                                   \
      .element_types = (element_types_),                              \
      .minimum_lane_count = (minimum_lane_count_),                    \
      .maximum_lane_count = (maximum_lane_count_),                    \
      .lane_domains = (lane_domains_),                                \
      .shape = (shape_),                                              \
      .required_fastmath_flags = (required_fastmath_flags_),          \
      .feature_class = (feature_class_),                              \
  }

enum loom_x86_math_rule_e {
  LOOM_X86_MATH_RULE_NATIVE_FP16 = 0,
  LOOM_X86_MATH_RULE_RSQRT_ESTIMATE_F64,
  LOOM_X86_MATH_RULE_RSQRT_ESTIMATE_F16,
  LOOM_X86_MATH_RULE_RSQRT_REFINE_F32,
  LOOM_X86_MATH_RULE_RSQRT_REFINE_F64,
  LOOM_X86_MATH_RULE_RSQRT_REWRITE_F32,
  LOOM_X86_MATH_RULE_RSQRT_REWRITE_F64_NATIVE,
  LOOM_X86_MATH_RULE_RSQRT_REWRITE_F16_NATIVE,
  LOOM_X86_MATH_RULE_RSQRT_REWRITE_F64,
  LOOM_X86_MATH_RULE_COUNT_,
};

static const loom_target_math_policy_rule_t
    kX86MathRules[LOOM_X86_MATH_RULE_COUNT_] = {
        [LOOM_X86_MATH_RULE_NATIVE_FP16] =
            {
                .match = LOOM_X86_MATH_MATCH(
                    LOOM_SCALAR_TYPE_SET_F16,
                    LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,
                    LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                    LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                    LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16),
                .decision_constraint = LOOM_X86_MATH_CONSTRAINT_OP_NATIVE_FLOAT,
            },
        [LOOM_X86_MATH_RULE_RSQRT_ESTIMATE_F64] =
            {
                .match =
                    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F64,
                                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,
                                        LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                                        LOOM_TARGET_MATH_FASTMATH_FLAG_AFN |
                                            LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP,
                                        LOOM_X86_MATH_FEATURE_CLASS_AVX512),
                .decision_constraint =
                    LOOM_X86_MATH_CONSTRAINT_OP_NATIVE_RSQRT_ESTIMATE,
            },
        [LOOM_X86_MATH_RULE_RSQRT_ESTIMATE_F16] =
            {
                .match = LOOM_X86_MATH_MATCH(
                    LOOM_SCALAR_TYPE_SET_F16,
                    LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,
                    LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                    LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
                    LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16),
                .decision_constraint =
                    LOOM_X86_MATH_CONSTRAINT_OP_NATIVE_RSQRT_ESTIMATE,
            },
        [LOOM_X86_MATH_RULE_RSQRT_REFINE_F32] =
            {
                .match =
                    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F32,
                                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,
                                        LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                                        LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
                                        LOOM_X86_MATH_FEATURE_CLASS_NONE),
                .recipe = LOOM_TARGET_MATH_RECIPE_RSQRT_NEWTON_ONE,
                .recipe_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN |
                                         LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP,
                .native_required_fastmath_flags =
                    LOOM_TARGET_MATH_FASTMATH_FLAG_AFN |
                    LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP,
                .decision_constraint =
                    LOOM_X86_MATH_CONSTRAINT_RECIPE_RSQRT_NEWTON_ONE,
            },
        [LOOM_X86_MATH_RULE_RSQRT_REFINE_F64] =
            {
                .match =
                    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F64,
                                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,
                                        LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                                        LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
                                        LOOM_X86_MATH_FEATURE_CLASS_AVX512),
                .recipe = LOOM_TARGET_MATH_RECIPE_RSQRT_NEWTON_TWO,
                .recipe_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN |
                                         LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP,
                .native_required_fastmath_flags =
                    LOOM_TARGET_MATH_FASTMATH_FLAG_AFN |
                    LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP,
                .decision_constraint =
                    LOOM_X86_MATH_CONSTRAINT_RECIPE_RSQRT_NEWTON_TWO,
            },
        [LOOM_X86_MATH_RULE_RSQRT_REWRITE_F32] =
            {
                .match =
                    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F32,
                                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,
                                        LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                                        LOOM_X86_MATH_FEATURE_CLASS_NONE),
                .recipe = LOOM_TARGET_MATH_RECIPE_RSQRT_SQRT_DIV,
                .native_required_fastmath_flags =
                    LOOM_TARGET_MATH_FASTMATH_FLAG_AFN |
                    LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP,
                .decision_constraint =
                    LOOM_X86_MATH_CONSTRAINT_RECIPE_RSQRT_SQRT_DIV,
            },
        [LOOM_X86_MATH_RULE_RSQRT_REWRITE_F64_NATIVE] =
            {
                .match =
                    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F64,
                                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,
                                        LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                                        LOOM_X86_MATH_FEATURE_CLASS_AVX512),
                .recipe = LOOM_TARGET_MATH_RECIPE_RSQRT_SQRT_DIV,
                .native_required_fastmath_flags =
                    LOOM_TARGET_MATH_FASTMATH_FLAG_AFN |
                    LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP,
                .decision_constraint =
                    LOOM_X86_MATH_CONSTRAINT_RECIPE_RSQRT_SQRT_DIV,
            },
        [LOOM_X86_MATH_RULE_RSQRT_REWRITE_F16_NATIVE] =
            {
                .match = LOOM_X86_MATH_MATCH(
                    LOOM_SCALAR_TYPE_SET_F16,
                    LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,
                    LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                    LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                    LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16),
                .recipe = LOOM_TARGET_MATH_RECIPE_RSQRT_SQRT_DIV,
                .native_required_fastmath_flags =
                    LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
                .decision_constraint =
                    LOOM_X86_MATH_CONSTRAINT_RECIPE_RSQRT_SQRT_DIV,
            },
        [LOOM_X86_MATH_RULE_RSQRT_REWRITE_F64] =
            {
                .match =
                    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F64,
                                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,
                                        LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                                        LOOM_X86_MATH_FEATURE_CLASS_NONE),
                .recipe = LOOM_TARGET_MATH_RECIPE_RSQRT_SQRT_DIV,
                .decision_constraint =
                    LOOM_X86_MATH_CONSTRAINT_RECIPE_RSQRT_SQRT_DIV,
            },
};

#define LOOM_X86_MATH_SPAN(start_, count_, fallback_) \
  {                                                   \
      .rule_start = (start_),                         \
      .rule_count = (count_),                         \
      .fallback_constraint = (fallback_),             \
  }

static const loom_target_math_policy_rule_span_t
    kX86MathOperationSpans[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_X86_MATH_NATIVE_SPAN(op_)                      \
  [LOOM_TARGET_MATH_OP_##op_] =                             \
      LOOM_X86_MATH_SPAN(LOOM_X86_MATH_RULE_NATIVE_FP16, 1, \
                         LOOM_X86_MATH_CONSTRAINT_ELEMENT_NATIVE_FLOAT)
        LOOM_X86_MATH_NATIVE_SPAN(ADDF),
        LOOM_X86_MATH_NATIVE_SPAN(SUBF),
        LOOM_X86_MATH_NATIVE_SPAN(MULF),
        LOOM_X86_MATH_NATIVE_SPAN(SQRTF),
#undef LOOM_X86_MATH_NATIVE_SPAN
        [LOOM_TARGET_MATH_OP_RSQRTF] = LOOM_X86_MATH_SPAN(
            LOOM_X86_MATH_RULE_RSQRT_ESTIMATE_F64,
            LOOM_X86_MATH_RULE_COUNT_ - LOOM_X86_MATH_RULE_RSQRT_ESTIMATE_F64,
            LOOM_X86_MATH_CONSTRAINT_ELEMENT_NATIVE_FLOAT),
#define LOOM_X86_MATH_F32_SPAN(op_) \
  [LOOM_TARGET_MATH_OP_##op_] =     \
      LOOM_X86_MATH_SPAN(0, 0, LOOM_X86_MATH_CONSTRAINT_ELEMENT_F32)
        LOOM_X86_MATH_F32_SPAN(LOGISTICF),
        LOOM_X86_MATH_F32_SPAN(SILUF),
        LOOM_X86_MATH_F32_SPAN(SOFTPLUSF),
        LOOM_X86_MATH_F32_SPAN(EXPF),
        LOOM_X86_MATH_F32_SPAN(ERFF),
        LOOM_X86_MATH_F32_SPAN(GELUF_ERF),
        LOOM_X86_MATH_F32_SPAN(GELUF_TANH),
        LOOM_X86_MATH_F32_SPAN(GELUF_LOGISTIC),
        LOOM_X86_MATH_F32_SPAN(LOGF),
        LOOM_X86_MATH_F32_SPAN(LOG2F),
        LOOM_X86_MATH_F32_SPAN(SINF),
        LOOM_X86_MATH_F32_SPAN(COSF),
        LOOM_X86_MATH_F32_SPAN(SINTURNSF),
        LOOM_X86_MATH_F32_SPAN(COSTURNSF),
        LOOM_X86_MATH_F32_SPAN(TANHF),
        LOOM_X86_MATH_F32_SPAN(POWF),
        LOOM_X86_MATH_F32_SPAN(CEILF),
        LOOM_X86_MATH_F32_SPAN(FLOORF),
        LOOM_X86_MATH_F32_SPAN(ROUNDF),
        LOOM_X86_MATH_F32_SPAN(ROUNDEVENF),
        LOOM_X86_MATH_F32_SPAN(TRUNCF),
        LOOM_X86_MATH_F32_SPAN(EXP2F),
        LOOM_X86_MATH_F32_SPAN(ASINHF),
        LOOM_X86_MATH_F32_SPAN(ACOSHF),
        LOOM_X86_MATH_F32_SPAN(ATANHF),
#undef LOOM_X86_MATH_F32_SPAN
};

#undef LOOM_X86_MATH_SPAN

static const loom_target_math_policy_match_t kX86MathFmaForms[] = {
    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64,
                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_SCALAR,
                        LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                        LOOM_X86_MATH_FEATURE_CLASS_NONE),
    LOOM_X86_MATH_MATCH(
        LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 2, 2,
        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, LOOM_X86_MATH_FEATURE_CLASS_AVX2),
    LOOM_X86_MATH_MATCH(
        LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 4, 4,
        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, LOOM_X86_MATH_FEATURE_CLASS_AVX2),
    LOOM_X86_MATH_MATCH(
        LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 8, 8,
        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, LOOM_X86_MATH_FEATURE_CLASS_AVX2),
    LOOM_X86_MATH_MATCH(
        LOOM_SCALAR_TYPE_SET_F64, LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 1, 1,
        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, LOOM_X86_MATH_FEATURE_CLASS_AVX2),
    LOOM_X86_MATH_MATCH(
        LOOM_SCALAR_TYPE_SET_F64, LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 2, 2,
        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, LOOM_X86_MATH_FEATURE_CLASS_AVX2),
    LOOM_X86_MATH_MATCH(
        LOOM_SCALAR_TYPE_SET_F64, LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 4, 4,
        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, LOOM_X86_MATH_FEATURE_CLASS_AVX2),
    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F32,
                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
                        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 16, 16,
                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                        LOOM_X86_MATH_FEATURE_CLASS_AVX512),
    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F64,
                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
                        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 8, 8,
                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                        LOOM_X86_MATH_FEATURE_CLASS_AVX512),
    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F16,
                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_SCALAR,
                        LOOM_TARGET_MATH_POLICY_SHAPE_ANY, 0, 0,
                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                        LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16),
    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F16,
                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
                        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 2, 2,
                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                        LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16_VL),
    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F16,
                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
                        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 4, 4,
                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                        LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16_VL),
    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F16,
                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
                        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 8, 8,
                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                        LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16_VL),
    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F16,
                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
                        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 16, 16,
                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                        LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16_VL),
    LOOM_X86_MATH_MATCH(LOOM_SCALAR_TYPE_SET_F16,
                        LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
                        LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE, 32, 32,
                        LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
                        LOOM_X86_MATH_FEATURE_CLASS_TARGET_FP16),
};

#undef LOOM_X86_MATH_MATCH

static const loom_target_math_policy_table_t kX86MathPolicyTable = {
    .operation_spans = kX86MathOperationSpans,
    .rules = kX86MathRules,
    .fma_forms = kX86MathFmaForms,
    .fma_form_count = IREE_ARRAYSIZE(kX86MathFmaForms),
    .default_constraint = LOOM_X86_MATH_CONSTRAINT_OP_SUPPORTED,
    .constraints = kX86MathConstraints,
    .constraint_strings =
        {
            .data = {(const char*)&kX86MathConstraintStrings},
            .data_length = {sizeof(kX86MathConstraintStrings)},
        },
    .target_feature_masks = kX86MathTargetFeatureMasks,
    .policy_feature_masks = kX86MathPolicyFeatureMasks,
};

static const loom_target_math_policy_t kX86Avx2MathPolicy = {
    .name = IREE_SVL("x86-math"),
    .direct_rules = kX86MathDirectRules,
    .table = &kX86MathPolicyTable,
    .feature_bits = LOOM_X86_MATH_POLICY_FEATURE_AVX2,
};

static const loom_target_math_policy_t kX86Avx512MathPolicy = {
    .name = IREE_SVL("x86-math"),
    .direct_rules = kX86MathDirectRules,
    .table = &kX86MathPolicyTable,
    .feature_bits = LOOM_X86_MATH_POLICY_FEATURE_AVX2 |
                    LOOM_X86_MATH_POLICY_FEATURE_AVX512 |
                    LOOM_X86_MATH_POLICY_FEATURE_TARGET_FP16,
};

static const loom_target_math_policy_registry_entry_t kX86MathPolicyEntries[] =
    {
        {/*.contract_set_key=*/IREE_SVL("x86.avx512.core"),
         /*.policy=*/&kX86Avx512MathPolicy},
        {/*.contract_set_key=*/IREE_SVL("x86.avx2.core"),
         /*.policy=*/&kX86Avx2MathPolicy},
        {/*.contract_set_key=*/IREE_SVL("x86.avx2_features.core"),
         /*.policy=*/&kX86Avx2MathPolicy},
        {/*.contract_set_key=*/IREE_SVL("x86.packed_dot.core"),
         /*.policy=*/&kX86Avx2MathPolicy},
        {/*.contract_set_key=*/IREE_SVL("x86.avx512_features.core"),
         /*.policy=*/&kX86Avx512MathPolicy},
};

void loom_x86_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kX86MathPolicyEntries,
      IREE_ARRAYSIZE(kX86MathPolicyEntries));
}
