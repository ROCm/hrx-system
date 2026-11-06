// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/math_policy.h"

#include <stddef.h>

#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

enum loom_amdgpu_math_policy_feature_bits_e {
  LOOM_AMDGPU_MATH_POLICY_FEATURE_NONE = 0u,
  LOOM_AMDGPU_MATH_POLICY_FEATURE_NATIVE_PACKED_BF16 = 1u << 0,
};

enum loom_amdgpu_math_feature_class_e {
  LOOM_AMDGPU_MATH_FEATURE_CLASS_NONE = 0,
  LOOM_AMDGPU_MATH_FEATURE_CLASS_NATIVE_PACKED_BF16 = 1,
};

static const loom_target_math_policy_feature_bits_t
    kAmdgpuMathPolicyFeatureMasks[] = {
        [LOOM_AMDGPU_MATH_FEATURE_CLASS_NATIVE_PACKED_BF16 - 1] =
            LOOM_AMDGPU_MATH_POLICY_FEATURE_NATIVE_PACKED_BF16,
};

#define LOOM_AMDGPU_MATH_CONSTRAINTS(_)                                      \
  _(BASIC_KEEP, basic_keep, "math.basic.keep")                               \
  _(ELEMENT_F32, element_f32, "math.element.f32")                            \
  _(EXP_EXACT_F32, exp_exact_f32, "math.exp.exact_f32")                      \
  _(INVERSE_HYPERBOLIC_AFN, inverse_hyperbolic_afn,                          \
    "math.inverse_hyperbolic.afn")                                           \
  _(INVERSE_HYPERBOLIC_F32_EVALUATION, inverse_hyperbolic_f32_evaluation,    \
    "math.inverse_hyperbolic.f32_evaluation")                                \
  _(LOG_EXACT_F32, log_exact_f32, "math.log.exact_f32")                      \
  _(LOG2_EXACT_F32, log2_exact_f32, "math.log2.exact_f32")                   \
  _(OP_NATIVE_APPROX_F32, op_native_approx_f32, "math.op.native_approx_f32") \
  _(OP_NATIVE_F32, op_native_f32, "math.op.native_f32")                      \
  _(OP_NATIVE_F64, op_native_f64, "math.op.native_f64")                      \
  _(OP_NATIVE_PACKED_BF16, op_native_packed_bf16, "math.op.native_pk_bf16")  \
  _(OP_NATIVE_RSQRT_AFN, op_native_rsqrt_afn, "math.op.native_rsqrt_afn")    \
  _(OP_NATIVE_SQRT, op_native_sqrt, "math.op.native_sqrt")                   \
  _(OP_SUPPORTED, op_supported, "math.op.supported")                         \
  _(POW_EXACT_F32, pow_exact_f32, "math.pow.exact_f32")                      \
  _(RECIPE_COS_TURNS_F32, recipe_cos_turns_f32, "math.recipe.cos_turns_f32") \
  _(RECIPE_ERF_RATIONAL_F32, recipe_erf_rational_f32,                        \
    "math.recipe.erf_rational_f32")                                          \
  _(RECIPE_EXP_EXP2_F32, recipe_exp_exp2_f32, "math.recipe.exp_exp2_f32")    \
  _(RECIPE_GELU_ERF_F32, recipe_gelu_erf_f32, "math.recipe.gelu_erf_f32")    \
  _(RECIPE_GELU_LOGISTIC_F32, recipe_gelu_logistic_f32,                      \
    "math.recipe.gelu_logistic_f32")                                         \
  _(RECIPE_GELU_TANH_F32, recipe_gelu_tanh_f32, "math.recipe.gelu_tanh_f32") \
  _(RECIPE_INVERSE_HYPERBOLIC_F32, recipe_inverse_hyperbolic_f32,            \
    "math.recipe.inverse_hyperbolic_f32")                                    \
  _(RECIPE_LOG_LOG2_F32, recipe_log_log2_f32, "math.recipe.log_log2_f32")    \
  _(RECIPE_LOGISTIC_EXP2_F32, recipe_logistic_exp2_f32,                      \
    "math.recipe.logistic_exp2_f32")                                         \
  _(RECIPE_POW_LOG2_EXP2_F32, recipe_pow_log2_exp2_f32,                      \
    "math.recipe.pow_log2_exp2_f32")                                         \
  _(RECIPE_ROUND_AWAY, recipe_round_away, "math.recipe.round_away")          \
  _(RECIPE_RSQRT_SQRT_DIV, recipe_rsqrt_sqrt_div,                            \
    "math.recipe.rsqrt_sqrt_div")                                            \
  _(RECIPE_SILU_LOGISTIC_F32, recipe_silu_logistic_f32,                      \
    "math.recipe.silu_logistic_f32")                                         \
  _(RECIPE_SIN_TURNS_F32, recipe_sin_turns_f32, "math.recipe.sin_turns_f32") \
  _(RECIPE_SOFTPLUS_EXP2_F32, recipe_softplus_exp2_f32,                      \
    "math.recipe.softplus_exp2_f32")                                         \
  _(RECIPE_TANH_LOGISTIC_F32, recipe_tanh_logistic_f32,                      \
    "math.recipe.tanh_logistic_f32")                                         \
  _(RECIPE_WIDEN_F32_ROUND, recipe_widen_f32_round,                          \
    "math.recipe.widen_f32_round")                                           \
  _(ROOT_F32_OR_SCALAR_F64, root_f32_or_scalar_f64,                          \
    "math.root.f32_or_scalar_f64")                                           \
  _(TANH_EXACT_F32, tanh_exact_f32, "math.tanh.exact_f32")                   \
  _(TRIG_EXACT_F32, trig_exact_f32, "math.trig.exact_f32")                   \
  _(TURNS_TRIG_EXACT_F32, turns_trig_exact_f32, "math.turns_trig.exact_f32")

enum loom_amdgpu_math_constraint_e {
  LOOM_AMDGPU_MATH_CONSTRAINT_NONE = 0,
#define LOOM_AMDGPU_MATH_CONSTRAINT_ENUM(name, field, value) \
  LOOM_AMDGPU_MATH_CONSTRAINT_##name,
  LOOM_AMDGPU_MATH_CONSTRAINTS(LOOM_AMDGPU_MATH_CONSTRAINT_ENUM)
#undef LOOM_AMDGPU_MATH_CONSTRAINT_ENUM
      LOOM_AMDGPU_MATH_CONSTRAINT_COUNT_,
};

typedef struct loom_amdgpu_math_constraint_strings_t {
#define LOOM_AMDGPU_MATH_CONSTRAINT_FIELD(name, field, value) \
  char field[sizeof(value)];
  LOOM_AMDGPU_MATH_CONSTRAINTS(LOOM_AMDGPU_MATH_CONSTRAINT_FIELD)
#undef LOOM_AMDGPU_MATH_CONSTRAINT_FIELD
} loom_amdgpu_math_constraint_strings_t;

static const loom_amdgpu_math_constraint_strings_t
    kAmdgpuMathConstraintStrings = {
#define LOOM_AMDGPU_MATH_CONSTRAINT_VALUE(name, field, value) .field = value,
        LOOM_AMDGPU_MATH_CONSTRAINTS(LOOM_AMDGPU_MATH_CONSTRAINT_VALUE)
#undef LOOM_AMDGPU_MATH_CONSTRAINT_VALUE
};

static const loom_string_ref_t
    kAmdgpuMathConstraints[LOOM_AMDGPU_MATH_CONSTRAINT_COUNT_ - 1] = {
#define LOOM_AMDGPU_MATH_CONSTRAINT_REF(name, field, value)      \
  [LOOM_AMDGPU_MATH_CONSTRAINT_##name - 1] = LOOM_STRING_REF(    \
      0, offsetof(loom_amdgpu_math_constraint_strings_t, field), \
      sizeof(value) - 1),
        LOOM_AMDGPU_MATH_CONSTRAINTS(LOOM_AMDGPU_MATH_CONSTRAINT_REF)
#undef LOOM_AMDGPU_MATH_CONSTRAINT_REF
};

#undef LOOM_AMDGPU_MATH_CONSTRAINTS

static_assert(LOOM_AMDGPU_MATH_CONSTRAINT_COUNT_ - 1 <= UINT8_MAX,
              "AMDGPU math constraints must fit in one byte");
static_assert(LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES <= UINT8_MAX,
              "AMDGPU math lane bounds must fit in one byte");

enum loom_amdgpu_math_evaluation_e {
  LOOM_AMDGPU_MATH_EVALUATION_NONE = 0,
  LOOM_AMDGPU_MATH_EVALUATION_SOURCE_F32 = 1,
  LOOM_AMDGPU_MATH_EVALUATION_WIDENED_F32 = 2,
};

static const loom_target_math_evaluation_t kAmdgpuMathEvaluations[] = {
    [LOOM_AMDGPU_MATH_EVALUATION_SOURCE_F32 - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_SOURCE,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
        },
    [LOOM_AMDGPU_MATH_EVALUATION_WIDENED_F32 - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_WIDENED,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
        },
};

#define LOOM_AMDGPU_MATH_FASTMATH_NONE LOOM_TARGET_MATH_FASTMATH_FLAG_NONE
#define LOOM_AMDGPU_MATH_FASTMATH_ARCP LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP
#define LOOM_AMDGPU_MATH_FASTMATH_AFN LOOM_TARGET_MATH_FASTMATH_FLAG_AFN
#define LOOM_AMDGPU_MATH_FASTMATH_ARCP_AFN \
  (LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP | LOOM_TARGET_MATH_FASTMATH_FLAG_AFN)

#define LOOM_AMDGPU_MATH_RULES(_)                                              \
  _(BINARY_NATIVE_BF16, LOOM_SCALAR_TYPE_SET_BF16, VECTOR, RANK_ONE_EVEN, 1,   \
    LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES, NONE, NATIVE_PACKED_BF16,        \
    UNKNOWN, NONE, NONE, NONE, OP_NATIVE_PACKED_BF16, NONE)                    \
  _(BINARY_WIDEN,                                                              \
    LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2 |                \
        LOOM_SCALAR_TYPE_SET_BF16,                                             \
    ALL, ANY, 0, 0, NONE, NONE, WIDEN_F32_ROUND, NONE, NONE, NONE,             \
    RECIPE_WIDEN_F32_ROUND, NONE)                                              \
  _(DIV_WIDEN, LOOM_SCALAR_TYPE_SET_FLOAT_LE16, ALL, ANY, 0, 0, NONE, NONE,    \
    WIDEN_F32_ROUND, NONE, NONE, NONE, RECIPE_WIDEN_F32_ROUND, NONE)           \
  _(SUB_NATIVE_BF16, LOOM_SCALAR_TYPE_SET_BF16, VECTOR, RANK_ONE_EVEN, 1,      \
    LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES, NONE, NATIVE_PACKED_BF16,        \
    UNKNOWN, NONE, NONE, NONE, OP_NATIVE_PACKED_BF16, NONE)                    \
  _(SUB_WIDEN_NARROW,                                                          \
    LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2 |                \
        LOOM_SCALAR_TYPE_SET_BF16,                                             \
    ALL, ANY, 0, 0, NONE, NONE, WIDEN_F32_ROUND, NONE, NONE, NONE,             \
    RECIPE_WIDEN_F32_ROUND, NONE)                                              \
  _(SUB_KEEP_PACKED_F16, LOOM_SCALAR_TYPE_SET_F16, VECTOR, RANK_ONE_EVEN, 1,   \
    LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES, NONE, NONE, UNKNOWN, NONE, NONE, \
    NONE, BASIC_KEEP, NONE)                                                    \
  _(SUB_WIDEN_F16_VECTOR, LOOM_SCALAR_TYPE_SET_F16, VECTOR, ANY, 0, 0, NONE,   \
    NONE, WIDEN_F32_ROUND, NONE, NONE, NONE, RECIPE_WIDEN_F32_ROUND, NONE)     \
  _(SUB_KEEP_SCALAR_F16, LOOM_SCALAR_TYPE_SET_F16, SCALAR, ANY, 0, 0, NONE,    \
    NONE, UNKNOWN, NONE, NONE, NONE, BASIC_KEEP, NONE)                         \
  _(SQRT_SCALAR_F64, LOOM_SCALAR_TYPE_SET_F64, SCALAR, ANY, 0, 0, NONE, NONE,  \
    UNKNOWN, NONE, NONE, NONE, OP_NATIVE_SQRT, NONE)                           \
  _(RSQRT_NATIVE_SCALAR_F64, LOOM_SCALAR_TYPE_SET_F64, SCALAR, ANY, 0, 0, AFN, \
    NONE, UNKNOWN, NONE, NONE, NONE, OP_NATIVE_RSQRT_AFN, NONE)                \
  _(RSQRT_REWRITE_F32, LOOM_SCALAR_TYPE_SET_F32, ALL, ANY, 0, 0, NONE, NONE,   \
    RSQRT_SQRT_DIV, NONE, AFN, NONE, RECIPE_RSQRT_SQRT_DIV, NONE)              \
  _(RSQRT_REWRITE_SCALAR_F64, LOOM_SCALAR_TYPE_SET_F64, SCALAR, ANY, 0, 0,     \
    NONE, NONE, RSQRT_SQRT_DIV, NONE, AFN, NONE, RECIPE_RSQRT_SQRT_DIV, NONE)  \
  _(ROUNDING_F64, LOOM_SCALAR_TYPE_SET_F64, ALL, ANY, 0, 0, NONE, NONE,        \
    UNKNOWN, NONE, NONE, NONE, OP_NATIVE_F64, NONE)                            \
  _(ROUND_REWRITE, LOOM_SCALAR_TYPE_SET_F32, ALL, ANY, 0, 0, NONE, NONE,       \
    ROUND_AWAY, NONE, AFN, NONE, RECIPE_ROUND_AWAY, NONE)                      \
  _(INVERSE_HYPERBOLIC_WIDENED_F32, LOOM_SCALAR_TYPE_SET_FLOAT_LE16, ALL, ANY, \
    0, 0, AFN, NONE, INVERSE_HYPERBOLIC_F32, NONE, NONE, WIDENED_F32,          \
    RECIPE_INVERSE_HYPERBOLIC_F32, INVERSE_HYPERBOLIC_AFN)

enum loom_amdgpu_math_rule_e {
#define LOOM_AMDGPU_MATH_RULE_ENUM(name, element_types, lane_domains, shape,  \
                                   minimum_lane_count, maximum_lane_count,    \
                                   required_fastmath, feature_class, recipe,  \
                                   recipe_fastmath, native_required_fastmath, \
                                   evaluation, decision, permission)          \
  LOOM_AMDGPU_MATH_RULE_##name,
  LOOM_AMDGPU_MATH_RULES(LOOM_AMDGPU_MATH_RULE_ENUM)
#undef LOOM_AMDGPU_MATH_RULE_ENUM
      LOOM_AMDGPU_MATH_RULE_COUNT_,
};

static const loom_target_math_policy_rule_t
    kAmdgpuMathRules[LOOM_AMDGPU_MATH_RULE_COUNT_] = {
#define LOOM_AMDGPU_MATH_RULE_ROW(                                           \
    name_, element_types_, lane_domains_, shape_, minimum_lane_count_,       \
    maximum_lane_count_, required_fastmath_, feature_class_, recipe_,        \
    recipe_fastmath_, native_required_fastmath_, evaluation_, decision_,     \
    permission_)                                                             \
  [LOOM_AMDGPU_MATH_RULE_##name_] = {                                        \
      .match =                                                               \
          {                                                                  \
              .element_types = (element_types_),                             \
              .minimum_lane_count = (minimum_lane_count_),                   \
              .maximum_lane_count = (maximum_lane_count_),                   \
              .lane_domains =                                                \
                  LOOM_TARGET_MATH_LANE_DOMAIN_SET_##lane_domains_,          \
              .shape = LOOM_TARGET_MATH_POLICY_SHAPE_##shape_,               \
              .required_fastmath_flags =                                     \
                  LOOM_AMDGPU_MATH_FASTMATH_##required_fastmath_,            \
              .feature_class =                                               \
                  LOOM_AMDGPU_MATH_FEATURE_CLASS_##feature_class_,           \
          },                                                                 \
      .recipe = LOOM_TARGET_MATH_RECIPE_##recipe_,                           \
      .recipe_fastmath_flags = LOOM_AMDGPU_MATH_FASTMATH_##recipe_fastmath_, \
      .native_required_fastmath_flags =                                      \
          LOOM_AMDGPU_MATH_FASTMATH_##native_required_fastmath_,             \
      .evaluation = LOOM_AMDGPU_MATH_EVALUATION_##evaluation_,               \
      .decision_constraint = LOOM_AMDGPU_MATH_CONSTRAINT_##decision_,        \
      .permission_constraint = LOOM_AMDGPU_MATH_CONSTRAINT_##permission_,    \
  },
        LOOM_AMDGPU_MATH_RULES(LOOM_AMDGPU_MATH_RULE_ROW)
#undef LOOM_AMDGPU_MATH_RULE_ROW
};

#undef LOOM_AMDGPU_MATH_RULES

#define LOOM_AMDGPU_MATH_DIRECT_RULES(_)                                       \
  _(LOGISTICF, LOOM_SCALAR_TYPE_SET_F32, NONE, LOGISTIC_EXP2_F32, ARCP, NONE,  \
    RECIPE_LOGISTIC_EXP2_F32, NONE)                                            \
  _(SILUF, LOOM_SCALAR_TYPE_SET_F32, NONE, SILU_LOGISTIC_F32, NONE, NONE,      \
    RECIPE_SILU_LOGISTIC_F32, NONE)                                            \
  _(SOFTPLUSF, LOOM_SCALAR_TYPE_SET_F32, NONE, SOFTPLUS_EXP2_F32, ARCP_AFN,    \
    NONE, RECIPE_SOFTPLUS_EXP2_F32, NONE)                                      \
  _(EXPF, LOOM_SCALAR_TYPE_SET_F32, AFN, EXP_EXP2_F32, NONE, NONE,             \
    RECIPE_EXP_EXP2_F32, EXP_EXACT_F32)                                        \
  _(ERFF, LOOM_SCALAR_TYPE_SET_F32, NONE, ERF_RATIONAL_F32, ARCP, NONE,        \
    RECIPE_ERF_RATIONAL_F32, NONE)                                             \
  _(GELUF_ERF, LOOM_SCALAR_TYPE_SET_F32, NONE, GELU_ERF_F32, ARCP, NONE,       \
    RECIPE_GELU_ERF_F32, NONE)                                                 \
  _(GELUF_TANH, LOOM_SCALAR_TYPE_SET_F32, NONE, GELU_TANH_F32, NONE, NONE,     \
    RECIPE_GELU_TANH_F32, NONE)                                                \
  _(GELUF_LOGISTIC, LOOM_SCALAR_TYPE_SET_F32, NONE, GELU_LOGISTIC_F32, NONE,   \
    NONE, RECIPE_GELU_LOGISTIC_F32, NONE)                                      \
  _(ADDF,                                                                      \
    LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32 |                      \
        LOOM_SCALAR_TYPE_SET_F64,                                              \
    NONE, UNKNOWN, NONE, NONE, BASIC_KEEP, NONE)                               \
  _(MULF,                                                                      \
    LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32 |                      \
        LOOM_SCALAR_TYPE_SET_F64,                                              \
    NONE, UNKNOWN, NONE, NONE, BASIC_KEEP, NONE)                               \
  _(DIVF, LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64, NONE, UNKNOWN,  \
    NONE, NONE, BASIC_KEEP, NONE)                                              \
  _(LOGF, LOOM_SCALAR_TYPE_SET_F32, AFN, LOG_LOG2_F32, NONE, NONE,             \
    RECIPE_LOG_LOG2_F32, LOG_EXACT_F32)                                        \
  _(LOG2F, LOOM_SCALAR_TYPE_SET_F32, AFN, UNKNOWN, NONE, NONE, OP_NATIVE_F32,  \
    LOG2_EXACT_F32)                                                            \
  _(SINF, LOOM_SCALAR_TYPE_SET_F32, AFN, SIN_TURNS_F32, NONE, NONE,            \
    RECIPE_SIN_TURNS_F32, TRIG_EXACT_F32)                                      \
  _(COSF, LOOM_SCALAR_TYPE_SET_F32, AFN, COS_TURNS_F32, NONE, NONE,            \
    RECIPE_COS_TURNS_F32, TRIG_EXACT_F32)                                      \
  _(SINTURNSF, LOOM_SCALAR_TYPE_SET_F32, AFN, UNKNOWN, NONE, NONE,             \
    OP_NATIVE_F32, TURNS_TRIG_EXACT_F32)                                       \
  _(COSTURNSF, LOOM_SCALAR_TYPE_SET_F32, AFN, UNKNOWN, NONE, NONE,             \
    OP_NATIVE_F32, TURNS_TRIG_EXACT_F32)                                       \
  _(TANHF, LOOM_SCALAR_TYPE_SET_F32, AFN, TANH_LOGISTIC_F32, NONE, NONE,       \
    RECIPE_TANH_LOGISTIC_F32, TANH_EXACT_F32)                                  \
  _(POWF, LOOM_SCALAR_TYPE_SET_F32, AFN, POW_LOG2_EXP2_F32, NONE, NONE,        \
    RECIPE_POW_LOG2_EXP2_F32, POW_EXACT_F32)                                   \
  _(CEILF, LOOM_SCALAR_TYPE_SET_F32, NONE, UNKNOWN, NONE, NONE, OP_NATIVE_F32, \
    NONE)                                                                      \
  _(FLOORF, LOOM_SCALAR_TYPE_SET_F32, NONE, UNKNOWN, NONE, NONE,               \
    OP_NATIVE_F32, NONE)                                                       \
  _(ROUNDF, LOOM_SCALAR_TYPE_SET_F32, AFN, UNKNOWN, NONE, NONE,                \
    OP_NATIVE_APPROX_F32, NONE)                                                \
  _(ROUNDEVENF, LOOM_SCALAR_TYPE_SET_F32, NONE, UNKNOWN, NONE, NONE,           \
    OP_NATIVE_F32, NONE)                                                       \
  _(TRUNCF, LOOM_SCALAR_TYPE_SET_F32, NONE, UNKNOWN, NONE, NONE,               \
    OP_NATIVE_F32, NONE)                                                       \
  _(SUBF, LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64, NONE, UNKNOWN,  \
    NONE, NONE, BASIC_KEEP, NONE)                                              \
  _(SQRTF, LOOM_SCALAR_TYPE_SET_F32, NONE, UNKNOWN, NONE, NONE,                \
    OP_NATIVE_SQRT, NONE)                                                      \
  _(RSQRTF, LOOM_SCALAR_TYPE_SET_F32, AFN, UNKNOWN, NONE, NONE,                \
    OP_NATIVE_RSQRT_AFN, NONE)                                                 \
  _(EXP2F, LOOM_SCALAR_TYPE_SET_F32, NONE, UNKNOWN, NONE, NONE, OP_NATIVE_F32, \
    NONE)                                                                      \
  _(ASINHF, LOOM_SCALAR_TYPE_SET_F32, AFN, INVERSE_HYPERBOLIC_F32, NONE,       \
    SOURCE_F32, RECIPE_INVERSE_HYPERBOLIC_F32, INVERSE_HYPERBOLIC_AFN)         \
  _(ACOSHF, LOOM_SCALAR_TYPE_SET_F32, AFN, INVERSE_HYPERBOLIC_F32, NONE,       \
    SOURCE_F32, RECIPE_INVERSE_HYPERBOLIC_F32, INVERSE_HYPERBOLIC_AFN)         \
  _(ATANHF, LOOM_SCALAR_TYPE_SET_F32, AFN, INVERSE_HYPERBOLIC_F32, NONE,       \
    SOURCE_F32, RECIPE_INVERSE_HYPERBOLIC_F32, INVERSE_HYPERBOLIC_AFN)

static const loom_target_math_policy_direct_rule_t
    kAmdgpuMathDirectRules[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_AMDGPU_MATH_DIRECT_RULE_ROW(                               \
    op_, element_types_, required_fastmath_, recipe_, recipe_fastmath_, \
    evaluation_, decision_, permission_)                                \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_TARGET_MATH_POLICY_DIRECT_RULE(    \
      element_types_, LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,             \
      LOOM_AMDGPU_MATH_FASTMATH_##required_fastmath_,                   \
      LOOM_TARGET_MATH_RECIPE_##recipe_,                                \
      LOOM_AMDGPU_MATH_FASTMATH_##recipe_fastmath_,                     \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,                              \
      LOOM_AMDGPU_MATH_EVALUATION_##evaluation_,                        \
      LOOM_AMDGPU_MATH_CONSTRAINT_##decision_,                          \
      LOOM_AMDGPU_MATH_CONSTRAINT_##permission_),
        LOOM_AMDGPU_MATH_DIRECT_RULES(LOOM_AMDGPU_MATH_DIRECT_RULE_ROW)
#undef LOOM_AMDGPU_MATH_DIRECT_RULE_ROW
};

#undef LOOM_AMDGPU_MATH_DIRECT_RULES
#undef LOOM_AMDGPU_MATH_FASTMATH_ARCP_AFN
#undef LOOM_AMDGPU_MATH_FASTMATH_AFN
#undef LOOM_AMDGPU_MATH_FASTMATH_ARCP
#undef LOOM_AMDGPU_MATH_FASTMATH_NONE

#define LOOM_AMDGPU_MATH_SPANS(_)                                            \
  _(LOGISTICF, 0, 0, ELEMENT_F32)                                            \
  _(SILUF, 0, 0, ELEMENT_F32)                                                \
  _(SOFTPLUSF, 0, 0, ELEMENT_F32)                                            \
  _(EXPF, 0, 0, ELEMENT_F32)                                                 \
  _(ERFF, 0, 0, ELEMENT_F32)                                                 \
  _(GELUF_ERF, 0, 0, ELEMENT_F32)                                            \
  _(GELUF_TANH, 0, 0, ELEMENT_F32)                                           \
  _(GELUF_LOGISTIC, 0, 0, ELEMENT_F32)                                       \
  _(ADDF, LOOM_AMDGPU_MATH_RULE_BINARY_NATIVE_BF16, 2, BASIC_KEEP)           \
  _(MULF, LOOM_AMDGPU_MATH_RULE_BINARY_NATIVE_BF16, 2, BASIC_KEEP)           \
  _(DIVF, LOOM_AMDGPU_MATH_RULE_DIV_WIDEN, 1, BASIC_KEEP)                    \
  _(LOGF, 0, 0, ELEMENT_F32)                                                 \
  _(LOG2F, 0, 0, ELEMENT_F32)                                                \
  _(SINF, 0, 0, ELEMENT_F32)                                                 \
  _(COSF, 0, 0, ELEMENT_F32)                                                 \
  _(SINTURNSF, 0, 0, ELEMENT_F32)                                            \
  _(COSTURNSF, 0, 0, ELEMENT_F32)                                            \
  _(TANHF, 0, 0, ELEMENT_F32)                                                \
  _(POWF, 0, 0, ELEMENT_F32)                                                 \
  _(CEILF, LOOM_AMDGPU_MATH_RULE_ROUNDING_F64, 1, ELEMENT_F32)               \
  _(FLOORF, LOOM_AMDGPU_MATH_RULE_ROUNDING_F64, 1, ELEMENT_F32)              \
  _(ROUNDF, LOOM_AMDGPU_MATH_RULE_ROUND_REWRITE, 1, ELEMENT_F32)             \
  _(ROUNDEVENF, LOOM_AMDGPU_MATH_RULE_ROUNDING_F64, 1, ELEMENT_F32)          \
  _(TRUNCF, LOOM_AMDGPU_MATH_RULE_ROUNDING_F64, 1, ELEMENT_F32)              \
  _(SUBF, LOOM_AMDGPU_MATH_RULE_SUB_NATIVE_BF16, 5, BASIC_KEEP)              \
  _(SQRTF, LOOM_AMDGPU_MATH_RULE_SQRT_SCALAR_F64, 1, ROOT_F32_OR_SCALAR_F64) \
  _(RSQRTF, LOOM_AMDGPU_MATH_RULE_RSQRT_NATIVE_SCALAR_F64, 3,                \
    ROOT_F32_OR_SCALAR_F64)                                                  \
  _(EXP2F, 0, 0, ELEMENT_F32)                                                \
  _(ASINHF, LOOM_AMDGPU_MATH_RULE_INVERSE_HYPERBOLIC_WIDENED_F32, 1,         \
    INVERSE_HYPERBOLIC_F32_EVALUATION)                                       \
  _(ACOSHF, LOOM_AMDGPU_MATH_RULE_INVERSE_HYPERBOLIC_WIDENED_F32, 1,         \
    INVERSE_HYPERBOLIC_F32_EVALUATION)                                       \
  _(ATANHF, LOOM_AMDGPU_MATH_RULE_INVERSE_HYPERBOLIC_WIDENED_F32, 1,         \
    INVERSE_HYPERBOLIC_F32_EVALUATION)

static const loom_target_math_policy_rule_span_t
    kAmdgpuMathOperationSpans[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_AMDGPU_MATH_SPAN_ROW(op_, start_, count_, fallback_)     \
  [LOOM_TARGET_MATH_OP_##op_] = {                                     \
      .rule_start = (start_),                                         \
      .rule_count = (count_),                                         \
      .fallback_constraint = LOOM_AMDGPU_MATH_CONSTRAINT_##fallback_, \
  },
        LOOM_AMDGPU_MATH_SPANS(LOOM_AMDGPU_MATH_SPAN_ROW)
#undef LOOM_AMDGPU_MATH_SPAN_ROW
};

#undef LOOM_AMDGPU_MATH_SPANS

static const loom_target_math_policy_match_t kAmdgpuMathFmaForms[] = {
    {
        .element_types = LOOM_SCALAR_TYPE_SET_F32,
        .lane_domains = LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,
        .shape = LOOM_TARGET_MATH_POLICY_SHAPE_ANY,
    },
    {
        .element_types = LOOM_SCALAR_TYPE_SET_F16,
        .minimum_lane_count = 1,
        .maximum_lane_count = LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES,
        .lane_domains = LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
        .shape = LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE_EVEN,
    },
    {
        .element_types = LOOM_SCALAR_TYPE_SET_BF16,
        .minimum_lane_count = 1,
        .maximum_lane_count = LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES,
        .lane_domains = LOOM_TARGET_MATH_LANE_DOMAIN_SET_VECTOR,
        .shape = LOOM_TARGET_MATH_POLICY_SHAPE_RANK_ONE_EVEN,
        .feature_class = LOOM_AMDGPU_MATH_FEATURE_CLASS_NATIVE_PACKED_BF16,
    },
};

static const loom_target_math_policy_table_t kAmdgpuMathPolicyTable = {
    .operation_spans = kAmdgpuMathOperationSpans,
    .rules = kAmdgpuMathRules,
    .fma_forms = kAmdgpuMathFmaForms,
    .fma_form_count = IREE_ARRAYSIZE(kAmdgpuMathFmaForms),
    .default_constraint = LOOM_AMDGPU_MATH_CONSTRAINT_OP_SUPPORTED,
    .constraints = kAmdgpuMathConstraints,
    .constraint_strings =
        {
            .data = {(const char*)&kAmdgpuMathConstraintStrings},
            .data_length = {sizeof(kAmdgpuMathConstraintStrings)},
        },
    .evaluations = kAmdgpuMathEvaluations,
    .policy_feature_masks = kAmdgpuMathPolicyFeatureMasks,
};

static const loom_target_math_policy_t kAmdgpuMathPolicy = {
    .name = IREE_SVL("amdgpu-math"),
    .direct_rules = kAmdgpuMathDirectRules,
    .table = &kAmdgpuMathPolicyTable,
};

static const loom_target_math_policy_t kAmdgpuNativePackedBf16MathPolicy = {
    .name = IREE_SVL("amdgpu-native-packed-bf16-math"),
    .direct_rules = kAmdgpuMathDirectRules,
    .table = &kAmdgpuMathPolicyTable,
    .feature_bits = LOOM_AMDGPU_MATH_POLICY_FEATURE_NATIVE_PACKED_BF16,
};

static const loom_target_math_policy_registry_entry_t
    kAmdgpuMathPolicyEntries[] = {
#define LOOM_AMDGPU_TARGET_DESCRIPTOR_SET(                                      \
    symbol_suffix, bundle_name, snapshot_name, key, stable_id,                  \
    descriptor_set_flags, wavefront_size, workgroup_storage_byte_limit)         \
  {                                                                             \
      .contract_set_stable_id = stable_id,                                      \
      .policy =                                                                 \
          ((descriptor_set_flags &                                              \
            LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_PACKED_BF16_ARITHMETIC) \
               ? &kAmdgpuNativePackedBf16MathPolicy                             \
               : &kAmdgpuMathPolicy),                                           \
  },
#include "loom/target/arch/amdgpu/target_records_tables.inl"
#undef LOOM_AMDGPU_TARGET_DESCRIPTOR_SET
};

void loom_amdgpu_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      kAmdgpuMathPolicyEntries, IREE_ARRAYSIZE(kAmdgpuMathPolicyEntries),
      out_registry);
}
