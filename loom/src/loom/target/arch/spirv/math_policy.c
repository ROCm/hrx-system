// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/math_policy.h"

#include <stddef.h>

#define LOOM_SPIRV_MATH_CONSTRAINTS(_)                                       \
  _(ACTIVATION_AFN, activation_afn, "math.activation.afn")                   \
  _(ACTIVATION_EXACT_FMA, activation_exact_fma, "math.activation.exact_fma") \
  _(ACTIVATION_F32_EVALUATION, activation_f32_evaluation,                    \
    "math.activation.f32_evaluation")                                        \
  _(ELEMENT_F16_F32_F64, element_f16_f32_f64, "math.element.f16_f32_f64")    \
  _(INVERSE_HYPERBOLIC_AFN, inverse_hyperbolic_afn,                          \
    "math.inverse_hyperbolic.afn")                                           \
  _(INVERSE_HYPERBOLIC_F32_EVALUATION, inverse_hyperbolic_f32_evaluation,    \
    "math.inverse_hyperbolic.f32_evaluation")                                \
  _(OP_NATIVE_ARITHMETIC, op_native_arithmetic, "math.op.native_arithmetic") \
  _(RECIPE_INVERSE_HYPERBOLIC_F32, recipe_inverse_hyperbolic_f32,            \
    "math.recipe.inverse_hyperbolic_f32")                                    \
  _(RECIPE_GELU_LOGISTIC_F32, recipe_gelu_logistic_f32,                      \
    "math.recipe.gelu_logistic_f32")                                         \
  _(RECIPE_GELU_TANH_F32, recipe_gelu_tanh_f32, "math.recipe.gelu_tanh_f32") \
  _(RECIPE_LOGISTIC_EXP2_F32, recipe_logistic_exp2_f32,                      \
    "math.recipe.logistic_exp2_f32")                                         \
  _(RECIPE_SILU_LOGISTIC_F32, recipe_silu_logistic_f32,                      \
    "math.recipe.silu_logistic_f32")                                         \
  _(RECIPE_SOFTPLUS_EXP2_F32, recipe_softplus_exp2_f32,                      \
    "math.recipe.softplus_exp2_f32")                                         \
  _(RECIPE_TANH_LOGISTIC_F32, recipe_tanh_logistic_f32,                      \
    "math.recipe.tanh_logistic_f32")                                         \
  _(RECIPE_TRIGONOMETRIC_F32, recipe_trigonometric_f32,                      \
    "math.recipe.trigonometric_f32")                                         \
  _(RECIPE_WIDEN_F32_ROUND, recipe_widen_f32_round,                          \
    "math.recipe.widen_f32_round")                                           \
  _(ROOT_AFN, root_afn, "math.root.afn")                                     \
  _(ROOT_AFN_FLOAT, root_afn_float, "math.root.afn_float")                   \
  _(ROOT_F16_F32_F64, root_f16_f32_f64, "math.root.f16_f32_f64")             \
  _(ROUNDING, rounding, "math.rounding")                                     \
  _(TRANSCENDENTAL_AFN, transcendental_afn, "math.transcendental.afn")       \
  _(TRANSCENDENTAL_AFN_F32, transcendental_afn_f32,                          \
    "math.transcendental.afn_f32")                                           \
  _(TRANSCENDENTAL_F32, transcendental_f32, "math.transcendental.f32")       \
  _(TRIGONOMETRIC_F32_EVALUATION, trigonometric_f32_evaluation,              \
    "math.trigonometric.f32_evaluation")                                     \
  _(TRIGONOMETRIC_F16_F32, trigonometric_f16_f32, "math.trigonometric.f16_f32")

enum loom_spirv_math_constraint_e {
  LOOM_SPIRV_MATH_CONSTRAINT_NONE = 0,
#define LOOM_SPIRV_MATH_CONSTRAINT_ENUM(name, field, value) \
  LOOM_SPIRV_MATH_CONSTRAINT_##name,
  LOOM_SPIRV_MATH_CONSTRAINTS(LOOM_SPIRV_MATH_CONSTRAINT_ENUM)
#undef LOOM_SPIRV_MATH_CONSTRAINT_ENUM
      LOOM_SPIRV_MATH_CONSTRAINT_COUNT_,
};

typedef struct loom_spirv_math_constraint_strings_t {
#define LOOM_SPIRV_MATH_CONSTRAINT_FIELD(name, field, value) \
  char field[sizeof(value)];
  LOOM_SPIRV_MATH_CONSTRAINTS(LOOM_SPIRV_MATH_CONSTRAINT_FIELD)
#undef LOOM_SPIRV_MATH_CONSTRAINT_FIELD
} loom_spirv_math_constraint_strings_t;

static const loom_spirv_math_constraint_strings_t kSpirvMathConstraintStrings =
    {
#define LOOM_SPIRV_MATH_CONSTRAINT_VALUE(name, field, value) .field = value,
        LOOM_SPIRV_MATH_CONSTRAINTS(LOOM_SPIRV_MATH_CONSTRAINT_VALUE)
#undef LOOM_SPIRV_MATH_CONSTRAINT_VALUE
};

static const loom_string_ref_t
    kSpirvMathConstraints[LOOM_SPIRV_MATH_CONSTRAINT_COUNT_ - 1] = {
#define LOOM_SPIRV_MATH_CONSTRAINT_REF(name, field, value)      \
  [LOOM_SPIRV_MATH_CONSTRAINT_##name - 1] = LOOM_STRING_REF(    \
      0, offsetof(loom_spirv_math_constraint_strings_t, field), \
      sizeof(value) - 1),
        LOOM_SPIRV_MATH_CONSTRAINTS(LOOM_SPIRV_MATH_CONSTRAINT_REF)
#undef LOOM_SPIRV_MATH_CONSTRAINT_REF
};

#undef LOOM_SPIRV_MATH_CONSTRAINTS

enum loom_spirv_math_evaluation_e {
  LOOM_SPIRV_MATH_EVALUATION_NONE = 0,
  LOOM_SPIRV_MATH_EVALUATION_SOURCE_F32 = 1,
  LOOM_SPIRV_MATH_EVALUATION_WIDENED_F32 = 2,
};

static const loom_target_math_evaluation_t kSpirvMathEvaluations[] = {
    [LOOM_SPIRV_MATH_EVALUATION_SOURCE_F32 - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_SOURCE,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
        },
    [LOOM_SPIRV_MATH_EVALUATION_WIDENED_F32 - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_WIDENED,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
        },
};

#define LOOM_SPIRV_MATH_ACTIVATION_OPS(_)          \
  _(LOGISTICF, LOGISTIC_EXP2_F32, NONE, NONE)      \
  _(SILUF, SILU_LOGISTIC_F32, NONE, NONE)          \
  _(SOFTPLUSF, SOFTPLUS_EXP2_F32, NONE, NONE)      \
  _(GELUF_TANH, GELU_TANH_F32, NONE, NONE)         \
  _(GELUF_LOGISTIC, GELU_LOGISTIC_F32, NONE, NONE) \
  _(TANHF, TANH_LOGISTIC_F32, AFN, ACTIVATION_AFN)

#define LOOM_SPIRV_MATH_ACTIVATION_RECIPE_FASTMATH \
  (LOOM_TARGET_MATH_FASTMATH_FLAG_AFN | LOOM_TARGET_MATH_FASTMATH_FLAG_ARCP)

#define LOOM_SPIRV_MATH_DIRECT_RULE(element_types_, required_fastmath_flags_, \
                                    recipe_, evaluation_, constraint_,        \
                                    permission_constraint_)                   \
  LOOM_TARGET_MATH_POLICY_DIRECT_RULE(                                        \
      element_types_, LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,                   \
      required_fastmath_flags_, recipe_, LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, evaluation_, constraint_,          \
      permission_constraint_)

static const loom_target_math_policy_direct_rule_t
    kSpirvMathDirectRules[LOOM_TARGET_MATH_OP_COUNT] = {
        [LOOM_TARGET_MATH_OP_ADDF] = LOOM_SPIRV_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32 |
                LOOM_SCALAR_TYPE_SET_F64,
            LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
            LOOM_TARGET_MATH_RECIPE_UNKNOWN, LOOM_SPIRV_MATH_EVALUATION_NONE,
            LOOM_SPIRV_MATH_CONSTRAINT_OP_NATIVE_ARITHMETIC,
            LOOM_SPIRV_MATH_CONSTRAINT_NONE),
        [LOOM_TARGET_MATH_OP_SUBF] = LOOM_SPIRV_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32 |
                LOOM_SCALAR_TYPE_SET_F64,
            LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
            LOOM_TARGET_MATH_RECIPE_UNKNOWN, LOOM_SPIRV_MATH_EVALUATION_NONE,
            LOOM_SPIRV_MATH_CONSTRAINT_OP_NATIVE_ARITHMETIC,
            LOOM_SPIRV_MATH_CONSTRAINT_NONE),
        [LOOM_TARGET_MATH_OP_MULF] = LOOM_SPIRV_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32 |
                LOOM_SCALAR_TYPE_SET_F64,
            LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
            LOOM_TARGET_MATH_RECIPE_UNKNOWN, LOOM_SPIRV_MATH_EVALUATION_NONE,
            LOOM_SPIRV_MATH_CONSTRAINT_OP_NATIVE_ARITHMETIC,
            LOOM_SPIRV_MATH_CONSTRAINT_NONE),
        [LOOM_TARGET_MATH_OP_DIVF] = LOOM_SPIRV_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32 |
                LOOM_SCALAR_TYPE_SET_F64,
            LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,
            LOOM_TARGET_MATH_RECIPE_UNKNOWN, LOOM_SPIRV_MATH_EVALUATION_NONE,
            LOOM_SPIRV_MATH_CONSTRAINT_OP_NATIVE_ARITHMETIC,
            LOOM_SPIRV_MATH_CONSTRAINT_NONE),
#define LOOM_SPIRV_MATH_TRIGONOMETRIC_RULE(op_)                            \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_SPIRV_MATH_DIRECT_RULE(               \
      LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32,                 \
      LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, LOOM_TARGET_MATH_RECIPE_UNKNOWN, \
      LOOM_SPIRV_MATH_EVALUATION_NONE,                                     \
      LOOM_SPIRV_MATH_CONSTRAINT_TRIGONOMETRIC_F16_F32,                    \
      LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN)
        LOOM_SPIRV_MATH_TRIGONOMETRIC_RULE(SINF),
        LOOM_SPIRV_MATH_TRIGONOMETRIC_RULE(COSF),
#undef LOOM_SPIRV_MATH_TRIGONOMETRIC_RULE
#define LOOM_SPIRV_MATH_TURNS_RULE(op_)                             \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_SPIRV_MATH_DIRECT_RULE(        \
      LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, \
      LOOM_TARGET_MATH_RECIPE_TRIGONOMETRIC_F32,                    \
      LOOM_SPIRV_MATH_EVALUATION_SOURCE_F32,                        \
      LOOM_SPIRV_MATH_CONSTRAINT_RECIPE_TRIGONOMETRIC_F32,          \
      LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN)
        LOOM_SPIRV_MATH_TURNS_RULE(SINTURNSF),
        LOOM_SPIRV_MATH_TURNS_RULE(COSTURNSF),
#undef LOOM_SPIRV_MATH_TURNS_RULE
        [LOOM_TARGET_MATH_OP_EXP2F] = LOOM_SPIRV_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
            LOOM_TARGET_MATH_RECIPE_UNKNOWN, LOOM_SPIRV_MATH_EVALUATION_NONE,
            LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN_F32,
            LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN),
        [LOOM_TARGET_MATH_OP_EXPF] = LOOM_SPIRV_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
            LOOM_TARGET_MATH_RECIPE_UNKNOWN, LOOM_SPIRV_MATH_EVALUATION_NONE,
            LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN_F32,
            LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN),
        [LOOM_TARGET_MATH_OP_LOGF] = LOOM_SPIRV_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
            LOOM_TARGET_MATH_RECIPE_UNKNOWN, LOOM_SPIRV_MATH_EVALUATION_NONE,
            LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN_F32,
            LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN),
        [LOOM_TARGET_MATH_OP_LOG2F] = LOOM_SPIRV_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,
            LOOM_TARGET_MATH_RECIPE_UNKNOWN, LOOM_SPIRV_MATH_EVALUATION_NONE,
            LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN_F32,
            LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN),
        [LOOM_TARGET_MATH_OP_SQRTF] = LOOM_SPIRV_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32 |
                LOOM_SCALAR_TYPE_SET_F64,
            LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_SPIRV_MATH_EVALUATION_NONE,
            LOOM_SPIRV_MATH_CONSTRAINT_ROOT_AFN_FLOAT,
            LOOM_SPIRV_MATH_CONSTRAINT_ROOT_AFN),
        [LOOM_TARGET_MATH_OP_RSQRTF] = LOOM_SPIRV_MATH_DIRECT_RULE(
            LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32 |
                LOOM_SCALAR_TYPE_SET_F64,
            LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_SPIRV_MATH_EVALUATION_NONE,
            LOOM_SPIRV_MATH_CONSTRAINT_ROOT_AFN_FLOAT,
            LOOM_SPIRV_MATH_CONSTRAINT_ROOT_AFN),
#define LOOM_SPIRV_MATH_ROUNDING_RULE(op_)                              \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_SPIRV_MATH_DIRECT_RULE(            \
      LOOM_SCALAR_TYPE_SET_FLOAT, LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,  \
      LOOM_TARGET_MATH_RECIPE_UNKNOWN, LOOM_SPIRV_MATH_EVALUATION_NONE, \
      LOOM_SPIRV_MATH_CONSTRAINT_ROUNDING, LOOM_SPIRV_MATH_CONSTRAINT_NONE)
        LOOM_SPIRV_MATH_ROUNDING_RULE(CEILF),
        LOOM_SPIRV_MATH_ROUNDING_RULE(FLOORF),
        LOOM_SPIRV_MATH_ROUNDING_RULE(ROUNDF),
        LOOM_SPIRV_MATH_ROUNDING_RULE(ROUNDEVENF),
        LOOM_SPIRV_MATH_ROUNDING_RULE(TRUNCF),
#undef LOOM_SPIRV_MATH_ROUNDING_RULE
#define LOOM_SPIRV_MATH_INVERSE_RULE(op_)                           \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_SPIRV_MATH_DIRECT_RULE(        \
      LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, \
      LOOM_TARGET_MATH_RECIPE_INVERSE_HYPERBOLIC_F32,               \
      LOOM_SPIRV_MATH_EVALUATION_SOURCE_F32,                        \
      LOOM_SPIRV_MATH_CONSTRAINT_RECIPE_INVERSE_HYPERBOLIC_F32,     \
      LOOM_SPIRV_MATH_CONSTRAINT_INVERSE_HYPERBOLIC_AFN)
        LOOM_SPIRV_MATH_INVERSE_RULE(ASINHF),
        LOOM_SPIRV_MATH_INVERSE_RULE(ACOSHF),
        LOOM_SPIRV_MATH_INVERSE_RULE(ATANHF),
#undef LOOM_SPIRV_MATH_INVERSE_RULE
#define LOOM_SPIRV_MATH_ACTIVATION_DIRECT_RULE(                       \
    op_, recipe_, required_fastmath_, permission_constraint_)         \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_TARGET_MATH_POLICY_DIRECT_RULE(  \
      LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL, \
      LOOM_TARGET_MATH_FASTMATH_FLAG_##required_fastmath_,            \
      LOOM_TARGET_MATH_RECIPE_##recipe_,                              \
      LOOM_SPIRV_MATH_ACTIVATION_RECIPE_FASTMATH,                     \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,                            \
      LOOM_SPIRV_MATH_EVALUATION_SOURCE_F32,                          \
      LOOM_SPIRV_MATH_CONSTRAINT_RECIPE_##recipe_,                    \
      LOOM_SPIRV_MATH_CONSTRAINT_##permission_constraint_),
        LOOM_SPIRV_MATH_ACTIVATION_OPS(LOOM_SPIRV_MATH_ACTIVATION_DIRECT_RULE)
#undef LOOM_SPIRV_MATH_ACTIVATION_DIRECT_RULE
};

#undef LOOM_SPIRV_MATH_DIRECT_RULE

#define LOOM_SPIRV_MATH_MATCH(element_types_, required_fastmath_flags_) \
  {                                                                     \
      .element_types = (element_types_),                                \
      .lane_domains = LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,             \
      .shape = LOOM_TARGET_MATH_POLICY_SHAPE_ANY,                       \
      .required_fastmath_flags = (required_fastmath_flags_),            \
  }

enum loom_spirv_math_rule_e {
  LOOM_SPIRV_MATH_RULE_ARITHMETIC_BF16 = 0,
  LOOM_SPIRV_MATH_RULE_DIV_NARROW,
  LOOM_SPIRV_MATH_RULE_INVERSE_NARROW,
  LOOM_SPIRV_MATH_RULE_TRIGONOMETRIC_NARROW,
#define LOOM_SPIRV_MATH_ACTIVATION_RULE_ENUM(op_, recipe_, required_fastmath_, \
                                             permission_constraint_)           \
  LOOM_SPIRV_MATH_RULE_ACTIVATION_##op_,
  LOOM_SPIRV_MATH_ACTIVATION_OPS(LOOM_SPIRV_MATH_ACTIVATION_RULE_ENUM)
#undef LOOM_SPIRV_MATH_ACTIVATION_RULE_ENUM
      LOOM_SPIRV_MATH_RULE_COUNT_,
};

static const loom_target_math_policy_rule_t
    kSpirvMathRules[LOOM_SPIRV_MATH_RULE_COUNT_] = {
        [LOOM_SPIRV_MATH_RULE_ARITHMETIC_BF16] =
            {
                .match =
                    LOOM_SPIRV_MATH_MATCH(LOOM_SCALAR_TYPE_SET_BF16,
                                          LOOM_TARGET_MATH_FASTMATH_FLAG_NONE),
                .recipe = LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND,
                .decision_constraint =
                    LOOM_SPIRV_MATH_CONSTRAINT_RECIPE_WIDEN_F32_ROUND,
            },
        [LOOM_SPIRV_MATH_RULE_DIV_NARROW] =
            {
                .match =
                    LOOM_SPIRV_MATH_MATCH(LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
                                          LOOM_TARGET_MATH_FASTMATH_FLAG_NONE),
                .recipe = LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND,
                .decision_constraint =
                    LOOM_SPIRV_MATH_CONSTRAINT_RECIPE_WIDEN_F32_ROUND,
            },
        [LOOM_SPIRV_MATH_RULE_INVERSE_NARROW] =
            {
                .match =
                    LOOM_SPIRV_MATH_MATCH(LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
                                          LOOM_TARGET_MATH_FASTMATH_FLAG_AFN),
                .recipe = LOOM_TARGET_MATH_RECIPE_INVERSE_HYPERBOLIC_F32,
                .evaluation = LOOM_SPIRV_MATH_EVALUATION_WIDENED_F32,
                .decision_constraint =
                    LOOM_SPIRV_MATH_CONSTRAINT_RECIPE_INVERSE_HYPERBOLIC_F32,
                .permission_constraint =
                    LOOM_SPIRV_MATH_CONSTRAINT_INVERSE_HYPERBOLIC_AFN,
            },
        [LOOM_SPIRV_MATH_RULE_TRIGONOMETRIC_NARROW] =
            {
                .match =
                    LOOM_SPIRV_MATH_MATCH(LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
                                          LOOM_TARGET_MATH_FASTMATH_FLAG_AFN),
                .recipe = LOOM_TARGET_MATH_RECIPE_TRIGONOMETRIC_F32,
                .evaluation = LOOM_SPIRV_MATH_EVALUATION_WIDENED_F32,
                .decision_constraint =
                    LOOM_SPIRV_MATH_CONSTRAINT_RECIPE_TRIGONOMETRIC_F32,
                .permission_constraint =
                    LOOM_SPIRV_MATH_CONSTRAINT_TRANSCENDENTAL_AFN,
            },
#define LOOM_SPIRV_MATH_ACTIVATION_RULE(op_, recipe_, required_fastmath_,  \
                                        permission_constraint_)            \
  [LOOM_SPIRV_MATH_RULE_ACTIVATION_##op_] = {                              \
      .match = LOOM_SPIRV_MATH_MATCH(                                      \
          LOOM_SCALAR_TYPE_SET_FLOAT_LE16,                                 \
          LOOM_TARGET_MATH_FASTMATH_FLAG_##required_fastmath_),            \
      .recipe = LOOM_TARGET_MATH_RECIPE_##recipe_,                         \
      .recipe_fastmath_flags = LOOM_SPIRV_MATH_ACTIVATION_RECIPE_FASTMATH, \
      .evaluation = LOOM_SPIRV_MATH_EVALUATION_WIDENED_F32,                \
      .decision_constraint = LOOM_SPIRV_MATH_CONSTRAINT_RECIPE_##recipe_,  \
      .permission_constraint =                                             \
          LOOM_SPIRV_MATH_CONSTRAINT_##permission_constraint_,             \
  },
        LOOM_SPIRV_MATH_ACTIVATION_OPS(LOOM_SPIRV_MATH_ACTIVATION_RULE)
#undef LOOM_SPIRV_MATH_ACTIVATION_RULE
};

#undef LOOM_SPIRV_MATH_MATCH

#define LOOM_SPIRV_MATH_SPAN(start_, count_, fallback_) \
  {                                                     \
      .rule_start = (start_),                           \
      .rule_count = (count_),                           \
      .fallback_constraint = (fallback_),               \
  }

static const loom_target_math_policy_rule_span_t
    kSpirvMathOperationSpans[LOOM_TARGET_MATH_OP_COUNT] = {
        [LOOM_TARGET_MATH_OP_ERFF] = LOOM_SPIRV_MATH_SPAN(
            0, 0, LOOM_SPIRV_MATH_CONSTRAINT_ACTIVATION_EXACT_FMA),
        [LOOM_TARGET_MATH_OP_GELUF_ERF] = LOOM_SPIRV_MATH_SPAN(
            0, 0, LOOM_SPIRV_MATH_CONSTRAINT_ACTIVATION_EXACT_FMA),
        [LOOM_TARGET_MATH_OP_ADDF] = LOOM_SPIRV_MATH_SPAN(
            LOOM_SPIRV_MATH_RULE_ARITHMETIC_BF16, 1,
            LOOM_SPIRV_MATH_CONSTRAINT_ELEMENT_F16_F32_F64),
        [LOOM_TARGET_MATH_OP_SUBF] = LOOM_SPIRV_MATH_SPAN(
            LOOM_SPIRV_MATH_RULE_ARITHMETIC_BF16, 1,
            LOOM_SPIRV_MATH_CONSTRAINT_ELEMENT_F16_F32_F64),
        [LOOM_TARGET_MATH_OP_MULF] = LOOM_SPIRV_MATH_SPAN(
            LOOM_SPIRV_MATH_RULE_ARITHMETIC_BF16, 1,
            LOOM_SPIRV_MATH_CONSTRAINT_ELEMENT_F16_F32_F64),
        [LOOM_TARGET_MATH_OP_DIVF] = LOOM_SPIRV_MATH_SPAN(
            LOOM_SPIRV_MATH_RULE_DIV_NARROW, 1,
            LOOM_SPIRV_MATH_CONSTRAINT_ELEMENT_F16_F32_F64),
#define LOOM_SPIRV_MATH_FALLBACK(op_, constraint_) \
  [LOOM_TARGET_MATH_OP_##op_] =                    \
      LOOM_SPIRV_MATH_SPAN(0, 0, LOOM_SPIRV_MATH_CONSTRAINT_##constraint_)
        LOOM_SPIRV_MATH_FALLBACK(EXP2F, TRANSCENDENTAL_F32),
        LOOM_SPIRV_MATH_FALLBACK(EXPF, TRANSCENDENTAL_F32),
        LOOM_SPIRV_MATH_FALLBACK(LOGF, TRANSCENDENTAL_F32),
        LOOM_SPIRV_MATH_FALLBACK(LOG2F, TRANSCENDENTAL_F32),
        LOOM_SPIRV_MATH_FALLBACK(SQRTF, ROOT_F16_F32_F64),
        LOOM_SPIRV_MATH_FALLBACK(RSQRTF, ROOT_F16_F32_F64),
#undef LOOM_SPIRV_MATH_FALLBACK
#define LOOM_SPIRV_MATH_TRIGONOMETRIC_SPAN(op_, constraint_)             \
  [LOOM_TARGET_MATH_OP_##op_] =                                          \
      LOOM_SPIRV_MATH_SPAN(LOOM_SPIRV_MATH_RULE_TRIGONOMETRIC_NARROW, 1, \
                           LOOM_SPIRV_MATH_CONSTRAINT_##constraint_)
        LOOM_SPIRV_MATH_TRIGONOMETRIC_SPAN(SINF, TRIGONOMETRIC_F16_F32),
        LOOM_SPIRV_MATH_TRIGONOMETRIC_SPAN(COSF, TRIGONOMETRIC_F16_F32),
        LOOM_SPIRV_MATH_TRIGONOMETRIC_SPAN(SINTURNSF,
                                           TRIGONOMETRIC_F32_EVALUATION),
        LOOM_SPIRV_MATH_TRIGONOMETRIC_SPAN(COSTURNSF,
                                           TRIGONOMETRIC_F32_EVALUATION),
#undef LOOM_SPIRV_MATH_TRIGONOMETRIC_SPAN
#define LOOM_SPIRV_MATH_INVERSE_SPAN(op_)             \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_SPIRV_MATH_SPAN( \
      LOOM_SPIRV_MATH_RULE_INVERSE_NARROW, 1,         \
      LOOM_SPIRV_MATH_CONSTRAINT_INVERSE_HYPERBOLIC_F32_EVALUATION)
        LOOM_SPIRV_MATH_INVERSE_SPAN(ASINHF),
        LOOM_SPIRV_MATH_INVERSE_SPAN(ACOSHF),
        LOOM_SPIRV_MATH_INVERSE_SPAN(ATANHF),
#undef LOOM_SPIRV_MATH_INVERSE_SPAN
#define LOOM_SPIRV_MATH_ACTIVATION_SPAN(op_, recipe_, required_fastmath_, \
                                        permission_constraint_)           \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_SPIRV_MATH_SPAN(                     \
      LOOM_SPIRV_MATH_RULE_ACTIVATION_##op_, 1,                           \
      LOOM_SPIRV_MATH_CONSTRAINT_ACTIVATION_F32_EVALUATION),
        LOOM_SPIRV_MATH_ACTIVATION_OPS(LOOM_SPIRV_MATH_ACTIVATION_SPAN)
#undef LOOM_SPIRV_MATH_ACTIVATION_SPAN
};

#undef LOOM_SPIRV_MATH_SPAN
#undef LOOM_SPIRV_MATH_ACTIVATION_RECIPE_FASTMATH
#undef LOOM_SPIRV_MATH_ACTIVATION_OPS

static const loom_target_math_policy_table_t kSpirvMathPolicyTable = {
    .operation_spans = kSpirvMathOperationSpans,
    .rules = kSpirvMathRules,
    .default_constraint = LOOM_SPIRV_MATH_CONSTRAINT_OP_NATIVE_ARITHMETIC,
    .constraints = kSpirvMathConstraints,
    .constraint_strings =
        {
            .data = {(const char*)&kSpirvMathConstraintStrings},
            .data_length = {sizeof(kSpirvMathConstraintStrings)},
        },
    .evaluations = kSpirvMathEvaluations,
};

static const loom_target_math_policy_t kSpirvMathPolicy = {
    .name = IREE_SVL("spirv-math"),
    .direct_rules = kSpirvMathDirectRules,
    .table = &kSpirvMathPolicyTable,
};

static const loom_target_math_policy_registry_entry_t
    kSpirvMathPolicyEntries[] = {
        {/*.contract_set_stable_id=*/UINT64_C(0x5b86e123b2b92ba9),
         /*.policy=*/&kSpirvMathPolicy},
};

void loom_spirv_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, kSpirvMathPolicyEntries,
      IREE_ARRAYSIZE(kSpirvMathPolicyEntries));
}
