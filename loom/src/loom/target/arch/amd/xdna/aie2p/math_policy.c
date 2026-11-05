// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/math_policy.h"

#include <stddef.h>

#define LOOM_AIE2P_MATH_CONSTRAINTS(_)                                        \
  _(ELEMENT_APPROXIMATE_ACTIVATION, element_approximate_activation,           \
    "math.element.aie2p_approximate_activation")                              \
  _(ELEMENT_F32, element_f32, "math.element.f32")                             \
  _(EXP_EXACT_F32, exp_exact_f32, "math.exp.exact_f32")                       \
  _(EXP2_EXACT_F32, exp2_exact_f32, "math.exp2.exact_f32")                    \
  _(GELU_ERF_EXACT, gelu_erf_exact, "math.gelu.erf.exact")                    \
  _(GELU_LOGISTIC_EXACT, gelu_logistic_exact, "math.gelu.logistic.exact")     \
  _(GELU_TANH_EXACT, gelu_tanh_exact, "math.gelu.tanh.exact")                 \
  _(LOGISTIC_EXACT, logistic_exact, "math.logistic.exact")                    \
  _(OP_EXACT_BINARY16, op_exact_binary16, "math.op.exact_binary16")           \
  _(OP_EXACT_BINARY32, op_exact_binary32, "math.op.exact_binary32")           \
  _(OP_EXACT_BINARY32_ROUNDEVEN, op_exact_binary32_roundeven,                 \
    "math.op.exact_binary32_roundeven")                                       \
  _(OP_EXACT_BINARY32_TRUNC, op_exact_binary32_trunc,                         \
    "math.op.exact_binary32_trunc")                                           \
  _(OP_NATIVE_APPROXIMATE_RSQRT_F32, op_native_approximate_rsqrt_f32,         \
    "math.op.native_approximate_rsqrt_f32")                                   \
  _(OP_NATIVE_APPROXIMATE_SQRT_F32, op_native_approximate_sqrt_f32,           \
    "math.op.native_approximate_sqrt_f32")                                    \
  _(OP_NATIVE_BF16_EXP, op_native_bf16_exp, "math.op.native_bf16_exp")        \
  _(OP_NATIVE_BF16_EXP2, op_native_bf16_exp2, "math.op.native_bf16_exp2")     \
  _(OP_NATIVE_BF16_MAC_TRIG, op_native_bf16_mac_trig,                         \
    "math.op.native_bf16_mac_trig")                                           \
  _(OP_NATIVE_BF16_TANH, op_native_bf16_tanh, "math.op.native_bf16_tanh")     \
  _(OP_NATIVE_VECTOR_F32X64, op_native_vector_f32x64,                         \
    "math.op.native_vector_f32x64")                                           \
  _(OP_PROMOTE_VECTOR_F32X64, op_promote_vector_f32x64,                       \
    "math.op.promote_vector_f32x64")                                          \
  _(OP_SUPPORTED, op_supported, "math.op.supported")                          \
  _(RECIPE_GELU_ERF_GROUPED_PRODUCT, recipe_gelu_erf_grouped_product,         \
    "math.recipe.gelu_erf_grouped_product")                                   \
  _(RECIPE_GELU_LOGISTIC_GROUPED_PRODUCT,                                     \
    recipe_gelu_logistic_grouped_product,                                     \
    "math.recipe.gelu_logistic_grouped_product")                              \
  _(RECIPE_GELU_TANH_GROUPED_PRODUCT, recipe_gelu_tanh_grouped_product,       \
    "math.recipe.gelu_tanh_grouped_product")                                  \
  _(RECIPE_LOGISTIC_TANH_GROUPED_PRODUCT,                                     \
    recipe_logistic_tanh_grouped_product,                                     \
    "math.recipe.logistic_tanh_grouped_product")                              \
  _(RECIPE_ROUND_AWAY_F32, recipe_round_away_f32,                             \
    "math.recipe.round_away_f32")                                             \
  _(RECIPE_SILU_LOGISTIC_GROUPED_PRODUCT,                                     \
    recipe_silu_logistic_grouped_product,                                     \
    "math.recipe.silu_logistic_grouped_product")                              \
  _(RECIPE_WIDEN_F32_ROUND, recipe_widen_f32_round,                           \
    "math.recipe.widen_f32_round")                                            \
  _(DIV_ARCP, div_arcp, "math.div.arcp")                                      \
  _(ROOT_RSQRT_EXACT_F32, root_rsqrt_exact_f32, "math.rsqrt.exact_f32")       \
  _(ROOT_SQRT_EXACT_F32, root_sqrt_exact_f32, "math.sqrt.exact_f32")          \
  _(SHAPE_APPROXIMATE_ACTIVATION_PACKET, shape_approximate_activation_packet, \
    "math.shape.aie2p_approximate_activation_packet")                         \
  _(SHAPE_F16_MULTIPLY, shape_f16_multiply, "math.shape.aie2p_f16_multiply")  \
  _(SHAPE_F32_ADD, shape_f32_add, "math.shape.aie2p_f32_add")                 \
  _(SHAPE_F32_EXP, shape_f32_exp, "math.shape.aie2p_f32_exp")                 \
  _(SHAPE_F32_MULTIPLY, shape_f32_multiply, "math.shape.aie2p_f32_multiply")  \
  _(SHAPE_F32_ROUND, shape_f32_round, "math.shape.aie2p_f32_round")           \
  _(SHAPE_F32_SUB, shape_f32_sub, "math.shape.aie2p_f32_sub")                 \
  _(SHAPE_F32_TANH, shape_f32_tanh, "math.shape.aie2p_f32_tanh")              \
  _(SHAPE_F32_TRIG, shape_f32_trig, "math.shape.aie2p_f32_trig")              \
  _(SHAPE_SCALAR_F32_ROOT, shape_scalar_f32_root,                             \
    "math.shape.aie2p_scalar_f32_root")                                       \
  _(SHAPE_STATIC_F32_EXP2, shape_static_f32_exp2,                             \
    "math.shape.aie2p_static_f32_exp2")                                       \
  _(SILU_EXACT, silu_exact, "math.silu.exact")                                \
  _(TANH_EXACT_F32, tanh_exact_f32, "math.tanh.exact_f32")                    \
  _(TRIG_EXACT_F32, trig_exact_f32, "math.trig.exact_f32")

enum loom_aie2p_math_constraint_e {
  LOOM_AIE2P_MATH_CONSTRAINT_NONE = 0,
#define LOOM_AIE2P_MATH_CONSTRAINT_ENUM(name, field, value) \
  LOOM_AIE2P_MATH_CONSTRAINT_##name,
  LOOM_AIE2P_MATH_CONSTRAINTS(LOOM_AIE2P_MATH_CONSTRAINT_ENUM)
#undef LOOM_AIE2P_MATH_CONSTRAINT_ENUM
      LOOM_AIE2P_MATH_CONSTRAINT_COUNT_,
};

typedef struct loom_aie2p_math_constraint_strings_t {
#define LOOM_AIE2P_MATH_CONSTRAINT_FIELD(name, field, value) \
  char field[sizeof(value)];
  LOOM_AIE2P_MATH_CONSTRAINTS(LOOM_AIE2P_MATH_CONSTRAINT_FIELD)
#undef LOOM_AIE2P_MATH_CONSTRAINT_FIELD
} loom_aie2p_math_constraint_strings_t;

static const loom_aie2p_math_constraint_strings_t kAie2pMathConstraintStrings =
    {
#define LOOM_AIE2P_MATH_CONSTRAINT_VALUE(name, field, value) .field = value,
        LOOM_AIE2P_MATH_CONSTRAINTS(LOOM_AIE2P_MATH_CONSTRAINT_VALUE)
#undef LOOM_AIE2P_MATH_CONSTRAINT_VALUE
};

static const loom_string_ref_t
    kAie2pMathConstraints[LOOM_AIE2P_MATH_CONSTRAINT_COUNT_ - 1] = {
#define LOOM_AIE2P_MATH_CONSTRAINT_REF(name, field, value)      \
  [LOOM_AIE2P_MATH_CONSTRAINT_##name - 1] = LOOM_STRING_REF(    \
      0, offsetof(loom_aie2p_math_constraint_strings_t, field), \
      sizeof(value) - 1),
        LOOM_AIE2P_MATH_CONSTRAINTS(LOOM_AIE2P_MATH_CONSTRAINT_REF)
#undef LOOM_AIE2P_MATH_CONSTRAINT_REF
};

#undef LOOM_AIE2P_MATH_CONSTRAINTS

enum loom_aie2p_math_evaluation_e {
  LOOM_AIE2P_MATH_EVALUATION_NONE = 0,
  LOOM_AIE2P_MATH_EVALUATION_GROUPED_PRODUCT = 1,
  LOOM_AIE2P_MATH_EVALUATION_GROUPED_PRODUCT_BF16 = 2,
};

static const loom_target_math_evaluation_t kAie2pMathEvaluations[] = {
    [LOOM_AIE2P_MATH_EVALUATION_GROUPED_PRODUCT - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT,
            .product_element_type = LOOM_SCALAR_TYPE_BF16,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
            .packet_lane_count = 16,
            .conversion =
                {
                    .product_flags = LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ,
                },
        },
    [LOOM_AIE2P_MATH_EVALUATION_GROUPED_PRODUCT_BF16 - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT,
            .product_element_type = LOOM_SCALAR_TYPE_BF16,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
            .packet_lane_count = 16,
            .conversion =
                {
                    .input_flags = LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ,
                    .product_flags = LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ,
                    .result_flags = LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ,
                },
        },
};

#define LOOM_AIE2P_APPROXIMATE_ACTIVATION_OTHER_ELEMENT_TYPES  \
  (LOOM_SCALAR_TYPE_SET_F8E4M3 | LOOM_SCALAR_TYPE_SET_F8E5M2 | \
   LOOM_SCALAR_TYPE_SET_F16 | LOOM_SCALAR_TYPE_SET_F32)

#define LOOM_AIE2P_MATH_RULES(_)                                               \
  _(EXP2, LOOM_SCALAR_TYPE_SET_F32, ALL, STATIC_ELEMENTS, 1, 0, AFN, UNKNOWN,  \
    NONE, OP_NATIVE_BF16_EXP2, ELEMENT_F32, SHAPE_STATIC_F32_EXP2,             \
    EXP2_EXACT_F32)                                                            \
  _(SQRT, LOOM_SCALAR_TYPE_SET_F32, SCALAR, RANK_ONE, 1, 1, AFN, UNKNOWN,      \
    NONE, OP_NATIVE_APPROXIMATE_SQRT_F32, ELEMENT_F32, SHAPE_SCALAR_F32_ROOT,  \
    ROOT_SQRT_EXACT_F32)                                                       \
  _(RSQRT, LOOM_SCALAR_TYPE_SET_F32, SCALAR, RANK_ONE, 1, 1, AFN, UNKNOWN,     \
    NONE, OP_NATIVE_APPROXIMATE_RSQRT_F32, ELEMENT_F32, SHAPE_SCALAR_F32_ROOT, \
    ROOT_RSQRT_EXACT_F32)                                                      \
  _(GELU_ERF_OTHER, LOOM_AIE2P_APPROXIMATE_ACTIVATION_OTHER_ELEMENT_TYPES,     \
    ALL, STATIC_ELEMENTS, 1, 16, AFN, GELU_TANH_F32, GROUPED_PRODUCT,          \
    RECIPE_GELU_ERF_GROUPED_PRODUCT, ELEMENT_APPROXIMATE_ACTIVATION,           \
    SHAPE_APPROXIMATE_ACTIVATION_PACKET, GELU_ERF_EXACT)                       \
  _(GELU_ERF_BF16, LOOM_SCALAR_TYPE_SET_BF16, ALL, STATIC_ELEMENTS, 1, 16,     \
    AFN, GELU_TANH_F32, GROUPED_PRODUCT_BF16, RECIPE_GELU_ERF_GROUPED_PRODUCT, \
    ELEMENT_APPROXIMATE_ACTIVATION, SHAPE_APPROXIMATE_ACTIVATION_PACKET,       \
    GELU_ERF_EXACT)                                                            \
  _(GELU_TANH_OTHER, LOOM_AIE2P_APPROXIMATE_ACTIVATION_OTHER_ELEMENT_TYPES,    \
    ALL, STATIC_ELEMENTS, 1, 16, AFN, GELU_TANH_F32, GROUPED_PRODUCT,          \
    RECIPE_GELU_TANH_GROUPED_PRODUCT, ELEMENT_APPROXIMATE_ACTIVATION,          \
    SHAPE_APPROXIMATE_ACTIVATION_PACKET, GELU_TANH_EXACT)                      \
  _(GELU_TANH_BF16, LOOM_SCALAR_TYPE_SET_BF16, ALL, STATIC_ELEMENTS, 1, 16,    \
    AFN, GELU_TANH_F32, GROUPED_PRODUCT_BF16,                                  \
    RECIPE_GELU_TANH_GROUPED_PRODUCT, ELEMENT_APPROXIMATE_ACTIVATION,          \
    SHAPE_APPROXIMATE_ACTIVATION_PACKET, GELU_TANH_EXACT)                      \
  _(GELU_LOGISTIC_OTHER,                                                       \
    LOOM_AIE2P_APPROXIMATE_ACTIVATION_OTHER_ELEMENT_TYPES, ALL,                \
    STATIC_ELEMENTS, 1, 16, AFN, GELU_LOGISTIC_F32, GROUPED_PRODUCT,           \
    RECIPE_GELU_LOGISTIC_GROUPED_PRODUCT, ELEMENT_APPROXIMATE_ACTIVATION,      \
    SHAPE_APPROXIMATE_ACTIVATION_PACKET, GELU_LOGISTIC_EXACT)                  \
  _(GELU_LOGISTIC_BF16, LOOM_SCALAR_TYPE_SET_BF16, ALL, STATIC_ELEMENTS, 1,    \
    16, AFN, GELU_LOGISTIC_F32, GROUPED_PRODUCT_BF16,                          \
    RECIPE_GELU_LOGISTIC_GROUPED_PRODUCT, ELEMENT_APPROXIMATE_ACTIVATION,      \
    SHAPE_APPROXIMATE_ACTIVATION_PACKET, GELU_LOGISTIC_EXACT)                  \
  _(LOGISTIC_OTHER, LOOM_AIE2P_APPROXIMATE_ACTIVATION_OTHER_ELEMENT_TYPES,     \
    ALL, STATIC_ELEMENTS, 1, 16, AFN, LOGISTIC_TANH_F32, GROUPED_PRODUCT,      \
    RECIPE_LOGISTIC_TANH_GROUPED_PRODUCT, ELEMENT_APPROXIMATE_ACTIVATION,      \
    SHAPE_APPROXIMATE_ACTIVATION_PACKET, LOGISTIC_EXACT)                       \
  _(LOGISTIC_BF16, LOOM_SCALAR_TYPE_SET_BF16, ALL, STATIC_ELEMENTS, 1, 16,     \
    AFN, LOGISTIC_TANH_F32, GROUPED_PRODUCT_BF16,                              \
    RECIPE_LOGISTIC_TANH_GROUPED_PRODUCT, ELEMENT_APPROXIMATE_ACTIVATION,      \
    SHAPE_APPROXIMATE_ACTIVATION_PACKET, LOGISTIC_EXACT)                       \
  _(SILU_OTHER, LOOM_AIE2P_APPROXIMATE_ACTIVATION_OTHER_ELEMENT_TYPES, ALL,    \
    STATIC_ELEMENTS, 1, 16, AFN, SILU_LOGISTIC_F32, GROUPED_PRODUCT,           \
    RECIPE_SILU_LOGISTIC_GROUPED_PRODUCT, ELEMENT_APPROXIMATE_ACTIVATION,      \
    SHAPE_APPROXIMATE_ACTIVATION_PACKET, SILU_EXACT)                           \
  _(SILU_BF16, LOOM_SCALAR_TYPE_SET_BF16, ALL, STATIC_ELEMENTS, 1, 16, AFN,    \
    SILU_LOGISTIC_F32, GROUPED_PRODUCT_BF16,                                   \
    RECIPE_SILU_LOGISTIC_GROUPED_PRODUCT, ELEMENT_APPROXIMATE_ACTIVATION,      \
    SHAPE_APPROXIMATE_ACTIVATION_PACKET, SILU_EXACT)                           \
  _(SIN, LOOM_SCALAR_TYPE_SET_F32, SCALAR, RANK_ONE, 1, 1, AFN, UNKNOWN, NONE, \
    OP_NATIVE_BF16_MAC_TRIG, SHAPE_F32_TRIG, SHAPE_F32_TRIG, TRIG_EXACT_F32)   \
  _(COS, LOOM_SCALAR_TYPE_SET_F32, SCALAR, RANK_ONE, 1, 1, AFN, UNKNOWN, NONE, \
    OP_NATIVE_BF16_MAC_TRIG, SHAPE_F32_TRIG, SHAPE_F32_TRIG, TRIG_EXACT_F32)   \
  _(SINTURNS, LOOM_SCALAR_TYPE_SET_F32, SCALAR, RANK_ONE, 1, 1, AFN, UNKNOWN,  \
    NONE, OP_NATIVE_BF16_MAC_TRIG, SHAPE_F32_TRIG, SHAPE_F32_TRIG,             \
    TRIG_EXACT_F32)                                                            \
  _(COSTURNS, LOOM_SCALAR_TYPE_SET_F32, SCALAR, RANK_ONE, 1, 1, AFN, UNKNOWN,  \
    NONE, OP_NATIVE_BF16_MAC_TRIG, SHAPE_F32_TRIG, SHAPE_F32_TRIG,             \
    TRIG_EXACT_F32)                                                            \
  _(EXP, LOOM_SCALAR_TYPE_SET_F32, ALL, RANK_ONE, 1, 16, AFN, UNKNOWN, NONE,   \
    OP_NATIVE_BF16_EXP, SHAPE_F32_EXP, SHAPE_F32_EXP, EXP_EXACT_F32)           \
  _(TANH, LOOM_SCALAR_TYPE_SET_F32, ALL, RANK_ONE, 1, 16, AFN, UNKNOWN, NONE,  \
    OP_NATIVE_BF16_TANH, SHAPE_F32_TANH, SHAPE_F32_TANH, TANH_EXACT_F32)       \
  _(DIV_F32, LOOM_SCALAR_TYPE_SET_F32, ALL, ANY, 0, 0, ARCP, UNKNOWN, NONE,    \
    OP_SUPPORTED, ELEMENT_F32, NONE, DIV_ARCP)                                 \
  _(DIV_NARROW, LOOM_SCALAR_TYPE_SET_FLOAT_LE16, ALL, ANY, 0, 0, ARCP,         \
    WIDEN_F32_ROUND, NONE, RECIPE_WIDEN_F32_ROUND, ELEMENT_F32, NONE,          \
    DIV_ARCP)                                                                  \
  _(MUL_F16, LOOM_SCALAR_TYPE_SET_F16, SCALAR, RANK_ONE, 1, 1, NONE, UNKNOWN,  \
    NONE, OP_EXACT_BINARY16, SHAPE_F16_MULTIPLY, SHAPE_F16_MULTIPLY, NONE)     \
  _(MUL_F32, LOOM_SCALAR_TYPE_SET_F32, ALL, RANK_ONE, 1, 32, NONE, UNKNOWN,    \
    NONE, OP_EXACT_BINARY32, SHAPE_F32_MULTIPLY, SHAPE_F32_MULTIPLY, NONE)     \
  _(ROUND, LOOM_SCALAR_TYPE_SET_F32, ALL, RANK_ONE, 1, 16, NONE, ROUND_AWAY,   \
    NONE, RECIPE_ROUND_AWAY_F32, SHAPE_F32_ROUND, SHAPE_F32_ROUND, NONE)       \
  _(ROUNDEVEN, LOOM_SCALAR_TYPE_SET_F32, ALL, RANK_ONE, 1, 16, NONE, UNKNOWN,  \
    NONE, OP_EXACT_BINARY32_ROUNDEVEN, SHAPE_F32_ROUND, SHAPE_F32_ROUND, NONE) \
  _(TRUNC, LOOM_SCALAR_TYPE_SET_F32, ALL, RANK_ONE, 1, 16, NONE, UNKNOWN,      \
    NONE, OP_EXACT_BINARY32_TRUNC, SHAPE_F32_ROUND, SHAPE_F32_ROUND, NONE)     \
  _(ADD_PROMOTE, LOOM_SCALAR_TYPE_SET_F32, ALL, RANK_ONE, 1, 32, NONE,         \
    UNKNOWN, NONE, OP_PROMOTE_VECTOR_F32X64, SHAPE_F32_ADD, SHAPE_F32_ADD,     \
    NONE)                                                                      \
  _(ADD_NATIVE, LOOM_SCALAR_TYPE_SET_F32, VECTOR, RANK_ONE, 64, 64, NONE,      \
    UNKNOWN, NONE, OP_NATIVE_VECTOR_F32X64, SHAPE_F32_ADD, SHAPE_F32_ADD,      \
    NONE)                                                                      \
  _(SUB_PROMOTE, LOOM_SCALAR_TYPE_SET_F32, ALL, RANK_ONE, 1, 32, NONE,         \
    UNKNOWN, NONE, OP_PROMOTE_VECTOR_F32X64, SHAPE_F32_SUB, SHAPE_F32_SUB,     \
    NONE)                                                                      \
  _(SUB_NATIVE, LOOM_SCALAR_TYPE_SET_F32, VECTOR, RANK_ONE, 64, 64, NONE,      \
    UNKNOWN, NONE, OP_NATIVE_VECTOR_F32X64, SHAPE_F32_SUB, SHAPE_F32_SUB,      \
    NONE)

enum loom_aie2p_math_rule_e {
#define LOOM_AIE2P_MATH_RULE_ENUM(                                       \
    name, element_types, lane_domains, shape, minimum_lane_count,        \
    maximum_lane_count, required_fastmath, recipe, evaluation, decision, \
    element_constraint, shape_constraint, permission_constraint)         \
  LOOM_AIE2P_MATH_RULE_##name,
  LOOM_AIE2P_MATH_RULES(LOOM_AIE2P_MATH_RULE_ENUM)
#undef LOOM_AIE2P_MATH_RULE_ENUM
      LOOM_AIE2P_MATH_RULE_COUNT_,
};

static const loom_target_math_policy_rule_t
    kAie2pMathRules[LOOM_AIE2P_MATH_RULE_COUNT_] = {
#define LOOM_AIE2P_MATH_RULE_ROW(                                             \
    name_, element_types_, lane_domains_, shape_, minimum_lane_count_,        \
    maximum_lane_count_, required_fastmath_, recipe_, evaluation_, decision_, \
    element_constraint_, shape_constraint_, permission_constraint_)           \
  [LOOM_AIE2P_MATH_RULE_##name_] = {                                          \
      .match =                                                                \
          {                                                                   \
              .element_types = (element_types_),                              \
              .minimum_lane_count = (minimum_lane_count_),                    \
              .maximum_lane_count = (maximum_lane_count_),                    \
              .lane_domains =                                                 \
                  LOOM_TARGET_MATH_LANE_DOMAIN_SET_##lane_domains_,           \
              .shape = LOOM_TARGET_MATH_POLICY_SHAPE_##shape_,                \
              .required_fastmath_flags =                                      \
                  LOOM_TARGET_MATH_FASTMATH_FLAG_##required_fastmath_,        \
          },                                                                  \
      .recipe = LOOM_TARGET_MATH_RECIPE_##recipe_,                            \
      .evaluation = LOOM_AIE2P_MATH_EVALUATION_##evaluation_,                 \
      .decision_constraint = LOOM_AIE2P_MATH_CONSTRAINT_##decision_,          \
      .element_constraint = LOOM_AIE2P_MATH_CONSTRAINT_##element_constraint_, \
      .shape_constraint = LOOM_AIE2P_MATH_CONSTRAINT_##shape_constraint_,     \
      .permission_constraint =                                                \
          LOOM_AIE2P_MATH_CONSTRAINT_##permission_constraint_,                \
  },
        LOOM_AIE2P_MATH_RULES(LOOM_AIE2P_MATH_RULE_ROW)
#undef LOOM_AIE2P_MATH_RULE_ROW
};

#undef LOOM_AIE2P_MATH_RULES
#undef LOOM_AIE2P_APPROXIMATE_ACTIVATION_OTHER_ELEMENT_TYPES

#define LOOM_AIE2P_MATH_SPAN(first_, count_)                          \
  {                                                                   \
      .rule_start = LOOM_AIE2P_MATH_RULE_##first_,                    \
      .rule_count = (count_),                                         \
      .fallback_constraint = LOOM_AIE2P_MATH_CONSTRAINT_OP_SUPPORTED, \
  }

static const loom_target_math_policy_rule_span_t
    kAie2pMathOperationSpans[LOOM_TARGET_MATH_OP_COUNT] = {
        [LOOM_TARGET_MATH_OP_EXP2F] = LOOM_AIE2P_MATH_SPAN(EXP2, 1),
        [LOOM_TARGET_MATH_OP_SQRTF] = LOOM_AIE2P_MATH_SPAN(SQRT, 1),
        [LOOM_TARGET_MATH_OP_RSQRTF] = LOOM_AIE2P_MATH_SPAN(RSQRT, 1),
        [LOOM_TARGET_MATH_OP_GELUF_ERF] =
            LOOM_AIE2P_MATH_SPAN(GELU_ERF_OTHER, 2),
        [LOOM_TARGET_MATH_OP_GELUF_TANH] =
            LOOM_AIE2P_MATH_SPAN(GELU_TANH_OTHER, 2),
        [LOOM_TARGET_MATH_OP_GELUF_LOGISTIC] =
            LOOM_AIE2P_MATH_SPAN(GELU_LOGISTIC_OTHER, 2),
        [LOOM_TARGET_MATH_OP_LOGISTICF] =
            LOOM_AIE2P_MATH_SPAN(LOGISTIC_OTHER, 2),
        [LOOM_TARGET_MATH_OP_SILUF] = LOOM_AIE2P_MATH_SPAN(SILU_OTHER, 2),
        [LOOM_TARGET_MATH_OP_SINF] = LOOM_AIE2P_MATH_SPAN(SIN, 1),
        [LOOM_TARGET_MATH_OP_COSF] = LOOM_AIE2P_MATH_SPAN(COS, 1),
        [LOOM_TARGET_MATH_OP_SINTURNSF] = LOOM_AIE2P_MATH_SPAN(SINTURNS, 1),
        [LOOM_TARGET_MATH_OP_COSTURNSF] = LOOM_AIE2P_MATH_SPAN(COSTURNS, 1),
        [LOOM_TARGET_MATH_OP_EXPF] = LOOM_AIE2P_MATH_SPAN(EXP, 1),
        [LOOM_TARGET_MATH_OP_TANHF] = LOOM_AIE2P_MATH_SPAN(TANH, 1),
        [LOOM_TARGET_MATH_OP_MULF] = LOOM_AIE2P_MATH_SPAN(MUL_F16, 2),
        [LOOM_TARGET_MATH_OP_DIVF] = LOOM_AIE2P_MATH_SPAN(DIV_F32, 2),
        [LOOM_TARGET_MATH_OP_ROUNDF] = LOOM_AIE2P_MATH_SPAN(ROUND, 1),
        [LOOM_TARGET_MATH_OP_ROUNDEVENF] = LOOM_AIE2P_MATH_SPAN(ROUNDEVEN, 1),
        [LOOM_TARGET_MATH_OP_TRUNCF] = LOOM_AIE2P_MATH_SPAN(TRUNC, 1),
        [LOOM_TARGET_MATH_OP_ADDF] = LOOM_AIE2P_MATH_SPAN(ADD_PROMOTE, 2),
        [LOOM_TARGET_MATH_OP_SUBF] = LOOM_AIE2P_MATH_SPAN(SUB_PROMOTE, 2),
};

#undef LOOM_AIE2P_MATH_SPAN

static const loom_target_math_policy_table_t kAie2pMathPolicyTable = {
    .operation_spans = kAie2pMathOperationSpans,
    .rules = kAie2pMathRules,
    .default_constraint = LOOM_AIE2P_MATH_CONSTRAINT_OP_SUPPORTED,
    .constraints = kAie2pMathConstraints,
    .constraint_strings =
        {
            .data = {(const char*)&kAie2pMathConstraintStrings},
            .data_length = {sizeof(kAie2pMathConstraintStrings)},
        },
    .evaluations = kAie2pMathEvaluations,
};

static const loom_target_math_policy_t kAie2pMathPolicy = {
    .name = IREE_SVL("aie2p-math"),
    .table = &kAie2pMathPolicyTable,
};

static const loom_target_math_policy_registry_entry_t
    kAie2pMathPolicyEntries[] = {
        {
            .contract_set_stable_id = UINT64_C(0x25ed3c7d25a1452a),
            .policy = &kAie2pMathPolicy,
        },
};

void loom_aie2p_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      kAie2pMathPolicyEntries, IREE_ARRAYSIZE(kAie2pMathPolicyEntries),
      out_registry);
}
