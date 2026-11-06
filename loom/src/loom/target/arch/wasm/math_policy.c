// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/math_policy.h"

#include <stddef.h>

#define LOOM_WASM_MATH_CONSTRAINTS(_)                                        \
  _(ELEMENT_NATIVE_FLOAT, element_native_float, "math.element.native_float") \
  _(OP_NATIVE_FLOAT, op_native_float, "math.op.native_float")                \
  _(OP_SUPPORTED, op_supported, "math.op.supported")                         \
  _(RECIPE_ROUND_AWAY, recipe_round_away, "math.recipe.round_away")          \
  _(RECIPE_RSQRT_SQRT_DIV, recipe_rsqrt_sqrt_div,                            \
    "math.recipe.rsqrt_sqrt_div")                                            \
  _(RECIPE_WIDEN_F32_ROUND, recipe_widen_f32_round,                          \
    "math.recipe.widen_f32_round")

enum loom_wasm_math_constraint_e {
  LOOM_WASM_MATH_CONSTRAINT_NONE = 0,
#define LOOM_WASM_MATH_CONSTRAINT_ENUM(name, field, value) \
  LOOM_WASM_MATH_CONSTRAINT_##name,
  LOOM_WASM_MATH_CONSTRAINTS(LOOM_WASM_MATH_CONSTRAINT_ENUM)
#undef LOOM_WASM_MATH_CONSTRAINT_ENUM
      LOOM_WASM_MATH_CONSTRAINT_COUNT_,
};

typedef struct loom_wasm_math_constraint_strings_t {
#define LOOM_WASM_MATH_CONSTRAINT_FIELD(name, field, value) \
  char field[sizeof(value)];
  LOOM_WASM_MATH_CONSTRAINTS(LOOM_WASM_MATH_CONSTRAINT_FIELD)
#undef LOOM_WASM_MATH_CONSTRAINT_FIELD
} loom_wasm_math_constraint_strings_t;

static const loom_wasm_math_constraint_strings_t kWasmMathConstraintStrings = {
#define LOOM_WASM_MATH_CONSTRAINT_VALUE(name, field, value) .field = value,
    LOOM_WASM_MATH_CONSTRAINTS(LOOM_WASM_MATH_CONSTRAINT_VALUE)
#undef LOOM_WASM_MATH_CONSTRAINT_VALUE
};

static const loom_string_ref_t
    kWasmMathConstraints[LOOM_WASM_MATH_CONSTRAINT_COUNT_ - 1] = {
#define LOOM_WASM_MATH_CONSTRAINT_REF(name, field, value)                      \
  [LOOM_WASM_MATH_CONSTRAINT_##name - 1] =                                     \
      LOOM_STRING_REF(0, offsetof(loom_wasm_math_constraint_strings_t, field), \
                      sizeof(value) - 1),
        LOOM_WASM_MATH_CONSTRAINTS(LOOM_WASM_MATH_CONSTRAINT_REF)
#undef LOOM_WASM_MATH_CONSTRAINT_REF
};

#undef LOOM_WASM_MATH_CONSTRAINTS

#define LOOM_WASM_MATH_DIRECT_RULE(element_types_, recipe_, evaluation_, \
                                   constraint_)                          \
  LOOM_TARGET_MATH_POLICY_DIRECT_RULE(                                   \
      element_types_, LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,              \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, recipe_,                      \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,                               \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, evaluation_, constraint_,     \
      LOOM_WASM_MATH_CONSTRAINT_NONE)

#define LOOM_WASM_NATIVE_FLOAT_TYPES \
  (LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64)

enum loom_wasm_math_evaluation_e {
  LOOM_WASM_MATH_EVALUATION_NONE = 0,
  LOOM_WASM_MATH_EVALUATION_WIDENED_F32 = 1,
};

static const loom_target_math_evaluation_t kWasmMathEvaluations[] = {
    [LOOM_WASM_MATH_EVALUATION_WIDENED_F32 - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_WIDENED,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
        },
};

static const loom_target_math_policy_direct_rule_t
    kWasmMathDirectRules[LOOM_TARGET_MATH_OP_COUNT] = {
        [LOOM_TARGET_MATH_OP_ADDF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_OP_NATIVE_FLOAT),
        [LOOM_TARGET_MATH_OP_SUBF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_OP_NATIVE_FLOAT),
        [LOOM_TARGET_MATH_OP_MULF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_OP_NATIVE_FLOAT),
        [LOOM_TARGET_MATH_OP_DIVF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_OP_NATIVE_FLOAT),
        [LOOM_TARGET_MATH_OP_CEILF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_OP_NATIVE_FLOAT),
        [LOOM_TARGET_MATH_OP_FLOORF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_OP_NATIVE_FLOAT),
        [LOOM_TARGET_MATH_OP_ROUNDEVENF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_OP_NATIVE_FLOAT),
        [LOOM_TARGET_MATH_OP_ROUNDF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES, LOOM_TARGET_MATH_RECIPE_ROUND_AWAY,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_RECIPE_ROUND_AWAY),
        [LOOM_TARGET_MATH_OP_TRUNCF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_OP_NATIVE_FLOAT),
        [LOOM_TARGET_MATH_OP_SQRTF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES, LOOM_TARGET_MATH_RECIPE_UNKNOWN,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_OP_NATIVE_FLOAT),
        [LOOM_TARGET_MATH_OP_RSQRTF] = LOOM_WASM_MATH_DIRECT_RULE(
            LOOM_WASM_NATIVE_FLOAT_TYPES,
            LOOM_TARGET_MATH_RECIPE_RSQRT_SQRT_DIV,
            LOOM_WASM_MATH_EVALUATION_NONE,
            LOOM_WASM_MATH_CONSTRAINT_RECIPE_RSQRT_SQRT_DIV),
};

#undef LOOM_WASM_MATH_DIRECT_RULE

#define LOOM_WASM_MATH_MATCH(element_types_)                          \
  {                                                                   \
      .element_types = (element_types_),                              \
      .lane_domains = LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,           \
      .shape = LOOM_TARGET_MATH_POLICY_SHAPE_ANY,                     \
      .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, \
  }

enum loom_wasm_math_rule_e {
  LOOM_WASM_MATH_RULE_WIDEN_F32_NARROW = 0,
  LOOM_WASM_MATH_RULE_ROUND_AWAY_NARROW,
  LOOM_WASM_MATH_RULE_COUNT_,
};

static const loom_target_math_policy_rule_t
    kWasmMathRules[LOOM_WASM_MATH_RULE_COUNT_] = {
        [LOOM_WASM_MATH_RULE_WIDEN_F32_NARROW] =
            {
                .match = LOOM_WASM_MATH_MATCH(LOOM_SCALAR_TYPE_SET_FLOAT_LE16),
                .recipe = LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND,
                .decision_constraint =
                    LOOM_WASM_MATH_CONSTRAINT_RECIPE_WIDEN_F32_ROUND,
            },
        [LOOM_WASM_MATH_RULE_ROUND_AWAY_NARROW] =
            {
                .match = LOOM_WASM_MATH_MATCH(LOOM_SCALAR_TYPE_SET_FLOAT_LE16),
                .recipe = LOOM_TARGET_MATH_RECIPE_ROUND_AWAY,
                .evaluation = LOOM_WASM_MATH_EVALUATION_WIDENED_F32,
                .decision_constraint =
                    LOOM_WASM_MATH_CONSTRAINT_RECIPE_ROUND_AWAY,
            },
};

#undef LOOM_WASM_MATH_MATCH

#define LOOM_WASM_MATH_SPAN(start_, count_, fallback_) \
  {                                                    \
      .rule_start = (start_),                          \
      .rule_count = (count_),                          \
      .fallback_constraint = (fallback_),              \
  }

#define LOOM_WASM_MATH_WIDEN_F32_SPAN                          \
  LOOM_WASM_MATH_SPAN(LOOM_WASM_MATH_RULE_WIDEN_F32_NARROW, 1, \
                      LOOM_WASM_MATH_CONSTRAINT_ELEMENT_NATIVE_FLOAT)

#define LOOM_WASM_MATH_ROUND_AWAY_SPAN                          \
  LOOM_WASM_MATH_SPAN(LOOM_WASM_MATH_RULE_ROUND_AWAY_NARROW, 1, \
                      LOOM_WASM_MATH_CONSTRAINT_ELEMENT_NATIVE_FLOAT)

#define LOOM_WASM_MATH_NATIVE_FLOAT_SPAN \
  LOOM_WASM_MATH_SPAN(0, 0, LOOM_WASM_MATH_CONSTRAINT_ELEMENT_NATIVE_FLOAT)

static const loom_target_math_policy_rule_span_t
    kWasmMathOperationSpans[LOOM_TARGET_MATH_OP_COUNT] = {
        [LOOM_TARGET_MATH_OP_ADDF] = LOOM_WASM_MATH_WIDEN_F32_SPAN,
        [LOOM_TARGET_MATH_OP_SUBF] = LOOM_WASM_MATH_WIDEN_F32_SPAN,
        [LOOM_TARGET_MATH_OP_MULF] = LOOM_WASM_MATH_WIDEN_F32_SPAN,
        [LOOM_TARGET_MATH_OP_DIVF] = LOOM_WASM_MATH_WIDEN_F32_SPAN,
        [LOOM_TARGET_MATH_OP_CEILF] = LOOM_WASM_MATH_WIDEN_F32_SPAN,
        [LOOM_TARGET_MATH_OP_FLOORF] = LOOM_WASM_MATH_WIDEN_F32_SPAN,
        [LOOM_TARGET_MATH_OP_ROUNDEVENF] = LOOM_WASM_MATH_WIDEN_F32_SPAN,
        [LOOM_TARGET_MATH_OP_ROUNDF] = LOOM_WASM_MATH_ROUND_AWAY_SPAN,
        [LOOM_TARGET_MATH_OP_TRUNCF] = LOOM_WASM_MATH_WIDEN_F32_SPAN,
        [LOOM_TARGET_MATH_OP_SQRTF] = LOOM_WASM_MATH_NATIVE_FLOAT_SPAN,
        [LOOM_TARGET_MATH_OP_RSQRTF] = LOOM_WASM_MATH_NATIVE_FLOAT_SPAN,
};

#undef LOOM_WASM_MATH_WIDEN_F32_SPAN
#undef LOOM_WASM_MATH_ROUND_AWAY_SPAN
#undef LOOM_WASM_MATH_NATIVE_FLOAT_SPAN
#undef LOOM_WASM_MATH_SPAN
#undef LOOM_WASM_NATIVE_FLOAT_TYPES

static const loom_target_math_policy_table_t kWasmMathPolicyTable = {
    .operation_spans = kWasmMathOperationSpans,
    .rules = kWasmMathRules,
    .evaluations = kWasmMathEvaluations,
    .default_constraint = LOOM_WASM_MATH_CONSTRAINT_OP_SUPPORTED,
    .constraints = kWasmMathConstraints,
    .constraint_strings =
        {
            .data = {(const char*)&kWasmMathConstraintStrings},
            .data_length = {sizeof(kWasmMathConstraintStrings)},
        },
};

static const loom_target_math_policy_t kWasmMathPolicy = {
    .name = IREE_SVL("wasm-math"),
    .direct_rules = kWasmMathDirectRules,
    .table = &kWasmMathPolicyTable,
};

static const loom_target_math_policy_registry_entry_t kWasmMathPolicyEntries[] =
    {
        {/*.contract_set_stable_id=*/UINT64_C(0x49e70aa103499a7e),
         /*.policy=*/&kWasmMathPolicy},
};

void loom_wasm_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  loom_target_math_policy_registry_initialize_from_entries(
      kWasmMathPolicyEntries, IREE_ARRAYSIZE(kWasmMathPolicyEntries),
      out_registry);
}
