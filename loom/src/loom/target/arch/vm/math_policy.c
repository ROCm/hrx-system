// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/math_policy.h"

#include <stddef.h>

#define LOOM_VM_MATH_CONSTRAINTS(_)                                 \
  _(ELEMENT_F32_F64, element_f32_f64, "math.element.f32_f64")       \
  _(OP_SELECTED_WIDTH, op_selected_width, "math.op.selected_width") \
  _(OP_SUPPORTED, op_supported, "math.op.supported")                \
  _(RECIPE_AFN_F32, recipe_afn_f32, "math.recipe.afn_f32")          \
  _(RECIPE_INVERSE_HYPERBOLIC_F32, recipe_inverse_hyperbolic_f32,   \
    "math.recipe.inverse_hyperbolic_f32")                           \
  _(RECIPE_WIDEN_F32_ROUND, recipe_widen_f32_round,                 \
    "math.recipe.widen_f32_round")

enum loom_vm_math_constraint_e {
  LOOM_VM_MATH_CONSTRAINT_NONE = 0,
#define LOOM_VM_MATH_CONSTRAINT_ENUM(name, field, value) \
  LOOM_VM_MATH_CONSTRAINT_##name,
  LOOM_VM_MATH_CONSTRAINTS(LOOM_VM_MATH_CONSTRAINT_ENUM)
#undef LOOM_VM_MATH_CONSTRAINT_ENUM
      LOOM_VM_MATH_CONSTRAINT_COUNT_,
};

typedef struct loom_vm_math_constraint_strings_t {
#define LOOM_VM_MATH_CONSTRAINT_FIELD(name, field, value) \
  char field[sizeof(value)];
  LOOM_VM_MATH_CONSTRAINTS(LOOM_VM_MATH_CONSTRAINT_FIELD)
#undef LOOM_VM_MATH_CONSTRAINT_FIELD
} loom_vm_math_constraint_strings_t;

static const loom_vm_math_constraint_strings_t kVmMathConstraintStrings = {
#define LOOM_VM_MATH_CONSTRAINT_VALUE(name, field, value) .field = value,
    LOOM_VM_MATH_CONSTRAINTS(LOOM_VM_MATH_CONSTRAINT_VALUE)
#undef LOOM_VM_MATH_CONSTRAINT_VALUE
};

static const loom_string_ref_t
    kVmMathConstraints[LOOM_VM_MATH_CONSTRAINT_COUNT_ - 1] = {
#define LOOM_VM_MATH_CONSTRAINT_REF(name, field, value)                      \
  [LOOM_VM_MATH_CONSTRAINT_##name - 1] =                                     \
      LOOM_STRING_REF(0, offsetof(loom_vm_math_constraint_strings_t, field), \
                      sizeof(value) - 1),
        LOOM_VM_MATH_CONSTRAINTS(LOOM_VM_MATH_CONSTRAINT_REF)
#undef LOOM_VM_MATH_CONSTRAINT_REF
};

#undef LOOM_VM_MATH_CONSTRAINTS

enum loom_vm_math_evaluation_e {
  LOOM_VM_MATH_EVALUATION_NONE = 0,
  LOOM_VM_MATH_EVALUATION_SOURCE_F32 = 1,
  LOOM_VM_MATH_EVALUATION_WIDENED_F32 = 2,
};

static const loom_target_math_evaluation_t kVmMathEvaluations[] = {
    [LOOM_VM_MATH_EVALUATION_SOURCE_F32 - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_SOURCE,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
        },
    [LOOM_VM_MATH_EVALUATION_WIDENED_F32 - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_WIDENED,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
        },
};

#define LOOM_VM_MATH_DIRECT_RULE(element_types_, required_fastmath_flags_,    \
                                 recipe_, evaluation_, constraint_,           \
                                 permission_constraint_)                      \
  LOOM_TARGET_MATH_POLICY_DIRECT_RULE(                                        \
      element_types_, LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,                   \
      required_fastmath_flags_, recipe_, LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, evaluation_, constraint_,          \
      permission_constraint_)

static const loom_target_math_policy_direct_rule_t
    kVmMathDirectRules[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_VM_MATH_SELECTED_WIDTH_RULE(op_)                                  \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_VM_MATH_DIRECT_RULE(                      \
      LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64,                     \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, LOOM_TARGET_MATH_RECIPE_UNKNOWN,    \
      LOOM_VM_MATH_EVALUATION_NONE, LOOM_VM_MATH_CONSTRAINT_OP_SELECTED_WIDTH, \
      LOOM_VM_MATH_CONSTRAINT_NONE)
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(ADDF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(SUBF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(MULF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(DIVF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(REMF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(CEILF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(FLOORF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(ROUNDEVENF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(ROUNDF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(TRUNCF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(EXPF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(EXP2F),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(EXPM1F),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(ERFF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(ERFCF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(LOGISTICF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(SILUF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(SOFTPLUSF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(GELUF_ERF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(GELUF_TANH),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(GELUF_LOGISTIC),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(TANF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(SINTURNSF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(COSTURNSF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(ASINF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(ACOSF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(ATANF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(ATAN2F),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(SINHF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(COSHF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(TANHF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(POWF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(SQRTF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(RSQRTF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(CBRTF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(LOGF),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(LOG2F),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(LOG10F),
        LOOM_VM_MATH_SELECTED_WIDTH_RULE(LOG1PF),
#undef LOOM_VM_MATH_SELECTED_WIDTH_RULE
#define LOOM_VM_MATH_AFN_RECIPE_RULE(op_, recipe_)                     \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_VM_MATH_DIRECT_RULE(              \
      LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,    \
      LOOM_TARGET_MATH_RECIPE_##recipe_, LOOM_VM_MATH_EVALUATION_NONE, \
      LOOM_VM_MATH_CONSTRAINT_RECIPE_AFN_F32, LOOM_VM_MATH_CONSTRAINT_NONE)
        LOOM_VM_MATH_AFN_RECIPE_RULE(SINF, SIN_TURNS_F32),
        LOOM_VM_MATH_AFN_RECIPE_RULE(COSF, COS_TURNS_F32),
#undef LOOM_VM_MATH_AFN_RECIPE_RULE
#define LOOM_VM_MATH_INVERSE_RULE(op_)                              \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_VM_MATH_DIRECT_RULE(           \
      LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, \
      LOOM_TARGET_MATH_RECIPE_INVERSE_HYPERBOLIC_F32,               \
      LOOM_VM_MATH_EVALUATION_SOURCE_F32,                           \
      LOOM_VM_MATH_CONSTRAINT_RECIPE_INVERSE_HYPERBOLIC_F32,        \
      LOOM_VM_MATH_CONSTRAINT_NONE)
        LOOM_VM_MATH_INVERSE_RULE(ASINHF),
        LOOM_VM_MATH_INVERSE_RULE(ACOSHF),
        LOOM_VM_MATH_INVERSE_RULE(ATANHF),
#undef LOOM_VM_MATH_INVERSE_RULE
};

#undef LOOM_VM_MATH_DIRECT_RULE

#define LOOM_VM_MATH_MATCH(element_types_, required_fastmath_flags_) \
  {                                                                  \
      .element_types = (element_types_),                             \
      .lane_domains = LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,          \
      .shape = LOOM_TARGET_MATH_POLICY_SHAPE_ANY,                    \
      .required_fastmath_flags = (required_fastmath_flags_),         \
  }

enum loom_vm_math_rule_e {
  LOOM_VM_MATH_RULE_INVERSE_NARROW = 0,
  LOOM_VM_MATH_RULE_ELEMENTARY_ORDINARY,
  LOOM_VM_MATH_RULE_ARITHMETIC_NARROW,
  LOOM_VM_MATH_RULE_COUNT_,
};

static const loom_target_math_policy_rule_t
    kVmMathRules[LOOM_VM_MATH_RULE_COUNT_] = {
        [LOOM_VM_MATH_RULE_ELEMENTARY_ORDINARY] =
            {
                .match = LOOM_VM_MATH_MATCH(
                    LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64,
                    LOOM_TARGET_MATH_FASTMATH_FLAG_NONE),
                .decision_constraint =
                    LOOM_VM_MATH_CONSTRAINT_OP_SELECTED_WIDTH,
            },
        [LOOM_VM_MATH_RULE_ARITHMETIC_NARROW] =
            {
                .match =
                    LOOM_VM_MATH_MATCH(LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
                                       LOOM_TARGET_MATH_FASTMATH_FLAG_NONE),
                .recipe = LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND,
                .decision_constraint =
                    LOOM_VM_MATH_CONSTRAINT_RECIPE_WIDEN_F32_ROUND,
            },
        [LOOM_VM_MATH_RULE_INVERSE_NARROW] =
            {
                .match = LOOM_VM_MATH_MATCH(LOOM_SCALAR_TYPE_SET_FLOAT_LE16,
                                            LOOM_TARGET_MATH_FASTMATH_FLAG_AFN),
                .recipe = LOOM_TARGET_MATH_RECIPE_INVERSE_HYPERBOLIC_F32,
                .evaluation = LOOM_VM_MATH_EVALUATION_WIDENED_F32,
                .decision_constraint =
                    LOOM_VM_MATH_CONSTRAINT_RECIPE_INVERSE_HYPERBOLIC_F32,
            },
};

#undef LOOM_VM_MATH_MATCH

#define LOOM_VM_MATH_SPAN(start_, count_, fallback_) \
  {                                                  \
      .rule_start = (start_),                        \
      .rule_count = (count_),                        \
      .fallback_constraint = (fallback_),            \
  }

static const loom_target_math_policy_rule_span_t
    kVmMathOperationSpans[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_VM_MATH_ARITHMETIC_SPAN(op_)                       \
  [LOOM_TARGET_MATH_OP_##op_] =                                 \
      LOOM_VM_MATH_SPAN(LOOM_VM_MATH_RULE_ARITHMETIC_NARROW, 1, \
                        LOOM_VM_MATH_CONSTRAINT_ELEMENT_F32_F64)
        LOOM_VM_MATH_ARITHMETIC_SPAN(ADDF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(SUBF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(MULF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(DIVF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(REMF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(EXPF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(EXP2F),
        LOOM_VM_MATH_ARITHMETIC_SPAN(EXPM1F),
        LOOM_VM_MATH_ARITHMETIC_SPAN(ERFF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(ERFCF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(LOGISTICF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(SILUF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(SOFTPLUSF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(GELUF_ERF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(GELUF_TANH),
        LOOM_VM_MATH_ARITHMETIC_SPAN(GELUF_LOGISTIC),
        LOOM_VM_MATH_ARITHMETIC_SPAN(TANF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(SINTURNSF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(COSTURNSF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(ASINF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(ACOSF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(ATANF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(ATAN2F),
        LOOM_VM_MATH_ARITHMETIC_SPAN(SINHF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(COSHF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(TANHF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(POWF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(SQRTF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(RSQRTF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(CBRTF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(LOGF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(LOG2F),
        LOOM_VM_MATH_ARITHMETIC_SPAN(LOG10F),
        LOOM_VM_MATH_ARITHMETIC_SPAN(LOG1PF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(CEILF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(FLOORF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(ROUNDEVENF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(ROUNDF),
        LOOM_VM_MATH_ARITHMETIC_SPAN(TRUNCF),
#undef LOOM_VM_MATH_ARITHMETIC_SPAN
        [LOOM_TARGET_MATH_OP_SINF] =
            LOOM_VM_MATH_SPAN(LOOM_VM_MATH_RULE_ELEMENTARY_ORDINARY, 2,
                              LOOM_VM_MATH_CONSTRAINT_ELEMENT_F32_F64),
        [LOOM_TARGET_MATH_OP_COSF] =
            LOOM_VM_MATH_SPAN(LOOM_VM_MATH_RULE_ELEMENTARY_ORDINARY, 2,
                              LOOM_VM_MATH_CONSTRAINT_ELEMENT_F32_F64),
#define LOOM_VM_MATH_INVERSE_SPAN(op_)                       \
  [LOOM_TARGET_MATH_OP_##op_] =                              \
      LOOM_VM_MATH_SPAN(LOOM_VM_MATH_RULE_INVERSE_NARROW, 3, \
                        LOOM_VM_MATH_CONSTRAINT_ELEMENT_F32_F64)
        LOOM_VM_MATH_INVERSE_SPAN(ASINHF),
        LOOM_VM_MATH_INVERSE_SPAN(ACOSHF),
        LOOM_VM_MATH_INVERSE_SPAN(ATANHF),
#undef LOOM_VM_MATH_INVERSE_SPAN
};

#undef LOOM_VM_MATH_SPAN

static const loom_target_math_policy_match_t kVmMathFmaForms[] = {
    {
        .element_types = LOOM_SCALAR_TYPE_SET_F32 | LOOM_SCALAR_TYPE_SET_F64,
        .lane_domains = LOOM_TARGET_MATH_LANE_DOMAIN_SET_SCALAR,
        .shape = LOOM_TARGET_MATH_POLICY_SHAPE_ANY,
    },
};

static const loom_target_math_policy_table_t kVmMathPolicyTable = {
    .operation_spans = kVmMathOperationSpans,
    .rules = kVmMathRules,
    .fma_forms = kVmMathFmaForms,
    .fma_form_count = IREE_ARRAYSIZE(kVmMathFmaForms),
    .default_constraint = LOOM_VM_MATH_CONSTRAINT_OP_SUPPORTED,
    .constraints = kVmMathConstraints,
    .constraint_strings =
        {
            .data = {(const char*)&kVmMathConstraintStrings},
            .data_length = {sizeof(kVmMathConstraintStrings)},
        },
    .evaluations = kVmMathEvaluations,
};

static const loom_target_math_policy_t kVmMathPolicy = {
    .name = IREE_SVL("vm-math"),
    .direct_rules = kVmMathDirectRules,
    .table = &kVmMathPolicyTable,
};

void loom_vm_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  static const loom_target_math_policy_registry_entry_t kEntries[] = {
      {.contract_set_stable_id = UINT64_C(0x32476f14332ccb19),
       .policy = &kVmMathPolicy},
  };
  loom_target_math_policy_registry_initialize_from_entries(
      kEntries, IREE_ARRAYSIZE(kEntries), out_registry);
}
