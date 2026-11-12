// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/provider.h"

#include <stddef.h>

#include "loom/ops/test/registry.h"
#include "loom/pass/test/registry.h"
#include "loom/target/facts.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/target/test/lower/registry.h"
#include "loom/target/test/target_records.h"

#define LOOM_TEST_MATH_CONSTRAINTS(_)                                      \
  _(GROUPED_FIXTURE, grouped_fixture, "test.math.grouped_product.fixture") \
  _(GROUPED_NATIVE, grouped_native, "test.math.grouped_product.native")    \
  _(GROUPED_REQUIRES_AFN, grouped_requires_afn,                            \
    "test.math.grouped_product.requires_afn")                              \
  _(RECIPE_GELU_LOGISTIC, recipe_gelu_logistic,                            \
    "test.math.recipe.gelu_logistic")                                      \
  _(RECIPE_GELU_TANH, recipe_gelu_tanh, "test.math.recipe.gelu_tanh")      \
  _(RECIPE_LOGISTIC_TANH, recipe_logistic_tanh,                            \
    "test.math.recipe.logistic_tanh")                                      \
  _(RECIPE_SILU_LOGISTIC, recipe_silu_logistic,                            \
    "test.math.recipe.silu_logistic")                                      \
  _(RECIPE_WIDEN_F32_ROUND, recipe_widen_f32_round,                        \
    "test.math.recipe.widen_f32_round")                                    \
  _(WIDEN_FIXTURE, widen_fixture, "test.math.widen_f32_round.fixture")

enum loom_test_math_constraint_e {
  LOOM_TEST_MATH_CONSTRAINT_NONE = 0,
#define LOOM_TEST_MATH_CONSTRAINT_ENUM(name, field, value) \
  LOOM_TEST_MATH_CONSTRAINT_##name,
  LOOM_TEST_MATH_CONSTRAINTS(LOOM_TEST_MATH_CONSTRAINT_ENUM)
#undef LOOM_TEST_MATH_CONSTRAINT_ENUM
      LOOM_TEST_MATH_CONSTRAINT_COUNT_,
};

typedef struct loom_test_math_constraint_strings_t {
#define LOOM_TEST_MATH_CONSTRAINT_FIELD(name, field, value) \
  char field[sizeof(value)];
  LOOM_TEST_MATH_CONSTRAINTS(LOOM_TEST_MATH_CONSTRAINT_FIELD)
#undef LOOM_TEST_MATH_CONSTRAINT_FIELD
} loom_test_math_constraint_strings_t;

static const loom_test_math_constraint_strings_t kTestMathConstraintStrings = {
#define LOOM_TEST_MATH_CONSTRAINT_VALUE(name, field, value) .field = value,
    LOOM_TEST_MATH_CONSTRAINTS(LOOM_TEST_MATH_CONSTRAINT_VALUE)
#undef LOOM_TEST_MATH_CONSTRAINT_VALUE
};

static const loom_string_ref_t
    kTestMathConstraints[LOOM_TEST_MATH_CONSTRAINT_COUNT_ - 1] = {
#define LOOM_TEST_MATH_CONSTRAINT_REF(name, field, value)                      \
  [LOOM_TEST_MATH_CONSTRAINT_##name - 1] =                                     \
      LOOM_STRING_REF(0, offsetof(loom_test_math_constraint_strings_t, field), \
                      sizeof(value) - 1),
        LOOM_TEST_MATH_CONSTRAINTS(LOOM_TEST_MATH_CONSTRAINT_REF)
#undef LOOM_TEST_MATH_CONSTRAINT_REF
};

#undef LOOM_TEST_MATH_CONSTRAINTS

#define LOOM_TEST_MATH_MATCH(element_types_)                \
  {                                                         \
      .element_types = (element_types_),                    \
      .lane_domains = LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL, \
      .shape = LOOM_TARGET_MATH_POLICY_SHAPE_ANY,           \
  }

static const loom_target_math_policy_direct_rule_t
    kTestWidenMathDirectRules[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_TEST_WIDEN_NATIVE_RULE(op_)                                    \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_TARGET_MATH_POLICY_DIRECT_RULE(        \
      LOOM_SCALAR_TYPE_SET_F32, LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,       \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, LOOM_TARGET_MATH_RECIPE_UNKNOWN, \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,                                  \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, 0,                               \
      LOOM_TEST_MATH_CONSTRAINT_WIDEN_FIXTURE, LOOM_TEST_MATH_CONSTRAINT_NONE)
        LOOM_TEST_WIDEN_NATIVE_RULE(MULF),
        LOOM_TEST_WIDEN_NATIVE_RULE(DIVF),
        LOOM_TEST_WIDEN_NATIVE_RULE(CEILF),
        LOOM_TEST_WIDEN_NATIVE_RULE(FLOORF),
        LOOM_TEST_WIDEN_NATIVE_RULE(ROUNDEVENF),
        LOOM_TEST_WIDEN_NATIVE_RULE(TRUNCF),
#undef LOOM_TEST_WIDEN_NATIVE_RULE
};

static const loom_target_math_policy_rule_t kTestWidenMathRules[] = {
    {
        .match = LOOM_TEST_MATH_MATCH(LOOM_SCALAR_TYPE_SET_FLOAT_LE16),
        .recipe = LOOM_TARGET_MATH_RECIPE_WIDEN_F32_ROUND,
        .decision_constraint = LOOM_TEST_MATH_CONSTRAINT_RECIPE_WIDEN_F32_ROUND,
    },
};

static const loom_target_math_policy_rule_span_t
    kTestWidenMathOperationSpans[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_TEST_WIDEN_SPAN(op_)                                     \
  [LOOM_TARGET_MATH_OP_##op_] = {                                     \
      .rule_count = IREE_ARRAYSIZE(kTestWidenMathRules),              \
      .fallback_constraint = LOOM_TEST_MATH_CONSTRAINT_WIDEN_FIXTURE, \
  }
        LOOM_TEST_WIDEN_SPAN(MULF),       LOOM_TEST_WIDEN_SPAN(DIVF),
        LOOM_TEST_WIDEN_SPAN(CEILF),      LOOM_TEST_WIDEN_SPAN(FLOORF),
        LOOM_TEST_WIDEN_SPAN(ROUNDEVENF), LOOM_TEST_WIDEN_SPAN(TRUNCF),
#undef LOOM_TEST_WIDEN_SPAN
};

static const loom_target_math_policy_table_t kTestWidenMathPolicyTable = {
    .operation_spans = kTestWidenMathOperationSpans,
    .rules = kTestWidenMathRules,
    .default_constraint = LOOM_TEST_MATH_CONSTRAINT_WIDEN_FIXTURE,
    .constraints = kTestMathConstraints,
    .constraint_strings =
        {
            .data = {(const char*)&kTestMathConstraintStrings},
            .data_length = {sizeof(kTestMathConstraintStrings)},
        },
};

enum loom_test_grouped_math_evaluation_e {
  LOOM_TEST_GROUPED_MATH_EVALUATION_NONE = 0,
  LOOM_TEST_GROUPED_MATH_EVALUATION_DEFAULT = 1,
  LOOM_TEST_GROUPED_MATH_EVALUATION_BF16_BOUNDARY = 2,
};

static const loom_target_math_evaluation_t kTestBf16x16MathEvaluations[] = {
    [LOOM_TEST_GROUPED_MATH_EVALUATION_DEFAULT - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT,
            .product_element_type = LOOM_SCALAR_TYPE_BF16,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
            .packet_lane_count = 16,
            .conversion =
                {
                    .product_flags =
                        LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ |
                        LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_FTZ,
                },
        },
    [LOOM_TEST_GROUPED_MATH_EVALUATION_BF16_BOUNDARY - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT,
            .product_element_type = LOOM_SCALAR_TYPE_BF16,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
            .packet_lane_count = 16,
            .conversion =
                {
                    .input_flags = LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ,
                    .product_flags =
                        LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ |
                        LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_FTZ,
                    .result_flags = LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ |
                                    LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_FTZ,
                },
        },
};

static const loom_target_math_evaluation_t kTestF16x4MathEvaluations[] = {
    [LOOM_TEST_GROUPED_MATH_EVALUATION_DEFAULT - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT,
            .product_element_type = LOOM_SCALAR_TYPE_F16,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
            .packet_lane_count = 4,
        },
    [LOOM_TEST_GROUPED_MATH_EVALUATION_BF16_BOUNDARY - 1] =
        {
            .kind = LOOM_TARGET_MATH_EVALUATION_GROUPED_PRODUCT,
            .product_element_type = LOOM_SCALAR_TYPE_F16,
            .accumulator_element_type = LOOM_SCALAR_TYPE_F32,
            .packet_lane_count = 4,
            .conversion =
                {
                    .input_flags = LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ,
                    .result_flags = LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_DAZ |
                                    LOOM_TARGET_MATH_FLOAT_CONVERSION_FLAG_FTZ,
                },
        },
};

#define LOOM_TEST_GROUPED_MATH_RULES(_)                                     \
  _(GELU_TANH, GELUF_TANH, GELU_TANH_F32, RECIPE_GELU_TANH)                 \
  _(GELU_LOGISTIC, GELUF_LOGISTIC, GELU_LOGISTIC_F32, RECIPE_GELU_LOGISTIC) \
  _(SILU, SILUF, SILU_LOGISTIC_F32, RECIPE_SILU_LOGISTIC)                   \
  _(LOGISTIC, LOGISTICF, LOGISTIC_TANH_F32, RECIPE_LOGISTIC_TANH)

enum loom_test_grouped_math_rule_e {
#define LOOM_TEST_GROUPED_MATH_RULE_ENUM(name, op, recipe, constraint) \
  LOOM_TEST_GROUPED_MATH_RULE_##name##_OTHER,                          \
      LOOM_TEST_GROUPED_MATH_RULE_##name##_BF16,
  LOOM_TEST_GROUPED_MATH_RULES(LOOM_TEST_GROUPED_MATH_RULE_ENUM)
#undef LOOM_TEST_GROUPED_MATH_RULE_ENUM
      LOOM_TEST_GROUPED_MATH_RULE_COUNT_,
};

static const loom_target_math_policy_rule_t
    kTestGroupedMathRules[LOOM_TEST_GROUPED_MATH_RULE_COUNT_] = {
#define LOOM_TEST_GROUPED_MATH_RULE_ROW(name_, op_, recipe_, constraint_) \
  [LOOM_TEST_GROUPED_MATH_RULE_##name_##_OTHER] =                         \
      {                                                                   \
          .match = LOOM_TEST_MATH_MATCH(LOOM_SCALAR_TYPE_SET_FLOAT &      \
                                        ~LOOM_SCALAR_TYPE_SET_BF16),      \
          .recipe = LOOM_TARGET_MATH_RECIPE_##recipe_,                    \
          .evaluation = LOOM_TEST_GROUPED_MATH_EVALUATION_DEFAULT,        \
          .decision_constraint = LOOM_TEST_MATH_CONSTRAINT_##constraint_, \
  },                                                                      \
  [LOOM_TEST_GROUPED_MATH_RULE_##name_##_BF16] = {                        \
      .match = LOOM_TEST_MATH_MATCH(LOOM_SCALAR_TYPE_SET_BF16),           \
      .recipe = LOOM_TARGET_MATH_RECIPE_##recipe_,                        \
      .evaluation = LOOM_TEST_GROUPED_MATH_EVALUATION_BF16_BOUNDARY,      \
      .decision_constraint = LOOM_TEST_MATH_CONSTRAINT_##constraint_,     \
  },
        LOOM_TEST_GROUPED_MATH_RULES(LOOM_TEST_GROUPED_MATH_RULE_ROW)
#undef LOOM_TEST_GROUPED_MATH_RULE_ROW
};

static const loom_target_math_policy_direct_rule_t
    kTestGroupedMathDirectRules[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_TEST_GROUPED_MATH_NATIVE_RULE(op_)                            \
  [LOOM_TARGET_MATH_OP_##op_] = LOOM_TARGET_MATH_POLICY_DIRECT_RULE(       \
      LOOM_SCALAR_TYPE_SET_FLOAT, LOOM_TARGET_MATH_LANE_DOMAIN_SET_ALL,    \
      LOOM_TARGET_MATH_FASTMATH_FLAG_AFN, LOOM_TARGET_MATH_RECIPE_UNKNOWN, \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE,                                 \
      LOOM_TARGET_MATH_FASTMATH_FLAG_NONE, 0,                              \
      LOOM_TEST_MATH_CONSTRAINT_GROUPED_NATIVE,                            \
      LOOM_TEST_MATH_CONSTRAINT_GROUPED_REQUIRES_AFN)
        LOOM_TEST_GROUPED_MATH_NATIVE_RULE(TANHF),
        LOOM_TEST_GROUPED_MATH_NATIVE_RULE(ADDF),
#undef LOOM_TEST_GROUPED_MATH_NATIVE_RULE
};

static const loom_target_math_policy_rule_span_t
    kTestGroupedMathOperationSpans[LOOM_TARGET_MATH_OP_COUNT] = {
#define LOOM_TEST_GROUPED_MATH_SPAN(name, op, recipe, constraint)       \
  [LOOM_TARGET_MATH_OP_##op] = {                                        \
      .rule_start = LOOM_TEST_GROUPED_MATH_RULE_##name##_OTHER,         \
      .rule_count = 2,                                                  \
      .fallback_constraint = LOOM_TEST_MATH_CONSTRAINT_GROUPED_FIXTURE, \
  },
        LOOM_TEST_GROUPED_MATH_RULES(LOOM_TEST_GROUPED_MATH_SPAN)
#undef LOOM_TEST_GROUPED_MATH_SPAN
};

#undef LOOM_TEST_GROUPED_MATH_RULES

#define LOOM_TEST_GROUPED_MATH_POLICY_TABLE(evaluations_)                      \
  {                                                                            \
      .operation_spans = kTestGroupedMathOperationSpans,                       \
      .rules = kTestGroupedMathRules,                                          \
      .default_constraint = LOOM_TEST_MATH_CONSTRAINT_GROUPED_FIXTURE,         \
      .required_fastmath_flags = LOOM_TARGET_MATH_FASTMATH_FLAG_AFN,           \
      .permission_constraint = LOOM_TEST_MATH_CONSTRAINT_GROUPED_REQUIRES_AFN, \
      .constraints = kTestMathConstraints,                                     \
      .constraint_strings =                                                    \
          {                                                                    \
              .data = {(const char*)&kTestMathConstraintStrings},              \
              .data_length = {sizeof(kTestMathConstraintStrings)},             \
          },                                                                   \
      .evaluations = (evaluations_),                                           \
  }

static const loom_target_math_policy_table_t kTestBf16x16MathPolicyTable =
    LOOM_TEST_GROUPED_MATH_POLICY_TABLE(kTestBf16x16MathEvaluations);

static const loom_target_math_policy_table_t kTestF16x4MathPolicyTable =
    LOOM_TEST_GROUPED_MATH_POLICY_TABLE(kTestF16x4MathEvaluations);

#undef LOOM_TEST_GROUPED_MATH_POLICY_TABLE

static const loom_target_math_policy_match_t kTestPreferFmaForms[] = {
    LOOM_TEST_MATH_MATCH(LOOM_SCALAR_TYPE_SET_FLOAT),
};

#undef LOOM_TEST_MATH_MATCH

static const loom_target_math_policy_table_t kTestPreferFmaPolicyTable = {
    .fma_forms = kTestPreferFmaForms,
    .fma_form_count = IREE_ARRAYSIZE(kTestPreferFmaForms),
    .default_constraint = LOOM_TEST_MATH_CONSTRAINT_GROUPED_FIXTURE,
    .constraints = kTestMathConstraints,
    .constraint_strings =
        {
            .data = {(const char*)&kTestMathConstraintStrings},
            .data_length = {sizeof(kTestMathConstraintStrings)},
        },
};

static void loom_test_math_policy_registry_initialize(
    loom_target_math_policy_registry_t* out_registry) {
  static const loom_target_math_policy_t kWidenF32RoundPolicy = {
      .name = IREE_SVL("test-widen-f32-round"),
      .direct_rules = kTestWidenMathDirectRules,
      .table = &kTestWidenMathPolicyTable,
  };
  static const loom_target_math_policy_t kBf16x16EvaluationPolicy = {
      .name = IREE_SVL("test-grouped-product-bf16x16"),
      .direct_rules = kTestGroupedMathDirectRules,
      .table = &kTestBf16x16MathPolicyTable,
  };
  static const loom_target_math_policy_t kF16x4EvaluationPolicy = {
      .name = IREE_SVL("test-grouped-product-f16x4"),
      .direct_rules = kTestGroupedMathDirectRules,
      .table = &kTestF16x4MathPolicyTable,
  };
  static const loom_target_math_policy_t kPreferFmaPolicy = {
      .name = IREE_SVL("test-prefer-fma"),
      .table = &kTestPreferFmaPolicyTable,
  };
  static const loom_target_math_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_stable_id = UINT64_C(0x382977ec10cc5a09),
          .policy = &kWidenF32RoundPolicy,
      },
      {
          .contract_set_stable_id = UINT64_C(0x145802c242758eb7),
          .policy = &kBf16x16EvaluationPolicy,
      },
      {
          .contract_set_stable_id = UINT64_C(0x23083cb600290ce9),
          .policy = &kF16x4EvaluationPolicy,
      },
      {
          .contract_set_stable_id = UINT64_C(0x5ddd1af00c682c82),
          .policy = &kPreferFmaPolicy,
      },
  };
  loom_target_math_policy_registry_initialize_from_entries(
      kEntries, IREE_ARRAYSIZE(kEntries), out_registry);
}

// Device construction has no native callable ABI. Its independently executing
// low_core workers do, so source policy must distinguish those environments.
static loom_target_call_policy_t loom_test_select_call_policy(
    const loom_resolved_target_t* target, const loom_module_t* module,
    loom_call_like_kind_t kind, loom_call_like_t call,
    loom_func_like_t callee) {
  (void)module;
  (void)call;
  (void)callee;
  switch (target->facts->selector) {
    case LOOM_TEST_TARGET_KIND_DEVICE:
      return LOOM_TARGET_CALL_POLICY_REQUIRE_INLINE;
    case LOOM_TEST_TARGET_KIND_LOW_LEAF:
      return LOOM_TARGET_CALL_POLICY_REQUIRE_LEAF |
             (kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL
                  ? LOOM_TARGET_CALL_POLICY_NO_RESOURCES |
                        LOOM_TARGET_CALL_POLICY_NO_STORAGE
                  : 0);
    default:
      return LOOM_TARGET_CALL_POLICY_DIRECT;
  }
}

const loom_target_provider_t loom_test_target_provider = {
    .select_call_policy = loom_test_select_call_policy,
    .register_context = loom_test_dialect_register,
    .initialize_low_descriptor_registry =
        loom_target_core_test_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_test_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_test_math_policy_registry_initialize,
    .pass_registry = &loom_test_pass_registry_storage,
    .view_boundary_carrier = LOOM_TARGET_VIEW_BOUNDARY_CARRIER_BUFFER_OFFSET,
    .loop_predicate_carrier = LOOM_TARGET_LOOP_PREDICATE_CARRIER_I32,
    .loop_predicate_max_vector_element_count = 32,
    .target_fact_type = &loom_test_target_fact_type,
};
