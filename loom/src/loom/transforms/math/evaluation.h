// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-neutral representations used to evaluate semantic math recipes.

#ifndef LOOM_TRANSFORMS_MATH_EVALUATION_H_
#define LOOM_TRANSFORMS_MATH_EVALUATION_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/math_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_math_evaluation_lane_ops_t loom_math_evaluation_lane_ops_t;

typedef enum loom_math_evaluation_unary_op_e {
  LOOM_MATH_EVALUATION_UNARY_ABSF = 0,
  LOOM_MATH_EVALUATION_UNARY_EXP2F = 1,
  LOOM_MATH_EVALUATION_UNARY_LOG2F = 2,
  LOOM_MATH_EVALUATION_UNARY_SINTURNSF = 3,
  LOOM_MATH_EVALUATION_UNARY_COSTURNSF = 4,
  LOOM_MATH_EVALUATION_UNARY_LOGISTICF = 5,
  LOOM_MATH_EVALUATION_UNARY_TANHF = 6,
  LOOM_MATH_EVALUATION_UNARY_TRUNCF = 7,
  LOOM_MATH_EVALUATION_UNARY_SQRTF = 8,
  LOOM_MATH_EVALUATION_UNARY_RSQRTF = 9,
  LOOM_MATH_EVALUATION_UNARY_SINF = 10,
  LOOM_MATH_EVALUATION_UNARY_COSF = 11,
  LOOM_MATH_EVALUATION_UNARY_ROUNDEVENF = 12,
} loom_math_evaluation_unary_op_t;

typedef enum loom_math_evaluation_binary_op_e {
  LOOM_MATH_EVALUATION_BINARY_ADDF = 0,
  LOOM_MATH_EVALUATION_BINARY_SUBF = 1,
  LOOM_MATH_EVALUATION_BINARY_MULF = 2,
  LOOM_MATH_EVALUATION_BINARY_DIVF = 3,
  LOOM_MATH_EVALUATION_BINARY_COPYSIGNF = 4,
} loom_math_evaluation_binary_op_t;

typedef struct loom_math_evaluation_value_t {
  // Value represented in the evaluation accumulator domain.
  loom_value_id_t accumulator;
  // Lazily materialized grouped-product operand, or invalid when absent.
  loom_value_id_t product;
} loom_math_evaluation_value_t;

typedef struct loom_math_evaluation_t {
  // Authored primary operand retained for source-fact queries.
  loom_value_id_t source_input;
  // Imported primary source operand.
  loom_math_evaluation_value_t input;
  // Imported secondary source operand for binary recipes.
  loom_math_evaluation_value_t secondary_input;
  // Authored source result type restored after recipe evaluation.
  loom_type_t source_result_type;
  // Type shared by evaluation constants and accumulator values.
  loom_type_t value_type;
  // Narrow operand type before grouped-product pairing.
  loom_type_t product_type;
  // Interleaved grouped-product operand type consumed by vector.dot2f.
  loom_type_t paired_product_type;
  // Number of logical elements represented by the source value.
  uint64_t source_element_count;
  // Source fast-math flags forwarded to replacement floating-point ops.
  uint8_t fastmath_flags;
  // Target-authorized flags used only by recipe-internal operations.
  uint8_t recipe_fastmath_flags;
  // Source location copied to replacement operations.
  loom_location_id_t location;
  // Policy-selected representation descriptor.
  loom_target_math_evaluation_t descriptor;
  // Scalar or vector operation table for the selected representation.
  const loom_math_evaluation_lane_ops_t* lane_ops;
  // Rewriter owning emitted operations and value facts.
  loom_rewriter_t* rewriter;
  // Lazily materialized zero in the accumulator domain.
  loom_value_id_t zero_accumulator;
  // Lazily materialized zero in the narrow product domain.
  loom_value_id_t zero_product;
} loom_math_evaluation_t;

// Imports the source operands into the representation selected by the policy
// decision. The source representation is preserved when no alternate
// evaluation was selected.
iree_status_t loom_math_evaluation_initialize(
    const loom_module_t* module, const loom_target_math_query_t* query,
    const loom_target_math_policy_decision_t* decision, const loom_op_t* op,
    loom_rewriter_t* rewriter, loom_math_evaluation_t* out_evaluation);

// Restores |value| to the authored source result type and shape.
iree_status_t loom_math_evaluation_export(
    loom_math_evaluation_t* evaluation,
    const loom_math_evaluation_value_t* value, loom_value_id_t* out_value);

// Builds a uniform constant in the evaluation accumulator representation.
iree_status_t loom_math_evaluation_build_constant(
    loom_math_evaluation_t* evaluation, double value,
    loom_math_evaluation_value_t* out_value);

// Builds one scalar or lane-wise unary operation in the evaluation domain.
iree_status_t loom_math_evaluation_build_unary(
    loom_math_evaluation_t* evaluation,
    loom_math_evaluation_unary_op_t unary_op,
    const loom_math_evaluation_value_t* input,
    loom_math_evaluation_value_t* out_value);

// Builds one scalar or lane-wise binary operation. Grouped-product evaluation
// routes multiplication through its narrow product representation.
iree_status_t loom_math_evaluation_build_binary(
    loom_math_evaluation_t* evaluation,
    loom_math_evaluation_binary_op_t binary_op,
    loom_math_evaluation_value_t* lhs, loom_math_evaluation_value_t* rhs,
    loom_math_evaluation_value_t* out_value);

// Builds a fused product and sum in the selected evaluation representation.
iree_status_t loom_math_evaluation_build_fma(
    loom_math_evaluation_t* evaluation, loom_math_evaluation_value_t* lhs,
    loom_math_evaluation_value_t* rhs,
    const loom_math_evaluation_value_t* accumulator,
    loom_math_evaluation_value_t* out_value);

// Clamps |value| to uniform floating-point bounds.
iree_status_t loom_math_evaluation_build_clamp(
    loom_math_evaluation_t* evaluation, loom_math_evaluation_value_t* value,
    double lower, double upper, loom_math_evaluation_value_t* out_value);

// Compares two values with ordered-greater-or-equal semantics.
iree_status_t loom_math_evaluation_build_ordered_greater_equal(
    loom_math_evaluation_t* evaluation, const loom_math_evaluation_value_t* lhs,
    const loom_math_evaluation_value_t* rhs, loom_value_id_t* out_value);

// Compares two values with ordered-equal semantics.
iree_status_t loom_math_evaluation_build_ordered_equal(
    loom_math_evaluation_t* evaluation, const loom_math_evaluation_value_t* lhs,
    const loom_math_evaluation_value_t* rhs, loom_value_id_t* out_value);

// Classifies positive normal finite lanes using their IEEE bit interval.
iree_status_t loom_math_evaluation_build_positive_normal_finite_condition(
    loom_math_evaluation_t* evaluation,
    const loom_math_evaluation_value_t* input, loom_value_id_t* out_condition);

// Selects between two values using a scalar or lane-wise condition.
iree_status_t loom_math_evaluation_build_select(
    loom_math_evaluation_t* evaluation, loom_value_id_t condition,
    const loom_math_evaluation_value_t* true_value,
    const loom_math_evaluation_value_t* false_value,
    loom_math_evaluation_value_t* out_value);

static inline loom_value_id_t loom_math_evaluation_value_id(
    const loom_math_evaluation_value_t* value) {
  return value->accumulator;
}

static inline loom_math_evaluation_value_t loom_math_evaluation_value_from_id(
    loom_value_id_t value) {
  return (loom_math_evaluation_value_t){
      .accumulator = value,
      .product = LOOM_VALUE_ID_INVALID,
  };
}

static inline loom_math_evaluation_value_t loom_math_evaluation_value_invalid(
    void) {
  return loom_math_evaluation_value_from_id(LOOM_VALUE_ID_INVALID);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_MATH_EVALUATION_H_
