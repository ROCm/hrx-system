// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/view/offset_expression.h"

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

static iree_status_t loom_view_offset_constant(loom_builder_t* builder,
                                               int64_t value,
                                               loom_location_id_t location,
                                               loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_I64),
      location, &op));
  *out_value = loom_scalar_constant_result(op);
  return iree_ok_status();
}

iree_status_t loom_view_materialize_offset_expression(
    loom_builder_t* builder, const loom_symbolic_expr_t* expression,
    loom_value_id_t base_value_id, loom_value_id_t anchor_value_id,
    loom_value_id_t* out_value) {
  if (base_value_id != LOOM_VALUE_ID_INVALID && expression->constant == 0 &&
      expression->term_count == 0) {
    *out_value = base_value_id;
    return iree_ok_status();
  }
  const loom_value_t* view =
      loom_module_value(builder->module, anchor_value_id);
  const loom_op_t* anchor = loom_value_is_block_arg(view)
                                ? loom_value_def_block(view)->first_op
                                : loom_value_def_op(view);
  if (loom_value_is_block_arg(view)) {
    loom_builder_set_before(builder, anchor);
  } else {
    loom_builder_set_after(builder, anchor);
  }
  const loom_location_id_t location = anchor->location;
  // Reuse the complete producer spelling. Its SSA identity carries facts
  // established for the aggregate expression that a term-by-term
  // reconstruction cannot attach to any one operand. The caller-owned
  // dominance contract guarantees that this value is available at the anchor.
  if (base_value_id == LOOM_VALUE_ID_INVALID &&
      loom_symbolic_expr_has_relation_value(expression)) {
    const loom_value_id_t value = expression->relation_value_id;
    const loom_type_t source_type =
        loom_module_value_type(builder->module, value);
    const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
    if (loom_type_equal(source_type, offset_type)) {
      *out_value = value;
      return iree_ok_status();
    }
    loom_op_t* cast = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cast_build(builder, value, source_type,
                                               offset_type, location, &cast));
    *out_value = loom_index_cast_result(cast);
    return iree_ok_status();
  }
  // A constant translation stays in the physical address domain. Converting
  // a carried base to signed arithmetic and back would require a new range
  // proof for the entire recurrence instead of just representing its step.
  // INT64_MIN has no nonnegative magnitude in the source offset domain.
  if (base_value_id != LOOM_VALUE_ID_INVALID && expression->term_count == 0 &&
      expression->constant != INT64_MIN) {
    const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
    const int64_t magnitude =
        expression->constant < 0 ? -expression->constant : expression->constant;
    loom_op_t* constant = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(magnitude), type, location, &constant));
    loom_op_t* translation = NULL;
    if (expression->constant < 0) {
      IREE_RETURN_IF_ERROR(loom_index_sub_build(
          builder, base_value_id, loom_index_constant_result(constant), type,
          location, &translation));
    } else {
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          builder, base_value_id, loom_index_constant_result(constant), type,
          location, &translation));
    }
    *out_value = loom_op_results(translation)[0];
    return iree_ok_status();
  }
  // A sign-known displacement is independent of the opaque carried base.
  // Keep the final translation in the offset domain so signed range analysis
  // need not prove that the complete address avoids overflowing i64.
  const bool subtract = expression->facts.range_lo > INT64_MIN &&
                        expression->facts.range_lo < 0 &&
                        expression->facts.range_hi <= 0;
  const bool translate = base_value_id != LOOM_VALUE_ID_INVALID &&
                         (expression->facts.range_lo >= 0 || subtract);
  loom_value_id_t sum = translate ? LOOM_VALUE_ID_INVALID : base_value_id;
  // Physical byte expressions use the full offset width. Logical index
  // carriers may be narrower on the selected target. Signed intermediates
  // also preserve negative affine coefficients before the complete offset.
  const loom_type_t arithmetic_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  if (sum != LOOM_VALUE_ID_INVALID) {
    loom_op_t* cast = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cast_build(
        builder, sum, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
        arithmetic_type, location, &cast));
    sum = loom_index_cast_result(cast);
  }
  if (expression->constant != 0 ||
      (sum == LOOM_VALUE_ID_INVALID && expression->term_count == 0)) {
    loom_value_id_t constant;
    IREE_RETURN_IF_ERROR(loom_view_offset_constant(
        builder, expression->constant, location, &constant));
    if (sum == LOOM_VALUE_ID_INVALID) {
      sum = constant;
    } else {
      loom_op_t* add = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
          builder, 0, sum, constant, arithmetic_type, location, &add));
      sum = loom_scalar_addi_result(add);
    }
  }
  for (iree_host_size_t i = 0; i < expression->term_count; ++i) {
    const loom_symbolic_term_t* term = &expression->terms[i];
    // The identity representative retains predicates established at its
    // producer while denoting the same numeric term as the canonical value.
    loom_value_id_t value = term->relation_value_id;
    const loom_type_t type = loom_module_value_type(builder->module, value);
    if (!loom_type_equal(type, arithmetic_type)) {
      loom_op_t* cast = NULL;
      const loom_scalar_type_t scalar_type = loom_type_element_type(type);
      if (scalar_type == LOOM_SCALAR_TYPE_I1) {
        IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
            builder, value, type, arithmetic_type, location, &cast));
      } else if (loom_scalar_type_is_integer(scalar_type)) {
        IREE_RETURN_IF_ERROR(loom_scalar_extsi_build(
            builder, value, type, arithmetic_type, location, &cast));
      } else {
        IREE_RETURN_IF_ERROR(loom_index_cast_build(
            builder, value, type, arithmetic_type, location, &cast));
      }
      value = loom_op_results(cast)[0];
    }
    if (term->coefficient != 1) {
      loom_value_id_t coefficient;
      IREE_RETURN_IF_ERROR(loom_view_offset_constant(builder, term->coefficient,
                                                     location, &coefficient));
      loom_op_t* multiply = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_muli_build(builder, 0, value,
                                                  coefficient, arithmetic_type,
                                                  location, &multiply));
      value = loom_scalar_muli_result(multiply);
    }
    if (sum == LOOM_VALUE_ID_INVALID) {
      sum = value;
    } else {
      loom_op_t* add = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
          builder, 0, sum, value, arithmetic_type, location, &add));
      sum = loom_scalar_addi_result(add);
    }
  }
  if (translate && subtract) {
    loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_view_offset_constant(builder, 0, location, &zero));
    loom_op_t* negate = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
        builder, 0, zero, sum, arithmetic_type, location, &negate));
    sum = loom_scalar_subi_result(negate);
  }
  loom_op_t* cast = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      builder, sum, arithmetic_type, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      location, &cast));
  *out_value = loom_index_cast_result(cast);
  if (translate) {
    const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
    loom_op_t* translation = NULL;
    if (subtract) {
      IREE_RETURN_IF_ERROR(loom_index_sub_build(
          builder, base_value_id, *out_value, type, location, &translation));
    } else {
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          builder, base_value_id, *out_value, type, location, &translation));
    }
    *out_value = loom_op_results(translation)[0];
  }
  return iree_ok_status();
}
