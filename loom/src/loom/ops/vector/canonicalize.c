// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/attribute.h"
#include "loom/ir/facts.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/combining.h"
#include "loom/ops/encoding/auxiliary.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

//===----------------------------------------------------------------------===//
// Fact queries
//===----------------------------------------------------------------------===//

static bool loom_vector_exact_facts_equal(loom_value_facts_t lhs,
                                          loom_value_facts_t rhs) {
  if (!loom_value_facts_is_exact(lhs) || !loom_value_facts_is_exact(rhs)) {
    return false;
  }
  return loom_value_facts_equal(lhs, rhs);
}

static bool loom_vector_exact_i64_facts_match(loom_value_facts_t facts,
                                              int64_t expected) {
  return loom_value_facts_is_exact(facts) &&
         !loom_value_facts_is_float(facts) && facts.range_lo == expected;
}

static bool loom_vector_query_all_equal_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_facts_t* out_element) {
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(context, facts, &uniform)) {
    *out_element = uniform.element;
    return true;
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_value_facts_query_small_static_lanes(context, facts, &lanes) ||
      lanes.count == 0 || !loom_value_facts_is_exact(lanes.lanes[0])) {
    return false;
  }
  for (iree_host_size_t i = 1; i < lanes.count; ++i) {
    if (!loom_vector_exact_facts_equal(lanes.lanes[0], lanes.lanes[i])) {
      return false;
    }
  }
  *out_element = lanes.lanes[0];
  return true;
}

static bool loom_vector_value_is_all_exact_i64(const loom_rewriter_t* rewriter,
                                               loom_value_id_t value_id,
                                               int64_t expected_value) {
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, value_id);
  loom_value_facts_t element = {0};
  if (!loom_vector_query_all_equal_element(&rewriter->fact_table->context,
                                           facts, &element)) {
    return false;
  }
  return loom_value_facts_is_exact(element) &&
         !loom_value_facts_is_float(element) &&
         element.range_lo == expected_value;
}

static bool loom_vector_query_all_exact_float(const loom_rewriter_t* rewriter,
                                              loom_value_id_t value_id,
                                              double* out_value) {
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, value_id);
  loom_value_facts_t element = {0};
  if (!loom_vector_query_all_equal_element(&rewriter->fact_table->context,
                                           facts, &element)) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(rewriter->module, value_id);
  return loom_value_facts_as_exact_float(loom_type_element_type(type), element,
                                         out_value);
}

static bool loom_vector_value_is_all_exact_float(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id,
    double expected_value) {
  double actual_value = 0.0;
  return loom_vector_query_all_exact_float(rewriter, value_id, &actual_value) &&
         actual_value == expected_value;
}

static bool loom_vector_value_is_unit_stride_iota(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id,
    loom_type_t vector_type) {
  if (!loom_type_is_vector(vector_type) || loom_type_rank(vector_type) != 1) {
    return false;
  }

  loom_value_fact_vector_iota_t iota = {0};
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, value_id);
  if (loom_value_facts_query_vector_iota(&rewriter->fact_table->context, facts,
                                         &iota)) {
    return loom_vector_exact_i64_facts_match(iota.base, 0) &&
           loom_vector_exact_i64_facts_match(iota.step, 1);
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_value_facts_query_small_static_lanes(&rewriter->fact_table->context,
                                                 facts, &lanes)) {
    return false;
  }
  uint64_t lane_count = 0;
  if (!loom_type_static_element_count(vector_type, &lane_count) ||
      lane_count != lanes.count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < lanes.count; ++i) {
    if (!loom_vector_exact_i64_facts_match(lanes.lanes[i], (int64_t)i)) {
      return false;
    }
  }
  return true;
}

static bool loom_vector_value_is_all_bool(const loom_rewriter_t* rewriter,
                                          loom_value_id_t value_id,
                                          bool expected) {
  return loom_vector_value_is_all_exact_i64(rewriter, value_id,
                                            expected ? 1 : 0);
}

static bool loom_vector_value_is_workgroup_view(const loom_rewriter_t* rewriter,
                                                loom_value_id_t value_id) {
  if (!rewriter->fact_table) return false;
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_value_facts_query_view_reference(
          &rewriter->fact_table->context,
          loom_rewriter_value_facts(rewriter, value_id), &reference)) {
    return false;
  }
  return reference.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
}

static bool loom_vector_facts_to_constant_attr(loom_value_facts_t facts,
                                               loom_scalar_type_t element_type,
                                               loom_attribute_t* out_attr) {
  if (!loom_value_facts_is_exact(facts)) return false;

  if (loom_scalar_type_is_float(element_type)) {
    double value = 0.0;
    if (!loom_value_facts_as_exact_float(element_type, facts, &value)) {
      return false;
    }
    *out_attr = loom_attr_f64(value);
    return true;
  }

  if (loom_value_facts_is_float(facts)) return false;
  if (element_type == LOOM_SCALAR_TYPE_I1) {
    if (facts.range_lo != 0 && facts.range_lo != 1) return false;
    *out_attr = loom_attr_bool(facts.range_lo != 0);
    return true;
  }

  *out_attr = loom_attr_i64(facts.range_lo);
  return true;
}

//===----------------------------------------------------------------------===//
// IR helpers
//===----------------------------------------------------------------------===//

static bool loom_vector_value_def_op(const loom_rewriter_t* rewriter,
                                     loom_value_id_t value_id,
                                     loom_op_t** out_def_op) {
  if (value_id >= rewriter->module->values.count) return false;
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op || (def_op->flags & LOOM_OP_FLAG_DEAD)) return false;
  *out_def_op = def_op;
  return true;
}

static bool loom_vector_index_value_as_static_index(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id,
    loom_type_t source_type, uint16_t axis, int64_t* out_static_index) {
  int64_t static_index = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_rewriter_value_facts(rewriter, value_id), &static_index) ||
      static_index < 0) {
    return false;
  }
  if (loom_type_is_vector(source_type) && axis < loom_type_rank(source_type) &&
      !loom_type_dim_is_dynamic_at(source_type, (uint8_t)axis) &&
      static_index >=
          loom_type_dim_static_size_at(source_type, (uint8_t)axis)) {
    return false;
  }
  *out_static_index = static_index;
  return true;
}

static bool loom_vector_static_ordinal_from_indices(
    loom_type_t type, const int64_t* indices, iree_host_size_t* out_ordinal) {
  iree_host_size_t ordinal = 0;
  uint8_t rank = loom_type_rank(type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    int64_t dimension_size = loom_type_dim_static_size_at(type, axis);
    if (dimension_size < 0 || indices[axis] < 0 ||
        indices[axis] >= dimension_size) {
      return false;
    }
    if (!iree_host_size_checked_mul(ordinal, (iree_host_size_t)dimension_size,
                                    &ordinal) ||
        !iree_host_size_checked_add(ordinal, (iree_host_size_t)indices[axis],
                                    &ordinal)) {
      return false;
    }
  }
  *out_ordinal = ordinal;
  return true;
}

static iree_status_t loom_vector_replace_single_result_with_value(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_vector_replace_single_result_with_constant_attr(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_type_t result_type,
    loom_attribute_t attr) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &rewriter->builder, attr, result_type, op->location, &constant_op));
  loom_value_id_t replacement = loom_vector_constant_result(constant_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_vector_replace_single_result_with_constant_facts(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_facts_t element_facts,
    loom_type_t result_type) {
  loom_attribute_t attr = {0};
  if (!loom_vector_facts_to_constant_attr(
          element_facts, loom_type_element_type(result_type), &attr)) {
    return iree_ok_status();
  }
  return loom_vector_replace_single_result_with_constant_attr(
      op, rewriter, result_type, attr);
}

static iree_status_t loom_vector_replace_single_result_with_splat(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t scalar,
    loom_type_t result_type) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* splat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      &rewriter->builder, scalar, result_type, op->location, &splat_op));
  loom_value_id_t replacement = loom_vector_splat_result(splat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_vector_replace_single_result_with_negf(
    loom_op_t* op, loom_rewriter_t* rewriter, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_negf_build(&rewriter->builder, instance_flags, input,
                             result_type, op->location, &replacement_op));
  loom_value_id_t replacement = loom_vector_negf_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_vector_replace_single_result_with_scalar_i64(
    loom_op_t* op, loom_rewriter_t* rewriter, int64_t value,
    loom_type_t result_type) {
  if (!rewriter->materialize_constant) {
    return iree_ok_status();
  }
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_build_constant(rewriter, loom_value_facts_exact_i64(value),
                                   result_type, op->location, &replacement));

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_vector_build_iota_lane_symbolic(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t base,
    loom_value_id_t step, iree_host_size_t lane, loom_type_t result_type,
    loom_value_id_t* out_value, loom_value_id_t value_checkpoint) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (lane == 0) {
    *out_value = base;
    return iree_ok_status();
  }

  int64_t step_value = 0;
  if (loom_value_facts_as_exact_i64(loom_rewriter_value_facts(rewriter, step),
                                    &step_value) &&
      step_value == 0) {
    *out_value = base;
    return iree_ok_status();
  }

  loom_scalar_type_t element_type = loom_type_element_type(result_type);
  loom_value_id_t scaled_step = step;
  loom_value_id_t lane_value = LOOM_VALUE_ID_INVALID;
  if (lane > 1) {
    if (!rewriter->materialize_constant || lane > (iree_host_size_t)INT64_MAX) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_build_constant(
        rewriter, loom_value_facts_exact_i64((int64_t)lane), result_type,
        op->location, &lane_value));
    if (element_type != LOOM_SCALAR_TYPE_INDEX) {
      loom_op_t* multiply_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_muli_build(
          &rewriter->builder, /*instance_flags=*/0, step, lane_value,
          result_type, op->location, &multiply_op));
      scaled_step = loom_scalar_muli_result(multiply_op);
    }
  }

  int64_t base_value = 0;
  if (loom_value_facts_as_exact_i64(loom_rewriter_value_facts(rewriter, base),
                                    &base_value) &&
      base_value == 0) {
    if (element_type == LOOM_SCALAR_TYPE_INDEX && lane > 1) {
      loom_op_t* multiply_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_mul_build(
          &rewriter->builder, step, lane_value, op->location, &multiply_op));
      scaled_step = loom_index_mul_result(multiply_op);
    }
    *out_value = scaled_step;
    return loom_rewriter_preserve_result_names_on_new_values(
        rewriter, op, out_value, 1, value_checkpoint);
  }

  loom_op_t* add_op = NULL;
  if (element_type == LOOM_SCALAR_TYPE_INDEX) {
    if (lane > 1) {
      IREE_RETURN_IF_ERROR(loom_index_madd_build(
          &rewriter->builder, step, lane_value, base, op->location, &add_op));
    } else {
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          &rewriter->builder, base, step, result_type, op->location, &add_op));
    }
    *out_value = loom_op_const_results(add_op)[0];
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
        &rewriter->builder, /*instance_flags=*/0, base, scaled_step,
        result_type, op->location, &add_op));
    *out_value = loom_scalar_addi_result(add_op);
  }

  return loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, out_value, 1, value_checkpoint);
}

static bool loom_vector_get_single_result_type(const loom_rewriter_t* rewriter,
                                               const loom_op_t* op,
                                               loom_type_t* out_result_type) {
  if (op->result_count != 1) return false;
  loom_value_id_t result = loom_op_const_results(op)[0];
  if (result >= rewriter->module->values.count) return false;
  *out_result_type = loom_module_value_type(rewriter->module, result);
  return true;
}

static int64_t loom_vector_integer_all_ones(loom_scalar_type_t element_type) {
  return element_type == LOOM_SCALAR_TYPE_I1 ? 1 : -1;
}

static bool loom_vector_reduce_integer_identity(loom_combining_kind_t kind,
                                                loom_type_t input_type,
                                                int64_t* out_identity) {
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
    case LOOM_COMBINING_KIND_ORI:
    case LOOM_COMBINING_KIND_XORI:
      *out_identity = 0;
      return true;
    case LOOM_COMBINING_KIND_MULI:
      *out_identity = 1;
      return true;
    case LOOM_COMBINING_KIND_ANDI:
      *out_identity =
          loom_vector_integer_all_ones(loom_type_element_type(input_type));
      return true;
    case LOOM_COMBINING_KIND_ADDF:
    case LOOM_COMBINING_KIND_MULF:
    case LOOM_COMBINING_KIND_MINSI:
    case LOOM_COMBINING_KIND_MAXSI:
    case LOOM_COMBINING_KIND_MINUI:
    case LOOM_COMBINING_KIND_MAXUI:
    case LOOM_COMBINING_KIND_MINIMUMF:
    case LOOM_COMBINING_KIND_MAXIMUMF:
    case LOOM_COMBINING_KIND_MINNUMF:
    case LOOM_COMBINING_KIND_MAXNUMF:
    case LOOM_COMBINING_KIND_COUNT_:
      return false;
  }
  return false;
}

static iree_status_t loom_vector_replace_single_result_with_integer_zero(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_type_t result_type) {
  loom_attribute_t attr =
      loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1
          ? loom_attr_bool(false)
          : loom_attr_i64(0);
  return loom_vector_replace_single_result_with_constant_attr(
      op, rewriter, result_type, attr);
}

static iree_status_t loom_vector_replace_single_result_with_new_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* new_op,
    loom_value_id_t value_checkpoint) {
  IREE_ASSERT_EQ(new_op->result_count, 1u);
  loom_value_id_t replacement = loom_op_const_results(new_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static bool loom_vector_type_is_float_vector(loom_type_t type) {
  return loom_type_is_vector(type) &&
         loom_scalar_type_is_float(loom_type_element_type(type));
}

static bool loom_vector_type_query_element_bitwidth(loom_type_t type,
                                                    int32_t* out_bitwidth) {
  if (!loom_vector_type_is_float_vector(type)) return false;
  int32_t bitwidth = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bitwidth <= 0) return false;
  *out_bitwidth = bitwidth;
  return true;
}

static iree_status_t loom_vector_replace_single_result_with_float_cast_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    loom_value_id_t input, loom_type_t input_type) {
  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* replacement_op = NULL;
  switch (kind) {
    case LOOM_OP_VECTOR_EXTF: {
      IREE_RETURN_IF_ERROR(
          loom_vector_extf_build(&rewriter->builder, input, input_type,
                                 result_type, op->location, &replacement_op));
      break;
    }
    case LOOM_OP_VECTOR_FPTRUNC: {
      IREE_RETURN_IF_ERROR(loom_vector_fptrunc_build(
          &rewriter->builder, input, input_type, result_type, op->location,
          &replacement_op));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported replacement vector cast kind");
      IREE_BUILTIN_UNREACHABLE();
  }

  return loom_vector_replace_single_result_with_new_op(
      op, rewriter, replacement_op, value_checkpoint);
}

static iree_status_t loom_vector_replace_single_result_with_splat_float_cast(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    loom_value_id_t scalar, loom_type_t scalar_type) {
  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  loom_type_t scalar_result_type =
      loom_type_scalar(loom_type_element_type(result_type));

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* scalar_cast_op = NULL;
  switch (kind) {
    case LOOM_OP_VECTOR_EXTF: {
      IREE_RETURN_IF_ERROR(loom_scalar_extf_build(
          &rewriter->builder, scalar, scalar_type, scalar_result_type,
          op->location, &scalar_cast_op));
      break;
    }
    case LOOM_OP_VECTOR_FPTRUNC: {
      IREE_RETURN_IF_ERROR(loom_scalar_fptrunc_build(
          &rewriter->builder, scalar, scalar_type, scalar_result_type,
          op->location, &scalar_cast_op));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported replacement vector cast kind");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_op_t* splat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      &rewriter->builder, loom_op_const_results(scalar_cast_op)[0], result_type,
      op->location, &splat_op));
  return loom_vector_replace_single_result_with_new_op(op, rewriter, splat_op,
                                                       value_checkpoint);
}

static iree_status_t loom_vector_replace_single_result_with_float_resize(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t input,
    loom_type_t input_type, bool* out_changed) {
  *out_changed = false;
  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  if (loom_type_equal(input_type, result_type)) {
    *out_changed = true;
    return loom_vector_replace_single_result_with_value(op, rewriter, input);
  }
  if (!loom_vector_type_is_float_vector(input_type) ||
      !loom_vector_type_is_float_vector(result_type) ||
      !loom_type_shape_equals(input_type, result_type)) {
    return iree_ok_status();
  }

  int32_t input_bitwidth = 0;
  int32_t result_bitwidth = 0;
  if (!loom_vector_type_query_element_bitwidth(input_type, &input_bitwidth) ||
      !loom_vector_type_query_element_bitwidth(result_type, &result_bitwidth) ||
      input_bitwidth == result_bitwidth) {
    return iree_ok_status();
  }

  loom_op_kind_t replacement_kind = result_bitwidth > input_bitwidth
                                        ? LOOM_OP_VECTOR_EXTF
                                        : LOOM_OP_VECTOR_FPTRUNC;
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_float_cast_op(
      op, rewriter, replacement_kind, input, input_type));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_float_cast_splat(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    loom_op_t* input_def, bool* out_changed) {
  *out_changed = false;
  if (!loom_vector_splat_isa(input_def)) {
    return iree_ok_status();
  }

  loom_value_id_t scalar = loom_vector_splat_scalar(input_def);
  loom_type_t scalar_type = loom_module_value_type(rewriter->module, scalar);
  loom_type_t result_type = {0};
  if (!loom_type_is_scalar(scalar_type) ||
      !loom_scalar_type_is_float(loom_type_element_type(scalar_type)) ||
      !loom_vector_get_single_result_type(rewriter, op, &result_type) ||
      !loom_vector_type_is_float_vector(result_type)) {
    return iree_ok_status();
  }

  int32_t scalar_bitwidth = 0;
  int32_t result_bitwidth = 0;
  scalar_bitwidth =
      loom_scalar_type_bitwidth(loom_type_element_type(scalar_type));
  if (scalar_bitwidth <= 0 ||
      !loom_vector_type_query_element_bitwidth(result_type, &result_bitwidth)) {
    return iree_ok_status();
  }
  if ((kind == LOOM_OP_VECTOR_EXTF && result_bitwidth <= scalar_bitwidth) ||
      (kind == LOOM_OP_VECTOR_FPTRUNC && result_bitwidth >= scalar_bitwidth)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_splat_float_cast(
      op, rewriter, kind, scalar, scalar_type));
  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Generic vector fact materialization
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_canonicalize_uniform_result(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;
  if (op->kind == LOOM_OP_VECTOR_CONSTANT || op->kind == LOOM_OP_VECTOR_EMPTY ||
      op->kind == LOOM_OP_VECTOR_POISON) {
    return iree_ok_status();
  }
  loom_trait_flags_t traits = loom_op_effective_traits(rewriter->module, op);
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) return iree_ok_status();
  if (loom_traits_are_convergent(traits)) return iree_ok_status();

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type) ||
      !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  loom_value_id_t result = loom_op_const_results(op)[0];
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, result);
  loom_value_facts_t element = {0};
  if (!loom_vector_query_all_equal_element(&rewriter->fact_table->context,
                                           facts, &element) ||
      !loom_value_facts_is_exact(element)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_constant_facts(
      op, rewriter, element, result_type));
  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Construction and access
//===----------------------------------------------------------------------===//

#define LOOM_VECTOR_LANE_CHAIN_MAX_STEPS 8

typedef struct loom_vector_lane_chain_step_t {
  // Scalar unary op kind matched for this lane-chain step.
  loom_op_kind_t scalar_kind;
  // Per-instance flags copied from the scalar op.
  uint8_t instance_flags;
  // Scalar input type observed on the matched scalar op.
  loom_type_t input_type;
  // Scalar result type observed on the matched scalar op.
  loom_type_t result_type;
} loom_vector_lane_chain_step_t;

typedef struct loom_vector_lane_chain_t {
  // Source vector whose static lanes feed every reconstructed lane.
  loom_value_id_t source;
  // Type of the source vector.
  loom_type_t source_type;
  // Number of valid entries in steps.
  iree_host_size_t step_count;
  // Scalar unary steps from outermost producer toward the source extract.
  loom_vector_lane_chain_step_t steps[LOOM_VECTOR_LANE_CHAIN_MAX_STEPS];
} loom_vector_lane_chain_t;

static loom_type_t loom_vector_same_shape_with_element_type(
    loom_type_t shape_type, loom_scalar_type_t element_type) {
  loom_type_t type = shape_type;
  type.header = loom_type_make_header(LOOM_TYPE_VECTOR, element_type,
                                      loom_type_rank(shape_type),
                                      loom_type_flags(shape_type));
  return type;
}

static bool loom_vector_type_is_static_1d_vector(loom_type_t type,
                                                 int64_t* out_lane_count) {
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      loom_type_dim_is_dynamic_at(type, 0)) {
    return false;
  }
  *out_lane_count = loom_type_dim_static_size_at(type, 0);
  return *out_lane_count >= 0;
}

static bool loom_vector_lane_chain_steps_equal(
    const loom_vector_lane_chain_step_t* lhs,
    const loom_vector_lane_chain_step_t* rhs) {
  return lhs->scalar_kind == rhs->scalar_kind &&
         lhs->instance_flags == rhs->instance_flags &&
         loom_type_equal(lhs->input_type, rhs->input_type) &&
         loom_type_equal(lhs->result_type, rhs->result_type);
}

static bool loom_vector_scalar_lane_chain_kind_supported(
    loom_op_kind_t scalar_kind) {
  switch (scalar_kind) {
    case LOOM_OP_SCALAR_CEILF:
    case LOOM_OP_SCALAR_FLOORF:
    case LOOM_OP_SCALAR_ROUNDF:
    case LOOM_OP_SCALAR_ROUNDEVENF:
    case LOOM_OP_SCALAR_TRUNCF:
    case LOOM_OP_SCALAR_FPTOSI:
    case LOOM_OP_SCALAR_FPTOUI:
    case LOOM_OP_SCALAR_TRUNCI:
    case LOOM_OP_SCALAR_SITOFP:
    case LOOM_OP_SCALAR_UITOFP:
      return true;
    default:
      return false;
  }
}

static bool loom_vector_describe_scalar_lane_chain_step(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_vector_lane_chain_step_t* out_step, loom_value_id_t* out_input) {
  *out_step = (loom_vector_lane_chain_step_t){0};
  *out_input = LOOM_VALUE_ID_INVALID;
  if (!loom_vector_scalar_lane_chain_kind_supported(source_op->kind) ||
      source_op->operand_count != 1 || source_op->result_count != 1) {
    return false;
  }
  out_step->scalar_kind = source_op->kind;
  out_step->instance_flags = source_op->instance_flags;
  *out_input = loom_op_const_operands(source_op)[0];
  if (*out_input >= module->values.count) return false;
  out_step->input_type = loom_module_value_type(module, *out_input);
  out_step->result_type =
      loom_module_value_type(module, loom_op_const_results(source_op)[0]);
  return loom_type_is_scalar(out_step->input_type) &&
         loom_type_is_scalar(out_step->result_type);
}

static bool loom_vector_value_is_static_lane_extract(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id,
    iree_host_size_t expected_lane, loom_value_id_t* out_source,
    loom_type_t* out_source_type) {
  *out_source = LOOM_VALUE_ID_INVALID;
  *out_source_type = loom_type_none();

  loom_op_t* extract_op = NULL;
  if (!loom_vector_value_def_op(rewriter, value_id, &extract_op) ||
      !loom_vector_extract_isa(extract_op) ||
      loom_vector_extract_indices(extract_op).count != 0) {
    return false;
  }

  loom_attribute_t static_indices =
      loom_vector_extract_static_indices(extract_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] < 0 ||
      (iree_host_size_t)static_indices.i64_array[0] != expected_lane) {
    return false;
  }

  loom_value_id_t source = loom_vector_extract_source(extract_op);
  if (source >= rewriter->module->values.count) return false;
  *out_source = source;
  *out_source_type = loom_module_value_type(rewriter->module, source);
  return true;
}

static bool loom_vector_match_scalar_lane_chain(
    const loom_rewriter_t* rewriter, loom_value_id_t lane_value,
    iree_host_size_t lane_index, loom_vector_lane_chain_t* out_chain) {
  *out_chain = (loom_vector_lane_chain_t){0};
  out_chain->source = LOOM_VALUE_ID_INVALID;

  loom_value_id_t value = lane_value;
  while (out_chain->step_count < LOOM_VECTOR_LANE_CHAIN_MAX_STEPS) {
    loom_op_t* source_op = NULL;
    if (!loom_vector_value_def_op(rewriter, value, &source_op)) break;

    loom_vector_lane_chain_step_t step = {0};
    loom_value_id_t input = LOOM_VALUE_ID_INVALID;
    if (!loom_vector_describe_scalar_lane_chain_step(
            rewriter->module, source_op, &step, &input)) {
      break;
    }
    out_chain->steps[out_chain->step_count++] = step;
    value = input;
  }

  loom_op_t* overflow_op = NULL;
  if (out_chain->step_count == LOOM_VECTOR_LANE_CHAIN_MAX_STEPS &&
      loom_vector_value_def_op(rewriter, value, &overflow_op)) {
    loom_vector_lane_chain_step_t overflow_step = {0};
    loom_value_id_t overflow_input = LOOM_VALUE_ID_INVALID;
    if (loom_vector_describe_scalar_lane_chain_step(
            rewriter->module, overflow_op, &overflow_step, &overflow_input)) {
      return false;
    }
  }

  return loom_vector_value_is_static_lane_extract(
      rewriter, value, lane_index, &out_chain->source, &out_chain->source_type);
}

static bool loom_vector_lane_chain_matches(
    const loom_vector_lane_chain_t* expected,
    const loom_vector_lane_chain_t* candidate) {
  if (candidate->source != expected->source ||
      candidate->step_count != expected->step_count ||
      !loom_type_equal(candidate->source_type, expected->source_type)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < expected->step_count; ++i) {
    if (!loom_vector_lane_chain_steps_equal(&candidate->steps[i],
                                            &expected->steps[i])) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_build_lane_chain_step(
    loom_builder_t* builder, const loom_vector_lane_chain_step_t* step,
    loom_value_id_t input, loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  switch (step->scalar_kind) {
    case LOOM_OP_SCALAR_CEILF:
      return loom_vector_ceilf_build(builder, step->instance_flags, input,
                                     result_type, location, out_op);
    case LOOM_OP_SCALAR_FLOORF:
      return loom_vector_floorf_build(builder, step->instance_flags, input,
                                      result_type, location, out_op);
    case LOOM_OP_SCALAR_ROUNDF:
      return loom_vector_roundf_build(builder, step->instance_flags, input,
                                      result_type, location, out_op);
    case LOOM_OP_SCALAR_ROUNDEVENF:
      return loom_vector_roundevenf_build(builder, step->instance_flags, input,
                                          result_type, location, out_op);
    case LOOM_OP_SCALAR_TRUNCF:
      return loom_vector_truncf_build(builder, step->instance_flags, input,
                                      result_type, location, out_op);
    case LOOM_OP_SCALAR_FPTOSI:
      return loom_vector_fptosi_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_FPTOUI:
      return loom_vector_fptoui_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_TRUNCI:
      return loom_vector_trunci_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_SITOFP:
      return loom_vector_sitofp_build(builder, input, input_type, result_type,
                                      location, out_op);
    case LOOM_OP_SCALAR_UITOFP:
      return loom_vector_uitofp_build(builder, input, input_type, result_type,
                                      location, out_op);
    default:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unsupported vector lane-chain step kind");
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_from_elements_lane_chain(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count == 0) return iree_ok_status();

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  int64_t result_lane_count = 0;
  if (!loom_vector_type_is_static_1d_vector(result_type, &result_lane_count) ||
      result_lane_count != (int64_t)elements.count) {
    return iree_ok_status();
  }

  loom_vector_lane_chain_t expected_chain = {0};
  if (!loom_vector_match_scalar_lane_chain(rewriter, elements.values[0], 0,
                                           &expected_chain)) {
    return iree_ok_status();
  }
  int64_t source_lane_count = 0;
  if (!loom_vector_type_is_static_1d_vector(expected_chain.source_type,
                                            &source_lane_count) ||
      source_lane_count != result_lane_count ||
      !loom_type_shape_equals(expected_chain.source_type, result_type)) {
    return iree_ok_status();
  }

  for (iree_host_size_t lane_index = 1; lane_index < elements.count;
       ++lane_index) {
    loom_vector_lane_chain_t candidate_chain = {0};
    if (!loom_vector_match_scalar_lane_chain(rewriter,
                                             elements.values[lane_index],
                                             lane_index, &candidate_chain) ||
        !loom_vector_lane_chain_matches(&expected_chain, &candidate_chain)) {
      return iree_ok_status();
    }
  }

  if (expected_chain.step_count == 0) {
    if (!loom_type_equal(expected_chain.source_type, result_type)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
        op, rewriter, expected_chain.source));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_type_t validated_input_type = expected_chain.source_type;
  for (iree_host_size_t step_ordinal = expected_chain.step_count;
       step_ordinal > 0; --step_ordinal) {
    const iree_host_size_t step_index = step_ordinal - 1;
    const loom_vector_lane_chain_step_t* step =
        &expected_chain.steps[step_index];
    if (loom_type_element_type(validated_input_type) !=
        loom_type_element_type(step->input_type)) {
      return iree_ok_status();
    }
    validated_input_type = loom_vector_same_shape_with_element_type(
        result_type, loom_type_element_type(step->result_type));
    if (step_index == 0) {
      validated_input_type = result_type;
    }
    if (loom_type_element_type(validated_input_type) !=
        loom_type_element_type(step->result_type)) {
      return iree_ok_status();
    }
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_value_id_t replacement = expected_chain.source;
  loom_type_t input_type = expected_chain.source_type;
  loom_op_t* replacement_op = NULL;
  for (iree_host_size_t step_ordinal = expected_chain.step_count;
       step_ordinal > 0; --step_ordinal) {
    const iree_host_size_t step_index = step_ordinal - 1;
    const loom_vector_lane_chain_step_t* step =
        &expected_chain.steps[step_index];
    loom_type_t step_result_type = loom_vector_same_shape_with_element_type(
        result_type, loom_type_element_type(step->result_type));
    if (step_index == 0) {
      step_result_type = result_type;
    }

    IREE_RETURN_IF_ERROR(loom_vector_build_lane_chain_step(
        &rewriter->builder, step, replacement, input_type, step_result_type,
        op->location, &replacement_op));
    replacement = loom_op_const_results(replacement_op)[0];
    input_type = step_result_type;
  }

  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, replacement_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_from_elements(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;
  loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count == 0) return iree_ok_status();

  loom_value_id_t first_element = elements.values[0];
  bool all_elements_equal = true;
  for (uint16_t i = 1; i < elements.count; ++i) {
    if (elements.values[i] != first_element) {
      all_elements_equal = false;
      break;
    }
  }
  if (!all_elements_equal) {
    return loom_vector_canonicalize_from_elements_lane_chain(op, rewriter,
                                                             out_changed);
  }

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_splat(
      op, rewriter, first_element, result_type));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_iota(loom_op_t* op,
                                                   loom_rewriter_t* rewriter,
                                                   bool* out_changed) {
  *out_changed = false;

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  uint64_t lane_count = 0;
  if (!loom_type_is_vector(result_type) ||
      !loom_type_static_element_count(result_type, &lane_count) ||
      lane_count == 0 || lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return iree_ok_status();
  }

  int64_t base = 0;
  int64_t step = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_rewriter_value_facts(rewriter, loom_vector_iota_base(op)),
          &base) ||
      !loom_value_facts_as_exact_i64(
          loom_rewriter_value_facts(rewriter, loom_vector_iota_step(op)),
          &step)) {
    return iree_ok_status();
  }
  if (!rewriter->materialize_constant) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_type_t element_type =
      loom_type_scalar(loom_type_element_type(result_type));
  loom_value_id_t elements[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {
      LOOM_VALUE_ID_INVALID};
  for (uint64_t i = 0; i < lane_count; ++i) {
    int64_t delta = 0;
    int64_t value = 0;
    if (!iree_checked_mul_i64((int64_t)i, step, &delta) ||
        !iree_checked_add_i64(base, delta, &value)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_build_constant(
        rewriter, loom_value_facts_exact_i64(value), element_type, op->location,
        &elements[i]));
  }

  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &rewriter->builder, elements, (iree_host_size_t)lane_count, result_type,
      op->location, &replacement_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, replacement_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_vector_static_indices_are_proven_in_bounds(
    loom_type_t source_type, loom_attribute_t static_indices) {
  if (!loom_type_is_all_static(source_type)) return false;
  if (static_indices.count > loom_type_rank(source_type)) return false;
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index < 0 || static_index == INT64_MIN) return false;
    if (static_index >= loom_type_dim_static_size_at(source_type, i)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_canonicalize_extract_from_splat(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t source_type, loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (!loom_vector_static_indices_are_proven_in_bounds(source_type,
                                                       static_indices)) {
    return iree_ok_status();
  }

  loom_value_id_t scalar = loom_vector_splat_scalar(source_def_op);
  if (loom_type_is_scalar(result_type)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, scalar));
  } else if (loom_type_is_vector(result_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_splat(
        op, rewriter, scalar, result_type));
  } else {
    return iree_ok_status();
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_extract_from_broadcast(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t source_type, loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  loom_value_id_t broadcast_source =
      loom_vector_broadcast_source(source_def_op);
  loom_type_t broadcast_source_type =
      loom_module_value_type(rewriter->module, broadcast_source);
  if (!loom_type_is_vector(source_type) ||
      !loom_type_is_vector(broadcast_source_type) ||
      !loom_type_equal(broadcast_source_type, result_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t broadcast_source_rank = loom_type_rank(broadcast_source_type);
  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (source_rank < broadcast_source_rank ||
      static_indices.count != source_rank - broadcast_source_rank ||
      loom_vector_extract_indices(op).count != 0 ||
      !loom_vector_static_indices_are_proven_in_bounds(source_type,
                                                       static_indices)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
      op, rewriter, broadcast_source));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_extract_from_elements(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  if (!loom_type_is_scalar(result_type)) {
    return iree_ok_status();
  }

  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (static_indices.count != 1 || static_indices.i64_array[0] < 0 ||
      static_indices.i64_array[0] == INT64_MIN) {
    return iree_ok_status();
  }
  loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_def_op);
  iree_host_size_t lane = (iree_host_size_t)static_indices.i64_array[0];
  if (lane >= elements.count) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
      op, rewriter, elements.values[lane]));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_extract_from_iota(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t source_type, loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  if (!loom_type_is_scalar(result_type)) return iree_ok_status();

  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (static_indices.count != loom_type_rank(source_type)) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] < 0 ||
        static_indices.i64_array[i] == INT64_MIN) {
      return iree_ok_status();
    }
  }

  iree_host_size_t lane = 0;
  if (!loom_vector_static_ordinal_from_indices(
          source_type, static_indices.i64_array, &lane)) {
    return iree_ok_status();
  }

  int64_t base = 0;
  int64_t step = 0;
  if (loom_value_facts_as_exact_i64(
          loom_rewriter_value_facts(rewriter,
                                    loom_vector_iota_base(source_def_op)),
          &base) &&
      loom_value_facts_as_exact_i64(
          loom_rewriter_value_facts(rewriter,
                                    loom_vector_iota_step(source_def_op)),
          &step)) {
    int64_t delta = 0;
    int64_t value = 0;
    if (!iree_checked_mul_i64((int64_t)lane, step, &delta) ||
        !iree_checked_add_i64(base, delta, &value)) {
      return iree_ok_status();
    }

    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_scalar_i64(
        op, rewriter, value, result_type));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_build_iota_lane_symbolic(
      op, rewriter, loom_vector_iota_base(source_def_op),
      loom_vector_iota_step(source_def_op), lane, result_type, &replacement,
      value_checkpoint));
  if (replacement == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_vector_replace_single_result_with_value(op, rewriter, replacement));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_extract_from_load(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t source_type, loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  if (!loom_type_is_scalar(result_type)) return iree_ok_status();

  if (source_def_op->result_count != 1) return iree_ok_status();
  loom_value_id_t load_result = loom_op_const_results(source_def_op)[0];
  if (load_result >= rewriter->module->values.count ||
      !loom_value_has_single_use(
          loom_module_value(rewriter->module, load_result))) {
    return iree_ok_status();
  }

  loom_attribute_t lane_indices = loom_vector_extract_static_indices(op);
  if (lane_indices.kind != LOOM_ATTR_I64_ARRAY ||
      lane_indices.count != loom_type_rank(source_type)) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < lane_indices.count; ++i) {
    if (lane_indices.i64_array[i] < 0 ||
        lane_indices.i64_array[i] == INT64_MIN) {
      return iree_ok_status();
    }
  }

  const loom_value_id_t view = loom_vector_load_view(source_def_op);
  if (loom_vector_value_is_workgroup_view(rewriter, view)) {
    return iree_ok_status();
  }

  const loom_type_t view_type = loom_module_value_type(rewriter->module, view);
  const loom_fact_context_t* fact_context =
      rewriter->fact_table ? &rewriter->fact_table->context : NULL;
  loom_vector_memory_access_t access = {0};
  if (!loom_vector_memory_access_describe(fact_context, rewriter->module,
                                          view_type, source_type, &access)) {
    return iree_ok_status();
  }

  loom_attribute_t load_static_indices =
      loom_vector_load_static_indices(source_def_op);
  if (load_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      load_static_indices.count != access.view_rank) {
    return iree_ok_status();
  }

  loom_value_slice_t load_indices = loom_vector_load_indices(source_def_op);
  int64_t static_indices[LOOM_TYPE_MAX_RANK] = {0};
  loom_value_id_t dynamic_indices[LOOM_TYPE_MAX_RANK] = {0};
  uint16_t dynamic_index_count = 0;
  uint16_t load_dynamic_index = 0;

  loom_builder_set_before(&rewriter->builder, op);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  for (uint8_t axis = 0; axis < access.view_rank; ++axis) {
    int64_t lane_index = 0;
    if (axis >= access.first_vector_axis) {
      lane_index = lane_indices.i64_array[axis - access.first_vector_axis];
    }

    int64_t static_index = load_static_indices.i64_array[axis];
    if (static_index != INT64_MIN) {
      int64_t combined_index = 0;
      if (!iree_checked_add_i64(static_index, lane_index, &combined_index)) {
        return iree_ok_status();
      }
      static_indices[axis] = combined_index;
      continue;
    }

    if (load_dynamic_index >= load_indices.count) {
      return iree_ok_status();
    }
    loom_value_id_t dynamic_index = load_indices.values[load_dynamic_index++];
    if (lane_index != 0) {
      loom_op_t* lane_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          &rewriter->builder, loom_attr_i64(lane_index), index_type,
          op->location, &lane_op));
      loom_op_t* add_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_index_add_build(&rewriter->builder, dynamic_index,
                               loom_index_constant_result(lane_op), index_type,
                               op->location, &add_op));
      dynamic_index = loom_index_add_result(add_op);
    }
    static_indices[axis] = INT64_MIN;
    dynamic_indices[dynamic_index_count++] = dynamic_index;
  }
  if (load_dynamic_index != load_indices.count) {
    return iree_ok_status();
  }

  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(rewriter->module, source_def_op,
                                               &cache_policy)) {
    return iree_ok_status();
  }
  uint32_t build_flags = 0;
  if (iree_any_bit_set(cache_policy.build_flags,
                       LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE)) {
    build_flags |= LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE;
  }
  if (iree_any_bit_set(cache_policy.build_flags,
                       LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL)) {
    build_flags |= LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL;
  }

  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_load_build(
      &rewriter->builder, build_flags, view, dynamic_indices,
      dynamic_index_count, static_indices, access.view_rank,
      cache_policy.cache_scope, cache_policy.cache_temporal, result_type,
      op->location, &load_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, load_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_extract_static_indices(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_type_t source_type,
    loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  loom_attribute_t old_static_indices = loom_vector_extract_static_indices(op);
  if (old_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      old_static_indices.count == 0) {
    return iree_ok_status();
  }

  loom_value_slice_t old_indices = loom_vector_extract_indices(op);
  if (old_indices.count == 0) {
    return iree_ok_status();
  }

  int64_t* new_static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, old_static_indices.count, sizeof(*new_static_indices),
      (void**)&new_static_indices));
  loom_value_id_t* new_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, old_indices.count,
                                sizeof(*new_indices), (void**)&new_indices));

  bool changed = false;
  uint16_t old_dynamic_index = 0;
  uint16_t new_dynamic_index = 0;
  for (uint16_t axis = 0; axis < old_static_indices.count; ++axis) {
    int64_t static_index = old_static_indices.i64_array[axis];
    if (static_index != INT64_MIN) {
      new_static_indices[axis] = static_index;
      continue;
    }
    if (old_dynamic_index >= old_indices.count) {
      return iree_ok_status();
    }

    loom_value_id_t dynamic_index = old_indices.values[old_dynamic_index++];
    if (loom_vector_index_value_as_static_index(
            rewriter, dynamic_index, source_type, axis, &static_index)) {
      new_static_indices[axis] = static_index;
      changed = true;
    } else {
      new_static_indices[axis] = INT64_MIN;
      new_indices[new_dynamic_index++] = dynamic_index;
    }
  }
  if (old_dynamic_index != old_indices.count || !changed) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &rewriter->builder, loom_vector_extract_source(op), new_indices,
      new_dynamic_index, new_static_indices, old_static_indices.count,
      result_type, op->location, &replacement_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, replacement_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_vector_exact_static_indices_match(
    loom_attribute_t lhs_static_indices, loom_value_slice_t lhs_indices,
    loom_attribute_t rhs_static_indices, loom_value_slice_t rhs_indices) {
  if (lhs_indices.count != 0 || rhs_indices.count != 0 ||
      lhs_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      rhs_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      lhs_static_indices.count == 0 ||
      lhs_static_indices.count != rhs_static_indices.count ||
      !lhs_static_indices.i64_array || !rhs_static_indices.i64_array) {
    return false;
  }
  for (uint16_t i = 0; i < lhs_static_indices.count; ++i) {
    if (lhs_static_indices.i64_array[i] < 0 ||
        lhs_static_indices.i64_array[i] == INT64_MIN ||
        lhs_static_indices.i64_array[i] != rhs_static_indices.i64_array[i]) {
      return false;
    }
  }
  return true;
}

static bool loom_vector_exact_static_indices_are_disjoint(
    loom_attribute_t lhs_static_indices, loom_value_slice_t lhs_indices,
    loom_attribute_t rhs_static_indices, loom_value_slice_t rhs_indices) {
  if (lhs_indices.count != 0 || rhs_indices.count != 0 ||
      lhs_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      rhs_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      lhs_static_indices.count == 0 ||
      lhs_static_indices.count != rhs_static_indices.count ||
      !lhs_static_indices.i64_array || !rhs_static_indices.i64_array) {
    return false;
  }
  bool differs = false;
  for (uint16_t i = 0; i < lhs_static_indices.count; ++i) {
    int64_t lhs = lhs_static_indices.i64_array[i];
    int64_t rhs = rhs_static_indices.i64_array[i];
    if (lhs < 0 || lhs == INT64_MIN || rhs < 0 || rhs == INT64_MIN) {
      return false;
    }
    differs |= lhs != rhs;
  }
  return differs;
}

static iree_status_t loom_vector_canonicalize_extract_from_insert(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* insert_op,
    loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  loom_attribute_t extract_static_indices =
      loom_vector_extract_static_indices(op);
  loom_value_slice_t extract_indices = loom_vector_extract_indices(op);
  loom_attribute_t insert_static_indices =
      loom_vector_insert_static_indices(insert_op);
  loom_value_slice_t insert_indices = loom_vector_insert_indices(insert_op);
  if (loom_vector_exact_static_indices_match(
          extract_static_indices, extract_indices, insert_static_indices,
          insert_indices)) {
    loom_value_id_t inserted_value = loom_vector_insert_value(insert_op);
    if (!loom_type_equal(
            loom_module_value_type(rewriter->module, inserted_value),
            result_type)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
        op, rewriter, inserted_value));
    *out_changed = true;
    return iree_ok_status();
  }
  if (!loom_vector_exact_static_indices_are_disjoint(
          extract_static_indices, extract_indices, insert_static_indices,
          insert_indices)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &rewriter->builder, loom_vector_insert_dest(insert_op),
      /*indices=*/NULL, /*indices_count=*/0, extract_static_indices.i64_array,
      extract_static_indices.count, result_type, op->location,
      &replacement_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, replacement_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_extract(loom_op_t* op,
                                                      loom_rewriter_t* rewriter,
                                                      bool* out_changed) {
  *out_changed = false;

  loom_value_id_t source = loom_vector_extract_source(op);
  loom_type_t source_type = loom_module_value_type(rewriter->module, source);
  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_canonicalize_extract_static_indices(
      op, rewriter, source_type, result_type, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }

  loom_op_t* source_def_op = NULL;
  if (!loom_vector_value_def_op(rewriter, source, &source_def_op)) {
    return iree_ok_status();
  }

  if (loom_vector_broadcast_isa(source_def_op)) {
    return loom_vector_canonicalize_extract_from_broadcast(
        op, rewriter, source_def_op, source_type, result_type, out_changed);
  }
  if (loom_vector_splat_isa(source_def_op)) {
    return loom_vector_canonicalize_extract_from_splat(
        op, rewriter, source_def_op, source_type, result_type, out_changed);
  }
  if (loom_vector_from_elements_isa(source_def_op)) {
    return loom_vector_canonicalize_extract_from_elements(
        op, rewriter, source_def_op, result_type, out_changed);
  }
  if (loom_vector_iota_isa(source_def_op)) {
    return loom_vector_canonicalize_extract_from_iota(
        op, rewriter, source_def_op, source_type, result_type, out_changed);
  }
  if (loom_vector_load_isa(source_def_op)) {
    return loom_vector_canonicalize_extract_from_load(
        op, rewriter, source_def_op, source_type, result_type, out_changed);
  }
  if (loom_vector_insert_isa(source_def_op)) {
    return loom_vector_canonicalize_extract_from_insert(
        op, rewriter, source_def_op, result_type, out_changed);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_insert_static_indices(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_type_t dest_type,
    loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  loom_attribute_t old_static_indices = loom_vector_insert_static_indices(op);
  if (old_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      old_static_indices.count == 0) {
    return iree_ok_status();
  }

  loom_value_slice_t old_indices = loom_vector_insert_indices(op);
  if (old_indices.count == 0) return iree_ok_status();

  int64_t* new_static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, old_static_indices.count, sizeof(*new_static_indices),
      (void**)&new_static_indices));
  loom_value_id_t* new_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, old_indices.count,
                                sizeof(*new_indices), (void**)&new_indices));

  bool changed = false;
  uint16_t old_dynamic_index = 0;
  uint16_t new_dynamic_index = 0;
  for (uint16_t axis = 0; axis < old_static_indices.count; ++axis) {
    int64_t static_index = old_static_indices.i64_array[axis];
    if (static_index != INT64_MIN) {
      new_static_indices[axis] = static_index;
      continue;
    }
    if (old_dynamic_index >= old_indices.count) return iree_ok_status();

    loom_value_id_t dynamic_index = old_indices.values[old_dynamic_index++];
    if (loom_vector_index_value_as_static_index(
            rewriter, dynamic_index, dest_type, axis, &static_index)) {
      new_static_indices[axis] = static_index;
      changed = true;
    } else {
      new_static_indices[axis] = INT64_MIN;
      new_indices[new_dynamic_index++] = dynamic_index;
    }
  }
  if (old_dynamic_index != old_indices.count || !changed) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_insert_build(
      &rewriter->builder, loom_vector_insert_value(op),
      loom_vector_insert_dest(op), new_indices, new_dynamic_index,
      new_static_indices, old_static_indices.count, result_type, op->location,
      &replacement_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, replacement_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_insert(loom_op_t* op,
                                                     loom_rewriter_t* rewriter,
                                                     bool* out_changed) {
  *out_changed = false;
  loom_value_id_t value = loom_vector_insert_value(op);
  loom_value_id_t dest = loom_vector_insert_dest(op);
  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_canonicalize_insert_static_indices(
      op, rewriter, loom_module_value_type(rewriter->module, dest), result_type,
      out_changed));
  if (*out_changed) return iree_ok_status();

  loom_attribute_t static_indices = loom_vector_insert_static_indices(op);
  loom_value_slice_t indices = loom_vector_insert_indices(op);

  loom_op_t* value_def_op = NULL;
  if (loom_vector_value_def_op(rewriter, value, &value_def_op) &&
      loom_vector_extract_isa(value_def_op) &&
      loom_vector_extract_source(value_def_op) == dest &&
      loom_vector_exact_static_indices_match(
          static_indices, indices,
          loom_vector_extract_static_indices(value_def_op),
          loom_vector_extract_indices(value_def_op))) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, dest));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_op_t* dest_def_op = NULL;
  if (!loom_vector_value_def_op(rewriter, dest, &dest_def_op) ||
      !loom_vector_insert_isa(dest_def_op) ||
      !loom_vector_exact_static_indices_match(
          static_indices, indices,
          loom_vector_insert_static_indices(dest_def_op),
          loom_vector_insert_indices(dest_def_op))) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_insert_build(
      &rewriter->builder, value, loom_vector_insert_dest(dest_def_op),
      /*indices=*/NULL, /*indices_count=*/0, static_indices.i64_array,
      static_indices.count, result_type, op->location, &replacement_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, replacement_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_shuffle(loom_op_t* op,
                                                      loom_rewriter_t* rewriter,
                                                      bool* out_changed) {
  *out_changed = false;

  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(op);
  for (uint16_t i = 0; i < source_lanes.count; ++i) {
    if (source_lanes.i64_array[i] != (int64_t)i) return iree_ok_status();
  }

  loom_value_id_t source = loom_vector_shuffle_source(op);
  IREE_RETURN_IF_ERROR(
      loom_vector_replace_single_result_with_value(op, rewriter, source));
  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Selection and comparison
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_canonicalize_select(loom_op_t* op,
                                                     loom_rewriter_t* rewriter,
                                                     bool* out_changed) {
  *out_changed = false;
  loom_value_id_t true_value = loom_vector_select_true_value(op);
  loom_value_id_t false_value = loom_vector_select_false_value(op);
  if (true_value == false_value) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, true_value));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_value_id_t condition = loom_vector_select_condition(op);
  if (loom_vector_value_is_all_bool(rewriter, condition, true)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, true_value));
    *out_changed = true;
    return iree_ok_status();
  }
  if (loom_vector_value_is_all_bool(rewriter, condition, false)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
        op, rewriter, false_value));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_op_t* condition_def_op = NULL;
  if (loom_vector_value_def_op(rewriter, condition, &condition_def_op) &&
      loom_vector_splat_isa(condition_def_op)) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, loom_vector_select_result(op));
    loom_builder_set_before(&rewriter->builder, op);
    loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
    loom_op_t* replacement_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scf_select_build(
        &rewriter->builder, loom_vector_splat_scalar(condition_def_op),
        true_value, false_value, result_type, op->location, &replacement_op));
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
        op, rewriter, replacement_op, value_checkpoint));
    *out_changed = true;
  }
  return iree_ok_status();
}

static bool loom_vector_cmpi_same_operand_result(uint8_t predicate,
                                                 bool* out_value) {
  switch ((loom_vector_cmpi_predicate_t)predicate) {
    case LOOM_VECTOR_CMPI_PREDICATE_EQ:
    case LOOM_VECTOR_CMPI_PREDICATE_SLE:
    case LOOM_VECTOR_CMPI_PREDICATE_SGE:
    case LOOM_VECTOR_CMPI_PREDICATE_ULE:
    case LOOM_VECTOR_CMPI_PREDICATE_UGE:
      *out_value = true;
      return true;
    case LOOM_VECTOR_CMPI_PREDICATE_NE:
    case LOOM_VECTOR_CMPI_PREDICATE_SLT:
    case LOOM_VECTOR_CMPI_PREDICATE_SGT:
    case LOOM_VECTOR_CMPI_PREDICATE_ULT:
    case LOOM_VECTOR_CMPI_PREDICATE_UGT:
      *out_value = false;
      return true;
    case LOOM_VECTOR_CMPI_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static bool loom_vector_cmpf_same_operand_result(uint8_t predicate,
                                                 bool* out_value) {
  switch ((loom_vector_cmpf_predicate_t)predicate) {
    case LOOM_VECTOR_CMPF_PREDICATE_OGT:
    case LOOM_VECTOR_CMPF_PREDICATE_OLT:
    case LOOM_VECTOR_CMPF_PREDICATE_ONE:
      *out_value = false;
      return true;
    case LOOM_VECTOR_CMPF_PREDICATE_UEQ:
    case LOOM_VECTOR_CMPF_PREDICATE_UGE:
    case LOOM_VECTOR_CMPF_PREDICATE_ULE:
      *out_value = true;
      return true;
    case LOOM_VECTOR_CMPF_PREDICATE_OEQ:
    case LOOM_VECTOR_CMPF_PREDICATE_OGE:
    case LOOM_VECTOR_CMPF_PREDICATE_OLE:
    case LOOM_VECTOR_CMPF_PREDICATE_ORD:
    case LOOM_VECTOR_CMPF_PREDICATE_UGT:
    case LOOM_VECTOR_CMPF_PREDICATE_ULT:
    case LOOM_VECTOR_CMPF_PREDICATE_UNE:
    case LOOM_VECTOR_CMPF_PREDICATE_UNO:
    case LOOM_VECTOR_CMPF_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static iree_status_t loom_vector_canonicalize_comparison(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_value_id_t lhs = loom_vector_cmpi_isa(op) ? loom_vector_cmpi_lhs(op)
                                                 : loom_vector_cmpf_lhs(op);
  loom_value_id_t rhs = loom_vector_cmpi_isa(op) ? loom_vector_cmpi_rhs(op)
                                                 : loom_vector_cmpf_rhs(op);
  if (lhs != rhs) return iree_ok_status();

  bool value = false;
  bool has_result = loom_vector_cmpi_isa(op)
                        ? loom_vector_cmpi_same_operand_result(
                              loom_vector_cmpi_predicate(op), &value)
                        : loom_vector_cmpf_same_operand_result(
                              loom_vector_cmpf_predicate(op), &value);
  if (!has_result) return iree_ok_status();

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_constant_attr(
      op, rewriter, result_type, loom_attr_bool(value)));
  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lanewise arithmetic
//===----------------------------------------------------------------------===//

static bool loom_vector_fastmath_has_all(const loom_op_t* op, uint8_t flags) {
  return (op->instance_flags & flags) == flags;
}

static float loom_vector_divide_f32(float lhs, float rhs) { return lhs / rhs; }
static double loom_vector_divide_f64(double lhs, double rhs) {
  return lhs / rhs;
}

static iree_status_t loom_vector_canonicalize_subf(loom_op_t* op,
                                                   loom_rewriter_t* rewriter,
                                                   bool* out_changed) {
  *out_changed = false;

  loom_value_id_t lhs = loom_vector_subf_lhs(op);
  loom_value_id_t rhs = loom_vector_subf_rhs(op);
  if (loom_vector_fastmath_has_all(op, LOOM_VECTOR_FASTMATHFLAGS_NSZ) &&
      loom_vector_value_is_all_exact_float(rewriter, rhs, 0.0)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, lhs));
    *out_changed = true;
    return iree_ok_status();
  }

  const uint8_t neg_flags =
      LOOM_VECTOR_FASTMATHFLAGS_NNAN | LOOM_VECTOR_FASTMATHFLAGS_NSZ;
  if (!loom_vector_fastmath_has_all(op, neg_flags) ||
      !loom_vector_value_is_all_exact_float(rewriter, lhs, 0.0)) {
    return iree_ok_status();
  }

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_negf(
      op, rewriter, loom_vector_subf_fastmath(op), rhs, result_type));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_vector_match_negf(const loom_rewriter_t* rewriter,
                                   loom_value_id_t value_id,
                                   loom_value_id_t* out_input) {
  loom_op_t* def_op = NULL;
  if (!loom_vector_value_def_op(rewriter, value_id, &def_op) ||
      !loom_vector_negf_isa(def_op)) {
    return false;
  }
  *out_input = loom_vector_negf_input(def_op);
  return true;
}

static iree_status_t loom_vector_replace_mulf_with_negated_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t input,
    double constant_value, loom_type_t result_type) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &rewriter->builder, loom_attr_f64(-constant_value), result_type,
      op->location, &constant_op));
  loom_value_id_t negated_constant = loom_vector_constant_result(constant_op);

  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_mulf_build(
      &rewriter->builder, loom_vector_mulf_fastmath(op), input,
      negated_constant, result_type, op->location, &replacement_op));
  loom_value_id_t replacement = loom_vector_mulf_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static bool loom_vector_match_contractable_mulf(const loom_rewriter_t* rewriter,
                                                loom_value_id_t value,
                                                loom_op_t** out_product_op) {
  loom_op_t* product_op = NULL;
  if (!loom_vector_value_def_op(rewriter, value, &product_op) ||
      !loom_vector_mulf_isa(product_op) ||
      !iree_any_bit_set(loom_vector_mulf_fastmath(product_op),
                        LOOM_VECTOR_FASTMATHFLAGS_CONTRACT)) {
    return false;
  }
  if (!loom_value_has_single_use(loom_module_value(rewriter->module, value))) {
    return false;
  }
  *out_product_op = product_op;
  return true;
}

static iree_status_t loom_vector_canonicalize_addf(loom_op_t* op,
                                                   loom_rewriter_t* rewriter,
                                                   bool* out_changed) {
  *out_changed = false;
  const uint8_t add_flags = loom_vector_addf_fastmath(op);
  if (!iree_any_bit_set(add_flags, LOOM_VECTOR_FASTMATHFLAGS_CONTRACT)) {
    return iree_ok_status();
  }

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }

  loom_value_id_t addend = loom_vector_addf_rhs(op);
  loom_op_t* product_op = NULL;
  if (!loom_vector_match_contractable_mulf(rewriter, loom_vector_addf_lhs(op),
                                           &product_op)) {
    addend = loom_vector_addf_lhs(op);
    if (!loom_vector_match_contractable_mulf(rewriter, loom_vector_addf_rhs(op),
                                             &product_op)) {
      return iree_ok_status();
    }
  }

  const uint8_t fmaf_flags = add_flags & loom_vector_mulf_fastmath(product_op);
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* fmaf_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_fmaf_build(
      &rewriter->builder, fmaf_flags, loom_vector_mulf_lhs(product_op),
      loom_vector_mulf_rhs(product_op), addend, result_type, op->location,
      &fmaf_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, fmaf_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_replace_divf_with_reciprocal_mulf(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t lhs,
    double divisor_value, loom_type_t result_type) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  const loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
  loom_value_facts_t one_facts = loom_value_facts_exact_float(scalar_type, 1.0);
  loom_value_facts_t divisor_facts =
      loom_value_facts_exact_float(scalar_type, divisor_value);
  loom_value_facts_t reciprocal_facts = loom_value_facts_unknown();
  loom_value_facts_eval_float_binary(scalar_type, &one_facts, &divisor_facts,
                                     loom_vector_divide_f32,
                                     loom_vector_divide_f64, &reciprocal_facts);
  double reciprocal_value = 0.0;
  const bool has_reciprocal = loom_value_facts_as_exact_float(
      scalar_type, reciprocal_facts, &reciprocal_value);
  IREE_ASSERT(has_reciprocal);

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &rewriter->builder, loom_attr_f64(reciprocal_value), result_type,
      op->location, &constant_op));
  loom_value_id_t reciprocal = loom_vector_constant_result(constant_op);

  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_mulf_build(
      &rewriter->builder, loom_vector_divf_fastmath(op), lhs, reciprocal,
      result_type, op->location, &replacement_op));
  loom_value_id_t replacement = loom_vector_mulf_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_vector_canonicalize_mulf(loom_op_t* op,
                                                   loom_rewriter_t* rewriter,
                                                   bool* out_changed) {
  *out_changed = false;

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  loom_value_id_t lhs = loom_vector_mulf_lhs(op);
  loom_value_id_t rhs = loom_vector_mulf_rhs(op);
  if (loom_vector_value_is_all_exact_float(rewriter, lhs, 1.0)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, rhs));
    *out_changed = true;
    return iree_ok_status();
  }
  if (loom_vector_value_is_all_exact_float(rewriter, rhs, 1.0)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, lhs));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_value_id_t negated_input = LOOM_VALUE_ID_INVALID;
  double constant_value = 0.0;
  if (loom_vector_match_negf(rewriter, lhs, &negated_input) &&
      loom_vector_query_all_exact_float(rewriter, rhs, &constant_value)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_mulf_with_negated_constant(
        op, rewriter, negated_input, constant_value, result_type));
    *out_changed = true;
    return iree_ok_status();
  }
  if (loom_vector_match_negf(rewriter, rhs, &negated_input) &&
      loom_vector_query_all_exact_float(rewriter, lhs, &constant_value)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_mulf_with_negated_constant(
        op, rewriter, negated_input, constant_value, result_type));
    *out_changed = true;
    return iree_ok_status();
  }

  const uint8_t zero_flags = LOOM_VECTOR_FASTMATHFLAGS_NNAN |
                             LOOM_VECTOR_FASTMATHFLAGS_NINF |
                             LOOM_VECTOR_FASTMATHFLAGS_NSZ;
  if (loom_vector_fastmath_has_all(op, zero_flags) &&
      (loom_vector_value_is_all_exact_float(rewriter, lhs, 0.0) ||
       loom_vector_value_is_all_exact_float(rewriter, rhs, 0.0))) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_constant_attr(
        op, rewriter, result_type, loom_attr_f64(0.0)));
    *out_changed = true;
    return iree_ok_status();
  }

  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_divf(loom_op_t* op,
                                                   loom_rewriter_t* rewriter,
                                                   bool* out_changed) {
  *out_changed = false;

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  loom_value_id_t lhs = loom_vector_divf_lhs(op);
  loom_value_id_t rhs = loom_vector_divf_rhs(op);
  if (loom_vector_value_is_all_exact_float(rewriter, rhs, 1.0)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, lhs));
    *out_changed = true;
    return iree_ok_status();
  }
  if (!loom_vector_fastmath_has_all(op, LOOM_VECTOR_FASTMATHFLAGS_ARCP)) {
    return iree_ok_status();
  }

  double divisor_value = 0.0;
  if (loom_vector_query_all_exact_float(rewriter, rhs, &divisor_value)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_divf_with_reciprocal_mulf(
        op, rewriter, lhs, divisor_value, result_type));
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_binary_identity(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  loom_scalar_type_t element_type = loom_type_element_type(result_type);
  int64_t all_ones = loom_vector_integer_all_ones(element_type);

  const loom_value_id_t* operands = loom_op_const_operands(op);
  loom_value_id_t lhs = operands[0];
  loom_value_id_t rhs = operands[1];
  if (lhs == rhs &&
      (op->kind == LOOM_OP_VECTOR_ANDI || op->kind == LOOM_OP_VECTOR_ORI ||
       op->kind == LOOM_OP_VECTOR_MINSI || op->kind == LOOM_OP_VECTOR_MAXSI ||
       op->kind == LOOM_OP_VECTOR_MINUI || op->kind == LOOM_OP_VECTOR_MAXUI)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, lhs));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  switch (op->kind) {
    case LOOM_OP_VECTOR_ADDI:
    case LOOM_OP_VECTOR_ORI:
      if (loom_vector_value_is_all_exact_i64(rewriter, lhs, 0)) {
        replacement = rhs;
      }
      if (loom_vector_value_is_all_exact_i64(rewriter, rhs, 0)) {
        replacement = lhs;
      }
      break;
    case LOOM_OP_VECTOR_SUBI:
      if (loom_vector_value_is_all_exact_i64(rewriter, rhs, 0)) {
        replacement = lhs;
      }
      break;
    case LOOM_OP_VECTOR_MULI:
      if (loom_vector_value_is_all_exact_i64(rewriter, lhs, 0)) {
        replacement = lhs;
      }
      if (loom_vector_value_is_all_exact_i64(rewriter, rhs, 0)) {
        replacement = rhs;
      }
      if (replacement == LOOM_VALUE_ID_INVALID &&
          loom_vector_value_is_all_exact_i64(rewriter, lhs, 1)) {
        replacement = rhs;
      }
      if (replacement == LOOM_VALUE_ID_INVALID &&
          loom_vector_value_is_all_exact_i64(rewriter, rhs, 1)) {
        replacement = lhs;
      }
      break;
    case LOOM_OP_VECTOR_ANDI:
      if (loom_vector_value_is_all_exact_i64(rewriter, lhs, 0)) {
        replacement = lhs;
      }
      if (loom_vector_value_is_all_exact_i64(rewriter, rhs, 0)) {
        replacement = rhs;
      }
      if (replacement == LOOM_VALUE_ID_INVALID &&
          loom_vector_value_is_all_exact_i64(rewriter, lhs, all_ones)) {
        replacement = rhs;
      }
      if (replacement == LOOM_VALUE_ID_INVALID &&
          loom_vector_value_is_all_exact_i64(rewriter, rhs, all_ones)) {
        replacement = lhs;
      }
      break;
    case LOOM_OP_VECTOR_XORI:
      if (loom_vector_value_is_all_exact_i64(rewriter, lhs, 0)) {
        replacement = rhs;
      }
      if (loom_vector_value_is_all_exact_i64(rewriter, rhs, 0)) {
        replacement = lhs;
      }
      break;
    default:
      break;
  }

  if (replacement != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
        op, rewriter, replacement));
    *out_changed = true;
    return iree_ok_status();
  }

  if (lhs == rhs &&
      (op->kind == LOOM_OP_VECTOR_SUBI || op->kind == LOOM_OP_VECTOR_XORI)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_integer_zero(
        op, rewriter, result_type));
    *out_changed = true;
    return iree_ok_status();
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Encoding boundaries
//===----------------------------------------------------------------------===//

static bool loom_vector_decode_schema_is_dense_q8_0(
    const loom_rewriter_t* rewriter, loom_value_id_t schema_value) {
  if (!rewriter->fact_table) return false;
  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(
          &rewriter->fact_table->context,
          loom_rewriter_value_facts(rewriter, schema_value), &summary)) {
    return false;
  }
  loom_value_fact_encoded_operand_schema_t schema =
      summary.storage_schema.encoded_operand;
  return schema.element_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8 &&
         schema.scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F16 &&
         schema.secondary_scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE &&
         schema.payload_packing ==
             LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES &&
         schema.scale_topology == LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D &&
         schema.affine_policy == LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY &&
         schema.rounding_policy == LOOM_VALUE_FACT_ROUNDING_POLICY_NONE &&
         schema.codebook_policy == LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE &&
         schema.sparsity_policy == LOOM_VALUE_FACT_SPARSITY_POLICY_NONE &&
         schema.flags == 0 && schema.payload_register_count == 0 &&
         schema.scale_operand_count == 1;
}

static bool loom_vector_type_is_single_lane_float_vector(
    loom_type_t type, loom_scalar_type_t* out_element_type) {
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type) ||
      loom_type_dim_static_size_at(type, 0) != 1) {
    return false;
  }
  loom_scalar_type_t element_type = loom_type_element_type(type);
  if (!loom_scalar_type_is_float(element_type)) return false;
  *out_element_type = element_type;
  return true;
}

static iree_status_t loom_vector_canonicalize_dense_q8_0_decode(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type) ||
      !loom_type_is_vector(result_type) ||
      !loom_scalar_type_is_float(loom_type_element_type(result_type))) {
    return iree_ok_status();
  }
  loom_type_t payload_type =
      loom_module_value_type(rewriter->module, loom_vector_decode_payload(op));
  if (!loom_type_is_vector(payload_type) ||
      loom_type_element_type(payload_type) != LOOM_SCALAR_TYPE_I8 ||
      !loom_type_shape_equals(payload_type, result_type)) {
    return iree_ok_status();
  }
  if (!loom_vector_decode_schema_is_dense_q8_0(rewriter,
                                               loom_vector_decode_schema(op))) {
    return iree_ok_status();
  }

  loom_encoding_auxiliary_view_t auxiliary = {0};
  if (!loom_encoding_auxiliary_view_resolve(
          rewriter->module, loom_vector_decode_auxiliary(op),
          loom_vector_decode_auxiliary_names(op), &auxiliary, NULL)) {
    return iree_ok_status();
  }
  loom_value_id_t scale_vector =
      auxiliary.values[LOOM_ENCODING_AUXILIARY_KEY_SCALE];
  if (scale_vector == LOOM_VALUE_ID_INVALID ||
      scale_vector >= rewriter->module->values.count) {
    return iree_ok_status();
  }

  loom_scalar_type_t scale_element_type = LOOM_SCALAR_TYPE_NONE;
  loom_type_t scale_vector_type =
      loom_module_value_type(rewriter->module, scale_vector);
  if (!loom_vector_type_is_single_lane_float_vector(scale_vector_type,
                                                    &scale_element_type)) {
    return iree_ok_status();
  }
  loom_scalar_type_t result_element_type = loom_type_element_type(result_type);
  if (scale_element_type != result_element_type &&
      !(scale_element_type == LOOM_SCALAR_TYPE_F16 &&
        result_element_type == LOOM_SCALAR_TYPE_F32)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_location_id_t location = op->location;

  int64_t scale_lane = 0;
  loom_op_t* scale_extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &rewriter->builder, scale_vector, /*indices=*/NULL,
      /*indices_count=*/0, &scale_lane, /*static_indices_count=*/1,
      loom_type_scalar(scale_element_type), location, &scale_extract_op));
  loom_value_id_t scale_scalar = loom_vector_extract_result(scale_extract_op);

  if (scale_element_type != result_element_type) {
    loom_op_t* scale_ext_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_extf_build(
        &rewriter->builder, scale_scalar, loom_type_scalar(scale_element_type),
        loom_type_scalar(result_element_type), location, &scale_ext_op));
    scale_scalar = loom_scalar_extf_result(scale_ext_op);
  }

  loom_op_t* scale_splat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(&rewriter->builder, scale_scalar,
                                               result_type, location,
                                               &scale_splat_op));
  loom_value_id_t scale_splat = loom_vector_splat_result(scale_splat_op);

  loom_type_t unpacked_type = result_type;
  unpacked_type.header = loom_type_make_header(
      loom_type_kind(result_type), LOOM_SCALAR_TYPE_I32,
      loom_type_rank(result_type), loom_type_flags(result_type));

  loom_op_t* unpacked_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_bitunpacks_build(
      &rewriter->builder, /*width=*/8, loom_vector_decode_payload(op),
      unpacked_type, location, &unpacked_op));
  loom_value_id_t unpacked = loom_vector_bitunpacks_result(unpacked_op);

  loom_op_t* converted_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_sitofp_build(&rewriter->builder, unpacked,
                                                unpacked_type, result_type,
                                                location, &converted_op));
  loom_value_id_t converted = loom_vector_sitofp_result(converted_op);

  loom_op_t* scaled_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_mulf_build(
      &rewriter->builder, /*instance_flags=*/0, converted, scale_splat,
      result_type, location, &scaled_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, scaled_op, value_checkpoint));
  *out_changed = true;
  return iree_ok_status();
}

iree_status_t loom_vector_reduce_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  loom_value_id_t input = loom_vector_reduce_input(op);
  loom_value_id_t init = loom_vector_reduce_init(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  loom_combining_kind_t kind = loom_vector_reduce_kind(op);

  if (kind == LOOM_COMBINING_KIND_ADDF &&
      loom_type_element_type(input_type) == LOOM_SCALAR_TYPE_F32 &&
      iree_any_bit_set(loom_vector_reduce_fastmath(op),
                       LOOM_VECTOR_FASTMATHFLAGS_CONTRACT)) {
    loom_op_t* product_op = NULL;
    if (loom_vector_value_def_op(rewriter, input, &product_op) &&
        loom_vector_mulf_isa(product_op) &&
        iree_any_bit_set(loom_vector_mulf_fastmath(product_op),
                         LOOM_VECTOR_FASTMATHFLAGS_CONTRACT)) {
      uint8_t dot_flags = loom_vector_reduce_fastmath(op) &
                          loom_vector_mulf_fastmath(product_op);
      loom_builder_set_before(&rewriter->builder, op);
      loom_value_id_t value_checkpoint =
          loom_rewriter_value_checkpoint(rewriter);
      loom_op_t* dot_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_dotf_build(
          &rewriter->builder, dot_flags, loom_vector_mulf_lhs(product_op),
          loom_vector_mulf_rhs(product_op), init,
          loom_module_value_type(rewriter->module,
                                 loom_vector_reduce_result(op)),
          op->location, &dot_op));
      return loom_vector_replace_single_result_with_new_op(op, rewriter, dot_op,
                                                           value_checkpoint);
    }
  }

  int64_t identity = 0;
  if (!loom_vector_reduce_integer_identity(kind, input_type, &identity)) {
    return iree_ok_status();
  }

  if (!loom_vector_value_is_all_exact_i64(rewriter, input, identity)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_vector_replace_single_result_with_value(op, rewriter, init));
  return iree_ok_status();
}

static bool loom_vector_reduce_axes_all_source_axes(loom_type_t input_type,
                                                    loom_attribute_t axes) {
  if (axes.count != loom_type_rank(input_type)) return false;
  for (uint16_t i = 0; i < axes.count; ++i) {
    if (axes.i64_array[i] != (int64_t)i) return false;
  }
  return true;
}

iree_status_t loom_vector_reduce_axes_canonicalize(loom_op_t* op,
                                                   loom_rewriter_t* rewriter) {
  loom_value_id_t input = loom_vector_reduce_axes_input(op);
  loom_value_id_t init = loom_vector_reduce_axes_init(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  loom_combining_kind_t kind = loom_vector_reduce_axes_kind(op);

  int64_t identity = 0;
  if (loom_vector_reduce_integer_identity(kind, input_type, &identity) &&
      loom_vector_value_is_all_exact_i64(rewriter, input, identity)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, init));
    return iree_ok_status();
  }

  loom_attribute_t axes = loom_vector_reduce_axes_axes(op);
  if (!loom_vector_reduce_axes_all_source_axes(input_type, axes)) {
    bool changed = false;
    return loom_vector_canonicalize_uniform_result(op, rewriter, &changed);
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_type_t result_type = loom_module_value_type(
      rewriter->module, loom_vector_reduce_axes_result(op));
  loom_op_t* reduce_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
      &rewriter->builder, kind, op->instance_flags, input, init, result_type,
      op->location, &reduce_op));
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
      op, rewriter, reduce_op, value_checkpoint));
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Masked memory
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_canonicalize_all_true_masked_memory(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD_MASK:
      mask = loom_vector_load_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_LOAD_EXPAND:
      mask = loom_vector_load_expand_mask(op);
      break;
    case LOOM_OP_VECTOR_GATHER_MASK:
      mask = loom_vector_gather_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_ATOMIC_RMW_MASK:
      mask = loom_vector_atomic_rmw_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_STORE_MASK:
      mask = loom_vector_store_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_STORE_COMPRESS:
      mask = loom_vector_store_compress_mask(op);
      break;
    case LOOM_OP_VECTOR_SCATTER_MASK:
      mask = loom_vector_scatter_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK:
      mask = loom_vector_atomic_reduce_mask_mask(op);
      break;
    default:
      return iree_ok_status();
  }
  if (!loom_vector_value_is_all_bool(rewriter, mask, true)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(rewriter->module, op,
                                               &cache_policy)) {
    return iree_ok_status();
  }
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD_MASK: {
      loom_value_slice_t indices = loom_vector_load_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_load_mask_static_indices(op);
      loom_type_t result_type = {0};
      if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_vector_load_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_load_mask_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          cache_policy.cache_scope, cache_policy.cache_temporal, result_type,
          op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_LOAD_EXPAND: {
      loom_value_slice_t indices = loom_vector_load_expand_indices(op);
      loom_attribute_t static_indices =
          loom_vector_load_expand_static_indices(op);
      loom_type_t result_type = {0};
      if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_vector_load_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_load_expand_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          cache_policy.cache_scope, cache_policy.cache_temporal, result_type,
          op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_GATHER_MASK: {
      loom_value_slice_t indices = loom_vector_gather_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_gather_mask_static_indices(op);
      loom_type_t result_type = {0};
      if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_vector_gather_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_gather_mask_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          loom_vector_gather_mask_offsets(op), cache_policy.cache_scope,
          cache_policy.cache_temporal, result_type, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_ATOMIC_RMW_MASK: {
      loom_value_slice_t indices = loom_vector_atomic_rmw_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_atomic_rmw_mask_static_indices(op);
      loom_type_t result_type = {0};
      if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_vector_atomic_rmw_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_atomic_rmw_mask_kind(op),
          loom_vector_atomic_rmw_mask_value(op),
          loom_vector_atomic_rmw_mask_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          loom_vector_atomic_rmw_mask_offsets(op),
          loom_vector_atomic_rmw_mask_ordering(op),
          loom_vector_atomic_rmw_mask_scope(op), cache_policy.cache_scope,
          cache_policy.cache_temporal, result_type, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_STORE_MASK: {
      loom_value_slice_t indices = loom_vector_store_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_store_mask_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_store_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_store_mask_value(op), loom_vector_store_mask_view(op),
          indices.values, indices.count, static_indices.i64_array,
          static_indices.count, cache_policy.cache_scope,
          cache_policy.cache_temporal, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    case LOOM_OP_VECTOR_STORE_COMPRESS: {
      loom_value_slice_t indices = loom_vector_store_compress_indices(op);
      loom_attribute_t static_indices =
          loom_vector_store_compress_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_store_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_store_compress_value(op),
          loom_vector_store_compress_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          cache_policy.cache_scope, cache_policy.cache_temporal, op->location,
          &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    case LOOM_OP_VECTOR_SCATTER_MASK: {
      loom_value_slice_t indices = loom_vector_scatter_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_scatter_mask_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_scatter_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_scatter_mask_value(op), loom_vector_scatter_mask_view(op),
          indices.values, indices.count, static_indices.i64_array,
          static_indices.count, loom_vector_scatter_mask_offsets(op),
          cache_policy.cache_scope, cache_policy.cache_temporal, op->location,
          &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    case LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK: {
      loom_value_slice_t indices = loom_vector_atomic_reduce_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_atomic_reduce_mask_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_atomic_reduce_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_atomic_reduce_mask_kind(op),
          loom_vector_atomic_reduce_mask_value(op),
          loom_vector_atomic_reduce_mask_view(op), indices.values,
          indices.count, static_indices.i64_array, static_indices.count,
          loom_vector_atomic_reduce_mask_offsets(op),
          loom_vector_atomic_reduce_mask_ordering(op),
          loom_vector_atomic_reduce_mask_scope(op), cache_policy.cache_scope,
          cache_policy.cache_temporal, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    default:
      return iree_ok_status();
  }

  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_all_false_masked_memory(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  loom_value_id_t passthrough = LOOM_VALUE_ID_INVALID;
  bool has_passthrough_result = false;
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD_MASK:
      mask = loom_vector_load_mask_mask(op);
      passthrough = loom_vector_load_mask_passthrough(op);
      has_passthrough_result = true;
      break;
    case LOOM_OP_VECTOR_LOAD_EXPAND:
      mask = loom_vector_load_expand_mask(op);
      passthrough = loom_vector_load_expand_passthrough(op);
      has_passthrough_result = true;
      break;
    case LOOM_OP_VECTOR_GATHER_MASK:
      mask = loom_vector_gather_mask_mask(op);
      passthrough = loom_vector_gather_mask_passthrough(op);
      has_passthrough_result = true;
      break;
    case LOOM_OP_VECTOR_ATOMIC_RMW_MASK:
      mask = loom_vector_atomic_rmw_mask_mask(op);
      passthrough = loom_vector_atomic_rmw_mask_passthrough(op);
      has_passthrough_result = true;
      break;
    case LOOM_OP_VECTOR_STORE_MASK:
      mask = loom_vector_store_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_STORE_COMPRESS:
      mask = loom_vector_store_compress_mask(op);
      break;
    case LOOM_OP_VECTOR_SCATTER_MASK:
      mask = loom_vector_scatter_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK:
      mask = loom_vector_atomic_reduce_mask_mask(op);
      break;
    default:
      return iree_ok_status();
  }

  if (!loom_vector_value_is_all_bool(rewriter, mask, false)) {
    return iree_ok_status();
  }

  if (has_passthrough_result) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
        op, rewriter, passthrough));
  } else {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_contiguous_gather_scatter(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_value_id_t offsets = LOOM_VALUE_ID_INVALID;
  loom_value_id_t shaped_value = LOOM_VALUE_ID_INVALID;
  switch (op->kind) {
    case LOOM_OP_VECTOR_GATHER:
      offsets = loom_vector_gather_offsets(op);
      shaped_value = loom_vector_gather_result(op);
      break;
    case LOOM_OP_VECTOR_GATHER_MASK:
      offsets = loom_vector_gather_mask_offsets(op);
      shaped_value = loom_vector_gather_mask_result(op);
      break;
    case LOOM_OP_VECTOR_SCATTER:
      offsets = loom_vector_scatter_offsets(op);
      shaped_value = loom_vector_scatter_value(op);
      break;
    case LOOM_OP_VECTOR_SCATTER_MASK:
      offsets = loom_vector_scatter_mask_offsets(op);
      shaped_value = loom_vector_scatter_mask_value(op);
      break;
    default:
      return iree_ok_status();
  }

  loom_type_t offsets_type = loom_module_value_type(rewriter->module, offsets);
  loom_type_t shaped_type =
      loom_module_value_type(rewriter->module, shaped_value);
  if (!loom_type_is_vector(shaped_type) || loom_type_rank(shaped_type) != 1 ||
      !loom_vector_value_is_unit_stride_iota(rewriter, offsets, offsets_type)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(rewriter->module, op,
                                               &cache_policy)) {
    return iree_ok_status();
  }
  switch (op->kind) {
    case LOOM_OP_VECTOR_GATHER: {
      loom_value_slice_t indices = loom_vector_gather_indices(op);
      loom_attribute_t static_indices = loom_vector_gather_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_load_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_gather_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          cache_policy.cache_scope, cache_policy.cache_temporal, shaped_type,
          op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_GATHER_MASK: {
      loom_value_slice_t indices = loom_vector_gather_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_gather_mask_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_load_mask_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_gather_mask_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          loom_vector_gather_mask_mask(op),
          loom_vector_gather_mask_passthrough(op), cache_policy.cache_scope,
          cache_policy.cache_temporal, shaped_type, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_SCATTER: {
      loom_value_slice_t indices = loom_vector_scatter_indices(op);
      loom_attribute_t static_indices = loom_vector_scatter_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_store_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_scatter_value(op), loom_vector_scatter_view(op),
          indices.values, indices.count, static_indices.i64_array,
          static_indices.count, cache_policy.cache_scope,
          cache_policy.cache_temporal, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    case LOOM_OP_VECTOR_SCATTER_MASK: {
      loom_value_slice_t indices = loom_vector_scatter_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_scatter_mask_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_store_mask_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_scatter_mask_value(op), loom_vector_scatter_mask_view(op),
          indices.values, indices.count, static_indices.i64_array,
          static_indices.count, loom_vector_scatter_mask_mask(op),
          cache_policy.cache_scope, cache_policy.cache_temporal, op->location,
          &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    default:
      return iree_ok_status();
  }

  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_canonicalize_extf(loom_op_t* op,
                                                   loom_rewriter_t* rewriter,
                                                   bool* out_changed) {
  *out_changed = false;
  loom_value_id_t input = loom_vector_extf_input(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  if (loom_type_equal(input_type, result_type)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, input));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_op_t* input_def = NULL;
  if (!loom_vector_value_def_op(rewriter, input, &input_def)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_canonicalize_float_cast_splat(
      op, rewriter, LOOM_OP_VECTOR_EXTF, input_def, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }
  if (loom_vector_extf_isa(input_def)) {
    loom_value_id_t inner_input = loom_vector_extf_input(input_def);
    return loom_vector_replace_single_result_with_float_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input), out_changed);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_fptrunc(loom_op_t* op,
                                                      loom_rewriter_t* rewriter,
                                                      bool* out_changed) {
  *out_changed = false;
  loom_value_id_t input = loom_vector_fptrunc_input(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  if (loom_type_equal(input_type, result_type)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, input));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_op_t* input_def = NULL;
  if (!loom_vector_value_def_op(rewriter, input, &input_def)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_canonicalize_float_cast_splat(
      op, rewriter, LOOM_OP_VECTOR_FPTRUNC, input_def, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }
  if (loom_vector_extf_isa(input_def)) {
    loom_value_id_t inner_input = loom_vector_extf_input(input_def);
    return loom_vector_replace_single_result_with_float_resize(
        op, rewriter, inner_input,
        loom_module_value_type(rewriter->module, inner_input), out_changed);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Entry points
//===----------------------------------------------------------------------===//

typedef iree_status_t (*loom_vector_try_canonicalize_fn_t)(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed);

iree_status_t loom_vector_uniform_result_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  bool changed = false;
  return loom_vector_canonicalize_uniform_result(op, rewriter, &changed);
}

static iree_status_t loom_vector_canonicalize_uniform_then(
    loom_op_t* op, loom_rewriter_t* rewriter,
    loom_vector_try_canonicalize_fn_t specific_canonicalize) {
  bool changed = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_canonicalize_uniform_result(op, rewriter, &changed));
  if (changed) return iree_ok_status();
  return specific_canonicalize(op, rewriter, &changed);
}

iree_status_t loom_vector_from_elements_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_from_elements);
}

iree_status_t loom_vector_iota_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(op, rewriter,
                                               loom_vector_canonicalize_iota);
}

iree_status_t loom_vector_extract_canonicalize(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_extract);
}

iree_status_t loom_vector_insert_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(op, rewriter,
                                               loom_vector_canonicalize_insert);
}

iree_status_t loom_vector_extf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(op, rewriter,
                                               loom_vector_canonicalize_extf);
}

iree_status_t loom_vector_fptrunc_canonicalize(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_fptrunc);
}

iree_status_t loom_vector_shuffle_canonicalize(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_shuffle);
}

iree_status_t loom_vector_select_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(op, rewriter,
                                               loom_vector_canonicalize_select);
}

iree_status_t loom_vector_comparison_canonicalize(loom_op_t* op,
                                                  loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_comparison);
}

iree_status_t loom_vector_binary_identity_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_binary_identity);
}

iree_status_t loom_vector_addf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(op, rewriter,
                                               loom_vector_canonicalize_addf);
}

iree_status_t loom_vector_subf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(op, rewriter,
                                               loom_vector_canonicalize_subf);
}

iree_status_t loom_vector_mulf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(op, rewriter,
                                               loom_vector_canonicalize_mulf);
}

iree_status_t loom_vector_divf_canonicalize(loom_op_t* op,
                                            loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(op, rewriter,
                                               loom_vector_canonicalize_divf);
}

iree_status_t loom_vector_decode_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  bool changed = false;
  return loom_vector_canonicalize_dense_q8_0_decode(op, rewriter, &changed);
}

iree_status_t loom_vector_gather_scatter_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  bool changed = false;
  return loom_vector_canonicalize_contiguous_gather_scatter(op, rewriter,
                                                            &changed);
}

iree_status_t loom_vector_masked_memory_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  bool changed = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_canonicalize_all_false_masked_memory(op, rewriter, &changed));
  if (changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_vector_canonicalize_contiguous_gather_scatter(
      op, rewriter, &changed));
  if (changed) return iree_ok_status();
  return loom_vector_canonicalize_all_true_masked_memory(op, rewriter,
                                                         &changed);
}
