// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the test dialect.
//
// test.constant: produces exact facts from the constant attribute.
// test.fact_*: expose individual analysis facts as observable values
// for testing. Each reads one field from its input's facts and returns
// it as an exact constant, which the rewriter materializes into a
// scalar.constant during canonicalization.

#include "loom/ir/facts.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/storage_facts.h"
#include "loom/ops/test/ops.h"
#include "loom/util/fact_table.h"

//===----------------------------------------------------------------------===//
// test.addi
//===----------------------------------------------------------------------===//

iree_status_t loom_test_addi_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  loom_value_facts_addi(&operand_facts[0], &operand_facts[1], &result_facts[0]);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.constant
//===----------------------------------------------------------------------===//

iree_status_t loom_test_constant_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_attribute_t attr = loom_op_attrs(op)[0];
  loom_value_id_t result_id = loom_test_constant_result(op);
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_scalar_type_is_float(loom_type_element_type(result_type))) {
    result_facts[0] = loom_value_facts_exact_f64(loom_attr_as_f64(attr));
  } else {
    result_facts[0] = loom_value_facts_exact_i64(loom_attr_as_i64(attr));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.fact_* — value facts inspection ops
//===----------------------------------------------------------------------===//
//
// Each reads one property from operand_facts[0] and returns an exact
// value. The rewriter's try_fold sees the exact output and materializes
// a scalar.constant, making the analysis state observable in .loom-test
// fixtures.

iree_status_t loom_test_fact_range_lo_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(operand_facts[0].range_lo);
  return iree_ok_status();
}

iree_status_t loom_test_fact_range_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(operand_facts[0].range_hi);
  return iree_ok_status();
}

iree_status_t loom_test_fact_all_equal_range_lo_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  if (!loom_value_facts_query_all_equal_element(context, operand_facts[0],
                                                &element_facts)) {
    result_facts[0] = loom_value_facts_exact_i64(INT64_MIN);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(element_facts.range_lo);
  return iree_ok_status();
}

iree_status_t loom_test_fact_all_equal_range_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  if (!loom_value_facts_query_all_equal_element(context, operand_facts[0],
                                                &element_facts)) {
    result_facts[0] = loom_value_facts_exact_i64(INT64_MIN);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(element_facts.range_hi);
  return iree_ok_status();
}

iree_status_t loom_test_fact_divisor_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(operand_facts[0].known_divisor);
  return iree_ok_status();
}

iree_status_t loom_test_fact_non_negative_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_non_negative(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_non_zero_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_non_zero(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_positive_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_positive(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_power_of_two_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_power_of_two(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_uniform_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_uniform(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_lane_varying_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_lane_varying(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_lane_predicate_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_lane_predicate(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_subgroup_lane_mask_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_subgroup_lane_mask(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_is_vector_iota_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_query_vector_iota(context, operand_facts[0], NULL) ? 1
                                                                          : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_is_vector_prefix_mask_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_query_vector_prefix_mask(context, operand_facts[0], NULL)
          ? 1
          : 0);
  return iree_ok_status();
}

static loom_value_fact_encoding_summary_t loom_test_encoding_summary_or_empty(
    const loom_fact_context_t* context, loom_value_facts_t facts) {
  loom_value_fact_encoding_summary_t summary = {0};
  (void)loom_value_facts_query_encoding_summary(context, facts, &summary);
  return summary;
}

iree_status_t loom_test_fact_encoding_layout_kind_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_encoding_summary_t summary =
      loom_test_encoding_summary_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64((int64_t)summary.address_layout.kind);
  return iree_ok_status();
}

iree_status_t loom_test_fact_encoding_layout_stride_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_encoding_summary_t summary =
      loom_test_encoding_summary_or_empty(context, operand_facts[0]);
  int64_t axis = loom_test_fact_encoding_layout_stride_hi_axis(op);
  if (summary.address_layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
      axis < 0 || axis >= summary.address_layout.rank ||
      !summary.address_layout.strides) {
    result_facts[0] = loom_value_facts_exact_i64(INT64_MIN);
    return iree_ok_status();
  }
  result_facts[0] =
      loom_value_facts_exact_i64(summary.address_layout.strides[axis].range_hi);
  return iree_ok_status();
}

static bool loom_test_string_id_equal(const loom_module_t* module,
                                      loom_string_id_t string_id,
                                      iree_string_view_t expected) {
  if (!module || string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

iree_status_t loom_test_fact_encoding_matrix_field_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_encoding_summary_t summary =
      loom_test_encoding_summary_or_empty(context, operand_facts[0]);
  loom_value_fact_storage_schema_t schema = summary.storage_schema;
  loom_value_fact_encoded_operand_schema_t encoded = schema.encoded_operand;
  loom_string_id_t field = loom_test_fact_encoding_matrix_field_field(op);
  int64_t value = INT64_MIN;
  if (loom_test_string_id_equal(module, field, IREE_SV("element_format"))) {
    value = (int64_t)encoded.element_format;
  } else if (loom_test_string_id_equal(module, field,
                                       IREE_SV("payload_packing"))) {
    value = (int64_t)encoded.payload_packing;
  } else if (loom_test_string_id_equal(module, field,
                                       IREE_SV("scale_topology"))) {
    value = (int64_t)encoded.scale_topology;
  } else if (loom_test_string_id_equal(module, field,
                                       IREE_SV("scale_format"))) {
    value = (int64_t)encoded.scale_format;
  } else if (loom_test_string_id_equal(module, field,
                                       IREE_SV("secondary_scale_format"))) {
    value = (int64_t)encoded.secondary_scale_format;
  } else if (loom_test_string_id_equal(module, field, IREE_SV("affine"))) {
    value = (int64_t)encoded.affine_policy;
  } else if (loom_test_string_id_equal(module, field, IREE_SV("rounding"))) {
    value = (int64_t)encoded.rounding_policy;
  } else if (loom_test_string_id_equal(module, field, IREE_SV("codebook"))) {
    value = (int64_t)encoded.codebook_policy;
  } else if (loom_test_string_id_equal(module, field, IREE_SV("sparsity"))) {
    value = (int64_t)encoded.sparsity_policy;
  } else if (loom_test_string_id_equal(module, field,
                                       IREE_SV("payload_registers"))) {
    value = (int64_t)encoded.payload_register_count;
  } else if (loom_test_string_id_equal(module, field,
                                       IREE_SV("payload_elements"))) {
    value = (int64_t)encoded.payload_element_count;
  } else if (loom_test_string_id_equal(module, field,
                                       IREE_SV("scale_group_elements"))) {
    value = (int64_t)encoded.scale_group_element_count;
  } else if (loom_test_string_id_equal(module, field,
                                       IREE_SV("scale_operands"))) {
    value = (int64_t)encoded.scale_operand_count;
  } else if (loom_test_string_id_equal(module, field,
                                       IREE_SV("zero_scale_fallback"))) {
    value = iree_any_bit_set(
                encoded.flags,
                LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK)
                ? 1
                : 0;
  } else if (loom_test_string_id_equal(module, field, IREE_SV("static_spec"))) {
    value = (int64_t)schema.static_spec_encoding_id;
  }
  result_facts[0] = loom_value_facts_exact_i64(value);
  return iree_ok_status();
}

iree_status_t loom_test_fact_is_buffer_reference_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_query_buffer_reference(context, operand_facts[0], NULL)
          ? 1
          : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_is_view_reference_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_query_view_reference(context, operand_facts[0], NULL)
          ? 1
          : 0);
  return iree_ok_status();
}

static int64_t loom_test_memory_space_or_unknown(
    loom_value_fact_memory_space_t memory_space) {
  if (memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN) return -1;
  return (int64_t)memory_space;
}

iree_status_t loom_test_fact_buffer_memory_space_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference = {0};
  if (!loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                               &reference)) {
    result_facts[0] = loom_value_facts_exact_i64(-1);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(
      loom_test_memory_space_or_unknown(reference.memory_space));
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_memory_space_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_value_facts_query_view_reference(context, operand_facts[0],
                                             &reference)) {
    result_facts[0] = loom_value_facts_exact_i64(-1);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(
      loom_test_memory_space_or_unknown(reference.memory_space));
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_root_matches_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(context, operand_facts[0],
                                             &view_reference)) {
    result_facts[0] = loom_value_facts_exact_i64(0);
    return iree_ok_status();
  }

  loom_value_id_t root_value_id = loom_op_const_operands(op)[1];
  loom_value_fact_buffer_reference_t buffer_reference = {0};
  loom_value_fact_view_reference_t other_view_reference = {0};
  if (loom_value_facts_query_buffer_reference(context, operand_facts[1],
                                              &buffer_reference)) {
    root_value_id = buffer_reference.root_value_id;
  } else if (loom_value_facts_query_view_reference(context, operand_facts[1],
                                                   &other_view_reference)) {
    root_value_id = other_view_reference.root_value_id;
  }

  result_facts[0] = loom_value_facts_exact_i64(
      view_reference.root_value_id == root_value_id ? 1 : 0);
  return iree_ok_status();
}

static loom_value_fact_alias_scope_id_t loom_test_alias_scope_or_none(
    const loom_fact_context_t* context, loom_value_facts_t facts) {
  loom_value_fact_buffer_reference_t buffer_reference = {0};
  if (loom_value_facts_query_buffer_reference(context, facts,
                                              &buffer_reference)) {
    return buffer_reference.alias_scope_id;
  }
  loom_value_fact_view_reference_t view_reference = {0};
  if (loom_value_facts_query_view_reference(context, facts, &view_reference)) {
    return view_reference.alias_scope_id;
  }
  return LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
}

iree_status_t loom_test_fact_alias_scope_known_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_test_alias_scope_or_none(context, operand_facts[0]) ==
              LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE
          ? 0
          : 1);
  return iree_ok_status();
}

iree_status_t loom_test_fact_alias_scope_matches_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_alias_scope_id_t lhs_scope =
      loom_test_alias_scope_or_none(context, operand_facts[0]);
  loom_value_fact_alias_scope_id_t rhs_scope =
      loom_test_alias_scope_or_none(context, operand_facts[1]);
  result_facts[0] = loom_value_facts_exact_i64(
      lhs_scope != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE && lhs_scope == rhs_scope
          ? 1
          : 0);
  return iree_ok_status();
}

static loom_value_fact_view_reference_t loom_test_view_reference_or_empty(
    const loom_fact_context_t* context, loom_value_facts_t facts) {
  loom_value_fact_view_reference_t reference = {
      .base_byte_offset = loom_value_facts_exact_i64(INT64_MIN),
      .footprint_byte_length = loom_value_facts_exact_i64(INT64_MIN),
      .minimum_alignment = 0,
      .root_minimum_alignment = 0,
      .static_element_byte_count = -1,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = LOOM_VALUE_ID_INVALID,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  (void)loom_value_facts_query_view_reference(context, facts, &reference);
  return reference;
}

iree_status_t loom_test_fact_view_byte_offset_lo_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.base_byte_offset.range_lo);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_byte_offset_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.base_byte_offset.range_hi);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_byte_length_lo_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.footprint_byte_length.range_lo);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_byte_length_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.footprint_byte_length.range_hi);
  return iree_ok_status();
}

iree_status_t loom_test_fact_buffer_min_alignment_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference = {
      .maximum_byte_extent = loom_value_facts_exact_i64(INT64_MIN),
      .minimum_alignment = 0,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = LOOM_VALUE_ID_INVALID,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  (void)loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                                &reference);
  result_facts[0] =
      loom_value_facts_exact_i64((int64_t)reference.minimum_alignment);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_min_alignment_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64((int64_t)reference.minimum_alignment);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_root_min_alignment_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64((int64_t)reference.root_minimum_alignment);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_element_bytes_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.static_element_byte_count);
  return iree_ok_status();
}

static loom_storage_reference_facts_t loom_test_storage_reference_or_empty(
    const loom_fact_context_t* context, loom_value_facts_t facts) {
  loom_storage_reference_facts_t reference = {
      .byte_offset = loom_value_facts_exact_i64(INT64_MIN),
      .target_byte_offset = loom_value_facts_exact_i64(INT64_MIN),
      .valid_byte_length = loom_value_facts_exact_i64(INT64_MIN),
      .minimum_alignment = 0,
      .backing_value_id = LOOM_VALUE_ID_INVALID,
      .storage_space = LOOM_STORAGE_SPACE_COUNT_,
  };
  (void)loom_storage_facts_query_reference(context, facts, &reference);
  return reference;
}

iree_status_t loom_test_fact_is_storage_reference_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_storage_facts_query_reference(context, operand_facts[0], NULL) ? 1
                                                                          : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_storage_same_backing_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_storage_reference_facts_t lhs = {0};
  loom_storage_reference_facts_t rhs = {0};
  if (!loom_storage_facts_query_reference(context, operand_facts[0], &lhs) ||
      !loom_storage_facts_query_reference(context, operand_facts[1], &rhs)) {
    result_facts[0] = loom_value_facts_exact_i64(0);
    return iree_ok_status();
  }
  result_facts[0] =
      loom_value_facts_exact_i64(lhs.backing_value_id == rhs.backing_value_id &&
                                         lhs.storage_space == rhs.storage_space
                                     ? 1
                                     : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_storage_byte_offset_lo_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_storage_reference_facts_t reference =
      loom_test_storage_reference_or_empty(context, operand_facts[0]);
  result_facts[0] = loom_value_facts_exact_i64(reference.byte_offset.range_lo);
  return iree_ok_status();
}

iree_status_t loom_test_fact_storage_byte_offset_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_storage_reference_facts_t reference =
      loom_test_storage_reference_or_empty(context, operand_facts[0]);
  result_facts[0] = loom_value_facts_exact_i64(reference.byte_offset.range_hi);
  return iree_ok_status();
}

iree_status_t loom_test_fact_storage_byte_offset_divisor_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_storage_reference_facts_t reference =
      loom_test_storage_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.byte_offset.known_divisor);
  return iree_ok_status();
}

iree_status_t loom_test_fact_storage_byte_length_lo_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_storage_reference_facts_t reference =
      loom_test_storage_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.valid_byte_length.range_lo);
  return iree_ok_status();
}

iree_status_t loom_test_fact_storage_byte_length_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_storage_reference_facts_t reference =
      loom_test_storage_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.valid_byte_length.range_hi);
  return iree_ok_status();
}

iree_status_t loom_test_fact_storage_min_alignment_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_storage_reference_facts_t reference =
      loom_test_storage_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64((int64_t)reference.minimum_alignment);
  return iree_ok_status();
}

iree_status_t loom_test_fact_storage_space_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_storage_reference_facts_t reference =
      loom_test_storage_reference_or_empty(context, operand_facts[0]);
  int64_t storage_space = reference.storage_space == LOOM_STORAGE_SPACE_COUNT_
                              ? -1
                              : (int64_t)reference.storage_space;
  result_facts[0] = loom_value_facts_exact_i64(storage_space);
  return iree_ok_status();
}
