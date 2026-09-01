// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/lower_rule_facts.h"
#include "loom/codegen/low/lower/lower_rule_source_memory.h"
#include "loom/codegen/low/lower/lower_rule_value.h"
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/ir/context.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/storage.h"

static iree_status_t loom_low_lower_rule_can_materialize_value(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, bool* out_can_materialize) {
  *out_can_materialize = false;
  if (match_context->can_materialize.fn == NULL) {
    return iree_ok_status();
  }
  return match_context->can_materialize.fn(
      match_context->can_materialize.user_data, match_context, rule_set,
      source_op, value_ref_index,
      loom_low_lower_rule_source_value(match_context->module, rule_set,
                                       source_op, value_ref_index),
      out_can_materialize);
}

static iree_status_t loom_low_lower_rule_descriptor_available(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref, bool* out_available) {
  *out_available = false;
  const loom_low_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
      match_context, rule_set, descriptor_ref, &descriptor));
  if (descriptor == NULL) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < descriptor->feature_mask_word_count; ++i) {
    const uint32_t word_index = descriptor->feature_mask_word_start + i;
    const uint64_t required_bits =
        match_context->descriptor_set->feature_mask_words[word_index];
    const uint64_t available_bits = i == 0 ? match_context->feature_bits : 0;
    if ((required_bits & ~available_bits) != 0) {
      return iree_ok_status();
    }
  }
  *out_available = true;
  return iree_ok_status();
}
static bool loom_low_lower_rule_type_matches(
    const loom_low_lower_type_pattern_t* pattern, loom_type_t type) {
  if (iree_any_bit_set(pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND) &&
      loom_type_kind(type) != pattern->type_kind) {
    return false;
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT) &&
      !iree_any_bit_set(
          pattern->element_type_mask,
          LOOM_LOW_LOWER_SCALAR_TYPE_BIT(loom_type_element_type(type)))) {
    return false;
  }
  if (iree_any_bit_set(pattern->flags, LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK) &&
      loom_type_rank(type) != pattern->rank) {
    return false;
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0)) {
    if (loom_type_rank(type) == 0 || loom_type_dim_is_dynamic_at(type, 0)) {
      return false;
    }
    if (loom_type_dim_static_size_at(type, 0) != pattern->static_dim0) {
      return false;
    }
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE)) {
    if (loom_type_rank(type) == 0 || loom_type_dim_is_dynamic_at(type, 0)) {
      return false;
    }
    const int64_t static_dim0 = loom_type_dim_static_size_at(type, 0);
    if (static_dim0 < pattern->static_dim0_min ||
        static_dim0 > pattern->static_dim0_max) {
      return false;
    }
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM1)) {
    if (loom_type_rank(type) < 2 || loom_type_dim_is_dynamic_at(type, 1)) {
      return false;
    }
    if (loom_type_dim_static_size_at(type, 1) != pattern->static_dim1) {
      return false;
    }
  }
  if (iree_any_bit_set(
          pattern->flags,
          LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_ELEMENT_COUNT_RANGE)) {
    uint64_t static_element_count = 0;
    if (!loom_type_static_element_count(type, &static_element_count)) {
      return false;
    }
    if (static_element_count < pattern->static_element_count_min ||
        static_element_count > pattern->static_element_count_max) {
      return false;
    }
  }
  return true;
}

static bool loom_low_lower_rule_vector_extract_tail_type_matches(
    loom_type_t source_type, uint16_t consumed_rank, loom_type_t result_type) {
  const uint8_t source_rank = loom_type_rank(source_type);
  if (consumed_rank > source_rank) return false;
  if (loom_type_element_type(source_type) !=
      loom_type_element_type(result_type)) {
    return false;
  }
  if (loom_type_is_scalar(result_type)) {
    return consumed_rank == source_rank;
  }
  if (!loom_type_is_vector(result_type)) return false;
  const uint8_t result_rank = loom_type_rank(result_type);
  if (consumed_rank + result_rank != source_rank) return false;
  for (uint8_t i = 0; i < result_rank; ++i) {
    if (loom_type_dim(source_type, consumed_rank + i) !=
        loom_type_dim(result_type, i)) {
      return false;
    }
  }
  return true;
}

static bool loom_low_lower_rule_vector_extract_shape_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_guard_t* guard) {
  if (!loom_vector_extract_isa(source_op)) return false;
  if (guard->attr_index >= source_op->attribute_count) return false;
  loom_attribute_t static_indices =
      loom_op_const_attrs(source_op)[guard->attr_index];
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) return false;

  const loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, guard->value_ref_index);
  const loom_value_id_t result_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, guard->other_value_ref_index);
  const loom_type_t source_type =
      loom_module_value_type(match_context->module, source_value_id);
  const loom_type_t result_type =
      loom_module_value_type(match_context->module, result_value_id);
  if (!loom_type_is_vector(source_type)) return false;

  const loom_value_slice_t dynamic_indices =
      loom_vector_extract_indices(source_op);
  if (static_indices.count == 1 && static_indices.i64_array[0] == INT64_MIN) {
    return dynamic_indices.count == 1 && loom_type_rank(source_type) == 1 &&
           loom_type_element_type(source_type) ==
               loom_type_element_type(result_type) &&
           loom_type_is_scalar(result_type);
  }

  if (dynamic_indices.count != 0 ||
      static_indices.count > loom_type_rank(source_type)) {
    return false;
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    const int64_t index = static_indices.i64_array[i];
    if (index < 0 || loom_type_dim_is_dynamic_at(source_type, i) ||
        index >= loom_type_dim_static_size_at(source_type, i)) {
      return false;
    }
  }
  return loom_low_lower_rule_vector_extract_tail_type_matches(
      source_type, static_indices.count, result_type);
}

static iree_status_t loom_low_lower_rule_mapped_value(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  IREE_ASSERT(match_context->map_value.fn != NULL);
  loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  return match_context->map_value.fn(match_context->map_value.user_data,
                                     match_context, source_op, source_value_id,
                                     out_mapped_value);
}

static iree_status_t loom_low_lower_rule_mapped_value_register_class_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    loom_low_lower_rule_mapped_value_t mapped_value,
    uint16_t descriptor_register_class_id, bool* out_matches) {
  *out_matches = false;
  if (!mapped_value.is_register) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(descriptor_register_class_id,
                 match_context->descriptor_set->reg_class_count);
  *out_matches =
      mapped_value.descriptor_register_class_id == descriptor_register_class_id;
  return iree_ok_status();
}

static bool loom_low_lower_rule_integer_element_range_facts(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  if (fact_table == NULL) {
    return false;
  }

  const loom_type_t type = loom_module_value_type(module, value_id);
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(fact_table, value_id);
  if (loom_type_is_scalar(type)) {
    if (loom_scalar_type_is_float(loom_type_element_type(type))) {
      return false;
    }
    *out_facts = facts;
    return true;
  }
  if (!loom_type_is_vector(type) ||
      loom_scalar_type_is_float(loom_type_element_type(type))) {
    return false;
  }

  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                             &uniform)) {
    *out_facts = uniform.element;
    return true;
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_value_facts_query_small_static_lanes(&fact_table->context, facts,
                                                &lanes)) {
    if (lanes.count == 0) {
      return false;
    }
    loom_value_facts_t aggregate = lanes.lanes[0];
    for (iree_host_size_t i = 1; i < lanes.count; ++i) {
      loom_value_facts_t next_aggregate;
      loom_value_facts_meet(&aggregate, &lanes.lanes[i], &next_aggregate);
      aggregate = next_aggregate;
    }
    *out_facts = aggregate;
    return true;
  }

  loom_value_fact_vector_iota_t iota = {0};
  uint64_t lane_count = 0;
  int64_t base = 0;
  int64_t step = 0;
  if (loom_value_facts_query_vector_iota(&fact_table->context, facts, &iota) &&
      loom_type_static_element_count(type, &lane_count) && lane_count > 0 &&
      loom_value_facts_as_exact_i64(iota.base, &base) &&
      loom_value_facts_as_exact_i64(iota.step, &step) &&
      lane_count <= (uint64_t)INT64_MAX) {
    int64_t final_delta = 0;
    int64_t final_value = 0;
    if (!iree_checked_mul_i64((int64_t)(lane_count - 1), step, &final_delta) ||
        !iree_checked_add_i64(base, final_delta, &final_value)) {
      return false;
    }
    *out_facts = loom_value_facts_make(iree_min(base, final_value),
                                       iree_max(base, final_value), 1);
    return true;
  }

  return false;
}

static bool loom_low_lower_rule_result_index_assume_facts(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_facts_t* out_facts) {
  *out_facts = loom_value_facts_unknown();
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  if (value_ref->kind != LOOM_LOW_LOWER_VALUE_REF_RESULT) {
    return false;
  }

  // A single-use identity assume is the only observable result contract.
  // Borrowing its facts here keeps target guards strict for unassumed or
  // multiply-used producers while allowing authored postconditions to prove a
  // dynamic address calculation.
  const loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  const loom_value_t* source_value =
      loom_module_value(match_context->module, source_value_id);
  if (!loom_value_has_single_use(source_value)) return false;

  const loom_use_t use = loom_value_uses(source_value)[0];
  const loom_op_t* user_op = loom_use_user_op(use);
  if (!user_op || !loom_index_assume_isa(user_op)) return false;

  const uint16_t operand_index = loom_use_operand_index(use);
  loom_value_slice_t assumed_values = loom_index_assume_values(user_op);
  loom_value_slice_t assumed_results = loom_index_assume_results(user_op);
  if (operand_index >= assumed_values.count ||
      operand_index >= assumed_results.count ||
      assumed_values.values[operand_index] != source_value_id) {
    return false;
  }

  return loom_low_lower_rule_integer_element_range_facts(
      match_context->module, match_context->fact_table,
      assumed_results.values[operand_index], out_facts);
}

static bool loom_low_lower_rule_value_integer_element_range_facts(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_facts_t* out_facts) {
  if (loom_low_lower_rule_result_index_assume_facts(
          match_context, rule_set, source_op, value_ref_index, out_facts)) {
    return true;
  }
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  return loom_low_lower_rule_integer_element_range_facts(
      match_context->module, match_context->fact_table, value_id, out_facts);
}

static iree_status_t loom_low_lower_rule_value_symbolically_fits_bit_count(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint8_t bit_count, bool is_signed_domain,
    bool* out_matches) {
  *out_matches = false;
  if (bit_count == 0 || bit_count > 64) return iree_ok_status();
  loom_symbolic_expr_context_t* expression_context =
      match_context->symbolic_expr_context;
  if (expression_context == NULL) return iree_ok_status();

  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_symbolic_expr_t value_expression = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
      expression_context, value_id, &value_expression));

  loom_symbolic_expr_t minimum_expression = {0};
  loom_symbolic_expr_constant(0, &minimum_expression);
  loom_symbolic_proof_result_t minimum_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (is_signed_domain) {
    int64_t minimum_value =
        bit_count >= 64 ? INT64_MIN : -(INT64_C(1) << (bit_count - 1));
    loom_symbolic_expr_constant(minimum_value, &minimum_expression);
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_prove_le(expression_context, &minimum_expression,
                                  &value_expression, &minimum_proof));
  if (minimum_proof != LOOM_SYMBOLIC_PROOF_TRUE) {
    return iree_ok_status();
  }

  if (!is_signed_domain && bit_count >= 63) {
    *out_matches = true;
    return iree_ok_status();
  }
  if (is_signed_domain && bit_count >= 64) {
    *out_matches = true;
    return iree_ok_status();
  }

  uint64_t unsigned_maximum =
      bit_count == 64 ? UINT64_MAX : (UINT64_C(1) << bit_count) - 1;
  int64_t maximum_value = is_signed_domain ? (INT64_C(1) << (bit_count - 1)) - 1
                                           : (int64_t)unsigned_maximum;
  loom_symbolic_expr_t maximum_expression = {0};
  loom_symbolic_expr_constant(maximum_value, &maximum_expression);
  loom_symbolic_proof_result_t maximum_proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_prove_le(expression_context, &value_expression,
                                  &maximum_expression, &maximum_proof));
  *out_matches = maximum_proof == LOOM_SYMBOLIC_PROOF_TRUE;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_value_facts_fit_bit_count(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint64_t bit_count, bool is_signed_domain,
    bool* out_matches) {
  *out_matches = false;
  if (bit_count > UINT8_MAX) {
    return iree_ok_status();
  }
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_value_integer_element_range_facts(
          match_context, rule_set, source_op, value_ref_index, &facts)) {
    return iree_ok_status();
  }
  if (is_signed_domain) {
    *out_matches =
        loom_value_facts_fit_signed_bit_count(facts, (uint8_t)bit_count);
  } else {
    *out_matches =
        loom_value_facts_fit_unsigned_bit_count(facts, (uint8_t)bit_count);
  }
  if (*out_matches) return iree_ok_status();
  return loom_low_lower_rule_value_symbolically_fits_bit_count(
      match_context, rule_set, source_op, value_ref_index, (uint8_t)bit_count,
      is_signed_domain, out_matches);
}

static bool loom_low_lower_rule_value_facts_exact_i64(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_immediate_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  int64_t exact_value = 0;
  return loom_value_facts_as_exact_i64(facts, &exact_value);
}

static bool loom_low_lower_rule_value_facts_exact_power_of_two_i64(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_immediate_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  int64_t exact_value = 0;
  return loom_value_facts_as_exact_i64(facts, &exact_value) &&
         iree_math_is_power_of_two_i64(exact_value);
}

static bool loom_low_lower_rule_value_facts_u32_divisor_magic_is_add(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, bool expected_is_add) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_low_lower_u32_divisor_magic_info_t info = {0};
  return loom_low_lower_rule_value_facts_u32_divisor_magic_info(
             match_context->module, match_context->fact_table, value_id,
             &info) &&
         info.is_add == expected_is_add;
}

static bool loom_low_lower_rule_value_facts_exact_float(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_float_immediate_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(
      loom_module_value_type(match_context->module, value_id));
  double value = 0.0;
  return loom_value_facts_as_exact_float(scalar_type, facts, &value);
}

static bool loom_low_lower_rule_value_facts_float_equals(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint64_t expected_bits) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_float_immediate_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(
      loom_module_value_type(match_context->module, value_id));
  double expected_value = 0.0;
  memcpy(&expected_value, &expected_bits, sizeof(expected_value));
  const loom_value_facts_t expected_facts =
      loom_value_facts_exact_float(scalar_type, expected_value);
  return loom_value_facts_is_exact(facts) &&
         facts.range_lo == expected_facts.range_lo;
}

static bool loom_low_lower_rule_value_facts_i64_range(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, int64_t minimum_i64, int64_t maximum_i64) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_element_range_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  return facts.range_lo >= minimum_i64 && facts.range_hi <= maximum_i64;
}

static bool loom_low_lower_rule_value_facts_i64_range_le(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint16_t other_value_ref_index) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_element_range_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  const loom_value_id_t other_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, other_value_ref_index);
  loom_value_facts_t other_facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_element_range_facts(
          match_context->module, match_context->fact_table, other_value_id,
          &other_facts)) {
    return false;
  }
  return facts.range_hi <= other_facts.range_lo;
}

static bool loom_low_lower_rule_value_facts_i64_range_ge(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint16_t other_value_ref_index) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_element_range_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  const loom_value_id_t other_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, other_value_ref_index);
  loom_value_facts_t other_facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_element_range_facts(
          match_context->module, match_context->fact_table, other_value_id,
          &other_facts)) {
    return false;
  }
  return facts.range_lo >= other_facts.range_hi;
}

static bool loom_low_lower_rule_value_storage_element_format(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index,
    loom_value_fact_numeric_format_flags_t expected_format) {
  if (expected_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN) {
    return false;
  }
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  const loom_type_t type =
      loom_module_value_type(match_context->module, value_id);
  const loom_fact_context_t* fact_context =
      match_context->fact_table != NULL ? &match_context->fact_table->context
                                        : NULL;
  loom_value_fact_storage_schema_t storage_schema = {0};
  return loom_encoding_query_type_storage_schema(
             fact_context, match_context->module, type, &storage_schema) &&
         storage_schema.encoded_operand.element_format == expected_format;
}

static bool loom_low_lower_rule_value_memory_space_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint64_t expected_memory_space_mask) {
  if (match_context->fact_table == NULL ||
      expected_memory_space_mask > UINT16_MAX) {
    return false;
  }
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(match_context->fact_table, value_id);
  loom_value_fact_view_reference_t view_reference = {0};
  if (loom_value_facts_query_view_reference(&match_context->fact_table->context,
                                            facts, &view_reference)) {
    return loom_low_lower_rule_memory_space_matches(
        (loom_low_lower_memory_space_mask_t)expected_memory_space_mask,
        view_reference.memory_space);
  }
  loom_value_fact_buffer_reference_t buffer_reference = {0};
  return loom_value_facts_query_buffer_reference(
             &match_context->fact_table->context, facts, &buffer_reference) &&
         loom_low_lower_rule_memory_space_matches(
             (loom_low_lower_memory_space_mask_t)expected_memory_space_mask,
             buffer_reference.memory_space);
}

static bool loom_low_lower_rule_storage_width(
    const loom_op_t* source_op, const loom_low_lower_guard_t* guard,
    uint32_t* out_storage_unit_bit_count, uint32_t* out_width) {
  *out_storage_unit_bit_count = 0;
  *out_width = 0;
  if (guard->attr_index >= source_op->attribute_count ||
      loom_op_const_attrs(source_op)[guard->attr_index].kind != LOOM_ATTR_I64) {
    return false;
  }
  const int64_t width_i64 =
      loom_op_const_attrs(source_op)[guard->attr_index].i64;
  if (width_i64 <= 0 || width_i64 > UINT32_MAX) {
    return false;
  }
  const uint32_t storage_unit_bit_count =
      guard->payload.packed_integer.storage_unit_bit_count;
  const uint32_t width = (uint32_t)width_i64;
  if ((storage_unit_bit_count % width) != 0) {
    return false;
  }
  *out_storage_unit_bit_count = storage_unit_bit_count;
  *out_width = width;
  return true;
}

static bool loom_low_lower_rule_packed_integer_payload_from_lanes_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_guard_t* guard) {
  uint32_t storage_unit_bit_count = 0;
  uint32_t width = 0;
  if (!loom_low_lower_rule_storage_width(source_op, guard,
                                         &storage_unit_bit_count, &width)) {
    return false;
  }

  const loom_value_id_t lane_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, guard->value_ref_index);
  const loom_value_id_t storage_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, guard->other_value_ref_index);
  loom_vector_packed_integer_payload_from_lanes_match_t match = {0};
  if (!loom_vector_packed_integer_payload_from_lanes_match(
          loom_module_value_type(match_context->module, lane_value_id),
          loom_module_value_type(match_context->module, storage_value_id),
          width, storage_unit_bit_count, UINT32_MAX, &match)) {
    return false;
  }

  return (match.result_shape.payload_bit_count %
          guard->payload.packed_integer.storage_payload_multiple) == 0;
}

static bool loom_low_lower_rule_packed_integer_lanes_from_payload_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_guard_t* guard) {
  uint32_t storage_unit_bit_count = 0;
  uint32_t width = 0;
  if (!loom_low_lower_rule_storage_width(source_op, guard,
                                         &storage_unit_bit_count, &width)) {
    return false;
  }

  const loom_value_id_t storage_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, guard->value_ref_index);
  const loom_value_id_t lane_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, guard->other_value_ref_index);
  if (!loom_vector_packed_integer_lanes_from_payload_match(
          loom_module_value_type(match_context->module, storage_value_id),
          loom_module_value_type(match_context->module, lane_value_id), width,
          storage_unit_bit_count,
          guard->payload.packed_integer.storage_payload_multiple,
          guard->payload.packed_integer.maximum_lane_count, NULL)) {
    return false;
  }
  return true;
}

static iree_status_t loom_low_lower_rule_guard_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_guard_t* guard, bool* out_matches) {
  *out_matches = false;
  switch (guard->kind) {
    case LOOM_LOW_LOWER_GUARD_VALUE_TYPE: {
      loom_value_id_t value_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op, guard->value_ref_index);
      loom_type_t type =
          loom_module_value_type(match_context->module, value_id);
      *out_matches = loom_low_lower_rule_type_matches(
          &rule_set->type_patterns[guard->index.type_pattern_index], type);
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_ATTR_KIND:
      if (guard->attr_index >= source_op->attribute_count) {
        return iree_ok_status();
      }
      *out_matches = loom_op_const_attrs(source_op)[guard->attr_index].kind ==
                     guard->attr_kind;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ:
      if (guard->attr_index >= source_op->attribute_count) {
        return iree_ok_status();
      }
      *out_matches = loom_op_const_attrs(source_op)[guard->attr_index].kind ==
                         LOOM_ATTR_ENUM &&
                     loom_op_const_attrs(source_op)[guard->attr_index].raw ==
                         guard->payload.u64;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE:
      if (guard->attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->attr_index].kind !=
              LOOM_ATTR_I64) {
        return iree_ok_status();
      }
      *out_matches = loom_op_const_attrs(source_op)[guard->attr_index].i64 >=
                         guard->payload.i64_range.minimum &&
                     loom_op_const_attrs(source_op)[guard->attr_index].i64 <=
                         guard->payload.i64_range.maximum;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_COUNT_EQ:
      if (guard->attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->attr_index].kind !=
              LOOM_ATTR_I64_ARRAY) {
        return iree_ok_status();
      }
      *out_matches =
          (uint64_t)loom_op_const_attrs(source_op)[guard->attr_index].count ==
          guard->payload.u64;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENT_RANGE:
      if (guard->attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->attr_index].kind !=
              LOOM_ATTR_I64_ARRAY) {
        return iree_ok_status();
      }
      if (guard->index.element_index >=
          loom_op_const_attrs(source_op)[guard->attr_index].count) {
        return iree_ok_status();
      }
      *out_matches = loom_op_const_attrs(source_op)[guard->attr_index]
                             .i64_array[guard->index.element_index] >=
                         guard->payload.i64_range.minimum &&
                     loom_op_const_attrs(source_op)[guard->attr_index]
                             .i64_array[guard->index.element_index] <=
                         guard->payload.i64_range.maximum;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENTS_RANGE: {
      if (guard->attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->attr_index].kind !=
              LOOM_ATTR_I64_ARRAY) {
        return iree_ok_status();
      }
      *out_matches = true;
      loom_attribute_t attr = loom_op_const_attrs(source_op)[guard->attr_index];
      for (uint16_t i = 0; i < attr.count; ++i) {
        if (attr.i64_array[i] < guard->payload.i64_range.minimum ||
            attr.i64_array[i] > guard->payload.i64_range.maximum) {
          *out_matches = false;
          break;
        }
      }
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE:
      return loom_low_lower_rule_descriptor_available(
          match_context, rule_set, guard->descriptor_ref, out_matches);
    case LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE:
      return loom_low_lower_rule_can_materialize_value(
          match_context, rule_set, source_op, guard->value_ref_index,
          out_matches);
    case LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS: {
      loom_low_lower_rule_mapped_value_t mapped_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_mapped_value(
          match_context, rule_set, source_op, guard->value_ref_index,
          &mapped_value));
      return loom_low_lower_rule_mapped_value_register_class_matches(
          match_context, mapped_value, guard->register_class_id, out_matches);
    }
    case LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT: {
      loom_low_lower_rule_mapped_value_t mapped_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_mapped_value(
          match_context, rule_set, source_op, guard->value_ref_index,
          &mapped_value));
      *out_matches = mapped_value.is_register &&
                     mapped_value.register_unit_count == guard->payload.u64;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_STATIC_DIM0_MULTIPLE: {
      IREE_ASSERT_GT(guard->payload.u64, 0);
      loom_value_id_t value_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op, guard->value_ref_index);
      loom_type_t type =
          loom_module_value_type(match_context->module, value_id);
      if (loom_type_rank(type) == 0 || loom_type_dim_is_dynamic_at(type, 0)) {
        return iree_ok_status();
      }
      const int64_t static_dim0 = loom_type_dim_static_size_at(type, 0);
      *out_matches =
          static_dim0 >= 0 && ((uint64_t)static_dim0 % guard->payload.u64) == 0;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT_EQ: {
      loom_low_lower_rule_mapped_value_t lhs_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(
          loom_low_lower_rule_mapped_value(match_context, rule_set, source_op,
                                           guard->value_ref_index, &lhs_value));
      if (!lhs_value.is_register) {
        return iree_ok_status();
      }
      loom_low_lower_rule_mapped_value_t rhs_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_mapped_value(
          match_context, rule_set, source_op, guard->other_value_ref_index,
          &rhs_value));
      if (!rhs_value.is_register) {
        return iree_ok_status();
      }
      *out_matches =
          lhs_value.register_unit_count == rhs_value.register_unit_count;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_STATIC_ELEMENT_COUNT_EQ: {
      const loom_value_id_t lhs_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op, guard->value_ref_index);
      const loom_value_id_t rhs_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op,
          guard->other_value_ref_index);
      uint64_t lhs_count = 0;
      uint64_t rhs_count = 0;
      if (!loom_type_static_element_count(
              loom_module_value_type(match_context->module, lhs_id),
              &lhs_count) ||
          !loom_type_static_element_count(
              loom_module_value_type(match_context->module, rhs_id),
              &rhs_count)) {
        return iree_ok_status();
      }
      *out_matches = lhs_count == rhs_count;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_OPERAND_SEGMENT_COUNT_EQ: {
      if (guard->attr_index > source_op->operand_count) {
        return iree_ok_status();
      }
      const loom_op_vtable_t* vtable = loom_context_resolve_op(
          match_context->module->context, source_op->kind);
      if (vtable == NULL) {
        return iree_ok_status();
      }
      uint16_t segment_count = 0;
      if (guard->attr_index < vtable->fixed_operand_count) {
        segment_count = 1;
      } else if (guard->attr_index == vtable->fixed_operand_count &&
                 iree_any_bit_set(vtable->vtable_flags,
                                  LOOM_OP_VTABLE_VARIADIC_OPERANDS)) {
        segment_count =
            (uint16_t)(source_op->operand_count - guard->attr_index);
      }
      *out_matches = segment_count == guard->payload.u64;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_SIGNED_BIT_COUNT:
      return loom_low_lower_rule_value_facts_fit_bit_count(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->payload.u64, /*is_signed_domain=*/true, out_matches);
    case LOOM_LOW_LOWER_GUARD_VALUE_UNSIGNED_BIT_COUNT:
      return loom_low_lower_rule_value_facts_fit_bit_count(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->payload.u64, /*is_signed_domain=*/false, out_matches);
    case LOOM_LOW_LOWER_GUARD_VALUE_EXACT_I64:
      *out_matches = loom_low_lower_rule_value_facts_exact_i64(
          match_context, rule_set, source_op, guard->value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_EXACT_POWER_OF_TWO_I64:
      *out_matches = loom_low_lower_rule_value_facts_exact_power_of_two_i64(
          match_context, rule_set, source_op, guard->value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_U32_DIVISOR_MAGIC_IS_ADD:
      *out_matches = loom_low_lower_rule_value_facts_u32_divisor_magic_is_add(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->payload.u64 != 0);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_EXACT_FLOAT:
      *out_matches = loom_low_lower_rule_value_facts_exact_float(
          match_context, rule_set, source_op, guard->value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE:
      *out_matches = loom_low_lower_rule_value_facts_i64_range(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->payload.i64_range.minimum, guard->payload.i64_range.maximum);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE_LE:
      *out_matches = loom_low_lower_rule_value_facts_i64_range_le(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->other_value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE_GE:
      *out_matches = loom_low_lower_rule_value_facts_i64_range_ge(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->other_value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_FLOAT_EQUALS:
      *out_matches = loom_low_lower_rule_value_facts_float_equals(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->payload.u64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_STORAGE_ELEMENT_FORMAT:
      *out_matches = loom_low_lower_rule_value_storage_element_format(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->payload.u64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_MEMORY_SPACE:
      *out_matches = loom_low_lower_rule_value_memory_space_matches(
          match_context, rule_set, source_op, guard->value_ref_index,
          guard->payload.u64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES:
      *out_matches =
          loom_low_lower_rule_packed_integer_payload_from_lanes_matches(
              match_context, rule_set, source_op, guard);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD:
      *out_matches =
          loom_low_lower_rule_packed_integer_lanes_from_payload_matches(
              match_context, rule_set, source_op, guard);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_NO_USES: {
      const loom_value_id_t value_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op, guard->value_ref_index);
      *out_matches = loom_value_has_no_uses(
          loom_module_value(match_context->module, value_id));
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VECTOR_EXTRACT_SHAPE:
      *out_matches = loom_low_lower_rule_vector_extract_shape_matches(
          match_context, rule_set, source_op, guard);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_ALL:
      *out_matches =
          iree_all_bits_set(source_op->instance_flags, guard->payload.u64);
      return iree_ok_status();
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated lower guard kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_low_lower_rule_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule, bool* out_matches,
    uint16_t* out_diagnostic_index, uint16_t* out_matched_guard_count,
    bool* out_source_memory_compatible, bool* out_uses_source_memory_access) {
  *out_matches = false;
  *out_diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  *out_matched_guard_count = 0;
  *out_source_memory_compatible = false;
  *out_uses_source_memory_access = false;
  for (uint16_t i = 0; i < rule->guard_count; ++i) {
    const uint16_t guard_ref_index = (uint16_t)(rule->guard_start + i);
    const loom_low_lower_guard_ref_t guard_index =
        rule_set->guard_refs[guard_ref_index];
    const loom_low_lower_guard_t* guard = &rule_set->guards[guard_index];
    bool guard_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_guard_matches(
        match_context, rule_set, source_op, guard, &guard_matches));
    if (!guard_matches) {
      *out_diagnostic_index = guard->diagnostic_index;
      *out_matched_guard_count = i;
      if (guard->kind == LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE) {
        const loom_low_lower_rule_source_memory_match_t source_memory_match =
            loom_low_lower_rule_source_memory_emits_match(
                match_context, rule_set, source_op, rule);
        *out_source_memory_compatible = source_memory_match.all_emits_match &&
                                        source_memory_match.has_source_memory;
      }
      return iree_ok_status();
    }
  }
  const loom_low_lower_rule_source_memory_match_t source_memory_match =
      loom_low_lower_rule_source_memory_emits_match(match_context, rule_set,
                                                    source_op, rule);
  if (!source_memory_match.all_emits_match) {
    *out_diagnostic_index = source_memory_match.diagnostic_index;
    *out_matched_guard_count = rule->guard_count;
    *out_source_memory_compatible = source_memory_match.constraints_compatible;
    return iree_ok_status();
  }
  *out_matches = true;
  *out_matched_guard_count = rule->guard_count;
  *out_uses_source_memory_access = source_memory_match.has_source_memory;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_set_select_rule_range_with_match_context(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t rule_start, uint16_t rule_count,
    loom_low_lower_rule_selection_t* out_selection) {
  *out_selection = (loom_low_lower_rule_selection_t){
      .rule = NULL,
      .rule_index = UINT16_MAX,
      .has_source_op_span = false,
      .diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE,
      .matched_guard_count = 0,
      .source_memory_compatible = false,
      .uses_source_memory_access = false,
  };

  if (rule_count == 0) {
    return iree_ok_status();
  }

  uint16_t best_diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  uint16_t best_matched_guard_count = 0;
  bool best_source_memory_compatible = false;
  for (uint16_t i = 0; i < rule_count; ++i) {
    uint16_t rule_index = (uint16_t)(rule_start + i);
    const loom_low_lower_rule_t* rule = &rule_set->rules[rule_index];
    if (iree_all_bits_set(rule->flags,
                          LOOM_LOW_LOWER_RULE_FLAG_CONTRACT_ONLY) &&
        !iree_all_bits_set(match_context->flags,
                           LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY)) {
      continue;
    }
    out_selection->has_source_op_span = true;
    bool rule_matches = false;
    uint16_t diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
    uint16_t matched_guard_count = 0;
    bool source_memory_compatible = false;
    bool uses_source_memory_access = false;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_matches(
        match_context, rule_set, source_op, rule, &rule_matches,
        &diagnostic_index, &matched_guard_count, &source_memory_compatible,
        &uses_source_memory_access));
    if (rule_matches) {
      out_selection->rule = rule;
      out_selection->rule_index = rule_index;
      out_selection->uses_source_memory_access = uses_source_memory_access;
      return iree_ok_status();
    }
    if (best_diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
        (source_memory_compatible && !best_source_memory_compatible) ||
        (source_memory_compatible == best_source_memory_compatible &&
         matched_guard_count > best_matched_guard_count)) {
      best_diagnostic_index = diagnostic_index;
      best_matched_guard_count = matched_guard_count;
      best_source_memory_compatible = source_memory_compatible;
    }
  }

  out_selection->diagnostic_index = best_diagnostic_index;
  out_selection->matched_guard_count = best_matched_guard_count;
  out_selection->source_memory_compatible = best_source_memory_compatible;
  return iree_ok_status();
}
