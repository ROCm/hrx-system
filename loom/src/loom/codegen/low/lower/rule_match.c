// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_match.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/consumption.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/rule_descriptor.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/codegen/low/lower/rule_value.h"
#include "loom/ir/context.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/storage.h"
#include "loom/target/registers.h"

static const loom_low_lower_rule_span_t* loom_low_lower_rule_set_find_span(
    const loom_low_lower_rule_set_t* rule_set, loom_op_kind_t source_op_kind) {
  uint16_t low = 0;
  uint16_t high = rule_set->span_count;
  while (low < high) {
    uint16_t mid = low + (uint16_t)((high - low) / 2);
    const loom_low_lower_rule_span_t* span = &rule_set->spans[mid];
    if (span->source_op_kind == source_op_kind) {
      return span;
    }
    if (span->source_op_kind < source_op_kind) {
      low = (uint16_t)(mid + 1);
    } else {
      high = mid;
    }
  }
  return NULL;
}

static iree_status_t loom_low_lower_rule_emit_no_mapping(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static iree_string_view_t loom_low_lower_rule_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_low_lower_rule_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return loom_string_table_get(&module->strings, symbol->name_id);
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_lower_rule_function_name(
    const loom_low_lower_rule_match_context_t* match_context) {
  if (!loom_func_like_isa(match_context->function)) {
    return IREE_SV("<module>");
  }
  return loom_low_lower_rule_symbol_name(
      match_context->module, loom_func_like_callee(match_context->function));
}

static iree_string_view_t loom_low_lower_rule_target_key(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_rule_nonempty(bundle->name, IREE_SV("<empty>"));
}

static iree_string_view_t loom_low_lower_rule_export_name(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_rule_nonempty(bundle->export_plan->name,
                                      IREE_SV("<empty>"));
}

static iree_string_view_t loom_low_lower_rule_config_key(
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_rule_nonempty(bundle->config->name, IREE_SV("<empty>"));
}

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

iree_status_t loom_low_lower_rule_resolve_descriptor_ref(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  if (descriptor_ref == LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(descriptor_ref, rule_set->descriptor_ref_count);
  if (match_context->descriptor_ref.fn != NULL) {
    return match_context->descriptor_ref.fn(
        match_context->descriptor_ref.user_data, match_context, rule_set,
        descriptor_ref, out_descriptor);
  }
  IREE_ASSERT(match_context->descriptor_set != NULL);
  IREE_ASSERT(rule_set->descriptor_refs != NULL);
  const iree_string_view_t key = loom_low_lower_rule_set_string(
      rule_set, rule_set->descriptor_refs[descriptor_ref].key_string_ref);
  const uint32_t descriptor_ordinal = loom_low_descriptor_set_lookup_descriptor(
      match_context->descriptor_set, key);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_ok_status();
  }
  *out_descriptor = loom_low_descriptor_set_descriptor_at(
      match_context->descriptor_set, descriptor_ordinal);
  IREE_ASSERT(*out_descriptor != NULL);
  return iree_ok_status();
}

static bool loom_low_lower_rule_descriptor_features_available(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_descriptor_t* descriptor) {
  for (uint16_t i = 0; i < descriptor->feature_mask_word_count; ++i) {
    const uint32_t word_index = descriptor->feature_mask_word_start + i;
    const uint64_t required_bits =
        match_context->descriptor_set->feature_mask_words[word_index];
    const uint64_t available_bits = i == 0 ? match_context->feature_bits : 0;
    if ((required_bits & ~available_bits) != 0) {
      return false;
    }
  }
  return true;
}

static bool loom_low_lower_rule_primary_feature_class_available(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, uint8_t feature_class) {
  if (feature_class == 0) {
    return true;
  }
  IREE_ASSERT_NE(feature_class,
                 LOOM_LOW_LOWER_RULE_PRIMARY_FEATURE_CLASS_DYNAMIC);
  const uint16_t feature_mask_index = (uint16_t)(feature_class - 1u);
  IREE_ASSERT_LT(feature_mask_index,
                 rule_set->primary_descriptor_feature_mask_count);
  const uint64_t required_features =
      rule_set->primary_descriptor_feature_masks[feature_mask_index];
  return (required_features & ~match_context->feature_bits) == 0;
}

static iree_status_t loom_low_lower_rule_descriptor_available(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref, bool* out_available) {
  *out_available = false;
  const loom_low_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
      match_context, rule_set, descriptor_ref, &descriptor));
  if (descriptor != NULL) {
    *out_available = loom_low_lower_rule_descriptor_features_available(
        match_context, descriptor);
  }
  return iree_ok_status();
}

static bool loom_low_lower_rule_type_matches(
    const loom_low_lower_rule_set_t* rule_set,
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
    if (loom_type_dim_static_size_at(type, 0) != pattern->shape.exact.dim0) {
      return false;
    }
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE)) {
    if (loom_type_rank(type) == 0 || loom_type_dim_is_dynamic_at(type, 0)) {
      return false;
    }
    const loom_low_lower_type_pattern_range_t* range =
        &rule_set->type_pattern_ranges[pattern->shape.range_ref - 1];
    const uint64_t static_dim0 =
        (uint64_t)loom_type_dim_static_size_at(type, 0);
    if (static_dim0 < range->minimum || static_dim0 > range->maximum) {
      return false;
    }
  }
  if (iree_any_bit_set(pattern->flags,
                       LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM1)) {
    if (loom_type_rank(type) < 2 || loom_type_dim_is_dynamic_at(type, 1)) {
      return false;
    }
    if (loom_type_dim_static_size_at(type, 1) != pattern->shape.exact.dim1) {
      return false;
    }
  }
  if (iree_any_bit_set(
          pattern->flags,
          LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_ELEMENT_COUNT_RANGE)) {
    const loom_low_lower_type_pattern_range_t* range =
        &rule_set->type_pattern_ranges[pattern->shape.range_ref - 1];
    uint64_t static_element_count = 0;
    if (!loom_type_static_element_count(type, &static_element_count)) {
      return false;
    }
    if (static_element_count < range->minimum ||
        static_element_count > range->maximum) {
      return false;
    }
  }
  return true;
}

static bool loom_low_lower_rule_vector_extract_tail_type_matches(
    loom_type_t source_type, uint16_t consumed_rank, loom_type_t result_type) {
  const uint8_t source_rank = loom_type_rank(source_type);
  if (consumed_rank > source_rank) {
    return false;
  }
  if (loom_type_element_type(source_type) !=
      loom_type_element_type(result_type)) {
    return false;
  }
  if (loom_type_is_scalar(result_type)) {
    return consumed_rank == source_rank;
  }
  if (!loom_type_is_vector(result_type)) {
    return false;
  }
  const uint8_t result_rank = loom_type_rank(result_type);
  if (consumed_rank + result_rank != source_rank) {
    return false;
  }
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
  if (!loom_vector_extract_isa(source_op)) {
    return false;
  }
  if (guard->selector.value.parameter_index >= source_op->attribute_count) {
    return false;
  }
  loom_attribute_t static_indices =
      loom_op_const_attrs(source_op)[guard->selector.value.parameter_index];
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }

  const loom_value_id_t source_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op,
      guard->selector.value.value_ref_index);
  const loom_value_id_t result_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op,
      guard->selector.value.other_value_ref_index);
  const loom_type_t source_type =
      loom_low_lower_rule_match_value_type(match_context, source_value_id);
  const loom_type_t result_type =
      loom_low_lower_rule_match_value_type(match_context, result_value_id);
  if (!loom_type_is_vector(source_type)) {
    return false;
  }

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

  uint64_t lane_count = 0;
  int64_t lower = 0;
  int64_t upper = 0;
  if (loom_type_static_element_count(type, &lane_count) && lane_count > 0 &&
      loom_value_facts_query_vector_integer_bounds(
          &fact_table->context, facts, lane_count, &lower, &upper)) {
    *out_facts = loom_value_facts_make(lower, upper, 1);
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
  if (!loom_value_has_single_use(source_value)) {
    return false;
  }

  const loom_use_t use = loom_value_uses(source_value)[0];
  const loom_op_t* user_op = loom_use_user_op(use);
  if (!user_op || !loom_index_assume_isa(user_op)) {
    return false;
  }

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
  if (bit_count == 0 || bit_count > 64) {
    return iree_ok_status();
  }
  loom_symbolic_expr_context_t* expression_context =
      match_context->symbolic_expr_context;
  if (expression_context == NULL) {
    return iree_ok_status();
  }

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
  if (*out_matches) {
    return iree_ok_status();
  }
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
    uint16_t value_ref_index, int64_t addend) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_integer_immediate_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  int64_t exact_value = 0;
  return loom_value_facts_as_exact_i64(facts, &exact_value) &&
         iree_checked_add_i64(exact_value, addend, &exact_value) &&
         iree_math_is_power_of_two_i64(exact_value);
}

static bool loom_low_lower_rule_value_facts_u32_divisor_magic_kind(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, uint16_t other_value_ref_index,
    loom_low_lower_unsigned_divisor_magic_kind_t expected_kind) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_low_lower_unsigned_divisor_magic_info_t info = {0};
  const loom_value_id_t numerator = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, other_value_ref_index);
  if (!loom_low_lower_rule_value_facts_u32_divisor_magic_info(
          match_context->module, match_context->fact_table, numerator, value_id,
          &info)) {
    return false;
  }
  const loom_low_lower_unsigned_divisor_magic_kind_t kind =
      info.is_add ? LOOM_LOW_LOWER_UNSIGNED_DIVISOR_MAGIC_MULTIPLY_ADD_SHIFT
      : info.post_shift ? LOOM_LOW_LOWER_UNSIGNED_DIVISOR_MAGIC_MULTIPLY_SHIFT
                        : LOOM_LOW_LOWER_UNSIGNED_DIVISOR_MAGIC_MULTIPLY;
  return kind == expected_kind;
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

static bool loom_low_lower_rule_value_facts_exact_power_of_two_float(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, int64_t minimum_exponent,
    int64_t maximum_exponent) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (!loom_low_lower_rule_float_immediate_facts(
          match_context->module, match_context->fact_table, value_id, &facts)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(
      loom_module_value_type(match_context->module, value_id));
  int32_t exponent = 0;
  return loom_value_facts_as_exact_power_of_two_float(scalar_type, facts,
                                                      &exponent) &&
         exponent >= minimum_exponent && exponent <= maximum_exponent;
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

static bool loom_low_lower_rule_value_storage_schema(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index,
    loom_value_fact_storage_schema_t* out_storage_schema) {
  const loom_value_id_t value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op, value_ref_index);
  if (match_context->fact_table != NULL) {
    loom_value_fact_encoding_summary_t summary = {0};
    if (loom_value_facts_query_encoding_summary(
            &match_context->fact_table->context,
            loom_value_fact_table_lookup(match_context->fact_table, value_id),
            &summary)) {
      *out_storage_schema = summary.storage_schema;
      return true;
    }
  }
  const loom_type_t type =
      loom_low_lower_rule_match_value_type(match_context, value_id);
  const loom_fact_context_t* fact_context =
      match_context->fact_table != NULL ? &match_context->fact_table->context
                                        : NULL;
  return loom_encoding_query_type_storage_schema(
      fact_context, match_context->module, type, out_storage_schema);
}

static bool loom_low_lower_rule_value_storage_element_format(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index,
    loom_value_fact_numeric_format_flags_t expected_format) {
  if (expected_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN) {
    return false;
  }
  loom_value_fact_storage_schema_t storage_schema = {0};
  return loom_low_lower_rule_value_storage_schema(match_context, rule_set,
                                                  source_op, value_ref_index,
                                                  &storage_schema) &&
         storage_schema.encoded_operand.element_format == expected_format;
}

static bool loom_low_lower_rule_value_storage_operand_schema(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index,
    loom_value_fact_encoded_operand_schema_t expected_schema) {
  loom_value_fact_storage_schema_t storage_schema = {0};
  return loom_low_lower_rule_value_storage_schema(match_context, rule_set,
                                                  source_op, value_ref_index,
                                                  &storage_schema) &&
         loom_value_fact_encoded_operand_schema_equal(
             storage_schema.encoded_operand, expected_schema);
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
    const loom_low_lower_guard_payload_t* payload,
    uint32_t* out_storage_unit_bit_count, uint32_t* out_width) {
  *out_storage_unit_bit_count = 0;
  *out_width = 0;
  if (guard->selector.value.parameter_index >= source_op->attribute_count ||
      loom_op_const_attrs(source_op)[guard->selector.value.parameter_index]
              .kind != LOOM_ATTR_I64) {
    return false;
  }
  const int64_t width_i64 =
      loom_op_const_attrs(source_op)[guard->selector.value.parameter_index].i64;
  if (width_i64 <= 0 || width_i64 > UINT32_MAX) {
    return false;
  }
  const uint32_t storage_unit_bit_count =
      payload->packed_integer.storage_unit_bit_count;
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
  const loom_low_lower_guard_payload_t* payload =
      loom_low_lower_rule_set_guard_payload(rule_set, guard);
  if (!loom_low_lower_rule_storage_width(source_op, guard, payload,
                                         &storage_unit_bit_count, &width)) {
    return false;
  }

  const loom_value_id_t lane_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op,
      guard->selector.value.value_ref_index);
  const loom_value_id_t storage_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op,
      guard->selector.value.other_value_ref_index);
  loom_vector_packed_integer_payload_from_lanes_match_t match = {0};
  if (!loom_vector_packed_integer_payload_from_lanes_match(
          loom_low_lower_rule_match_value_type(match_context, lane_value_id),
          loom_low_lower_rule_match_value_type(match_context, storage_value_id),
          width, storage_unit_bit_count, UINT32_MAX, &match)) {
    return false;
  }

  return (match.result_shape.payload_bit_count %
          payload->packed_integer.storage_payload_multiple) == 0;
}

static bool loom_low_lower_rule_packed_integer_lanes_from_payload_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_guard_t* guard) {
  uint32_t storage_unit_bit_count = 0;
  uint32_t width = 0;
  const loom_low_lower_guard_payload_t* payload =
      loom_low_lower_rule_set_guard_payload(rule_set, guard);
  if (!loom_low_lower_rule_storage_width(source_op, guard, payload,
                                         &storage_unit_bit_count, &width)) {
    return false;
  }

  const loom_value_id_t storage_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op,
      guard->selector.value.value_ref_index);
  const loom_value_id_t lane_value_id = loom_low_lower_rule_source_value(
      match_context->module, rule_set, source_op,
      guard->selector.value.other_value_ref_index);
  if (!loom_vector_packed_integer_lanes_from_payload_match(
          loom_low_lower_rule_match_value_type(match_context, storage_value_id),
          loom_low_lower_rule_match_value_type(match_context, lane_value_id),
          width, storage_unit_bit_count,
          payload->packed_integer.storage_payload_multiple,
          payload->packed_integer.maximum_lane_count, NULL)) {
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
      loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
      if (!loom_low_lower_rule_resolve_source_value_from_nodes(
              match_context->module, match_context->fact_table,
              match_context->vector_lane_projection, rule_set, source_op,
              /*source_nodes=*/NULL, /*source_node_count=*/1,
              guard->selector.value.value_ref_index, &value_id)) {
        return iree_ok_status();
      }
      const loom_type_t type =
          loom_low_lower_rule_match_value_type(match_context, value_id);
      *out_matches = loom_low_lower_rule_type_matches(
          rule_set,
          &rule_set->type_patterns[guard->selector.value.parameter_index],
          type);
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_ATTR_KIND:
      if (guard->selector.attribute.attr_index >= source_op->attribute_count) {
        return iree_ok_status();
      }
      *out_matches =
          loom_op_const_attrs(source_op)[guard->selector.attribute.attr_index]
              .kind == guard->attr_kind;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ:
      if (guard->selector.attribute.attr_index >= source_op->attribute_count) {
        return iree_ok_status();
      }
      *out_matches =
          loom_op_const_attrs(source_op)[guard->selector.attribute.attr_index]
                  .kind == LOOM_ATTR_ENUM &&
          loom_op_const_attrs(source_op)[guard->selector.attribute.attr_index]
                  .raw ==
              loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_ENUM_IN: {
      const uint16_t attr_index = guard->selector.attribute.attr_index;
      if (attr_index >= source_op->attribute_count) {
        return iree_ok_status();
      }
      const loom_attribute_t source_attr =
          loom_op_const_attrs(source_op)[attr_index];
      if (source_attr.kind != LOOM_ATTR_ENUM || source_attr.raw >= 64) {
        return iree_ok_status();
      }
      const uint64_t enum_mask =
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64;
      *out_matches = ((enum_mask >> source_attr.raw) & UINT64_C(1)) != 0;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE: {
      if (guard->selector.attribute.attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->selector.attribute.attr_index]
                  .kind != LOOM_ATTR_I64) {
        return iree_ok_status();
      }
      const int64_t value =
          loom_op_const_attrs(source_op)[guard->selector.attribute.attr_index]
              .i64;
      const loom_low_lower_guard_payload_t* payload =
          loom_low_lower_rule_set_guard_payload(rule_set, guard);
      *out_matches = value >= payload->i64_range.minimum &&
                     value <= payload->i64_range.maximum;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_SUM_EQ: {
      const uint16_t attr_index = guard->selector.attribute.attr_index;
      const uint16_t other_attr_index =
          guard->selector.attribute.other_attr_index;
      if (attr_index >= source_op->attribute_count ||
          other_attr_index >= source_op->attribute_count) {
        return iree_ok_status();
      }
      const loom_attribute_t attr = loom_op_const_attrs(source_op)[attr_index];
      const loom_attribute_t other_attr =
          loom_op_const_attrs(source_op)[other_attr_index];
      if (attr.kind != LOOM_ATTR_I64 || other_attr.kind != LOOM_ATTR_I64) {
        return iree_ok_status();
      }
      if ((other_attr.i64 > 0 && attr.i64 > INT64_MAX - other_attr.i64) ||
          (other_attr.i64 < 0 && attr.i64 < INT64_MIN - other_attr.i64)) {
        return iree_ok_status();
      }
      *out_matches =
          attr.i64 + other_attr.i64 ==
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->i64;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_COUNT_EQ:
      if (guard->selector.attribute.attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->selector.attribute.attr_index]
                  .kind != LOOM_ATTR_I64_ARRAY) {
        return iree_ok_status();
      }
      *out_matches =
          (uint64_t)loom_op_const_attrs(
              source_op)[guard->selector.attribute.attr_index]
              .count ==
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64;
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENT_RANGE: {
      if (guard->selector.attribute.attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->selector.attribute.attr_index]
                  .kind != LOOM_ATTR_I64_ARRAY) {
        return iree_ok_status();
      }
      const loom_attribute_t attr =
          loom_op_const_attrs(source_op)[guard->selector.attribute.attr_index];
      if (guard->selector.attribute.element_index >= attr.count) {
        return iree_ok_status();
      }
      const int64_t value =
          attr.i64_array[guard->selector.attribute.element_index];
      const loom_low_lower_guard_payload_t* payload =
          loom_low_lower_rule_set_guard_payload(rule_set, guard);
      *out_matches = value >= payload->i64_range.minimum &&
                     value <= payload->i64_range.maximum;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENTS_RANGE: {
      if (guard->selector.attribute.attr_index >= source_op->attribute_count ||
          loom_op_const_attrs(source_op)[guard->selector.attribute.attr_index]
                  .kind != LOOM_ATTR_I64_ARRAY) {
        return iree_ok_status();
      }
      *out_matches = true;
      const loom_attribute_t attr =
          loom_op_const_attrs(source_op)[guard->selector.attribute.attr_index];
      const loom_low_lower_guard_payload_t* payload =
          loom_low_lower_rule_set_guard_payload(rule_set, guard);
      for (uint16_t i = 0; i < attr.count; ++i) {
        if (attr.i64_array[i] < payload->i64_range.minimum ||
            attr.i64_array[i] > payload->i64_range.maximum) {
          *out_matches = false;
          break;
        }
      }
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE:
      return loom_low_lower_rule_descriptor_available(
          match_context, rule_set, guard->selector.descriptor.descriptor_ref,
          out_matches);
    case LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE:
      return loom_low_lower_rule_can_materialize_value(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index, out_matches);
    case LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS: {
      loom_low_lower_rule_mapped_value_t mapped_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_mapped_value(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index, &mapped_value));
      return loom_low_lower_rule_mapped_value_register_class_matches(
          match_context, mapped_value, guard->selector.value.parameter_index,
          out_matches);
    }
    case LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT: {
      loom_low_lower_rule_mapped_value_t mapped_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_mapped_value(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index, &mapped_value));
      *out_matches =
          mapped_value.is_register &&
          mapped_value.register_unit_count ==
              loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_STATIC_DIM0_MULTIPLE: {
      const uint64_t multiple =
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64;
      IREE_ASSERT_GT(multiple, 0);
      loom_value_id_t value_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op,
          guard->selector.value.value_ref_index);
      const loom_type_t type =
          loom_low_lower_rule_match_value_type(match_context, value_id);
      if (loom_type_rank(type) == 0 || loom_type_dim_is_dynamic_at(type, 0)) {
        return iree_ok_status();
      }
      const int64_t static_dim0 = loom_type_dim_static_size_at(type, 0);
      *out_matches =
          static_dim0 >= 0 && ((uint64_t)static_dim0 % multiple) == 0;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT_EQ: {
      loom_low_lower_rule_mapped_value_t lhs_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_mapped_value(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index, &lhs_value));
      if (!lhs_value.is_register) {
        return iree_ok_status();
      }
      loom_low_lower_rule_mapped_value_t rhs_value =
          loom_low_lower_rule_mapped_value_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_rule_mapped_value(
          match_context, rule_set, source_op,
          guard->selector.value.other_value_ref_index, &rhs_value));
      if (!rhs_value.is_register) {
        return iree_ok_status();
      }
      *out_matches =
          lhs_value.register_unit_count == rhs_value.register_unit_count;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_STATIC_ELEMENT_COUNT_EQ: {
      const loom_value_id_t lhs_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op,
          guard->selector.value.value_ref_index);
      const loom_value_id_t rhs_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op,
          guard->selector.value.other_value_ref_index);
      uint64_t lhs_count = 0;
      uint64_t rhs_count = 0;
      if (!loom_type_static_element_count(
              loom_low_lower_rule_match_value_type(match_context, lhs_id),
              &lhs_count) ||
          !loom_type_static_element_count(
              loom_low_lower_rule_match_value_type(match_context, rhs_id),
              &rhs_count)) {
        return iree_ok_status();
      }
      *out_matches = lhs_count == rhs_count;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_OPERAND_SEGMENT_COUNT_EQ: {
      const loom_op_vtable_t* vtable =
          loom_op_vtable(match_context->module, source_op);
      const loom_value_slice_t segment = loom_op_operand_field_span(
          vtable, source_op, guard->selector.attribute.attr_index);
      *out_matches =
          segment.count ==
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_SIGNED_BIT_COUNT:
      return loom_low_lower_rule_value_facts_fit_bit_count(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index,
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64,
          /*is_signed_domain=*/true, out_matches);
    case LOOM_LOW_LOWER_GUARD_VALUE_UNSIGNED_BIT_COUNT:
      return loom_low_lower_rule_value_facts_fit_bit_count(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index,
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64,
          /*is_signed_domain=*/false, out_matches);
    case LOOM_LOW_LOWER_GUARD_VALUE_EXACT_I64:
      *out_matches = loom_low_lower_rule_value_facts_exact_i64(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_EXACT_POWER_OF_TWO_I64:
      *out_matches = loom_low_lower_rule_value_facts_exact_power_of_two_i64(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index,
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->addend);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_U32_DIVISOR_MAGIC_KIND:
      *out_matches = loom_low_lower_rule_value_facts_u32_divisor_magic_kind(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index,
          guard->selector.value.other_value_ref_index,
          (loom_low_lower_unsigned_divisor_magic_kind_t)
              loom_low_lower_rule_set_guard_payload(rule_set, guard)
                  ->u64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_EXACT_FLOAT:
      *out_matches = loom_low_lower_rule_value_facts_exact_float(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_EXACT_POWER_OF_TWO_FLOAT: {
      const loom_low_lower_guard_payload_t* payload =
          loom_low_lower_rule_set_guard_payload(rule_set, guard);
      *out_matches = loom_low_lower_rule_value_facts_exact_power_of_two_float(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index, payload->i64_range.minimum,
          payload->i64_range.maximum);
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_NOT_NAN: {
      const loom_value_id_t value_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op,
          guard->selector.value.value_ref_index);
      *out_matches = match_context->fact_table &&
                     loom_value_facts_is_not_nan(loom_value_fact_table_lookup(
                         match_context->fact_table, value_id));
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_NOT_SUBNORMAL_OR_INSTANCE_FLAGS_HAS_ALL: {
      if (iree_all_bits_set(
              source_op->instance_flags,
              loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64)) {
        *out_matches = true;
        return iree_ok_status();
      }
      const loom_value_id_t value_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op,
          guard->selector.value.value_ref_index);
      *out_matches =
          match_context->fact_table &&
          loom_value_facts_is_not_subnormal(loom_value_fact_table_lookup(
              match_context->fact_table, value_id));
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE: {
      const loom_low_lower_guard_payload_t* payload =
          loom_low_lower_rule_set_guard_payload(rule_set, guard);
      *out_matches = loom_low_lower_rule_value_facts_i64_range(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index, payload->i64_range.minimum,
          payload->i64_range.maximum);
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE_LE:
      *out_matches = loom_low_lower_rule_value_facts_i64_range_le(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index,
          guard->selector.value.other_value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE_GE:
      *out_matches = loom_low_lower_rule_value_facts_i64_range_ge(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index,
          guard->selector.value.other_value_ref_index);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_FLOAT_EQUALS:
      *out_matches = loom_low_lower_rule_value_facts_float_equals(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index,
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_STORAGE_ELEMENT_FORMAT:
      *out_matches = loom_low_lower_rule_value_storage_element_format(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index,
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_STORAGE_OPERAND_SCHEMA:
      *out_matches = loom_low_lower_rule_value_storage_operand_schema(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index,
          rule_set
              ->storage_operand_schemas[guard->selector.value.parameter_index]);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_VALUE_MEMORY_SPACE:
      *out_matches = loom_low_lower_rule_value_memory_space_matches(
          match_context, rule_set, source_op,
          guard->selector.value.value_ref_index,
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64);
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
          match_context->module, rule_set, source_op,
          guard->selector.value.value_ref_index);
      *out_matches = loom_value_has_no_uses(
          loom_module_value(match_context->module, value_id));
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VALUE_NO_USES_AFTER: {
      if (match_context->consumption_query == NULL) {
        *out_matches = false;
        return iree_ok_status();
      }
      const loom_value_id_t value_id = loom_low_lower_rule_source_value(
          match_context->module, rule_set, source_op,
          guard->selector.value.value_ref_index);
      loom_consumption_use_t use = {0};
      bool use_found = false;
      IREE_RETURN_IF_ERROR(loom_consumption_find_use_after(
          match_context->consumption_query, source_op, value_id, &use,
          &use_found));
      *out_matches = !use_found;
      return iree_ok_status();
    }
    case LOOM_LOW_LOWER_GUARD_VECTOR_EXTRACT_SHAPE:
      *out_matches = loom_low_lower_rule_vector_extract_shape_matches(
          match_context, rule_set, source_op, guard);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_ALL:
      *out_matches = iree_all_bits_set(
          source_op->instance_flags,
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_NONE:
      *out_matches = !iree_any_bit_set(
          source_op->instance_flags,
          loom_low_lower_rule_set_guard_payload(rule_set, guard)->u64);
      return iree_ok_status();
    case LOOM_LOW_LOWER_GUARD_TARGET_SUBGROUP_SIZE_RANGE: {
      const int64_t subgroup_size =
          (int64_t)match_context->bundle->snapshot->subgroup_size;
      const loom_low_lower_guard_payload_t* payload =
          loom_low_lower_rule_set_guard_payload(rule_set, guard);
      *out_matches = subgroup_size != 0 &&
                     subgroup_size >= payload->i64_range.minimum &&
                     subgroup_size <= payload->i64_range.maximum;
      return iree_ok_status();
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated lower guard kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_low_lower_rule_source_connection_is_exclusive(
    const loom_low_lower_rule_match_context_t* match_context,
    loom_value_id_t value_id) {
  const loom_value_t* value =
      loom_module_value(match_context->module, value_id);
  return loom_value_has_single_use(value) &&
         !loom_value_has_attribute_uses(value) &&
         !loom_module_value_has_type_uses(match_context->module, value_id);
}

static bool loom_low_lower_rule_source_node_op_is_eligible(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_source_node_t* source_node,
    const loom_op_t* source_op) {
  return source_op != NULL && source_op->kind == source_node->source_op_kind &&
         source_op->region_count == 0 &&
         iree_any_bit_set(
             loom_op_effective_traits(match_context->module, source_op),
             LOOM_TRAIT_PURE);
}

static bool loom_low_lower_rule_resolve_source_node(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_node_t* source_node, uint8_t source_node_index,
    const loom_op_t** source_nodes) {
  IREE_ASSERT_GT(source_node_index, 0);
  IREE_ASSERT_LT(source_node_index, LOOM_LOW_LOWER_MAX_SOURCE_NODES);
  IREE_ASSERT_LT(source_node->parent_node_index, source_node_index);
  const loom_low_lower_value_ref_t* parent_value_ref =
      &rule_set->value_refs[source_node->parent_value_ref_index];
  const loom_low_lower_value_ref_t* node_value_ref =
      &rule_set->value_refs[source_node->node_value_ref_index];
  IREE_ASSERT_EQ(parent_value_ref->source_node_index,
                 source_node->parent_node_index);
  IREE_ASSERT_EQ(node_value_ref->source_node_index, source_node_index);

  const loom_op_t* parent_op = source_nodes[source_node->parent_node_index];
  const loom_value_id_t connection =
      loom_low_lower_rule_source_value_from_nodes(
          match_context->module, rule_set, source_nodes[0], source_nodes,
          source_node_index, source_node->parent_value_ref_index);
  if (!loom_low_lower_rule_source_connection_is_exclusive(match_context,
                                                          connection)) {
    return false;
  }

  const loom_value_t* connection_value =
      loom_module_value(match_context->module, connection);
  const loom_op_t* node_op = NULL;
  switch (source_node->relation) {
    case LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_UNIQUE_USER:
    case LOOM_LOW_LOWER_SOURCE_NODE_EXCLUSIVE_USER: {
      IREE_ASSERT_EQ(parent_value_ref->kind, LOOM_LOW_LOWER_VALUE_REF_RESULT);
      IREE_ASSERT_EQ(node_value_ref->kind, LOOM_LOW_LOWER_VALUE_REF_OPERAND);
      node_op = loom_use_user_op(loom_value_uses(connection_value)[0]);
      if (source_node->relation ==
                  LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_UNIQUE_USER
              ? parent_op->next_op != node_op
              : parent_op->parent_block != node_op->parent_block) {
        return false;
      }
      break;
    }
    case LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_DEFINITION:
    case LOOM_LOW_LOWER_SOURCE_NODE_EXCLUSIVE_DEFINITION: {
      IREE_ASSERT_EQ(parent_value_ref->kind, LOOM_LOW_LOWER_VALUE_REF_OPERAND);
      IREE_ASSERT_EQ(node_value_ref->kind, LOOM_LOW_LOWER_VALUE_REF_RESULT);
      if (loom_value_is_block_arg(connection_value)) {
        return false;
      }
      node_op = loom_value_def_op(connection_value);
      if ((source_node->relation ==
                   LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_DEFINITION
               ? parent_op->prev_op != node_op
               : parent_op->parent_block != node_op->parent_block) ||
          loom_use_user_op(loom_value_uses(connection_value)[0]) != parent_op) {
        return false;
      }
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown generated source-node relation");
      IREE_BUILTIN_UNREACHABLE();
  }
  if (!loom_low_lower_rule_source_node_op_is_eligible(match_context,
                                                      source_node, node_op)) {
    return false;
  }
  for (uint8_t i = 0; i < source_node_index; ++i) {
    if (source_nodes[i] == node_op) {
      return false;
    }
  }
  source_nodes[source_node_index] = node_op;
  const loom_value_id_t node_connection =
      loom_low_lower_rule_source_value_from_nodes(
          match_context->module, rule_set, source_nodes[0], source_nodes,
          (uint8_t)(source_node_index + 1), source_node->node_value_ref_index);
  if (node_connection != connection) {
    source_nodes[source_node_index] = NULL;
    return false;
  }
  return true;
}

static bool loom_low_lower_rule_source_value_available_at_root(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* root_op, loom_value_id_t value_id) {
  const loom_value_t* value =
      loom_module_value(match_context->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return true;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  IREE_ASSERT(defining_op != NULL);
  return defining_op->parent_block != root_op->parent_block ||
         defining_op->block_ordinal < root_op->block_ordinal;
}

static bool loom_low_lower_rule_nonlocal_source_operands_available(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule, const loom_op_t* const* source_nodes,
    uint8_t source_node_count) {
  for (uint16_t emit_ordinal = 0; emit_ordinal < rule->emit_count;
       ++emit_ordinal) {
    const loom_low_lower_emit_t* emit = loom_low_lower_rule_set_emit_at(
        rule_set, (uint16_t)(rule->action.emit_start + emit_ordinal));
    for (uint16_t operand_ordinal = 0;
         operand_ordinal < emit->operand_ref_count; ++operand_ordinal) {
      const uint16_t value_ref_index =
          (uint16_t)(emit->operand_ref_start + operand_ordinal);
      const loom_low_lower_value_ref_kind_t value_ref_kind =
          rule_set->value_refs[value_ref_index].kind;
      switch (value_ref_kind) {
        case LOOM_LOW_LOWER_VALUE_REF_TEMPORARY:
        case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM:
        case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
        case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET:
        case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS:
        case LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ROOT:
          continue;
        case LOOM_LOW_LOWER_VALUE_REF_OPERAND:
        case LOOM_LOW_LOWER_VALUE_REF_RESULT:
        case LOOM_LOW_LOWER_VALUE_REF_EXACT_LANE_ORIGIN_OPERAND:
        case LOOM_LOW_LOWER_VALUE_REF_EXACT_UNIFORM_ELEMENT_ORIGIN_OPERAND:
        case LOOM_LOW_LOWER_VALUE_REF_UNIFORM_ELEMENT_ORIGIN_OPERAND:
          break;
        default:
          IREE_ASSERT_UNREACHABLE("unknown generated value ref kind");
          IREE_BUILTIN_UNREACHABLE();
      }
      loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
      if (!loom_low_lower_rule_resolve_source_value_from_nodes(
              match_context->module, match_context->fact_table,
              match_context->vector_lane_projection, rule_set, source_op,
              source_nodes, source_node_count, value_ref_index, &value_id)) {
        return false;
      }
      if (!loom_low_lower_rule_source_value_available_at_root(
              match_context, source_op, value_id)) {
        return false;
      }
    }
  }
  return true;
}

static iree_status_t loom_low_lower_rule_guard_program_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t guard_start, uint16_t guard_count, bool* out_matches,
    uint16_t* out_diagnostic_index, uint16_t* out_matched_guard_count) {
  *out_matches = false;
  *out_diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  *out_matched_guard_count = 0;
  for (uint16_t i = 0; i < guard_count; ++i) {
    const uint16_t guard_ref_index = (uint16_t)(guard_start + i);
    const loom_low_lower_guard_ref_t guard_index =
        rule_set->guard_refs[guard_ref_index];
    const loom_low_lower_guard_t* guard = &rule_set->guards[guard_index];
    bool guard_matches = false;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_guard_matches(
        match_context, rule_set, source_op, guard, &guard_matches));
    if (!guard_matches) {
      *out_diagnostic_index = guard->diagnostic_index;
      *out_matched_guard_count = i;
      return iree_ok_status();
    }
  }
  *out_matches = true;
  *out_matched_guard_count = guard_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_rule_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule, bool* out_matches,
    uint16_t* out_diagnostic_index, uint16_t* out_matched_guard_count,
    const loom_op_t** out_diagnostic_source_op,
    bool* out_source_memory_compatible, bool* out_uses_source_memory_access,
    const loom_op_t** out_source_nodes, uint8_t* out_source_node_count) {
  *out_matches = false;
  *out_diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  *out_matched_guard_count = 0;
  *out_diagnostic_source_op = source_op;
  *out_source_memory_compatible = false;
  *out_uses_source_memory_access = false;
  *out_source_node_count = 0;
  bool guards_match = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_guard_program_matches(
      match_context, rule_set, source_op, rule->guard_start, rule->guard_count,
      &guards_match, out_diagnostic_index, out_matched_guard_count));
  if (!guards_match) {
    if (*out_matched_guard_count < rule->guard_count) {
      const uint16_t guard_ref_index =
          (uint16_t)(rule->guard_start + *out_matched_guard_count);
      const loom_low_lower_guard_ref_t guard_index =
          rule_set->guard_refs[guard_ref_index];
      const loom_low_lower_guard_t* guard = &rule_set->guards[guard_index];
      if (guard->kind == LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE) {
        const loom_low_lower_rule_source_memory_match_t source_memory_match =
            loom_low_lower_rule_source_memory_emits_match(
                match_context, rule_set, source_op, rule);
        *out_source_memory_compatible = source_memory_match.all_emits_match &&
                                        source_memory_match.has_source_memory;
      }
    }
    return iree_ok_status();
  }

  out_source_nodes[0] = source_op;
  uint8_t source_node_count = 1;
  uint16_t matched_guard_count = rule->guard_count;
  const uint8_t related_source_node_count =
      loom_low_lower_rule_source_node_count(rule);
  const uint16_t source_node_start =
      loom_low_lower_rule_source_node_start(rule);
  for (uint8_t i = 0; i < related_source_node_count; ++i) {
    const loom_low_lower_source_node_t* source_node =
        &rule_set->source_nodes[source_node_start + i];
    const uint8_t source_node_index = (uint8_t)(i + 1);
    if (!loom_low_lower_rule_resolve_source_node(match_context, rule_set,
                                                 source_node, source_node_index,
                                                 out_source_nodes)) {
      *out_matched_guard_count = matched_guard_count;
      return iree_ok_status();
    }
    source_node_count = (uint8_t)(source_node_index + 1);
    uint16_t node_matched_guard_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_guard_program_matches(
        match_context, rule_set, out_source_nodes[source_node_index],
        source_node->guard_start, source_node->guard_count, &guards_match,
        out_diagnostic_index, &node_matched_guard_count));
    if (!guards_match) {
      *out_diagnostic_source_op = out_source_nodes[source_node_index];
      *out_matched_guard_count =
          (uint16_t)(matched_guard_count + node_matched_guard_count);
      return iree_ok_status();
    }
    matched_guard_count =
        (uint16_t)(matched_guard_count + source_node->guard_count);
  }
  const loom_low_lower_rule_source_memory_match_t source_memory_match =
      loom_low_lower_rule_source_memory_emits_match(match_context, rule_set,
                                                    source_op, rule);
  if (!source_memory_match.all_emits_match) {
    *out_diagnostic_index = source_memory_match.diagnostic_index;
    *out_matched_guard_count = matched_guard_count;
    *out_source_memory_compatible = source_memory_match.constraints_compatible;
    return iree_ok_status();
  }
  if (iree_any_bit_set(rule->flags,
                       LOOM_LOW_LOWER_RULE_FLAG_NONLOCAL_SOURCE_GRAPH) &&
      !loom_low_lower_rule_nonlocal_source_operands_available(
          match_context, rule_set, source_op, rule, out_source_nodes,
          source_node_count)) {
    *out_matched_guard_count = matched_guard_count;
    return iree_ok_status();
  }
  *out_matches = true;
  *out_matched_guard_count = matched_guard_count;
  *out_uses_source_memory_access = source_memory_match.has_source_memory;
  *out_source_node_count = source_node_count;
  return iree_ok_status();
}

bool loom_low_lower_rule_failure_is_better(
    loom_low_lower_rule_failure_t candidate,
    loom_low_lower_rule_failure_t incumbent) {
  if (!candidate.has_source_op_span) {
    return false;
  }
  if (!incumbent.has_source_op_span) {
    return true;
  }
  const bool candidate_has_diagnostic =
      candidate.diagnostic_index != LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  const bool incumbent_has_diagnostic =
      incumbent.diagnostic_index != LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  if (candidate_has_diagnostic != incumbent_has_diagnostic) {
    return candidate_has_diagnostic;
  }
  if (candidate.source_memory_compatible !=
      incumbent.source_memory_compatible) {
    return candidate.source_memory_compatible;
  }
  return candidate.matched_guard_count > incumbent.matched_guard_count;
}

bool loom_low_lower_rule_selections_have_same_source_graph(
    const loom_low_lower_rule_selection_t* lhs,
    const loom_low_lower_rule_selection_t* rhs) {
  if (lhs->source_node_count != rhs->source_node_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->source_node_count; ++i) {
    if (lhs->source_nodes[i] != rhs->source_nodes[i]) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_low_lower_rule_set_select_rule_range_with_match_context(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t rule_start, uint16_t rule_count,
    loom_low_lower_rule_selection_t* out_selection) {
  out_selection->rule = NULL;
  out_selection->failure = (loom_low_lower_rule_failure_t){
      .diagnostic_source_op = source_op,
      .has_source_op_span = false,
      .diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE,
      .matched_guard_count = 0,
      .source_memory_compatible = false,
  };
  out_selection->rule_index = UINT16_MAX;
  out_selection->uses_source_memory_access = false;
  out_selection->primary_descriptor_unavailable = false;
  out_selection->source_node_count = 0;

  if (rule_count == 0) {
    return iree_ok_status();
  }

  loom_low_lower_rule_failure_t best_failure = out_selection->failure;
  loom_low_lower_rule_selection_t unavailable_selection = *out_selection;
  bool has_unavailable_selection = false;
  for (uint16_t i = 0; i < rule_count; ++i) {
    uint16_t rule_index = (uint16_t)(rule_start + i);
    const loom_low_lower_rule_t* rule = &rule_set->rules[rule_index];
    if (iree_all_bits_set(rule->flags,
                          LOOM_LOW_LOWER_RULE_FLAG_CONTRACT_ONLY) &&
        !iree_all_bits_set(match_context->flags,
                           LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY)) {
      continue;
    }
    out_selection->failure.has_source_op_span = true;
    bool rule_matches = false;
    uint16_t diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
    uint16_t matched_guard_count = 0;
    const loom_op_t* diagnostic_source_op = source_op;
    bool source_memory_compatible = false;
    bool uses_source_memory_access = false;
    // Matching writes the root and each related node before reading it. Only
    // the populated prefix is published after a successful match.
    const loom_op_t* source_nodes[LOOM_LOW_LOWER_MAX_SOURCE_NODES];
    uint8_t source_node_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_matches(
        match_context, rule_set, source_op, rule, &rule_matches,
        &diagnostic_index, &matched_guard_count, &diagnostic_source_op,
        &source_memory_compatible, &uses_source_memory_access, source_nodes,
        &source_node_count));
    bool primary_descriptor_unavailable = false;
    if (rule_matches && rule->emit_count != 0) {
      const uint16_t primary_emit_ordinal =
          loom_low_lower_rule_primary_emit_ordinal(rule);
      if (primary_emit_ordinal != LOOM_LOW_LOWER_RULE_PRIMARY_EMIT_NONE) {
        const uint8_t feature_class =
            loom_low_lower_rule_primary_feature_class(rule);
        if (iree_all_bits_set(
                match_context->flags,
                LOOM_LOW_LOWER_RULE_MATCH_FLAG_PRIMARY_DESCRIPTORS_BOUND) &&
            feature_class !=
                LOOM_LOW_LOWER_RULE_PRIMARY_FEATURE_CLASS_DYNAMIC) {
          primary_descriptor_unavailable =
              !loom_low_lower_rule_primary_feature_class_available(
                  match_context, rule_set, feature_class);
        } else {
          const loom_low_lower_descriptor_ref_t primary_descriptor_ref =
              loom_low_lower_rule_primary_descriptor_ref(rule_set, rule);
          if (primary_descriptor_ref != LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE) {
            const loom_low_descriptor_t* primary_descriptor = NULL;
            IREE_RETURN_IF_ERROR(loom_low_lower_rule_resolve_descriptor_ref(
                match_context, rule_set, primary_descriptor_ref,
                &primary_descriptor));
            rule_matches = primary_descriptor != NULL;
            primary_descriptor_unavailable =
                primary_descriptor != NULL &&
                !loom_low_lower_rule_descriptor_features_available(
                    match_context, primary_descriptor);
          }
        }
      }
    }
    if (rule_matches && primary_descriptor_unavailable) {
      if (!has_unavailable_selection) {
        unavailable_selection.rule = rule;
        unavailable_selection.rule_index = rule_index;
        unavailable_selection.uses_source_memory_access =
            uses_source_memory_access;
        unavailable_selection.primary_descriptor_unavailable = true;
        unavailable_selection.source_node_count = source_node_count;
        memcpy((void*)unavailable_selection.source_nodes, source_nodes,
               source_node_count * sizeof(*source_nodes));
        has_unavailable_selection = true;
      }
      rule_matches = false;
    }
    if (rule_matches) {
      out_selection->rule = rule;
      out_selection->rule_index = rule_index;
      out_selection->uses_source_memory_access = uses_source_memory_access;
      out_selection->primary_descriptor_unavailable = false;
      out_selection->source_node_count = source_node_count;
      memcpy((void*)out_selection->source_nodes, source_nodes,
             source_node_count * sizeof(*source_nodes));
      if (!has_unavailable_selection ||
          loom_low_lower_rule_selections_have_same_source_graph(
              &unavailable_selection, out_selection)) {
        return iree_ok_status();
      }
      continue;
    }
    const loom_low_lower_rule_failure_t candidate_failure = {
        .diagnostic_source_op = diagnostic_source_op,
        .has_source_op_span = true,
        .diagnostic_index = diagnostic_index,
        .matched_guard_count = matched_guard_count,
        .source_memory_compatible = source_memory_compatible,
    };
    if (loom_low_lower_rule_failure_is_better(candidate_failure,
                                              best_failure)) {
      best_failure = candidate_failure;
    }
    if (matched_guard_count < rule->guard_count) {
      const uint16_t guard_ref_index =
          (uint16_t)(rule->guard_start + matched_guard_count);
      const loom_low_lower_guard_ref_t guard_index =
          rule_set->guard_refs[guard_ref_index];
      const loom_low_lower_guard_t* guard = &rule_set->guards[guard_index];
      if (guard->kind != LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE) {
        // An interned root guard program has the same pure result for every
        // adjacent rule that references it. Descriptor availability is the
        // exception because its failure also ranks rule-specific memory forms.
        while ((uint16_t)(i + 1) < rule_count) {
          const loom_low_lower_rule_t* next_rule =
              &rule_set->rules[rule_start + i + 1];
          if (next_rule->guard_start != rule->guard_start ||
              next_rule->guard_count != rule->guard_count) {
            break;
          }
          ++i;
        }
      }
    }
  }

  if (has_unavailable_selection) {
    *out_selection = unavailable_selection;
    return iree_ok_status();
  }
  out_selection->failure = best_failure;
  return iree_ok_status();
}

iree_status_t loom_low_lower_rule_set_select_with_match_context(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_low_lower_rule_span_t* span =
      loom_low_lower_rule_set_find_span(rule_set, source_op->kind);
  const uint16_t rule_start = span ? span->rule_start : 0;
  const uint16_t rule_count = span ? span->rule_count : 0;
  return loom_low_lower_rule_set_select_rule_range_with_match_context(
      match_context, rule_set, source_op, rule_start, rule_count,
      out_selection);
}

static iree_status_t loom_low_lower_rule_match_map_value_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)match_context;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_query_value(context, source_op,
                                                  source_value_id, &low_type));
  if (!loom_low_type_is_register(low_type)) {
    return iree_ok_status();
  }
  *out_mapped_value = (loom_low_lower_rule_mapped_value_t){
      .is_register = true,
      .descriptor_register_class_id = loom_low_register_type_class_id(low_type),
      .register_unit_count = loom_low_register_type_unit_count(low_type),
  };
  return iree_ok_status();
}

static uint64_t
loom_low_lower_rule_match_source_memory_root_byte_offset_from_lowering(
    void* user_data,
    const loom_low_source_memory_access_plan_t* source_memory_access) {
  const loom_low_lower_context_t* context =
      (const loom_low_lower_context_t*)user_data;
  const loom_low_lower_source_memory_root_byte_offset_callback_t callback =
      context->policy->source_memory_root_byte_offset;
  IREE_ASSERT(callback.fn != NULL);
  return callback.fn(callback.user_data, context, source_memory_access);
}

static iree_status_t loom_low_lower_rule_match_can_materialize_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t source_value_id,
    bool* out_can_materialize) {
  (void)match_context;
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  const loom_low_lower_value_materializer_t* materializer =
      loom_low_lower_rule_value_materializer(rule_set, value_ref);
  return materializer->can_materialize(context, source_op, source_value_id,
                                       out_can_materialize);
}

static iree_status_t loom_low_lower_rule_descriptor_cache_initialize(
    loom_low_lower_context_t* context,
    const loom_low_descriptor_set_t* descriptor_set) {
  // A frame's lowering policy is immutable. Only descriptor-set probes may
  // change the selected binding during target legality queries.
  if (context->lowering->rule_descriptor_cache != NULL &&
      context->lowering->rule_descriptor_cache->descriptor_set ==
          descriptor_set) {
    return iree_ok_status();
  }
  const loom_low_lower_rule_set_list_t rule_sets =
      context->policy->contract.rule_sets;
  if (context->module_state != NULL) {
    return loom_low_lower_module_state_rule_descriptor_cache(
        context->module_state, rule_sets, descriptor_set,
        &context->lowering->rule_descriptor_cache);
  }
  return loom_low_lower_rule_descriptor_cache_select(
      rule_sets, descriptor_set, context->function_arena,
      &context->lowering->rule_descriptor_cache);
}

iree_status_t loom_low_lower_rule_match_descriptor_ref_from_lowering(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_descriptor_cache_initialize(
      context, match_context->descriptor_set));
  loom_low_lower_rule_descriptor_cache_t* cache =
      context->lowering->rule_descriptor_cache;
  uint16_t rule_set_index = 0;
  if (match_context->policy_rule_set_ordinal == 0) {
    while (rule_set_index < cache->rule_sets.count &&
           cache->rule_sets.values[rule_set_index] != rule_set) {
      ++rule_set_index;
    }
  } else {
    rule_set_index = (uint16_t)(match_context->policy_rule_set_ordinal - 1u);
  }
  IREE_ASSERT_LT(rule_set_index, cache->rule_sets.count);
  IREE_ASSERT_EQ(cache->rule_sets.values[rule_set_index], rule_set);
  return loom_low_lower_rule_descriptor_cache_resolve(
      cache, rule_set_index, descriptor_ref, out_descriptor);
}

void loom_low_lower_rule_match_context_initialize_from_lowering(
    loom_low_lower_context_t* context,
    const loom_view_region_table_t* view_regions,
    loom_low_lower_rule_source_memory_state_t* source_memory_state,
    loom_low_lower_rule_match_context_t* out_match_context) {
  *out_match_context = (loom_low_lower_rule_match_context_t){
      .module = loom_low_lower_context_module(context),
      .function = loom_low_lower_context_source_function(context),
      .bundle = loom_low_lower_context_bundle(context),
      .descriptor_set = loom_low_lower_context_descriptor_set(context),
      .feature_bits =
          loom_low_lower_context_bundle(context)->config->contract_feature_bits,
      .map_value =
          {
              .fn = loom_low_lower_rule_match_map_value_from_lowering,
              .user_data = context,
          },
      .can_materialize =
          {
              .fn = loom_low_lower_rule_match_can_materialize_from_lowering,
              .user_data = context,
          },
      .descriptor_ref =
          {
              .fn = loom_low_lower_rule_match_descriptor_ref_from_lowering,
              .user_data = context,
          },
      .source_memory_root_byte_offset =
          {
              .fn =
                  context->policy->source_memory_root_byte_offset.fn != NULL
                      ? loom_low_lower_rule_match_source_memory_root_byte_offset_from_lowering
                      : NULL,
              .user_data = context,
          },
      .fact_table = loom_low_lower_context_fact_table(context),
      .view_regions = view_regions,
      .source_memory_state = source_memory_state,
      .symbolic_expr_context =
          loom_low_lower_context_symbolic_expr_context(context),
  };
}

static iree_status_t loom_low_lower_rule_set_match_view_regions(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_view_region_table_t** out_view_regions) {
  *out_view_regions = NULL;
  if (rule_set->source_memory_count == 0) {
    return iree_ok_status();
  }
  return loom_low_lower_context_view_regions(context, out_view_regions);
}

static iree_status_t loom_low_lower_rule_set_select_range_from_lowering(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_match_flags_t match_flags, uint16_t rule_start,
    uint16_t rule_count, loom_low_source_memory_access_plan_t* access_plan,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_match_view_regions(
      context, rule_set, &view_regions));
  loom_low_lower_rule_source_memory_state_t source_memory_state;
  loom_low_lower_rule_source_memory_state_initialize(source_op, access_plan,
                                                     &source_memory_state);
  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, view_regions, &source_memory_state, &match_context);
  match_context.flags |= match_flags;
  return loom_low_lower_rule_set_select_rule_range_with_match_context(
      &match_context, rule_set, source_op, rule_start, rule_count,
      out_selection);
}

static iree_status_t loom_low_lower_rule_set_select_range(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_match_flags_t match_flags, uint16_t rule_start,
    uint16_t rule_count, loom_low_lower_rule_selection_t* out_selection) {
  if (rule_set->source_memory_count != 0) {
    loom_low_source_memory_access_plan_t access_plan;
    return loom_low_lower_rule_set_select_range_from_lowering(
        context, rule_set, source_op, match_flags, rule_start, rule_count,
        &access_plan, out_selection);
  }
  return loom_low_lower_rule_set_select_range_from_lowering(
      context, rule_set, source_op, match_flags, rule_start, rule_count,
      /*access_plan=*/NULL, out_selection);
}

void loom_low_lower_rule_materialize_diagnostic_params(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_diagnostic_t* diagnostic,
    loom_diagnostic_param_t* out_params) {
  uint8_t param_index = 0;
  if (iree_any_bit_set(
          diagnostic->flags,
          LOOM_LOW_LOWER_DIAGNOSTIC_FLAG_IMPLICIT_TARGET_CONTEXT)) {
    out_params[0] = loom_param_string(
        loom_low_lower_rule_target_key(match_context->bundle));
    out_params[1] = loom_param_string(
        loom_low_lower_rule_export_name(match_context->bundle));
    out_params[2] = loom_param_string(
        loom_low_lower_rule_config_key(match_context->bundle));
    out_params[3] =
        loom_param_string(loom_low_lower_rule_function_name(match_context));
    out_params[4] =
        loom_param_string(loom_op_name(match_context->module, source_op));
    param_index = LOOM_LOW_LOWER_TARGET_CONTEXT_PARAM_COUNT;
  }
  const uint8_t stored_param_count = diagnostic->param_count - param_index;
  for (uint8_t stored_param_index = 0; stored_param_index < stored_param_count;
       ++stored_param_index, ++param_index) {
    const uint16_t param_ref_index =
        (uint16_t)(diagnostic->param_start + stored_param_index);
    const loom_low_lower_diagnostic_param_ref_t param_ordinal =
        rule_set->diagnostic_param_refs[param_ref_index];
    const loom_low_lower_diagnostic_param_t* row =
        &rule_set->diagnostic_params[param_ordinal];
    switch (row->kind) {
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_TARGET_KEY:
        out_params[param_index] = loom_param_string(
            loom_low_lower_rule_target_key(match_context->bundle));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_EXPORT_NAME:
        out_params[param_index] = loom_param_string(
            loom_low_lower_rule_export_name(match_context->bundle));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_CONFIG_KEY:
        out_params[param_index] = loom_param_string(
            loom_low_lower_rule_config_key(match_context->bundle));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_FUNCTION_NAME:
        out_params[param_index] =
            loom_param_string(loom_low_lower_rule_function_name(match_context));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_OP_NAME:
        out_params[param_index] =
            loom_param_string(loom_op_name(match_context->module, source_op));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL:
        out_params[param_index] =
            loom_param_string(loom_low_lower_rule_set_string(
                rule_set, row->value.string_value_ref));
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_VALUE_TYPE: {
        const loom_value_id_t value_id = loom_low_lower_rule_source_value(
            match_context->module, rule_set, source_op,
            row->value.value_ref_index);
        out_params[param_index] = loom_param_type(
            loom_low_lower_rule_match_value_type(match_context, value_id));
        break;
      }
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_I64_LITERAL:
        out_params[param_index] = loom_param_i64(row->value.i64_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U32_LITERAL:
        out_params[param_index] = loom_param_u32(row->value.u32_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U64_LITERAL:
        out_params[param_index] = loom_param_u64(row->value.u64_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_BOOL_LITERAL:
        out_params[param_index] = loom_param_bool(row->value.bool_value);
        break;
      case LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_MEMORY_MINIMUM_ALIGNMENT:
        out_params[param_index] =
            loom_param_u32(loom_low_lower_rule_source_memory_minimum_alignment(
                match_context, source_op));
        break;
      default:
        IREE_ASSERT_UNREACHABLE("unknown generated diagnostic param kind");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
}

static iree_status_t loom_low_lower_rule_emit_diagnostic(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t diagnostic_index,
    loom_low_lower_rule_source_memory_state_t* source_memory_state) {
  if (diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
      diagnostic_index >= rule_set->diagnostic_count) {
    return loom_low_lower_rule_emit_no_mapping(context, source_op);
  }
  const loom_low_lower_diagnostic_t* diagnostic =
      &rule_set->diagnostics[diagnostic_index];
  loom_diagnostic_param_t params[LOOM_LOW_LOWER_MAX_DIAGNOSTIC_PARAMS] = {0};
  IREE_ASSERT_LE(diagnostic->param_count, IREE_ARRAYSIZE(params));
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_match_view_regions(
      context, rule_set, &view_regions));
  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, view_regions, source_memory_state, &match_context);
  loom_low_lower_rule_materialize_diagnostic_params(
      &match_context, rule_set, source_op, diagnostic, params);
  return loom_low_lower_emit_error_ref(context, source_op,
                                       diagnostic->error_ref, params,
                                       diagnostic->param_count);
}

iree_status_t loom_low_lower_rule_set_select(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_low_lower_rule_span_t* span =
      loom_low_lower_rule_set_find_span(rule_set, source_op->kind);
  return loom_low_lower_rule_set_select_range(
      context, rule_set, source_op, /*match_flags=*/0,
      span ? span->rule_start : 0, span ? span->rule_count : 0, out_selection);
}

iree_status_t loom_low_lower_rule_set_select_contract(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_selection_t* out_selection) {
  const loom_low_lower_rule_span_t* span =
      loom_low_lower_rule_set_find_span(rule_set, source_op->kind);
  return loom_low_lower_rule_set_select_range(
      context, rule_set, source_op,
      LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY, span ? span->rule_start : 0,
      span ? span->rule_count : 0, out_selection);
}

iree_status_t loom_low_lower_rule_set_select_rule_range(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t rule_start, uint16_t rule_count,
    loom_low_lower_rule_selection_t* out_selection) {
  return loom_low_lower_rule_set_select_range(context, rule_set, source_op,
                                              /*match_flags=*/0, rule_start,
                                              rule_count, out_selection);
}

const loom_low_lower_diagnostic_t* loom_low_lower_rule_set_failure_diagnostic(
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_rule_failure_t failure) {
  if (failure.diagnostic_index == LOOM_LOW_LOWER_DIAGNOSTIC_NONE ||
      failure.diagnostic_index >= rule_set->diagnostic_count) {
    return NULL;
  }
  return &rule_set->diagnostics[failure.diagnostic_index];
}

loom_low_lower_descriptor_ref_t loom_low_lower_rule_primary_descriptor_ref(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_t* rule) {
  if (rule->emit_count == 0) {
    return LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE;
  }
  const uint16_t primary_emit_ordinal =
      loom_low_lower_rule_primary_emit_ordinal(rule);
  if (primary_emit_ordinal == LOOM_LOW_LOWER_RULE_PRIMARY_EMIT_NONE) {
    return LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE;
  }
  IREE_ASSERT_LT(primary_emit_ordinal, rule->emit_count);
  const uint16_t emit_ref_index =
      (uint16_t)(rule->action.emit_start + primary_emit_ordinal);
  return loom_low_lower_rule_set_emit_at(rule_set, emit_ref_index)
      ->descriptor_ref;
}

iree_status_t loom_low_lower_rule_set_emit_selection_failure(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    loom_low_lower_rule_failure_t failure,
    loom_low_lower_rule_source_memory_state_t* source_memory_state) {
  if (!failure.has_source_op_span) {
    return loom_low_lower_rule_emit_no_mapping(context, source_op);
  }
  const loom_op_t* diagnostic_source_op = failure.diagnostic_source_op != NULL
                                              ? failure.diagnostic_source_op
                                              : source_op;
  return loom_low_lower_rule_emit_diagnostic(
      context, rule_set, diagnostic_source_op, failure.diagnostic_index,
      source_memory_state);
}
