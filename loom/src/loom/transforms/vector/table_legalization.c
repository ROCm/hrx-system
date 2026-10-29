// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/table_legalization.h"

#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"

typedef struct loom_vector_table_quantize_builder_t {
  // Builder positioned immediately before the quantization operation.
  loom_builder_t* builder;
  // Source location of the quantization operation.
  loom_location_id_t location;
  // Rank-one view of the input value captured by the source program.
  loom_value_id_t input;
  // Ordered rank-one threshold value captured by the source program.
  loom_value_id_t thresholds;
  // Number of logical input lanes.
  uint64_t input_count;
  // Number of ordered thresholds; also the maximum result code.
  uint64_t threshold_count;
  // Source floating-point element type, before native comparison widening.
  loom_scalar_type_t input_element_type;
  // Native floating-point comparison element type.
  loom_scalar_type_t comparison_element_type;
  // Native ordinal type, or i1 for a predicate result.
  loom_scalar_type_t ordinal_element_type;
  // Logical result element type, after ordinal narrowing.
  loom_scalar_type_t result_element_type;
  // Threshold lanes fitting one native comparison carrier.
  uint32_t threshold_packet_count;
  // Floating relation implementing both the NaN and tie policies.
  uint8_t predicate;
} loom_vector_table_quantize_builder_t;

static loom_type_t loom_vector_table_packet_type(loom_scalar_type_t element,
                                                 uint64_t count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element,
                             loom_dim_pack_static((int64_t)count), 0);
}

static int64_t loom_vector_table_quantize_integer_bits(
    loom_scalar_type_t element_type, uint64_t value) {
  const uint32_t bit_count = loom_scalar_type_bitwidth(element_type);
  if (bit_count < 64 && value >= (UINT64_C(1) << (bit_count - 1))) {
    return (int64_t)(value - (UINT64_C(1) << bit_count));
  }
  return (int64_t)value;
}

static iree_status_t loom_vector_table_quantize_constant(
    const loom_vector_table_quantize_builder_t* quantize, loom_type_t type,
    uint64_t value, loom_value_id_t* out_value) {
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_splati_build(quantize->builder,
                               loom_vector_table_quantize_integer_bits(
                                   loom_type_element_type(type), value),
                               type, quantize->location, &constant_op));
  *out_value = loom_vector_splati_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_table_quantize_comparison_packet(
    const loom_vector_table_quantize_builder_t* quantize,
    loom_value_id_t source, uint64_t source_count, uint64_t offset,
    uint32_t count, loom_value_id_t* out_value) {
  loom_value_id_t value = source;
  const loom_type_t input_type =
      loom_vector_table_packet_type(quantize->input_element_type, count);
  loom_op_t* packet_op = NULL;
  if (count != source_count) {
    const int64_t static_offset = (int64_t)offset;
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        quantize->builder, source, NULL, 0, &static_offset, 1, input_type,
        quantize->location, &packet_op));
    value = loom_vector_slice_result(packet_op);
  }
  if (quantize->comparison_element_type != quantize->input_element_type) {
    const loom_type_t comparison_type =
        loom_vector_table_packet_type(quantize->comparison_element_type, count);
    IREE_RETURN_IF_ERROR(loom_vector_extf_build(
        quantize->builder, /*instance_flags=*/0, value, input_type,
        comparison_type, quantize->location, &packet_op));
    value = loom_vector_extf_result(packet_op);
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_vector_table_quantize_linear_search(
    const loom_vector_table_quantize_builder_t* quantize, loom_value_id_t input,
    loom_type_t comparison_type, loom_type_t ordinal_type,
    loom_type_t predicate_type, loom_value_id_t* out_ordinal) {
  const loom_type_t index_type = loom_vector_table_packet_type(
      LOOM_SCALAR_TYPE_I8, loom_type_dim_static_size_at(ordinal_type, 0));
  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_table_quantize_constant(quantize, ordinal_type, 0, &ordinal));
  loom_value_id_t unit_increment = LOOM_VALUE_ID_INVALID;
  if (quantize->ordinal_element_type != LOOM_SCALAR_TYPE_I1 &&
      quantize->threshold_count != 0) {
    IREE_RETURN_IF_ERROR(loom_vector_table_quantize_constant(
        quantize, ordinal_type, 1, &unit_increment));
  }
  loom_op_t* value_op = NULL;
  for (uint64_t base = 0; base < quantize->threshold_count;
       base += quantize->threshold_packet_count) {
    const uint32_t threshold_count = (uint32_t)iree_min(
        quantize->threshold_packet_count, quantize->threshold_count - base);
    loom_value_id_t thresholds = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_table_quantize_comparison_packet(
        quantize, quantize->thresholds, quantize->threshold_count, base,
        threshold_count, &thresholds));
    for (uint32_t index = 0; index < threshold_count; ++index) {
      loom_value_id_t index_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_vector_table_quantize_constant(
          quantize, index_type, index, &index_value));
      IREE_RETURN_IF_ERROR(loom_vector_table_lookup_build(
          quantize->builder, thresholds, index_value, comparison_type,
          quantize->location, &value_op));
      IREE_RETURN_IF_ERROR(loom_vector_cmpf_build(
          quantize->builder, 0, quantize->predicate,
          loom_vector_table_lookup_result(value_op), input, comparison_type,
          predicate_type, quantize->location, &value_op));
      const loom_value_id_t passed = loom_vector_cmpf_result(value_op);
      if (quantize->ordinal_element_type == LOOM_SCALAR_TYPE_I1) {
        // An i1 result admits at most one threshold.
        ordinal = passed;
        continue;
      }

      IREE_RETURN_IF_ERROR(
          loom_vector_addi_build(quantize->builder, 0, ordinal, unit_increment,
                                 ordinal_type, quantize->location, &value_op));
      IREE_RETURN_IF_ERROR(loom_vector_select_build(
          quantize->builder, passed, loom_vector_addi_result(value_op), ordinal,
          ordinal_type, quantize->location, &value_op));
      ordinal = loom_vector_select_result(value_op);
    }
  }
  *out_ordinal = ordinal;
  return iree_ok_status();
}

static iree_status_t loom_vector_table_quantize_packet(
    const loom_vector_table_quantize_builder_t* quantize, uint64_t offset,
    uint32_t count, loom_value_id_t* out_value) {
  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_table_quantize_comparison_packet(
      quantize, quantize->input, quantize->input_count, offset, count, &input));
  const loom_type_t comparison_type =
      loom_vector_table_packet_type(quantize->comparison_element_type, count);
  const loom_type_t ordinal_type =
      loom_vector_table_packet_type(quantize->ordinal_element_type, count);
  const loom_type_t predicate_type =
      loom_vector_table_packet_type(LOOM_SCALAR_TYPE_I1, count);
  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_table_quantize_linear_search(
      quantize, input, comparison_type, ordinal_type, predicate_type,
      &ordinal));

  loom_op_t* value_op = NULL;
  if (quantize->ordinal_element_type != quantize->result_element_type) {
    const loom_type_t result_type =
        loom_vector_table_packet_type(quantize->result_element_type, count);
    if (loom_scalar_type_bitwidth(quantize->ordinal_element_type) >
        loom_scalar_type_bitwidth(quantize->result_element_type)) {
      IREE_RETURN_IF_ERROR(
          loom_vector_trunci_build(quantize->builder, ordinal, ordinal_type,
                                   result_type, quantize->location, &value_op));
      ordinal = loom_vector_trunci_result(value_op);
    } else {
      IREE_RETURN_IF_ERROR(
          loom_vector_extui_build(quantize->builder, ordinal, ordinal_type,
                                  result_type, quantize->location, &value_op));
      ordinal = loom_vector_extui_result(value_op);
    }
  }
  *out_value = ordinal;
  return iree_ok_status();
}

iree_status_t loom_vector_table_quantize_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_table_quantize_policy_t* policy, bool* out_rewritten) {
  *out_rewritten = false;
  const loom_value_id_t input = loom_vector_table_quantize_input(op);
  const loom_value_id_t thresholds = loom_vector_table_quantize_thresholds(op);
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  const loom_type_t threshold_type =
      loom_module_value_type(context->module, thresholds);
  const loom_type_t result_type = loom_module_value_type(
      context->module, loom_vector_table_quantize_result(op));
  uint64_t input_count = 0;
  uint64_t threshold_count = 0;
  if (!loom_type_static_element_count(input_type, &input_count) ||
      !loom_type_static_element_count(threshold_type, &threshold_count) ||
      input_count == 0 || input_count > INT64_MAX ||
      threshold_count > INT64_MAX) {
    return iree_ok_status();
  }

  const loom_scalar_type_t result_element_type =
      loom_type_element_type(result_type);
  const loom_scalar_type_t ordinal_element_type =
      result_element_type == LOOM_SCALAR_TYPE_I1 ? LOOM_SCALAR_TYPE_I1
                                                 : policy->ordinal_element_type;
  const uint32_t comparison_bit_count =
      loom_scalar_type_bitwidth(policy->comparison_element_type);
  const uint32_t lane_bit_count = iree_max(
      iree_max(comparison_bit_count,
               (uint32_t)loom_scalar_type_bitwidth(ordinal_element_type)),
      (uint32_t)loom_scalar_type_bitwidth(result_element_type));
  const uint32_t packet_count = policy->packet_bit_count / lane_bit_count;
  const uint64_t result_packet_count = (input_count - 1) / packet_count + 1;
  loom_value_id_t single_packet = LOOM_VALUE_ID_INVALID;
  loom_value_id_t* packets = &single_packet;
  if (result_packet_count > 1) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(context->arena, result_packet_count,
                                  sizeof(*packets), (void**)&packets));
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t flat_input_type = loom_vector_table_packet_type(
      loom_type_element_type(input_type), input_count);
  loom_value_id_t flat_input = input;
  if (!loom_type_equal(input_type, flat_input_type)) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(&rewriter->builder, input,
                                                   input_type, flat_input_type,
                                                   op->location, &bitcast_op));
    flat_input = loom_vector_bitcast_result(bitcast_op);
  }
  const bool upper = loom_vector_table_quantize_tie(op) ==
                     LOOM_VECTOR_TABLE_QUANTIZE_TIE_UPPER;
  const bool nan_max =
      loom_vector_table_quantize_nan(op) == LOOM_VECTOR_TABLE_QUANTIZE_NAN_MAX;
  loom_vector_table_quantize_builder_t quantize = {
      .builder = &rewriter->builder,
      .location = op->location,
      .input = flat_input,
      .thresholds = thresholds,
      .input_count = input_count,
      .threshold_count = threshold_count,
      .input_element_type = loom_type_element_type(input_type),
      .comparison_element_type = policy->comparison_element_type,
      .ordinal_element_type = ordinal_element_type,
      .result_element_type = result_element_type,
      .threshold_packet_count = policy->packet_bit_count / comparison_bit_count,
      .predicate = nan_max ? (upper ? LOOM_VECTOR_CMPF_PREDICATE_ULE
                                    : LOOM_VECTOR_CMPF_PREDICATE_ULT)
                           : (upper ? LOOM_VECTOR_CMPF_PREDICATE_OLE
                                    : LOOM_VECTOR_CMPF_PREDICATE_OLT),
  };
  for (uint64_t packet = 0; packet < result_packet_count; ++packet) {
    const uint64_t offset = packet * packet_count;
    const uint32_t count =
        (uint32_t)iree_min(packet_count, input_count - offset);
    IREE_RETURN_IF_ERROR(loom_vector_table_quantize_packet(
        &quantize, offset, count, &packets[packet]));
  }
  const loom_type_t flat_result_type =
      loom_vector_table_packet_type(result_element_type, input_count);
  loom_value_id_t replacement = packets[0];
  if (result_packet_count > 1) {
    loom_op_t* value_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        quantize.builder, 0, packets, result_packet_count, flat_result_type,
        op->location, &value_op));
    replacement = loom_vector_concat_result(value_op);
  }
  if (!loom_type_equal(flat_result_type, result_type)) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        quantize.builder, replacement, flat_result_type, result_type,
        op->location, &bitcast_op));
    replacement = loom_vector_bitcast_result(bitcast_op);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}
