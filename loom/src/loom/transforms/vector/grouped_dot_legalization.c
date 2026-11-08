// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/grouped_dot_legalization.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"

static loom_type_t loom_vector_grouped_dot_flat_type(loom_type_t type,
                                                     uint32_t lane_count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(type),
                             loom_dim_pack_static(lane_count),
                             loom_type_rank(type) == 1 ? type.encoding_id : 0);
}

static iree_status_t loom_vector_grouped_dot_packet_is_legal(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    uint32_t input_lane_count, uint32_t result_lane_count,
    uint32_t packet_input_lane_count, bool* out_legal) {
  const loom_target_contract_vector_lane_projection_t projection = {
      .source_lane_count = input_lane_count,
      .projected_lane_count = packet_input_lane_count,
      .secondary_source_lane_count = result_lane_count,
      .secondary_projected_lane_count =
          packet_input_lane_count / (input_lane_count / result_lane_count),
  };
  loom_target_contract_query_result_t query_result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(
      loom_target_legalization_query_contract_with_vector_lane_projection(
          context, op, projection, &query_result));
  *out_legal = query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  return iree_ok_status();
}

iree_status_t loom_vector_grouped_dot_packetize(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  const loom_target_vector_packet_policy_t* policy =
      context->vector_packet_policy;
  if (policy == NULL ||
      context->policy == LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY ||
      context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_type_t input_type =
      loom_module_value_type(context->module, operands[0]);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_const_results(op)[0]);
  uint64_t input_lane_count = 0;
  uint64_t result_lane_count = 0;
  if (!loom_type_static_element_count(input_type, &input_lane_count) ||
      !loom_type_static_element_count(result_type, &result_lane_count) ||
      input_lane_count == 0 || input_lane_count > UINT32_MAX) {
    return iree_ok_status();
  }
  const uint32_t group_size = input_lane_count / result_lane_count;
  const bool ranked = loom_type_rank(input_type) != 1;
  const uint32_t operation_limit =
      policy->maximum_component_expansion_op_count != 0
          ? policy->maximum_component_expansion_op_count
          : 64u;
  uint32_t packet_lane_count = 0;
  // A ranked operation may only need a row-major view. Otherwise consider the
  // target's native input widths and require both full and tail packets to be
  // legal before changing any IR.
  for (uint8_t i = 0; i <= policy->native_lane_count_count; ++i) {
    const uint32_t candidate = i == policy->native_lane_count_count
                                   ? (ranked ? input_lane_count : 0)
                                   : policy->native_lane_counts[i];
    if (candidate == 0 || candidate > input_lane_count ||
        (candidate == input_lane_count && !ranked) ||
        candidate <= packet_lane_count || candidate % group_size != 0) {
      continue;
    }
    const uint32_t packet_count = (input_lane_count - 1u) / candidate + 1u;
    if ((uint64_t)packet_count * 4u + 1u + (ranked ? 4u : 0u) >
        operation_limit) {
      continue;
    }
    bool legal = false;
    IREE_RETURN_IF_ERROR(loom_vector_grouped_dot_packet_is_legal(
        context, op, input_lane_count, result_lane_count, candidate, &legal));
    if (!legal) {
      continue;
    }
    const uint32_t tail = input_lane_count % candidate;
    if (tail != 0) {
      IREE_RETURN_IF_ERROR(loom_vector_grouped_dot_packet_is_legal(
          context, op, input_lane_count, result_lane_count, tail, &legal));
    }
    if (legal) {
      packet_lane_count = candidate;
    }
  }
  if (packet_lane_count == 0) {
    return iree_ok_status();
  }

  const uint32_t packet_count =
      (input_lane_count - 1u) / packet_lane_count + 1u;
  loom_value_id_t* packet_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, packet_count,
                                                 sizeof(*packet_results),
                                                 (void**)&packet_results));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t sources[3];
  for (uint16_t i = 0; i < 3; ++i) {
    sources[i] = operands[i];
    if (ranked) {
      const loom_type_t source_type =
          loom_module_value_type(context->module, sources[i]);
      const loom_type_t flat_type = loom_vector_grouped_dot_flat_type(
          source_type, i < 2 ? input_lane_count : result_lane_count);
      loom_op_t* bitcast = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
          builder, sources[i], source_type, flat_type, op->location, &bitcast));
      sources[i] = loom_vector_bitcast_result(bitcast);
    }
  }

  for (uint32_t i = 0; i < packet_count; ++i) {
    const uint32_t input_offset = i * packet_lane_count;
    const uint32_t input_count =
        iree_min(input_lane_count - input_offset, packet_lane_count);
    loom_value_id_t packet_operands[3];
    for (uint16_t j = 0; j < 3; ++j) {
      const int64_t offset = j < 2 ? input_offset : input_offset / group_size;
      const uint32_t count = j < 2 ? input_count : input_count / group_size;
      const loom_type_t source_type =
          loom_module_value_type(context->module, sources[j]);
      if (count == (j < 2 ? input_lane_count : result_lane_count)) {
        packet_operands[j] = sources[j];
      } else {
        loom_op_t* slice = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_slice_build(
            builder, sources[j], NULL, 0, &offset, 1,
            loom_vector_grouped_dot_flat_type(source_type, count), op->location,
            &slice));
        packet_operands[j] = loom_vector_slice_result(slice);
      }
    }
    loom_op_t* packet_op = NULL;
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        builder, op->kind, /*operand_count=*/3, /*result_count=*/1,
        /*region_count=*/0, /*tied_result_count=*/0, op->attribute_count,
        op->location, &packet_op));
    packet_op->instance_flags = op->instance_flags;
    memcpy(loom_op_operands(packet_op), packet_operands,
           sizeof(packet_operands));
    memcpy(loom_op_attrs(packet_op), loom_op_const_attrs(op),
           op->attribute_count * sizeof(loom_attribute_t));
    IREE_RETURN_IF_ERROR(
        loom_builder_define_value(builder,
                                  loom_vector_grouped_dot_flat_type(
                                      result_type, input_count / group_size),
                                  &packet_results[i]));
    loom_op_results(packet_op)[0] = packet_results[i];
    IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, packet_op));
  }
  loom_value_id_t replacement = packet_results[0];
  const loom_type_t flat_result_type =
      loom_vector_grouped_dot_flat_type(result_type, result_lane_count);
  if (packet_count > 1) {
    loom_op_t* concat = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_concat_build(builder, 0, packet_results, packet_count,
                                 flat_result_type, op->location, &concat));
    replacement = loom_vector_concat_result(concat);
  }
  if (ranked) {
    loom_op_t* bitcast = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_bitcast_build(builder, replacement, flat_result_type,
                                  result_type, op->location, &bitcast));
    replacement = loom_vector_bitcast_result(bitcast);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}
