// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/grouped_packet_legalization.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"

static loom_type_t loom_vector_grouped_packet_flat_type(loom_type_t type,
                                                        uint32_t lane_count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(type),
                             loom_dim_pack_static(lane_count),
                             loom_type_rank(type) == 1 ? type.encoding_id : 0);
}

static iree_status_t loom_vector_grouped_packet_is_legal(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    uint32_t expanded_lane_count, uint32_t group_lane_count,
    uint32_t packet_expanded_lane_count, bool* out_legal) {
  const loom_target_contract_vector_lane_projection_t projection = {
      .source_lane_count = expanded_lane_count,
      .projected_lane_count = packet_expanded_lane_count,
      .secondary_source_lane_count = group_lane_count,
      .secondary_projected_lane_count =
          packet_expanded_lane_count / (expanded_lane_count / group_lane_count),
  };
  loom_target_contract_query_result_t query_result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(
      loom_target_legalization_query_contract_with_vector_lane_projection(
          context, op, projection, &query_result));
  *out_legal = query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  return iree_ok_status();
}

static iree_status_t loom_vector_grouped_packetize(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_type_t expanded_type, loom_type_t group_type, bool* out_rewritten) {
  *out_rewritten = false;
  const loom_target_vector_packet_policy_t* policy =
      context->vector_packet_policy;
  if (policy == NULL ||
      context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  uint64_t expanded_lane_count = 0;
  uint64_t group_lane_count = 0;
  if (!loom_type_static_element_count(expanded_type, &expanded_lane_count) ||
      !loom_type_static_element_count(group_type, &group_lane_count) ||
      expanded_lane_count == 0 || expanded_lane_count > UINT32_MAX) {
    return iree_ok_status();
  }
  const uint32_t group_size = expanded_lane_count / group_lane_count;
  const bool ranked = loom_type_rank(expanded_type) != 1;
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
                                   ? (ranked ? expanded_lane_count : 0)
                                   : policy->native_lane_counts[i];
    if (candidate == 0 || candidate > expanded_lane_count ||
        (candidate == expanded_lane_count && !ranked) ||
        candidate <= packet_lane_count || candidate % group_size != 0) {
      continue;
    }
    const uint32_t packet_count = (expanded_lane_count - 1u) / candidate + 1u;
    if ((uint64_t)packet_count * (op->operand_count + 1u) + op->result_count +
            (ranked ? op->operand_count + op->result_count : 0u) >
        operation_limit) {
      continue;
    }
    bool legal = false;
    IREE_RETURN_IF_ERROR(loom_vector_grouped_packet_is_legal(
        context, op, expanded_lane_count, group_lane_count, candidate, &legal));
    if (!legal) {
      continue;
    }
    const uint32_t tail = expanded_lane_count % candidate;
    if (tail != 0) {
      IREE_RETURN_IF_ERROR(loom_vector_grouped_packet_is_legal(
          context, op, expanded_lane_count, group_lane_count, tail, &legal));
    }
    if (legal) {
      packet_lane_count = candidate;
    }
  }
  if (packet_lane_count == 0) {
    return iree_ok_status();
  }

  const uint32_t packet_count =
      (expanded_lane_count - 1u) / packet_lane_count + 1u;
  loom_value_id_t* packet_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, packet_count * op->result_count, sizeof(*packet_results),
      (void**)&packet_results));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  // These grouped operations have at most three inputs and two outputs.
  loom_value_id_t sources[3];
  uint32_t operand_divisors[3];
  uint32_t result_divisors[2];
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    sources[i] = operands[i];
    const loom_type_t source_type =
        loom_module_value_type(context->module, sources[i]);
    uint64_t source_lane_count = 0;
    (void)loom_type_static_element_count(source_type, &source_lane_count);
    operand_divisors[i] = expanded_lane_count / source_lane_count;
    if (ranked) {
      const loom_type_t flat_type =
          loom_vector_grouped_packet_flat_type(source_type, source_lane_count);
      loom_op_t* bitcast = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
          builder, sources[i], source_type, flat_type, op->location, &bitcast));
      sources[i] = loom_vector_bitcast_result(bitcast);
    }
  }

  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(context->module, loom_op_const_results(op)[i]);
    uint64_t lane_count = 0;
    (void)loom_type_static_element_count(type, &lane_count);
    result_divisors[i] = expanded_lane_count / lane_count;
  }

  for (uint32_t i = 0; i < packet_count; ++i) {
    const uint32_t expanded_offset = i * packet_lane_count;
    const uint32_t expanded_count =
        iree_min(expanded_lane_count - expanded_offset, packet_lane_count);
    loom_value_id_t packet_operands[3];
    for (uint16_t j = 0; j < op->operand_count; ++j) {
      const loom_type_t source_type =
          loom_module_value_type(context->module, sources[j]);
      const uint32_t divisor = operand_divisors[j];
      const uint32_t source_lane_count = expanded_lane_count / divisor;
      const int64_t offset = expanded_offset / divisor;
      const uint32_t count = expanded_count / divisor;
      if (count == source_lane_count) {
        packet_operands[j] = sources[j];
      } else {
        loom_op_t* slice = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_slice_build(
            builder, sources[j], NULL, 0, &offset, 1,
            loom_vector_grouped_packet_flat_type(source_type, count),
            op->location, &slice));
        packet_operands[j] = loom_vector_slice_result(slice);
      }
    }
    loom_op_t* packet_op = NULL;
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        builder, op->kind, op->operand_count, op->result_count,
        /*region_count=*/0, /*tied_result_count=*/0, op->attribute_count,
        op->location, &packet_op));
    packet_op->instance_flags = op->instance_flags;
    memcpy(loom_op_operands(packet_op), packet_operands,
           op->operand_count * sizeof(*packet_operands));
    memcpy(loom_op_attrs(packet_op), loom_op_const_attrs(op),
           op->attribute_count * sizeof(loom_attribute_t));
    for (uint16_t j = 0; j < op->result_count; ++j) {
      const loom_type_t type =
          loom_module_value_type(context->module, loom_op_const_results(op)[j]);
      const loom_type_t packet_type = loom_vector_grouped_packet_flat_type(
          type, expanded_count / result_divisors[j]);
      loom_value_id_t* result = &packet_results[j * packet_count + i];
      IREE_RETURN_IF_ERROR(
          loom_builder_define_value(builder, packet_type, result));
      loom_op_results(packet_op)[j] = *result;
    }
    IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, packet_op));
  }
  loom_value_id_t replacements[2];
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(context->module, loom_op_const_results(op)[i]);
    const uint32_t lane_count = expanded_lane_count / result_divisors[i];
    const loom_type_t flat_type =
        loom_vector_grouped_packet_flat_type(type, lane_count);
    replacements[i] = packet_results[i * packet_count];
    if (packet_count > 1) {
      loom_op_t* concat = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_concat_build(
          builder, 0, &packet_results[i * packet_count], packet_count,
          flat_type, op->location, &concat));
      replacements[i] = loom_vector_concat_result(concat);
    }
    if (ranked) {
      loom_op_t* bitcast = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
          builder, replacements[i], flat_type, type, op->location, &bitcast));
      replacements[i] = loom_vector_bitcast_result(bitcast);
    }
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, op->result_count));
  *out_rewritten = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_grouped_dot(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_grouped_packetize(
      context, op,
      loom_module_value_type(context->module, loom_op_const_operands(op)[0]),
      loom_module_value_type(context->module, loom_op_const_results(op)[0]),
      &rewritten));
  *out_result = (loom_target_legalizer_result_t){
      .action = rewritten ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                          : LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_interleave_packets(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  const bool is_zip = entry->root_kind == LOOM_OP_VECTOR_INTERLEAVE;
  const loom_type_t full_type = loom_module_value_type(
      context->module,
      is_zip ? loom_op_const_results(op)[0] : loom_op_const_operands(op)[0]);
  const loom_type_t half_type = loom_module_value_type(
      context->module,
      is_zip ? loom_op_const_operands(op)[0] : loom_op_const_results(op)[0]);
  bool rewritten = false;
  if (loom_type_rank(full_type) == 1) {
    IREE_RETURN_IF_ERROR(loom_vector_grouped_packetize(context, op, full_type,
                                                       half_type, &rewritten));
  }
  *out_result = (loom_target_legalizer_result_t){
      .action = rewritten ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                          : LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kGroupedPacketLegalizerRules[] = {
    {.root_kind = LOOM_OP_VECTOR_INTERLEAVE,
     .legalize = loom_vector_legalize_interleave_packets},
    {.root_kind = LOOM_OP_VECTOR_DEINTERLEAVE,
     .legalize = loom_vector_legalize_interleave_packets},
    {.root_kind = LOOM_OP_VECTOR_DOT2F,
     .legalize = loom_vector_legalize_grouped_dot},
    {.root_kind = LOOM_OP_VECTOR_DOT4I,
     .legalize = loom_vector_legalize_grouped_dot},
    {.root_kind = LOOM_OP_VECTOR_DOT8I4,
     .legalize = loom_vector_legalize_grouped_dot},
    {.root_kind = LOOM_OP_VECTOR_DOT4F8,
     .legalize = loom_vector_legalize_grouped_dot},
};

static const loom_target_legalizer_provider_t kGroupedPacketLegalizerProvider =
    {
        .name = IREE_SVL("vector-grouped-packet"),
        .scope = LOOM_LOW_DESCRIPTOR_SCOPE_SHARED,
        .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
        .rules = kGroupedPacketLegalizerRules,
        .rule_count = IREE_ARRAYSIZE(kGroupedPacketLegalizerRules),
};

const loom_target_legalizer_provider_t*
loom_vector_grouped_packet_legalizer_provider(void) {
  return &kGroupedPacketLegalizerProvider;
}
