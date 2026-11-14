// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/coordinate_legalization.h"

#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector/packet_plan.h"

static iree_status_t loom_vector_coordinate_packet_base(
    loom_builder_t* builder, loom_value_id_t base, loom_value_id_t step,
    loom_type_t type, uint32_t offset, loom_location_id_t location,
    loom_value_id_t* out_base) {
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  const bool is_index = element_type == LOOM_SCALAR_TYPE_INDEX;
  int64_t ordinal = offset;
  const int32_t bit_width = loom_scalar_type_bitwidth(element_type);
  if (!is_index && bit_width < 64) {
    // The ordinal participates in the coordinate's modular arithmetic, even
    // when the vector contains more lanes than that integer domain has values.
    const int64_t sign = (int64_t)1 << (bit_width - 1);
    ordinal = ((ordinal & (sign * 2 - 1)) ^ sign) - sign;
  }
  loom_op_t* constant = NULL;
  loom_op_t* multiply = NULL;
  loom_op_t* add = NULL;
  if (is_index) {
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(ordinal), type, location, &constant));
    IREE_RETURN_IF_ERROR(
        loom_index_mul_build(builder, loom_index_constant_result(constant),
                             step, location, &multiply));
    IREE_RETURN_IF_ERROR(loom_index_add_build(
        builder, base, loom_index_mul_result(multiply), type, location, &add));
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder, loom_attr_i64(ordinal), type, location, &constant));
    IREE_RETURN_IF_ERROR(loom_scalar_muli_build(
        builder, 0, loom_scalar_constant_result(constant), step, type, location,
        &multiply));
    IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
        builder, 0, base, loom_scalar_muli_result(multiply), type, location,
        &add));
  }
  *out_base = loom_op_results(add)[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_iota_packets(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_target_vector_packet_policy_t* policy =
      context->vector_packet_policy;
  if (policy == NULL ||
      context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }
  const loom_type_t type =
      loom_module_value_type(context->module, loom_vector_iota_result(op));
  uint64_t lane_count = 0;
  if (!loom_type_static_element_count(type, &lane_count) || lane_count == 0 ||
      lane_count > UINT32_MAX) {
    return iree_ok_status();
  }
  const bool ranked = loom_type_rank(type) != 1;
  const uint32_t operation_limit =
      policy->maximum_component_expansion_op_count != 0
          ? policy->maximum_component_expansion_op_count
          : LOOM_VECTOR_PACKET_STATIC_OP_LIMIT;
  uint64_t candidates = 0;
  uint64_t producer_closure_candidates = 0;
  IREE_RETURN_IF_ERROR(loom_vector_packet_component_candidate_bits(
      context, op, lane_count, &candidates, &producer_closure_candidates));
  // Coordinate construction has scalar inputs, not a projected producer graph.
  candidates &= ~producer_closure_candidates;
  uint32_t packet_lane_count = 0;
  for (uint8_t i = 0; i < policy->native_lane_count_count; ++i) {
    const uint32_t candidate = policy->native_lane_counts[i];
    if ((candidates & ((uint64_t)1 << i)) == 0 ||
        candidate <= packet_lane_count) {
      continue;
    }
    const uint64_t packet_count = (lane_count - 1) / candidate + 1;
    if (packet_count * 4 - 2 + ranked <= operation_limit) {
      packet_lane_count = candidate;
    }
  }
  if (ranked) {
    const loom_target_contract_vector_lane_projection_t projection = {
        .source_lane_count = lane_count,
        .projected_lane_count = lane_count,
    };
    loom_target_contract_query_result_t result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(
        loom_target_legalization_query_contract_with_vector_lane_projection(
            context, op, projection, &result));
    if (result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      packet_lane_count = lane_count;
    }
  }
  if (packet_lane_count == 0) {
    return iree_ok_status();
  }

  const uint32_t packet_count = (lane_count - 1) / packet_lane_count + 1;
  loom_value_id_t* packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, packet_count, sizeof(*packets), (void**)&packets));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t base = loom_vector_iota_base(op);
  const loom_value_id_t step = loom_vector_iota_step(op);
  const loom_type_t scalar_type = loom_module_value_type(context->module, base);
  const loom_type_t flat_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(type),
      loom_dim_pack_static(lane_count), ranked ? 0 : type.encoding_id);
  for (uint32_t i = 0; i < packet_count; ++i) {
    const uint32_t offset = i * packet_lane_count;
    const uint32_t count = iree_min(packet_lane_count, lane_count - offset);
    loom_value_id_t packet_base = base;
    if (offset != 0) {
      IREE_RETURN_IF_ERROR(loom_vector_coordinate_packet_base(
          builder, base, step, scalar_type, offset, op->location,
          &packet_base));
    }
    loom_op_t* packet = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_iota_build(
        builder, packet_base, step, loom_vector_packet_type(flat_type, count),
        op->location, &packet));
    packets[i] = loom_vector_iota_result(packet);
  }
  loom_value_id_t replacement = packets[0];
  if (packet_count > 1) {
    loom_op_t* concat = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        builder, 0, packets, packet_count, flat_type, op->location, &concat));
    replacement = loom_vector_concat_result(concat);
  }
  if (ranked) {
    loom_op_t* bitcast = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        builder, replacement, flat_type, type, op->location, &bitcast));
    replacement = loom_vector_bitcast_result(bitcast);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_mask_coordinates(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  const loom_value_id_t lower = loom_vector_mask_range_lower_bound(op);
  const loom_type_t scalar_type =
      loom_module_value_type(context->module, lower);
  const loom_type_t mask_type = loom_module_value_type(
      context->module, loom_vector_mask_range_result(op));
  loom_type_t coordinate_type = mask_type;
  coordinate_type.header = loom_type_make_header(
      LOOM_TYPE_VECTOR, loom_type_element_type(scalar_type),
      loom_type_rank(mask_type), loom_type_flags(mask_type));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* iota = NULL;
  loom_op_t* upper = NULL;
  loom_op_t* compare = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_iota_build(builder, lower, loom_vector_mask_range_step(op),
                             coordinate_type, op->location, &iota));
  loom_value_id_t coordinates = loom_vector_iota_result(iota);
  loom_value_id_t upper_bound = loom_vector_mask_range_upper_bound(op);
  if (loom_type_element_type(scalar_type) == LOOM_SCALAR_TYPE_INDEX) {
    // Compare after index arithmetic, so its target-width wrap is preserved.
    // vector.cmpi owns fixed integers; index_cast preserves signed ordering.
    const loom_target_vector_packet_policy_t* policy =
        context->vector_packet_policy;
    const loom_scalar_type_t integer_type =
        policy != NULL && policy->index_bit_count != 0 &&
                policy->index_bit_count <= 32
            ? LOOM_SCALAR_TYPE_I32
            : LOOM_SCALAR_TYPE_I64;
    loom_type_t integer_vector_type = coordinate_type;
    integer_vector_type.header = loom_type_make_header(
        LOOM_TYPE_VECTOR, integer_type, loom_type_rank(coordinate_type),
        loom_type_flags(coordinate_type));
    loom_op_t* coordinate_cast = NULL;
    loom_op_t* bound_cast = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_index_cast_build(
        builder, coordinates, coordinate_type, integer_vector_type,
        op->location, &coordinate_cast));
    IREE_RETURN_IF_ERROR(loom_index_cast_build(
        builder, upper_bound, scalar_type, loom_type_scalar(integer_type),
        op->location, &bound_cast));
    coordinates = loom_vector_index_cast_result(coordinate_cast);
    upper_bound = loom_index_cast_result(bound_cast);
    coordinate_type = integer_vector_type;
  }
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      builder, upper_bound, coordinate_type, op->location, &upper));
  IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
      builder, LOOM_VECTOR_CMPI_PREDICATE_SLT, coordinates,
      loom_vector_splat_result(upper), coordinate_type, mask_type, op->location,
      &compare));
  const loom_value_id_t replacement = loom_vector_cmpi_result(compare);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kCoordinateLegalizerRules[] = {
    {.root_kind = LOOM_OP_VECTOR_IOTA,
     .legalize = loom_vector_legalize_iota_packets},
    {.root_kind = LOOM_OP_VECTOR_MASK_RANGE,
     .legalize = loom_vector_legalize_mask_coordinates},
};

static const loom_target_legalizer_provider_t kCoordinateLegalizerProvider = {
    .name = IREE_SVL("vector-coordinate"),
    .scope = LOOM_LOW_DESCRIPTOR_SCOPE_SHARED,
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
    .rules = kCoordinateLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kCoordinateLegalizerRules),
};

const loom_target_legalizer_provider_t*
loom_vector_coordinate_legalizer_provider(void) {
  return &kCoordinateLegalizerProvider;
}
