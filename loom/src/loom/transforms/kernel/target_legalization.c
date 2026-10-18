// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/target_legalization.h"

#include <stdint.h>
#include <string.h>

#include "loom/analysis/control_uniformity.h"
#include "loom/ops/combining.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/combining.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/combining.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/kernel/workgroup_collective_legalization.h"
#include "loom/transforms/vector/packet_plan.h"

#define LOOM_KERNEL_SCAN_MAX_SELECTOR_LEVELS 6u

// The subgroup payload family has one componentwise value/result pair.
// Addressing, clustering, and participation do not depend on vector components.
// Keep every packet at the authored convergence point and consume its captured
// SSA value rather than replaying the producer across the collective.
static iree_status_t loom_kernel_legalize_subgroup_packets(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->vector_packet_policy == NULL ||
      context->source_function_has_unsupported_vector_carrier) {
    return iree_ok_status();
  }
  const loom_value_id_t source = loom_op_const_operands(op)[0];
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  if (!loom_type_is_vector(source_type) ||
      loom_target_legalization_has_source_vector_carrier(context, source_type,
                                                         source)) {
    return iree_ok_status();
  }
  loom_vector_packet_shape_t shape;
  if (!loom_vector_packet_shape_from_type(
          context, context->vector_packet_policy, source_type, &shape) ||
      shape.chunk_count > UINT16_MAX) {
    return iree_ok_status();
  }

  // Native packets avoid the portable shuffle tree. Otherwise expand the tree
  // or normalize dynamic addressing once for the logical payload and packetize
  // the resulting shuffles. Splitting first would duplicate scalar lane work.
  const uint32_t tail_lane_count = shape.lane_count % shape.chunk_lane_count;
  const uint32_t lane_counts[] = {shape.chunk_lane_count, tail_lane_count};
  iree_status_t status = iree_ok_status();
  bool native_packets = true;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(lane_counts) && iree_status_is_ok(status); ++i) {
    if (lane_counts[i] == 0) {
      continue;
    }
    const loom_target_contract_vector_lane_projection_t projection = {
        .source_lane_count = shape.lane_count,
        .projected_lane_count = lane_counts[i],
    };
    loom_target_contract_query_result_t result =
        loom_target_contract_query_result_empty();
    status =
        loom_target_legalization_query_contract_with_vector_lane_projection(
            context, op, projection, &result);
    if (result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      native_packets = false;
      break;
    }
  }
  if (!iree_status_is_ok(status) || !native_packets) {
    return status;
  }

  loom_value_id_t* packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, shape.chunk_count, sizeof(*packets), (void**)&packets));
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t replacements[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < shape.chunk_count && iree_status_is_ok(status);
       ++i) {
    const int64_t offset = i * shape.chunk_lane_count;
    const uint32_t lane_count =
        iree_min(shape.chunk_lane_count, shape.lane_count - offset);
    const loom_type_t packet_type =
        loom_vector_packet_type(source_type, lane_count);
    loom_op_t* slice = NULL;
    status = loom_vector_slice_build(builder, source, NULL, 0, &offset, 1,
                                     packet_type, op->location, &slice);
    loom_op_t* packet = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_builder_allocate_op(
          builder, op->kind, op->operand_count, op->result_count, 0, 0,
          op->attribute_count, op->location, &packet);
    }
    if (iree_status_is_ok(status)) {
      packet->instance_flags = op->instance_flags;
      memcpy(loom_op_attrs(packet), loom_op_const_attrs(op),
             op->attribute_count * sizeof(loom_attribute_t));
      memcpy(loom_op_operands(packet), loom_op_const_operands(op),
             op->operand_count * sizeof(loom_value_id_t));
      loom_op_operands(packet)[0] = loom_vector_slice_result(slice);
      status = loom_builder_define_result(builder, packet_type,
                                          &loom_op_results(packet)[0]);
    }
    // Only shuffle has a second result. Its validity depends on the common
    // source lane and active participants, not on any packet's payload.
    if (iree_status_is_ok(status) && op->result_count == 2) {
      status = loom_builder_define_result(builder,
                                          loom_type_scalar(LOOM_SCALAR_TYPE_I1),
                                          &loom_op_results(packet)[1]);
      if (i == 0) {
        replacements[1] = loom_op_results(packet)[1];
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_builder_finalize_op(builder, packet);
      packets[i] = loom_op_results(packet)[0];
    }
  }
  loom_op_t* assembled = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_vector_concat_build(builder, 0, packets, shape.chunk_count,
                                      source_type, op->location, &assembled);
  }
  if (iree_status_is_ok(status)) {
    replacements[0] = loom_vector_concat_result(assembled);
    status = loom_rewriter_preserve_result_names_on_new_values(
        rewriter, op, replacements, op->result_count, checkpoint);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_replace_all_uses_and_erase(
        rewriter, op, replacements, op->result_count);
  }
  if (iree_status_is_ok(status)) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Subgroup shuffle reference legalization
//===----------------------------------------------------------------------===//

typedef struct loom_kernel_shuffle_state_t {
  loom_builder_t* builder;
  loom_location_id_t location;
  loom_type_t value_type;
  loom_type_t i1_type;
  loom_type_t i32_type;
  uint32_t subgroup_size;
  loom_value_id_t zero_i32;
  loom_value_id_t one_i32;
  loom_value_id_t subgroup_size_i32;
} loom_kernel_shuffle_state_t;

static bool loom_kernel_value_as_exact_i64(
    const loom_target_legalization_context_t* context, loom_value_id_t value,
    int64_t* out_value) {
  if (context->fact_table != NULL &&
      loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(context->fact_table, value),
          out_value)) {
    return true;
  }
  const loom_value_t* ir_value = loom_module_value(context->module, value);
  if (loom_value_is_block_arg(ir_value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(ir_value);
  if (defining_op == NULL || !loom_scalar_constant_isa(defining_op)) {
    return false;
  }
  const loom_attribute_t attribute = loom_scalar_constant_value(defining_op);
  if (attribute.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = loom_attr_as_i64(attribute);
  return true;
}

static bool loom_kernel_match_subgroup_shuffle(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  const loom_target_bundle_t* bundle =
      loom_target_legalization_context_bundle(context);
  if (bundle == NULL || bundle->snapshot == NULL ||
      bundle->snapshot->subgroup_size == 0 ||
      bundle->snapshot->subgroup_size > INT32_MAX) {
    return false;
  }

  // An indexed physical-subgroup shuffle is the normal form produced below.
  // Leaving it alone both lets native target contracts claim it and prevents a
  // reference-only pipeline from repeatedly normalizing its own output.
  int64_t width = 0;
  return loom_kernel_subgroup_shuffle_mode(op) !=
             LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_INDEX ||
         !loom_kernel_value_as_exact_i64(
             context, loom_kernel_subgroup_shuffle_width(op), &width) ||
         width != (int64_t)bundle->snapshot->subgroup_size;
}

static iree_status_t loom_kernel_shuffle_build_integer_constant(
    loom_kernel_shuffle_state_t* state, int64_t value,
    loom_value_id_t* out_value) {
  loom_op_t* constant = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_constant_build(state->builder, loom_attr_i64(value),
                                 state->i32_type, state->location, &constant));
  *out_value = loom_scalar_constant_result(constant);
  return iree_ok_status();
}

static iree_status_t loom_kernel_shuffle_build_compare(
    loom_kernel_shuffle_state_t* state, loom_scalar_cmpi_predicate_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_value) {
  loom_op_t* compare = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(state->builder, predicate, lhs,
                                              rhs, state->location, &compare));
  *out_value = loom_scalar_cmpi_result(compare);
  return iree_ok_status();
}

static iree_status_t loom_kernel_shuffle_build_and(
    loom_kernel_shuffle_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_value) {
  loom_op_t* and_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
      state->builder, lhs, rhs, state->i1_type, state->location, &and_op));
  *out_value = loom_scalar_andi_result(and_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_shuffle_build_select(
    loom_kernel_shuffle_state_t* state, loom_value_id_t condition,
    loom_value_id_t true_value, loom_value_id_t false_value,
    loom_value_id_t* out_value) {
  loom_op_t* select = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_select_build(state->builder, condition, true_value, false_value,
                            state->i32_type, state->location, &select));
  *out_value = loom_scf_select_result(select);
  return iree_ok_status();
}

static iree_status_t loom_kernel_shuffle_build_source_lane(
    loom_kernel_shuffle_state_t* state,
    loom_kernel_subgroup_shuffle_mode_t mode, loom_value_id_t lane,
    loom_value_id_t cluster_base, loom_value_id_t offset,
    loom_value_id_t* out_source_lane) {
  loom_op_t* source_op = NULL;
  switch (mode) {
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_XOR: {
      IREE_RETURN_IF_ERROR(loom_scalar_xori_build(state->builder, lane, offset,
                                                  state->i32_type,
                                                  state->location, &source_op));
      *out_source_lane = loom_scalar_xori_result(source_op);
      return iree_ok_status();
    }
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_UP: {
      IREE_RETURN_IF_ERROR(loom_scalar_subi_build(state->builder, 0, lane,
                                                  offset, state->i32_type,
                                                  state->location, &source_op));
      *out_source_lane = loom_scalar_subi_result(source_op);
      return iree_ok_status();
    }
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_DOWN: {
      IREE_RETURN_IF_ERROR(loom_scalar_addi_build(state->builder, 0, lane,
                                                  offset, state->i32_type,
                                                  state->location, &source_op));
      *out_source_lane = loom_scalar_addi_result(source_op);
      return iree_ok_status();
    }
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_INDEX: {
      IREE_RETURN_IF_ERROR(
          loom_scalar_addi_build(state->builder, 0, cluster_base, offset,
                                 state->i32_type, state->location, &source_op));
      *out_source_lane = loom_scalar_addi_result(source_op);
      return iree_ok_status();
    }
    case LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_COUNT_:
      IREE_ASSERT_UNREACHABLE("verified subgroup shuffle has invalid mode");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_kernel_target_legalize_subgroup_shuffle(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_target_bundle_t* bundle =
      loom_target_legalization_context_bundle(context);
  if (bundle == NULL || bundle->snapshot == NULL ||
      bundle->snapshot->subgroup_size == 0 ||
      bundle->snapshot->subgroup_size > INT32_MAX) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t value = loom_kernel_subgroup_shuffle_value(op);
  const loom_value_id_t offset = loom_kernel_subgroup_shuffle_offset(op);
  const loom_value_id_t width = loom_kernel_subgroup_shuffle_width(op);
  loom_kernel_shuffle_state_t state = {
      .builder = &rewriter->builder,
      .location = op->location,
      .value_type = loom_module_value_type(context->module, value),
      .i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1),
      .i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      .subgroup_size = bundle->snapshot->subgroup_size,
  };
  IREE_RETURN_IF_ERROR(
      loom_kernel_shuffle_build_integer_constant(&state, 0, &state.zero_i32));
  IREE_RETURN_IF_ERROR(
      loom_kernel_shuffle_build_integer_constant(&state, 1, &state.one_i32));
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_integer_constant(
      &state, state.subgroup_size, &state.subgroup_size_i32));

  // Widths outside [1, subgroup_size] and offsets outside [0, width) name no
  // logical source lane. Substitute benign operands for the arithmetic while
  // retaining the invalid predicate for the result contract.
  loom_value_id_t width_nonzero = LOOM_VALUE_ID_INVALID;
  loom_value_id_t width_in_subgroup = LOOM_VALUE_ID_INVALID;
  loom_value_id_t width_valid = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_kernel_shuffle_build_compare(&state, LOOM_SCALAR_CMPI_PREDICATE_NE,
                                        width, state.zero_i32, &width_nonzero));
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_compare(
      &state, LOOM_SCALAR_CMPI_PREDICATE_ULE, width, state.subgroup_size_i32,
      &width_in_subgroup));
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_and(
      &state, width_nonzero, width_in_subgroup, &width_valid));
  loom_value_id_t safe_width = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_select(
      &state, width_valid, width, state.one_i32, &safe_width));

  loom_value_id_t offset_in_width = LOOM_VALUE_ID_INVALID;
  loom_value_id_t offset_valid = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_kernel_shuffle_build_compare(&state, LOOM_SCALAR_CMPI_PREDICATE_ULT,
                                        offset, safe_width, &offset_in_width));
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_and(
      &state, width_valid, offset_in_width, &offset_valid));
  loom_value_id_t safe_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_select(
      &state, offset_valid, offset, state.zero_i32, &safe_offset));

  loom_op_t* lane_id_op = NULL;
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_lane_id_build(
      state.builder, index_type, state.location, &lane_id_op));
  loom_op_t* lane_i32_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      state.builder, loom_kernel_subgroup_lane_id_result(lane_id_op),
      index_type, state.i32_type, state.location, &lane_i32_op));
  const loom_value_id_t lane = loom_index_cast_result(lane_i32_op);

  loom_op_t* cluster_index_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_divui_build(state.builder, lane, safe_width,
                                               state.i32_type, state.location,
                                               &cluster_index_op));
  const loom_value_id_t cluster_index =
      loom_scalar_divui_result(cluster_index_op);
  loom_op_t* cluster_base_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_muli_build(state.builder, 0, cluster_index, safe_width,
                             state.i32_type, state.location, &cluster_base_op));
  const loom_value_id_t cluster_base = loom_scalar_muli_result(cluster_base_op);

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_source_lane(
      &state, loom_kernel_subgroup_shuffle_mode(op), lane, cluster_base,
      safe_offset, &source_lane));
  loom_op_t* source_cluster_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_divui_build(
      state.builder, source_lane, safe_width, state.i32_type, state.location,
      &source_cluster_op));
  loom_value_id_t same_cluster = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_compare(
      &state, LOOM_SCALAR_CMPI_PREDICATE_EQ,
      loom_scalar_divui_result(source_cluster_op), cluster_index,
      &same_cluster));
  loom_value_id_t source_in_subgroup = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_compare(
      &state, LOOM_SCALAR_CMPI_PREDICATE_ULT, source_lane,
      state.subgroup_size_i32, &source_in_subgroup));
  loom_value_id_t logical_valid = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_and(
      &state, offset_valid, same_cluster, &logical_valid));
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_and(
      &state, logical_valid, source_in_subgroup, &logical_valid));

  loom_value_id_t safe_source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_select(
      &state, logical_valid, source_lane, lane, &safe_source_lane));
  // Preserve a directly queryable [0, subgroup_size) fact for native target
  // selection instead of asking each target to rediscover the guarding select.
  loom_op_t* bounded_source_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_remui_build(
      state.builder, safe_source_lane, state.subgroup_size_i32, state.i32_type,
      state.location, &bounded_source_op));

  loom_op_t* shuffle_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_shuffle_build(
      state.builder, LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_INDEX, value,
      loom_scalar_remui_result(bounded_source_op), state.subgroup_size_i32,
      state.value_type, state.location, &shuffle_op));
  loom_value_id_t replacements[] = {
      loom_kernel_subgroup_shuffle_result(shuffle_op),
      LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_kernel_shuffle_build_and(
      &state, logical_valid, loom_kernel_subgroup_shuffle_valid(shuffle_op),
      &replacements[1]));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements), checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements)));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Subgroup scan reference legalization
//===----------------------------------------------------------------------===//

typedef struct loom_kernel_scan_cluster_t {
  // Number of logical lanes in one cluster.
  uint32_t size;
  // Physical lane distance between adjacent logical lanes.
  uint32_t stride;
} loom_kernel_scan_cluster_t;

typedef struct loom_kernel_scan_selector_level_t {
  // Width of the lower half examined by this level.
  loom_value_id_t half;
  // Mask selecting that lower half.
  loom_value_id_t lower_mask;
} loom_kernel_scan_selector_level_t;

typedef struct loom_kernel_scan_state_t {
  loom_builder_t* builder;
  loom_location_id_t location;
  // Authored collective value and result type.
  loom_type_t logical_type;
  // Type carried through shuffle and combine operations.
  loom_type_t value_type;
  // Native arithmetic packet shape, or zero when the complete carrier exists.
  loom_vector_packet_shape_t packet_shape;
  // Scoped scratch reused for the result packets of each arithmetic operation.
  loom_value_id_t* packet_values;
  loom_type_t i1_type;
  loom_type_t i32_type;
  loom_type_t mask_type;
  loom_combining_kind_t kind;
  uint32_t subgroup_size;
  uint32_t mask_bit_count;
  bool direct_physical_lanes;
  loom_value_id_t zero_mask;
  loom_value_id_t one_mask;
  loom_value_id_t subgroup_size_i32;
  loom_value_id_t cluster_width_i32;
  loom_value_id_t lane_i32;
  loom_value_id_t cluster_active_mask;
  loom_value_id_t rank;
  loom_value_id_t cluster_active_count;
  loom_kernel_scan_selector_level_t
      selector_levels[LOOM_KERNEL_SCAN_MAX_SELECTOR_LEVELS];
  uint32_t selector_level_count;
} loom_kernel_scan_state_t;

// Returns whether authored participation already follows physical subgroup
// lanes. A subgroup-uniform operation executes in every hardware-active lane;
// contiguous clusters can therefore use direct up/down shuffles without the
// active-rank selection network required by sparse or strided participation.
static bool loom_kernel_scan_has_direct_physical_lanes(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_kernel_scan_cluster_t* cluster) {
  if (cluster->stride != 1 || context->fact_table == NULL) {
    return false;
  }
  loom_control_uniformity_info_t control_uniformity;
  loom_control_uniformity_info_initialize(context->module, context->fact_table,
                                          context->arena, &control_uniformity);
  return loom_control_uniformity_prove_execution(
      &control_uniformity, op, LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP, NULL);
}

static bool loom_kernel_scan_cluster_from_attributes(
    const loom_target_legalization_context_t* context, bool has_cluster_size,
    int64_t cluster_size_attr, bool has_cluster_stride,
    int64_t cluster_stride_attr, loom_kernel_scan_cluster_t* out_cluster) {
  const loom_target_bundle_t* bundle =
      loom_target_legalization_context_bundle(context);
  if (bundle == NULL || bundle->snapshot == NULL ||
      bundle->snapshot->subgroup_size == 0 ||
      bundle->snapshot->subgroup_size > 64) {
    return false;
  }
  const uint32_t subgroup_size = bundle->snapshot->subgroup_size;
  const uint32_t cluster_size =
      has_cluster_size ? (uint32_t)cluster_size_attr : subgroup_size;
  const uint32_t cluster_stride =
      has_cluster_stride ? (uint32_t)cluster_stride_attr : 1u;
  const uint64_t cluster_span =
      (uint64_t)cluster_size * (uint64_t)cluster_stride;
  if (cluster_size == 0 || cluster_stride == 0 ||
      cluster_span > subgroup_size) {
    return false;
  }
  *out_cluster = (loom_kernel_scan_cluster_t){
      .size = cluster_size,
      .stride = cluster_stride,
  };
  return true;
}

static bool loom_kernel_scan_cluster_from_op(
    const loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_kernel_scan_cluster_t* out_cluster) {
  const bool has_cluster_size = loom_kernel_subgroup_scan_has_cluster_size(op);
  const bool has_cluster_stride =
      loom_kernel_subgroup_scan_has_cluster_stride(op);
  return loom_kernel_scan_cluster_from_attributes(
      context, has_cluster_size,
      has_cluster_size ? loom_kernel_subgroup_scan_cluster_size(op) : 0,
      has_cluster_stride,
      has_cluster_stride ? loom_kernel_subgroup_scan_cluster_stride(op) : 0,
      out_cluster);
}

static bool loom_kernel_reduce_cluster_from_op(
    const loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_kernel_scan_cluster_t* out_cluster) {
  const bool has_cluster_size =
      loom_kernel_subgroup_reduce_has_cluster_size(op);
  const bool has_cluster_stride =
      loom_kernel_subgroup_reduce_has_cluster_stride(op);
  return loom_kernel_scan_cluster_from_attributes(
      context, has_cluster_size,
      has_cluster_size ? loom_kernel_subgroup_reduce_cluster_size(op) : 0,
      has_cluster_stride,
      has_cluster_stride ? loom_kernel_subgroup_reduce_cluster_stride(op) : 0,
      out_cluster);
}

static bool loom_kernel_match_subgroup_scan(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  loom_kernel_scan_cluster_t cluster = {0};
  return loom_kernel_scan_cluster_from_op(context, op, &cluster);
}

static bool loom_kernel_match_subgroup_reduce(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  loom_kernel_scan_cluster_t cluster = {0};
  return loom_kernel_reduce_cluster_from_op(context, op, &cluster);
}

static iree_status_t loom_kernel_scan_build_integer_constant(
    loom_kernel_scan_state_t* state, loom_type_t type, int64_t value,
    loom_value_id_t* out_value) {
  loom_op_t* constant = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      state->builder, loom_attr_i64(value), type, state->location, &constant));
  *out_value = loom_scalar_constant_result(constant);
  return iree_ok_status();
}

static iree_status_t loom_kernel_scan_build_transport_value(
    loom_kernel_scan_state_t* state, loom_value_id_t logical_value,
    loom_value_id_t* out_value) {
  if (loom_type_equal(state->logical_type, state->value_type)) {
    *out_value = logical_value;
    return iree_ok_status();
  }
  loom_op_t* extend_op = NULL;
  if (loom_type_is_scalar(state->logical_type)) {
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
        state->builder, logical_value, state->logical_type, state->value_type,
        state->location, &extend_op));
    *out_value = loom_scalar_extui_result(extend_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_extui_build(
        state->builder, logical_value, state->logical_type, state->value_type,
        state->location, &extend_op));
    *out_value = loom_vector_extui_result(extend_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_scan_build_logical_value(
    loom_kernel_scan_state_t* state, loom_value_id_t value,
    loom_value_id_t* out_logical_value) {
  if (loom_type_equal(state->logical_type, state->value_type)) {
    *out_logical_value = value;
    return iree_ok_status();
  }
  loom_op_t* truncate_op = NULL;
  if (loom_type_is_scalar(state->logical_type)) {
    IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
        state->builder, value, state->value_type, state->logical_type,
        state->location, &truncate_op));
    *out_logical_value = loom_scalar_trunci_result(truncate_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_trunci_build(
        state->builder, value, state->value_type, state->logical_type,
        state->location, &truncate_op));
    *out_logical_value = loom_vector_trunci_result(truncate_op);
  }
  return iree_ok_status();
}

// Construct arithmetic in native carriers before scalar consumers can expand
// an unsupported aggregate's producer graph. Participation and lane selection
// remain shared by all packets, and slices consume already-captured SSA values.
static iree_status_t loom_kernel_scan_build_packet_operation(
    loom_kernel_scan_state_t* state, loom_value_id_t condition,
    loom_value_id_t left, loom_value_id_t right, loom_value_id_t* out_value) {
  const loom_vector_packet_shape_t* shape = &state->packet_shape;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < shape->chunk_count && iree_status_is_ok(status);
       ++i) {
    const int64_t offset = i * shape->chunk_lane_count;
    const uint32_t lane_count =
        iree_min(shape->chunk_lane_count, shape->lane_count - offset);
    const loom_type_t packet_type =
        loom_vector_packet_type(state->value_type, lane_count);
    loom_op_t* left_slice = NULL;
    status = loom_vector_slice_build(state->builder, left, NULL, 0, &offset, 1,
                                     packet_type, state->location, &left_slice);
    loom_op_t* right_slice = NULL;
    if (iree_status_is_ok(status)) {
      status =
          loom_vector_slice_build(state->builder, right, NULL, 0, &offset, 1,
                                  packet_type, state->location, &right_slice);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (condition == LOOM_VALUE_ID_INVALID) {
      status = loom_vector_combining_build(
          state->builder, state->kind, /*fastmath_flags=*/0,
          loom_vector_slice_result(left_slice),
          loom_vector_slice_result(right_slice), packet_type, state->location,
          &state->packet_values[i]);
    } else {
      loom_op_t* select = NULL;
      status = loom_scf_select_build(state->builder, condition,
                                     loom_vector_slice_result(left_slice),
                                     loom_vector_slice_result(right_slice),
                                     packet_type, state->location, &select);
      if (iree_status_is_ok(status)) {
        state->packet_values[i] = loom_scf_select_result(select);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* assembled = NULL;
    status = loom_vector_concat_build(state->builder, 0, state->packet_values,
                                      shape->chunk_count, state->value_type,
                                      state->location, &assembled);
    if (iree_status_is_ok(status)) {
      *out_value = loom_vector_concat_result(assembled);
    }
  }
  return status;
}

static iree_status_t loom_kernel_scan_build_select(
    loom_kernel_scan_state_t* state, loom_value_id_t condition,
    loom_value_id_t true_value, loom_value_id_t false_value, loom_type_t type,
    loom_value_id_t* out_value) {
  if (state->packet_shape.chunk_count != 0 && loom_type_is_vector(type)) {
    return loom_kernel_scan_build_packet_operation(state, condition, true_value,
                                                   false_value, out_value);
  }
  loom_op_t* select = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(state->builder, condition,
                                             true_value, false_value, type,
                                             state->location, &select));
  *out_value = loom_scf_select_result(select);
  return iree_ok_status();
}

static iree_status_t loom_kernel_scan_build_and(loom_kernel_scan_state_t* state,
                                                loom_value_id_t lhs,
                                                loom_value_id_t rhs,
                                                loom_type_t type,
                                                loom_value_id_t* out_value) {
  loom_op_t* and_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(state->builder, lhs, rhs, type,
                                              state->location, &and_op));
  *out_value = loom_scalar_andi_result(and_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_scan_build_cluster_state(
    loom_kernel_scan_state_t* state,
    const loom_kernel_scan_cluster_t* cluster) {
  loom_op_t* lane_id_op = NULL;
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_lane_id_build(
      state->builder, index_type, state->location, &lane_id_op));
  loom_op_t* lane_i32_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      state->builder, loom_kernel_subgroup_lane_id_result(lane_id_op),
      index_type, state->i32_type, state->location, &lane_i32_op));
  const loom_value_id_t lane_i32 = loom_index_cast_result(lane_i32_op);
  state->lane_i32 = lane_i32;

  loom_value_id_t lane_mask = lane_i32;
  if (state->mask_bit_count == 64) {
    loom_op_t* lane_mask_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
        state->builder, lane_i32, state->i32_type, state->mask_type,
        state->location, &lane_mask_op));
    lane_mask = loom_scalar_extui_result(lane_mask_op);
  }

  uint64_t cluster_pattern = 0;
  for (uint32_t i = 0; i < cluster->size; ++i) {
    cluster_pattern |= UINT64_C(1) << (i * cluster->stride);
  }
  const int64_t signed_pattern =
      state->mask_bit_count == 32 ? (int64_t)(int32_t)(uint32_t)cluster_pattern
                                  : (int64_t)cluster_pattern;
  loom_value_id_t cluster_pattern_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
      state, state->mask_type, signed_pattern, &cluster_pattern_value));

  loom_value_id_t cluster_base_i32 = LOOM_VALUE_ID_INVALID;
  const uint32_t cluster_span = cluster->size * cluster->stride;
  if (cluster_span == state->subgroup_size && cluster->stride == 1) {
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
        state, state->i32_type, 0, &cluster_base_i32));
  } else {
    loom_value_id_t span_i32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
        state, state->i32_type, cluster_span, &span_i32));
    loom_op_t* quotient_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_divui_build(
        state->builder, lane_i32, span_i32, state->i32_type, state->location,
        &quotient_op));
    loom_op_t* cluster_base_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_muli_build(
        state->builder, 0, loom_scalar_divui_result(quotient_op), span_i32,
        state->i32_type, state->location, &cluster_base_op));
    cluster_base_i32 = loom_scalar_muli_result(cluster_base_op);
    if (cluster->stride != 1) {
      loom_value_id_t stride_i32 = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
          state, state->i32_type, cluster->stride, &stride_i32));
      loom_op_t* remainder_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_remui_build(
          state->builder, lane_i32, stride_i32, state->i32_type,
          state->location, &remainder_op));
      loom_op_t* adjusted_base_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
          state->builder, 0, cluster_base_i32,
          loom_scalar_remui_result(remainder_op), state->i32_type,
          state->location, &adjusted_base_op));
      cluster_base_i32 = loom_scalar_addi_result(adjusted_base_op);
    }
  }

  loom_value_id_t cluster_base_mask = cluster_base_i32;
  if (state->mask_bit_count == 64) {
    loom_op_t* cluster_base_mask_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
        state->builder, cluster_base_i32, state->i32_type, state->mask_type,
        state->location, &cluster_base_mask_op));
    cluster_base_mask = loom_scalar_extui_result(cluster_base_mask_op);
  }
  loom_op_t* cluster_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
      state->builder, 0, cluster_pattern_value, cluster_base_mask,
      state->mask_type, state->location, &cluster_mask_op));

  loom_op_t* active_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_active_mask_build(
      state->builder, state->mask_type, state->location, &active_mask_op));
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_and(
      state, loom_kernel_subgroup_active_mask_mask(active_mask_op),
      loom_scalar_shli_result(cluster_mask_op), state->mask_type,
      &state->cluster_active_mask));

  loom_op_t* lane_bit_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_shli_build(state->builder, 0, state->one_mask, lane_mask,
                             state->mask_type, state->location, &lane_bit_op));
  loom_op_t* lower_lane_bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
      state->builder, 0, loom_scalar_shli_result(lane_bit_op), state->one_mask,
      state->mask_type, state->location, &lower_lane_bits_op));
  loom_value_id_t lower_active_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_kernel_scan_build_and(state, state->cluster_active_mask,
                                 loom_scalar_subi_result(lower_lane_bits_op),
                                 state->mask_type, &lower_active_mask));

  loom_op_t* rank_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_ctpopi_build(state->builder, lower_active_mask,
                               state->mask_type, state->location, &rank_op));
  state->rank = loom_scalar_ctpopi_result(rank_op);
  loom_op_t* count_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_ctpopi_build(state->builder, state->cluster_active_mask,
                               state->mask_type, state->location, &count_op));
  state->cluster_active_count = loom_scalar_ctpopi_result(count_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_scan_build_direct_state(
    loom_kernel_scan_state_t* state, const loom_kernel_scan_cluster_t* cluster,
    bool need_active_count) {
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
      state, state->i32_type, cluster->size, &state->cluster_width_i32));
  if (!need_active_count) {
    return iree_ok_status();
  }

  loom_op_t* active_mask_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_active_mask_build(
      state->builder, state->mask_type, state->location, &active_mask_op));
  state->cluster_active_mask =
      loom_kernel_subgroup_active_mask_mask(active_mask_op);
  if (cluster->size != state->subgroup_size) {
    loom_op_t* lane_id_op = NULL;
    const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    IREE_RETURN_IF_ERROR(loom_kernel_subgroup_lane_id_build(
        state->builder, index_type, state->location, &lane_id_op));
    loom_op_t* lane_i32_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cast_build(
        state->builder, loom_kernel_subgroup_lane_id_result(lane_id_op),
        index_type, state->i32_type, state->location, &lane_i32_op));
    state->lane_i32 = loom_index_cast_result(lane_i32_op);

    loom_op_t* cluster_index_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_divui_build(
        state->builder, state->lane_i32, state->cluster_width_i32,
        state->i32_type, state->location, &cluster_index_op));
    loom_op_t* cluster_base_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_muli_build(
        state->builder, 0, loom_scalar_divui_result(cluster_index_op),
        state->cluster_width_i32, state->i32_type, state->location,
        &cluster_base_op));
    loom_value_id_t cluster_base = loom_scalar_muli_result(cluster_base_op);
    if (state->mask_bit_count == 64) {
      loom_op_t* extend_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_extui_build(
          state->builder, cluster_base, state->i32_type, state->mask_type,
          state->location, &extend_op));
      cluster_base = loom_scalar_extui_result(extend_op);
    }

    const uint64_t cluster_pattern =
        cluster->size == 64 ? UINT64_MAX : (UINT64_C(1) << cluster->size) - 1u;
    const int64_t signed_pattern =
        state->mask_bit_count == 32
            ? (int64_t)(int32_t)(uint32_t)cluster_pattern
            : (int64_t)cluster_pattern;
    loom_value_id_t cluster_pattern_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
        state, state->mask_type, signed_pattern, &cluster_pattern_value));
    loom_op_t* cluster_mask_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
        state->builder, 0, cluster_pattern_value, cluster_base,
        state->mask_type, state->location, &cluster_mask_op));
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_and(
        state, state->cluster_active_mask,
        loom_scalar_shli_result(cluster_mask_op), state->mask_type,
        &state->cluster_active_mask));
  }

  loom_op_t* count_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_ctpopi_build(state->builder, state->cluster_active_mask,
                               state->mask_type, state->location, &count_op));
  state->cluster_active_count = loom_scalar_ctpopi_result(count_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_scan_build_selector_constants(
    loom_kernel_scan_state_t* state) {
  uint32_t first_half = state->subgroup_size > 1 ? 1u : 0u;
  while ((first_half << 1) < state->subgroup_size) {
    first_half <<= 1;
  }
  for (uint32_t half = first_half; half != 0; half >>= 1) {
    loom_kernel_scan_selector_level_t* level =
        &state->selector_levels[state->selector_level_count++];
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
        state, state->mask_type, half, &level->half));
    const uint64_t lower_mask = (UINT64_C(1) << half) - 1u;
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
        state, state->mask_type, (int64_t)lower_mask, &level->lower_mask));
  }
  return iree_ok_status();
}

// Maps one compact active-lane rank to its physical subgroup lane. The fixed
// depth selection network costs ceil(log2(subgroup size)) steps and avoids a
// data-dependent loop over sparse participation masks.
static iree_status_t loom_kernel_scan_build_physical_lane(
    loom_kernel_scan_state_t* state, loom_value_id_t desired_rank,
    loom_value_id_t* out_lane_i32) {
  loom_value_id_t remaining = state->cluster_active_mask;
  loom_value_id_t rank = desired_rank;
  loom_value_id_t base = state->zero_mask;
  for (uint32_t i = 0; i < state->selector_level_count; ++i) {
    const loom_kernel_scan_selector_level_t* level = &state->selector_levels[i];
    loom_value_id_t lower = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_and(
        state, remaining, level->lower_mask, state->mask_type, &lower));
    loom_op_t* lower_count_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_ctpopi_build(state->builder, lower, state->mask_type,
                                 state->location, &lower_count_op));
    const loom_value_id_t lower_count =
        loom_scalar_ctpopi_result(lower_count_op);
    loom_op_t* choose_upper_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
        state->builder, LOOM_SCALAR_CMPI_PREDICATE_UGE, rank, lower_count,
        state->location, &choose_upper_op));
    const loom_value_id_t choose_upper =
        loom_scalar_cmpi_result(choose_upper_op);

    loom_op_t* upper_rank_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
        state->builder, 0, rank, lower_count, state->mask_type, state->location,
        &upper_rank_op));
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_select(
        state, choose_upper, loom_scalar_subi_result(upper_rank_op), rank,
        state->mask_type, &rank));

    loom_op_t* upper_remaining_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(
        state->builder, remaining, level->half, state->mask_type,
        state->location, &upper_remaining_op));
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_select(
        state, choose_upper, loom_scalar_shrui_result(upper_remaining_op),
        remaining, state->mask_type, &remaining));

    loom_op_t* upper_base_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
        state->builder, 0, base, level->half, state->mask_type, state->location,
        &upper_base_op));
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_select(
        state, choose_upper, loom_scalar_addi_result(upper_base_op), base,
        state->mask_type, &base));
  }

  if (state->mask_bit_count == 32) {
    *out_lane_i32 = base;
    return iree_ok_status();
  }
  loom_op_t* lane_i32_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_trunci_build(state->builder, base, state->mask_type,
                               state->i32_type, state->location, &lane_i32_op));
  *out_lane_i32 = loom_scalar_trunci_result(lane_i32_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_scan_build_source_rank(
    loom_kernel_scan_state_t* state,
    loom_kernel_subgroup_scan_direction_t direction, uint32_t distance,
    loom_value_id_t* out_rank, loom_value_id_t* out_valid) {
  loom_value_id_t distance_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
      state, state->mask_type, distance, &distance_value));
  loom_op_t* rank_op = NULL;
  loom_op_t* valid_op = NULL;
  if (direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD) {
    IREE_RETURN_IF_ERROR(
        loom_scalar_subi_build(state->builder, 0, state->rank, distance_value,
                               state->mask_type, state->location, &rank_op));
    IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
        state->builder, LOOM_SCALAR_CMPI_PREDICATE_UGE, state->rank,
        distance_value, state->location, &valid_op));
    *out_rank = loom_scalar_subi_result(rank_op);
  } else {
    IREE_RETURN_IF_ERROR(
        loom_scalar_addi_build(state->builder, 0, state->rank, distance_value,
                               state->mask_type, state->location, &rank_op));
    IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
        state->builder, LOOM_SCALAR_CMPI_PREDICATE_ULT,
        loom_scalar_addi_result(rank_op), state->cluster_active_count,
        state->location, &valid_op));
    *out_rank = loom_scalar_addi_result(rank_op);
  }
  *out_valid = loom_scalar_cmpi_result(valid_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_scan_build_shuffle(
    loom_kernel_scan_state_t* state, loom_value_id_t value,
    loom_value_id_t source_rank, loom_value_id_t source_rank_valid,
    loom_value_id_t* out_value, loom_value_id_t* out_valid) {
  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_kernel_scan_build_physical_lane(state, source_rank, &source_lane));
  loom_op_t* shuffle_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_shuffle_build(
      state->builder, LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_INDEX, value,
      source_lane, state->subgroup_size_i32, state->value_type, state->location,
      &shuffle_op));
  *out_value = loom_kernel_subgroup_shuffle_result(shuffle_op);
  return loom_kernel_scan_build_and(
      state, source_rank_valid, loom_kernel_subgroup_shuffle_valid(shuffle_op),
      state->i1_type, out_valid);
}

static iree_status_t loom_kernel_scan_build_direct_shuffle(
    loom_kernel_scan_state_t* state, loom_value_id_t value,
    loom_kernel_subgroup_scan_direction_t direction, uint32_t distance,
    loom_value_id_t* out_value, loom_value_id_t* out_valid) {
  loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
      state, state->i32_type, distance, &offset));
  loom_op_t* shuffle_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_shuffle_build(
      state->builder,
      direction == LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD
          ? LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_UP
          : LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_DOWN,
      value, offset, state->cluster_width_i32, state->value_type,
      state->location, &shuffle_op));
  *out_value = loom_kernel_subgroup_shuffle_result(shuffle_op);
  *out_valid = loom_kernel_subgroup_shuffle_valid(shuffle_op);
  return iree_ok_status();
}

static iree_status_t loom_kernel_scan_build_combine(
    loom_kernel_scan_state_t* state, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t* out_result) {
  if (loom_type_is_scalar(state->value_type)) {
    return loom_scalar_combining_build(
        state->builder, state->kind, /*fastmath_flags=*/0, lhs, rhs,
        state->value_type, state->location, out_result);
  }
  if (state->packet_shape.chunk_count != 0) {
    return loom_kernel_scan_build_packet_operation(state, LOOM_VALUE_ID_INVALID,
                                                   lhs, rhs, out_result);
  }
  return loom_vector_combining_build(
      state->builder, state->kind, /*fastmath_flags=*/0, lhs, rhs,
      state->value_type, state->location, out_result);
}

static iree_status_t loom_kernel_scan_build_seed(
    loom_kernel_scan_state_t* state, loom_value_id_t* out_seed) {
  const loom_scalar_type_t element_type =
      loom_type_element_type(state->value_type);
  loom_attribute_t seed_attr = loom_attr_absent();
  if (loom_scalar_type_is_integer(element_type)) {
    int64_t seed = 0;
    if (!loom_combining_kind_integer_identity(state->kind, element_type,
                                              &seed)) {
      IREE_ASSERT_UNREACHABLE("verified integer scan has no seed");
      IREE_BUILTIN_UNREACHABLE();
    }
    seed_attr = loom_attr_i64(seed);
  } else {
    double seed = 0.0;
    if (!loom_combining_kind_float_scan_seed(state->kind, &seed)) {
      IREE_ASSERT_UNREACHABLE("verified floating-point scan has no seed");
      IREE_BUILTIN_UNREACHABLE();
    }
    seed_attr = loom_attr_f64(seed);
  }
  const loom_type_t element_value_type = loom_type_scalar(element_type);
  loom_op_t* scalar_seed_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_constant_build(state->builder, seed_attr, element_value_type,
                                 state->location, &scalar_seed_op));
  if (loom_type_is_scalar(state->value_type)) {
    *out_seed = loom_scalar_constant_result(scalar_seed_op);
    return iree_ok_status();
  }
  loom_op_t* vector_seed_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      state->builder, loom_scalar_constant_result(scalar_seed_op),
      state->value_type, state->location, &vector_seed_op));
  *out_seed = loom_vector_splat_result(vector_seed_op);
  return iree_ok_status();
}

static bool loom_kernel_scan_math_is_supported(loom_combining_kind_t kind,
                                               uint8_t math_flags) {
  // Ordinary arithmetic support does not establish source-format
  // preservation. Extrema retain their fixed semantics independently of the
  // arithmetic mode.
  return math_flags == 0 ||
         (kind != LOOM_COMBINING_KIND_ADDF && kind != LOOM_COMBINING_KIND_MULF);
}

static iree_status_t loom_kernel_scan_state_initialize(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_value_id_t source_value, loom_combining_kind_t kind,
    const loom_kernel_scan_cluster_t* cluster, bool need_active_count,
    loom_kernel_scan_state_t* out_state) {
  const loom_target_bundle_t* bundle =
      loom_target_legalization_context_bundle(context);
  const uint32_t subgroup_size = bundle->snapshot->subgroup_size;
  const loom_type_t logical_type =
      loom_module_value_type(context->module, source_value);
  loom_type_t value_type = logical_type;
  loom_combining_kind_t value_kind = kind;
  if (loom_type_element_type(logical_type) == LOOM_SCALAR_TYPE_I1) {
    const bool has_i1_equivalent =
        loom_combining_kind_i1_equivalent(kind, &value_kind);
    IREE_ASSERT(has_i1_equivalent);
    value_type.header = loom_type_make_header(
        loom_type_kind(logical_type), LOOM_SCALAR_TYPE_I32,
        loom_type_rank(logical_type), loom_type_flags(logical_type));
  }
  *out_state = (loom_kernel_scan_state_t){
      .builder = &context->rewriter->builder,
      .location = op->location,
      .logical_type = logical_type,
      .value_type = value_type,
      .i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1),
      .i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      .mask_type = loom_type_scalar(subgroup_size <= 32 ? LOOM_SCALAR_TYPE_I32
                                                        : LOOM_SCALAR_TYPE_I64),
      .kind = value_kind,
      .subgroup_size = subgroup_size,
      .mask_bit_count = subgroup_size <= 32 ? 32u : 64u,
      .direct_physical_lanes =
          loom_kernel_scan_has_direct_physical_lanes(context, op, cluster),
  };
  loom_vector_packet_shape_t packet_shape;
  if (context->vector_packet_policy != NULL &&
      loom_vector_packet_shape_from_type(context, context->vector_packet_policy,
                                         value_type, &packet_shape) &&
      packet_shape.chunk_count <= UINT16_MAX) {
    out_state->packet_shape = packet_shape;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, packet_shape.chunk_count,
        sizeof(*out_state->packet_values), (void**)&out_state->packet_values));
  }
  if (out_state->direct_physical_lanes) {
    if (need_active_count) {
      IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
          out_state, out_state->mask_type, 1, &out_state->one_mask));
    }
    return loom_kernel_scan_build_direct_state(out_state, cluster,
                                               need_active_count);
  }
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
      out_state, out_state->mask_type, 0, &out_state->zero_mask));
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
      out_state, out_state->mask_type, 1, &out_state->one_mask));
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
      out_state, out_state->i32_type, subgroup_size,
      &out_state->subgroup_size_i32));
  IREE_RETURN_IF_ERROR(
      loom_kernel_scan_build_cluster_state(out_state, cluster));
  return loom_kernel_scan_build_selector_constants(out_state);
}

static iree_status_t loom_kernel_scan_build_inclusive(
    loom_kernel_scan_state_t* state, loom_value_id_t source_value,
    const loom_kernel_scan_cluster_t* cluster,
    loom_kernel_subgroup_scan_direction_t direction,
    loom_value_id_t* out_result) {
  loom_value_id_t accumulator = source_value;
  for (uint32_t distance = 1; distance < cluster->size; distance <<= 1) {
    loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
    loom_value_id_t peer_valid = LOOM_VALUE_ID_INVALID;
    if (state->direct_physical_lanes) {
      IREE_RETURN_IF_ERROR(loom_kernel_scan_build_direct_shuffle(
          state, accumulator, direction, distance, &peer, &peer_valid));
    } else {
      loom_value_id_t source_rank = LOOM_VALUE_ID_INVALID;
      loom_value_id_t source_rank_valid = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_kernel_scan_build_source_rank(
          state, direction, distance, &source_rank, &source_rank_valid));
      IREE_RETURN_IF_ERROR(loom_kernel_scan_build_shuffle(
          state, accumulator, source_rank, source_rank_valid, &peer,
          &peer_valid));
    }
    loom_value_id_t combined = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_kernel_scan_build_combine(state, peer, accumulator, &combined));
    IREE_RETURN_IF_ERROR(
        loom_kernel_scan_build_select(state, peer_valid, combined, accumulator,
                                      state->value_type, &accumulator));
  }
  *out_result = accumulator;
  return iree_ok_status();
}

iree_status_t loom_kernel_target_legalize_subgroup_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL &&
      context->policy != LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
    return iree_ok_status();
  }
  const loom_combining_kind_t kind = loom_kernel_subgroup_reduce_kind(op);
  if (!loom_kernel_scan_math_is_supported(
          kind, loom_kernel_subgroup_reduce_math(op))) {
    return iree_ok_status();
  }
  loom_kernel_scan_cluster_t cluster = {0};
  if (!loom_kernel_reduce_cluster_from_op(context, op, &cluster)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t source_value = loom_kernel_subgroup_reduce_value(op);
  loom_kernel_scan_state_t state;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_state_initialize(
      context, op, source_value, kind, &cluster,
      /*need_active_count=*/true, &state));
  loom_value_id_t transport_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_transport_value(
      &state, source_value, &transport_source));
  loom_value_id_t inclusive = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_inclusive(
      &state, transport_source, &cluster,
      LOOM_KERNEL_SUBGROUP_SCAN_DIRECTION_FORWARD, &inclusive));

  loom_value_id_t reduction = LOOM_VALUE_ID_INVALID;
  if (state.direct_physical_lanes) {
    loom_op_t* last_offset_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
        state.builder, 0, state.cluster_active_count, state.one_mask,
        state.mask_type, state.location, &last_offset_op));
    loom_value_id_t last_offset = loom_scalar_subi_result(last_offset_op);
    if (state.mask_bit_count == 64) {
      loom_op_t* truncate_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
          state.builder, last_offset, state.mask_type, state.i32_type,
          state.location, &truncate_op));
      last_offset = loom_scalar_trunci_result(truncate_op);
    }
    loom_op_t* shuffle_op = NULL;
    IREE_RETURN_IF_ERROR(loom_kernel_subgroup_shuffle_build(
        state.builder, LOOM_KERNEL_SUBGROUP_SHUFFLE_MODE_INDEX, inclusive,
        last_offset, state.cluster_width_i32, state.value_type, state.location,
        &shuffle_op));
    reduction = loom_kernel_subgroup_shuffle_result(shuffle_op);
  } else {
    loom_op_t* last_rank_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
        state.builder, 0, state.cluster_active_count, state.one_mask,
        state.mask_type, state.location, &last_rank_op));
    loom_op_t* valid_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_cmpi_build(state.builder, LOOM_SCALAR_CMPI_PREDICATE_UGT,
                               state.cluster_active_count, state.zero_mask,
                               state.location, &valid_op));
    loom_value_id_t reduction_valid = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_shuffle(
        &state, inclusive, loom_scalar_subi_result(last_rank_op),
        loom_scalar_cmpi_result(valid_op), &reduction, &reduction_valid));
  }

  loom_value_id_t logical_reduction = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_logical_value(
      &state, reduction, &logical_reduction));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &logical_reduction, 1, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, &logical_reduction, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

iree_status_t loom_kernel_target_legalize_subgroup_scan(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  // Normalize and share the source values before expanding one scan into its
  // shuffle/select network. Native target contracts are queried first and do
  // not enter this portable expansion.
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL &&
      context->policy != LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY) {
    out_result->action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
    return iree_ok_status();
  }
  const loom_combining_kind_t kind = loom_kernel_subgroup_scan_kind(op);
  if (!loom_kernel_scan_math_is_supported(kind,
                                          loom_kernel_subgroup_scan_math(op))) {
    return iree_ok_status();
  }
  loom_kernel_scan_cluster_t cluster = {0};
  const bool has_cluster =
      loom_kernel_scan_cluster_from_op(context, op, &cluster);
  if (!has_cluster) {
    return iree_ok_status();
  }
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t source_value = loom_kernel_subgroup_scan_value(op);
  loom_kernel_scan_state_t state;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_state_initialize(
      context, op, source_value, kind, &cluster,
      /*need_active_count=*/false, &state));
  loom_value_id_t transport_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_transport_value(
      &state, source_value, &transport_source));

  const loom_kernel_subgroup_scan_direction_t direction =
      loom_kernel_subgroup_scan_direction(op);
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_inclusive(
      &state, transport_source, &cluster, direction, &accumulator));

  if (loom_kernel_subgroup_scan_mode(op) ==
      LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE) {
    loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
    loom_value_id_t shifted_valid = LOOM_VALUE_ID_INVALID;
    if (state.direct_physical_lanes) {
      IREE_RETURN_IF_ERROR(loom_kernel_scan_build_direct_shuffle(
          &state, accumulator, direction, 1, &shifted, &shifted_valid));
    } else {
      loom_value_id_t source_rank = LOOM_VALUE_ID_INVALID;
      loom_value_id_t source_rank_valid = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_kernel_scan_build_source_rank(
          &state, direction, 1, &source_rank, &source_rank_valid));
      IREE_RETURN_IF_ERROR(loom_kernel_scan_build_shuffle(
          &state, accumulator, source_rank, source_rank_valid, &shifted,
          &shifted_valid));
    }
    loom_value_id_t seed = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_seed(&state, &seed));
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_select(
        &state, shifted_valid, shifted, seed, state.value_type, &accumulator));
  }

  loom_value_id_t logical_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_logical_value(&state, accumulator,
                                                            &logical_result));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &logical_result, 1, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, &logical_result, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kKernelLegalizerRules[] = {
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_REDUCE,
        .legalize = loom_kernel_legalize_subgroup_packets,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_SCAN,
        .legalize = loom_kernel_legalize_subgroup_packets,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_SHUFFLE,
        .legalize = loom_kernel_legalize_subgroup_packets,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_SHUFFLE,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .match = loom_kernel_match_subgroup_shuffle,
        .legalize = loom_kernel_target_legalize_subgroup_shuffle,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_REDUCE,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .match = loom_kernel_match_subgroup_reduce,
        .legalize = loom_kernel_target_legalize_subgroup_reduce,
    },
    {
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_SCAN,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .match = loom_kernel_match_subgroup_scan,
        .legalize = loom_kernel_target_legalize_subgroup_scan,
    },
    {
        .root_kind = LOOM_OP_KERNEL_WORKGROUP_REDUCE,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_kernel_target_legalize_workgroup_reduce,
    },
    {
        .root_kind = LOOM_OP_KERNEL_WORKGROUP_SCAN,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_kernel_target_legalize_workgroup_scan,
    },
};

static const loom_target_legalizer_provider_t kKernelLegalizerProvider = {
    .name = IREE_SVL("kernel"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .rules = kKernelLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kKernelLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_kernel_target_legalizer_provider(
    void) {
  return &kKernelLegalizerProvider;
}

#undef LOOM_KERNEL_SCAN_MAX_SELECTOR_LEVELS
