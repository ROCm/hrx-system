// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/target_legalization.h"

#include <stdint.h>

#include "loom/ops/combining.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/combining.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/combining.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"

#define LOOM_KERNEL_SCAN_MAX_SELECTOR_LEVELS 6u

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
  loom_type_t result_type;
  loom_type_t i1_type;
  loom_type_t i32_type;
  loom_type_t mask_type;
  loom_combining_kind_t kind;
  uint32_t subgroup_size;
  uint32_t mask_bit_count;
  loom_value_id_t zero_mask;
  loom_value_id_t one_mask;
  loom_value_id_t subgroup_size_i32;
  loom_value_id_t cluster_active_mask;
  loom_value_id_t rank;
  loom_value_id_t cluster_active_count;
  loom_kernel_scan_selector_level_t
      selector_levels[LOOM_KERNEL_SCAN_MAX_SELECTOR_LEVELS];
  uint32_t selector_level_count;
} loom_kernel_scan_state_t;

static bool loom_kernel_scan_cluster_from_op(
    const loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_kernel_scan_cluster_t* out_cluster) {
  const loom_target_bundle_t* bundle =
      loom_target_legalization_context_bundle(context);
  if (bundle == NULL || bundle->snapshot == NULL ||
      bundle->snapshot->subgroup_size == 0 ||
      bundle->snapshot->subgroup_size > 64) {
    return false;
  }
  const uint32_t subgroup_size = bundle->snapshot->subgroup_size;
  const uint32_t cluster_size =
      loom_kernel_subgroup_scan_has_cluster_size(op)
          ? (uint32_t)loom_kernel_subgroup_scan_cluster_size(op)
          : subgroup_size;
  const uint32_t cluster_stride =
      loom_kernel_subgroup_scan_has_cluster_stride(op)
          ? (uint32_t)loom_kernel_subgroup_scan_cluster_stride(op)
          : 1u;
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

static bool loom_kernel_match_subgroup_scan(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op) {
  (void)entry;
  loom_kernel_scan_cluster_t cluster = {0};
  return loom_kernel_scan_cluster_from_op(context, op, &cluster);
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

static iree_status_t loom_kernel_scan_build_select(
    loom_kernel_scan_state_t* state, loom_value_id_t condition,
    loom_value_id_t true_value, loom_value_id_t false_value, loom_type_t type,
    loom_value_id_t* out_value) {
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

static iree_status_t loom_kernel_scan_build_selector_constants(
    loom_kernel_scan_state_t* state) {
  for (uint32_t half = state->mask_bit_count / 2; half != 0; half >>= 1) {
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
// depth selection network costs log2(mask width) steps for every target and
// avoids a data-dependent loop over sparse participation masks.
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
      source_lane, state->subgroup_size_i32, state->result_type,
      state->location, &shuffle_op));
  *out_value = loom_kernel_subgroup_shuffle_result(shuffle_op);
  return loom_kernel_scan_build_and(
      state, source_rank_valid, loom_kernel_subgroup_shuffle_valid(shuffle_op),
      state->i1_type, out_valid);
}

static iree_status_t loom_kernel_scan_build_combine(
    loom_kernel_scan_state_t* state, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t* out_result) {
  if (loom_type_is_scalar(state->result_type)) {
    return loom_scalar_combining_build(
        state->builder, state->kind, /*fastmath_flags=*/0, lhs, rhs,
        state->result_type, state->location, out_result);
  }
  return loom_vector_combining_build(
      state->builder, state->kind, /*fastmath_flags=*/0, lhs, rhs,
      state->result_type, state->location, out_result);
}

static iree_status_t loom_kernel_scan_build_seed(
    loom_kernel_scan_state_t* state, loom_value_id_t* out_seed) {
  const loom_scalar_type_t element_type =
      loom_type_element_type(state->result_type);
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
  if (loom_type_is_scalar(state->result_type)) {
    *out_seed = loom_scalar_constant_result(scalar_seed_op);
    return iree_ok_status();
  }
  loom_op_t* vector_seed_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      state->builder, loom_scalar_constant_result(scalar_seed_op),
      state->result_type, state->location, &vector_seed_op));
  *out_seed = loom_vector_splat_result(vector_seed_op);
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
  // Ordinary arithmetic support does not establish source-format preservation.
  // Extrema retain their fixed semantics independently of arithmetic mode.
  if (loom_kernel_subgroup_scan_math(op) != 0 &&
      (kind == LOOM_COMBINING_KIND_ADDF || kind == LOOM_COMBINING_KIND_MULF)) {
    return iree_ok_status();
  }
  loom_kernel_scan_cluster_t cluster = {0};
  const bool has_cluster =
      loom_kernel_scan_cluster_from_op(context, op, &cluster);
  if (!has_cluster) {
    return iree_ok_status();
  }
  const loom_target_bundle_t* bundle =
      loom_target_legalization_context_bundle(context);
  const uint32_t subgroup_size = bundle->snapshot->subgroup_size;

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t source_value = loom_kernel_subgroup_scan_value(op);
  const loom_type_t result_type =
      loom_module_value_type(context->module, source_value);
  loom_kernel_scan_state_t state = {
      .builder = &rewriter->builder,
      .location = op->location,
      .result_type = result_type,
      .i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1),
      .i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      .mask_type = loom_type_scalar(subgroup_size <= 32 ? LOOM_SCALAR_TYPE_I32
                                                        : LOOM_SCALAR_TYPE_I64),
      .kind = kind,
      .subgroup_size = subgroup_size,
      .mask_bit_count = subgroup_size <= 32 ? 32u : 64u,
  };
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
      &state, state.mask_type, 0, &state.zero_mask));
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
      &state, state.mask_type, 1, &state.one_mask));
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_integer_constant(
      &state, state.i32_type, subgroup_size, &state.subgroup_size_i32));
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_cluster_state(&state, &cluster));
  IREE_RETURN_IF_ERROR(loom_kernel_scan_build_selector_constants(&state));

  const loom_kernel_subgroup_scan_direction_t direction =
      loom_kernel_subgroup_scan_direction(op);
  loom_value_id_t accumulator = source_value;
  for (uint32_t distance = 1; distance < cluster.size; distance <<= 1) {
    loom_value_id_t source_rank = LOOM_VALUE_ID_INVALID;
    loom_value_id_t source_rank_valid = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_source_rank(
        &state, direction, distance, &source_rank, &source_rank_valid));
    loom_value_id_t peer = LOOM_VALUE_ID_INVALID;
    loom_value_id_t peer_valid = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_kernel_scan_build_shuffle(&state, accumulator, source_rank,
                                       source_rank_valid, &peer, &peer_valid));
    loom_value_id_t combined = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_kernel_scan_build_combine(&state, peer, accumulator, &combined));
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_select(
        &state, peer_valid, combined, accumulator, result_type, &accumulator));
  }

  if (loom_kernel_subgroup_scan_mode(op) ==
      LOOM_KERNEL_SUBGROUP_SCAN_MODE_EXCLUSIVE) {
    loom_value_id_t source_rank = LOOM_VALUE_ID_INVALID;
    loom_value_id_t source_rank_valid = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_source_rank(
        &state, direction, 1, &source_rank, &source_rank_valid));
    loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
    loom_value_id_t shifted_valid = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_shuffle(
        &state, accumulator, source_rank, source_rank_valid, &shifted,
        &shifted_valid));
    loom_value_id_t seed = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_seed(&state, &seed));
    IREE_RETURN_IF_ERROR(loom_kernel_scan_build_select(
        &state, shifted_valid, shifted, seed, result_type, &accumulator));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &accumulator, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &accumulator, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kKernelLegalizerRules[] = {
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
        .root_kind = LOOM_OP_KERNEL_SUBGROUP_SCAN,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER | LOOM_SCALAR_TYPE_SET_FLOAT,
        .match = loom_kernel_match_subgroup_scan,
        .legalize = loom_kernel_target_legalize_subgroup_scan,
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
