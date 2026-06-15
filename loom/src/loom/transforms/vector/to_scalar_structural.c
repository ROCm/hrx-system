// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_structural.h"

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"

//===----------------------------------------------------------------------===//
// Structural lane programs
//===----------------------------------------------------------------------===//

iree_status_t loom_vector_to_scalar_build_broadcast_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t source = loom_vector_broadcast_source(state->op);
  loom_type_t source_type =
      loom_module_value_type(state->rewriter->module, source);
  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(state->vector_type);

  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (source_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, source_rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  uint8_t leading_broadcast_rank = (uint8_t)(result_rank - source_rank);
  for (uint8_t source_axis = 0; source_axis < source_rank; ++source_axis) {
    if (!loom_type_dim_is_dynamic_at(source_type, source_axis) &&
        loom_type_dim_static_size_at(source_type, source_axis) == 1) {
      source_terms[source_axis] = loom_vector_to_scalar_static_term(0);
    } else {
      uint8_t result_axis = (uint8_t)(leading_broadcast_rank + source_axis);
      source_terms[source_axis] =
          loom_vector_to_scalar_lane_term(state, indices, result_axis);
    }
  }

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, source_rank, &source_indices));
  return loom_vector_to_scalar_materialize_lane(state, source, source_indices,
                                                out_lane);
}

iree_status_t loom_vector_to_scalar_build_extract_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t source = loom_vector_extract_source(state->op);
  loom_type_t source_type =
      loom_module_value_type(state->rewriter->module, source);
  loom_vector_to_scalar_index_term_t* explicit_terms = NULL;
  uint8_t explicit_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_from_explicit_indices(
      state, loom_vector_extract_static_indices(state->op),
      loom_vector_extract_indices(state->op), &explicit_terms,
      &explicit_count));

  uint8_t source_rank = loom_type_rank(source_type);
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (source_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, source_rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t i = 0; i < explicit_count; ++i) {
    source_terms[i] = explicit_terms[i];
  }
  for (uint8_t i = 0; i < indices.rank; ++i) {
    source_terms[explicit_count + i] =
        loom_vector_to_scalar_lane_term(state, indices, i);
  }

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, source_rank, &source_indices));
  return loom_vector_to_scalar_materialize_lane(state, source, source_indices,
                                                out_lane);
}

iree_status_t loom_vector_to_scalar_build_insert_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_vector_to_scalar_index_term_t* explicit_terms = NULL;
  uint8_t explicit_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_from_explicit_indices(
      state, loom_vector_insert_static_indices(state->op),
      loom_vector_insert_indices(state->op), &explicit_terms, &explicit_count));

  bool condition_is_statically_true = true;
  loom_value_id_t dynamic_condition = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < explicit_count; ++i) {
    loom_vector_to_scalar_index_term_t lane_term =
        loom_vector_to_scalar_lane_term(state, indices, i);
    bool equal = false;
    if (loom_vector_to_scalar_terms_equal_static(lane_term, explicit_terms[i],
                                                 &equal)) {
      if (!equal) {
        return loom_vector_to_scalar_materialize_lane(
            state, loom_vector_insert_dest(state->op), indices, out_lane);
      }
      continue;
    }
    condition_is_statically_true = false;
    loom_value_id_t axis_equal = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_term_cmp(
        state, LOOM_INDEX_CMP_PREDICATE_EQ, lane_term, explicit_terms[i],
        &axis_equal));
    if (dynamic_condition == LOOM_VALUE_ID_INVALID) {
      dynamic_condition = axis_equal;
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_i1_and(
          state, dynamic_condition, axis_equal, &dynamic_condition));
    }
  }

  loom_vector_to_scalar_index_term_t* value_terms = NULL;
  uint8_t value_rank = (uint8_t)(indices.rank - explicit_count);
  if (value_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, value_rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&value_terms));
  }
  for (uint8_t i = 0; i < value_rank; ++i) {
    value_terms[i] = loom_vector_to_scalar_lane_term(
        state, indices, (uint8_t)(explicit_count + i));
  }
  loom_vector_to_scalar_index_list_t value_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, value_terms, value_rank, &value_indices));
  loom_value_id_t value_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_insert_value(state->op), value_indices, &value_lane));
  if (condition_is_statically_true) {
    *out_lane = value_lane;
    return iree_ok_status();
  }

  loom_value_id_t dest_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_insert_dest(state->op), indices, &dest_lane));
  return loom_vector_to_scalar_build_select_lane(
      state, dynamic_condition, value_lane, dest_lane, out_lane);
}

iree_status_t loom_vector_to_scalar_build_slice_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_vector_to_scalar_index_term_t* offset_terms = NULL;
  uint8_t offset_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_from_explicit_indices(
      state, loom_vector_slice_static_offsets(state->op),
      loom_vector_slice_offsets(state->op), &offset_terms, &offset_count));
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (indices.rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, indices.rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t i = 0; i < indices.rank; ++i) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, offset_terms[i],
        loom_vector_to_scalar_lane_term(state, indices, i), &source_terms[i]));
  }

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, indices.rank, &source_indices));
  return loom_vector_to_scalar_materialize_lane(
      state, loom_vector_slice_source(state->op), source_indices, out_lane);
}

static bool loom_vector_to_scalar_concat_axis_extents_are_static(
    const loom_module_t* module, loom_value_slice_t inputs, uint8_t axis) {
  for (uint16_t i = 0; i < inputs.count; ++i) {
    loom_type_t input_type =
        loom_module_value_type(module, loom_value_slice_get(inputs, i));
    if (loom_type_dim_is_dynamic_at(input_type, axis)) return false;
  }
  return true;
}

static iree_status_t loom_vector_to_scalar_build_concat_dynamic_lane(
    loom_vector_to_scalar_state_t* state, loom_value_slice_t inputs,
    uint16_t input_ordinal, loom_vector_to_scalar_index_term_t prefix,
    uint8_t axis, loom_vector_to_scalar_index_list_t indices,
    loom_value_id_t* out_lane) {
  if (input_ordinal >= inputs.count) {
    return loom_vector_to_scalar_build_poison_lane(state, out_lane);
  }

  loom_value_id_t input = loom_value_slice_get(inputs, input_ordinal);
  loom_type_t input_type =
      loom_module_value_type(state->rewriter->module, input);
  loom_vector_to_scalar_index_term_t axis_index =
      loom_vector_to_scalar_lane_term(state, indices, axis);
  loom_vector_to_scalar_index_term_t extent =
      loom_vector_to_scalar_dim_bound_term(state, input_type, axis);
  loom_vector_to_scalar_index_term_t end = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, prefix, extent, &end));
  loom_value_id_t within_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_term_cmp(
      state, LOOM_INDEX_CMP_PREDICATE_SLT, axis_index, end, &within_input));

  loom_vector_to_scalar_index_term_t source_axis = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_SUB, axis_index, prefix,
      &source_axis));
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (indices.rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, indices.rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t i = 0; i < indices.rank; ++i) {
    source_terms[i] = loom_vector_to_scalar_lane_term(state, indices, i);
  }
  source_terms[axis] = source_axis;
  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, indices.rank, &source_indices));

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      &state->rewriter->builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
      within_input, &state->result_scalar_type, 1, NULL, 0, state->location,
      &if_op));

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  loom_value_id_t then_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, input, source_indices, &then_lane));
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->rewriter->builder, &then_lane, 1, state->location, &then_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  saved = loom_builder_enter_region(&state->rewriter->builder, if_op,
                                    loom_scf_if_else_region(if_op));
  loom_value_id_t else_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_concat_dynamic_lane(
      state, inputs, (uint16_t)(input_ordinal + 1), end, axis, indices,
      &else_lane));
  loom_op_t* else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->rewriter->builder, &else_lane, 1, state->location, &else_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_lane = loom_scf_if_results(if_op).values[0];
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_concat_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  int64_t axis = loom_vector_concat_axis(state->op);
  uint8_t axis_u8 = (uint8_t)axis;
  loom_value_slice_t inputs = loom_vector_concat_inputs(state->op);
  if (loom_vector_to_scalar_indices_are_dynamic(indices) ||
      !loom_vector_to_scalar_concat_axis_extents_are_static(
          state->rewriter->module, inputs, axis_u8)) {
    return loom_vector_to_scalar_build_concat_dynamic_lane(
        state, inputs, 0, loom_vector_to_scalar_static_term(0), axis_u8,
        indices, out_lane);
  }

  int64_t axis_index = indices.static_indices[axis_u8];
  int64_t prefix = 0;
  for (uint16_t input_ordinal = 0; input_ordinal < inputs.count;
       ++input_ordinal) {
    loom_value_id_t input = loom_value_slice_get(inputs, input_ordinal);
    loom_type_t input_type =
        loom_module_value_type(state->rewriter->module, input);
    int64_t extent = (int64_t)loom_type_dim_static_size_at(input_type, axis_u8);
    if (input_ordinal + 1 < inputs.count && axis_index >= prefix + extent) {
      prefix += extent;
      continue;
    }

    loom_vector_to_scalar_index_term_t* source_terms = NULL;
    if (indices.rank > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->rewriter->arena, indices.rank,
          sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
    }
    for (uint8_t i = 0; i < indices.rank; ++i) {
      source_terms[i] =
          loom_vector_to_scalar_static_term(indices.static_indices[i]);
    }
    source_terms[axis_u8] =
        loom_vector_to_scalar_static_term(axis_index - prefix);
    loom_vector_to_scalar_index_list_t source_indices = {0};
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
        state, source_terms, indices.rank, &source_indices));
    return loom_vector_to_scalar_materialize_lane(state, input, source_indices,
                                                  out_lane);
  }
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_transpose_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_attribute_t permutation = loom_vector_transpose_permutation(state->op);
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (indices.rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, indices.rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t result_axis = 0; result_axis < indices.rank; ++result_axis) {
    int64_t source_axis = permutation.i64_array[result_axis];
    source_terms[source_axis] =
        loom_vector_to_scalar_lane_term(state, indices, result_axis);
  }
  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, indices.rank, &source_indices));
  return loom_vector_to_scalar_materialize_lane(
      state, loom_vector_transpose_source(state->op), source_indices, out_lane);
}

iree_status_t loom_vector_to_scalar_build_shuffle_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(state->op);
  int64_t result_lane = indices.static_indices[0];
  int64_t source_lane = source_lanes.i64_array[result_lane];
  loom_vector_to_scalar_index_list_t source_indices = {
      .static_indices = &source_lane,
      .rank = 1,
  };
  return loom_vector_to_scalar_materialize_lane(
      state, loom_vector_shuffle_source(state->op), source_indices, out_lane);
}

iree_status_t loom_vector_to_scalar_build_interleave_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  int64_t axis = loom_vector_interleave_axis(state->op);
  uint8_t axis_u8 = (uint8_t)axis;
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (indices.rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, indices.rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t i = 0; i < indices.rank; ++i) {
    source_terms[i] = loom_vector_to_scalar_lane_term(state, indices, i);
  }

  loom_vector_to_scalar_index_term_t axis_index = source_terms[axis_u8];
  loom_vector_to_scalar_index_term_t source_axis_index = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV, axis_index,
      loom_vector_to_scalar_static_term(2), &source_axis_index));
  source_terms[axis_u8] = source_axis_index;

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, indices.rank, &source_indices));

  if (!loom_vector_to_scalar_indices_are_dynamic(indices)) {
    loom_value_id_t source = (indices.static_indices[axis_u8] & 1) == 0
                                 ? loom_vector_interleave_even(state->op)
                                 : loom_vector_interleave_odd(state->op);
    return loom_vector_to_scalar_materialize_lane(state, source, source_indices,
                                                  out_lane);
  }

  loom_value_id_t even_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_interleave_even(state->op), source_indices,
      &even_lane));
  loom_value_id_t odd_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_interleave_odd(state->op), source_indices, &odd_lane));

  loom_vector_to_scalar_index_term_t remainder = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM, axis_index,
      loom_vector_to_scalar_static_term(2), &remainder));
  loom_value_id_t remainder_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, remainder, &remainder_value));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &zero));
  loom_op_t* is_even_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      &state->rewriter->builder, LOOM_INDEX_CMP_PREDICATE_EQ, remainder_value,
      zero, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location, &is_even_op));
  return loom_vector_to_scalar_build_select_lane(
      state, loom_index_cmp_result(is_even_op), even_lane, odd_lane, out_lane);
}

iree_status_t loom_vector_to_scalar_build_deinterleave_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  int64_t axis = loom_vector_deinterleave_axis(state->op);
  uint8_t axis_u8 = (uint8_t)axis;
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (indices.rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, indices.rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t i = 0; i < indices.rank; ++i) {
    source_terms[i] = loom_vector_to_scalar_lane_term(state, indices, i);
  }

  loom_vector_to_scalar_index_term_t scaled = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL, source_terms[axis_u8],
      loom_vector_to_scalar_static_term(2), &scaled));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, scaled,
      loom_vector_to_scalar_static_term((int64_t)state->result_ordinal),
      &source_terms[axis_u8]));

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, indices.rank, &source_indices));
  return loom_vector_to_scalar_materialize_lane(
      state, loom_vector_deinterleave_source(state->op), source_indices,
      out_lane);
}

static loom_type_t loom_vector_to_scalar_bitcast_integer_type(
    loom_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I1:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I1);
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I8);
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I16);
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F32:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F64:
      return loom_type_scalar(LOOM_SCALAR_TYPE_I64);
    default:
      IREE_ASSERT_UNREACHABLE("unknown scalar type");
      IREE_BUILTIN_UNREACHABLE();
  }
  return loom_type_scalar(LOOM_SCALAR_TYPE_I64);
}

static iree_status_t loom_vector_to_scalar_bitcast_lane_to_integer(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lane,
    loom_type_t lane_type, loom_type_t integer_type,
    loom_value_id_t* out_integer_lane) {
  if (loom_type_equal(lane_type, integer_type)) {
    *out_integer_lane = lane;
    return iree_ok_status();
  }
  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(&state->rewriter->builder,
                                                 lane, lane_type, integer_type,
                                                 state->location, &bitcast_op));
  *out_integer_lane = loom_scalar_bitcast_result(bitcast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_bitcast_piece(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source_lane,
    loom_type_t source_integer_type, int64_t source_shift, int64_t bit_count,
    int64_t dest_shift, loom_type_t result_integer_type,
    loom_value_id_t accumulator, loom_value_id_t* out_accumulator) {
  loom_value_id_t source_bits = LOOM_VALUE_ID_INVALID;
  int32_t source_width =
      loom_scalar_type_bitwidth(loom_type_element_type(source_integer_type));
  if (source_shift == 0 && bit_count == source_width) {
    source_bits = source_lane;
  } else {
    loom_value_id_t shifted_source = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
        state, LOOM_OP_SCALAR_SHRUI, source_lane, source_integer_type,
        source_shift, &shifted_source));
    loom_value_id_t source_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
        state, source_integer_type, bit_count, &source_mask));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
        state, LOOM_OP_SCALAR_ANDI, shifted_source, source_mask,
        source_integer_type, &source_bits));
  }
  loom_value_id_t result_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, source_bits, source_integer_type, result_integer_type,
      /*signed_extend=*/false, &result_bits));
  loom_value_id_t shifted_result_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHLI, result_bits, result_integer_type, dest_shift,
      &shifted_result_bits));
  if (accumulator == LOOM_VALUE_ID_INVALID) {
    *out_accumulator = shifted_result_bits;
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ORI, accumulator, shifted_result_bits,
      result_integer_type, out_accumulator);
}

static iree_status_t loom_vector_to_scalar_build_dynamic_bitcast_bit(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_type_t input_type, loom_type_t source_scalar_type,
    loom_type_t source_integer_type, loom_type_t result_integer_type,
    loom_vector_to_scalar_index_term_t result_base_bit,
    loom_vector_to_scalar_index_term_t result_bit, int32_t source_width,
    loom_value_id_t source_one, loom_value_id_t accumulator,
    loom_value_id_t* out_accumulator) {
  loom_vector_to_scalar_index_term_t global_bit = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, result_base_bit,
      result_bit, &global_bit));
  loom_vector_to_scalar_index_term_t source_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV, global_bit,
      loom_vector_to_scalar_static_term(source_width), &source_ordinal));
  loom_vector_to_scalar_index_term_t source_shift = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM, global_bit,
      loom_vector_to_scalar_static_term(source_width), &source_shift));

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_linear_lane(
      state, input, input_type, source_ordinal, &source_lane));
  loom_value_id_t source_integer_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_bitcast_lane_to_integer(
      state, source_lane, source_scalar_type, source_integer_type,
      &source_integer_lane));
  loom_value_id_t source_bit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_single_bit_extract(
      state, source_integer_lane, source_integer_type, source_shift, source_one,
      &source_bit));
  loom_value_id_t result_bit_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, source_bit, source_integer_type, result_integer_type,
      /*signed_extend=*/false, &result_bit_value));
  loom_value_id_t shifted_result_bit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift_term(
      state, LOOM_OP_SCALAR_SHLI, result_bit_value, result_integer_type,
      result_bit, &shifted_result_bit));
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ORI, accumulator, shifted_result_bit,
      result_integer_type, out_accumulator);
}

static iree_status_t
loom_vector_to_scalar_build_dynamic_shape_changing_bitcast_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t input = loom_vector_bitcast_input(state->op);
  loom_type_t input_type =
      loom_module_value_type(state->rewriter->module, input);
  loom_type_t source_scalar_type = loom_vector_to_scalar_lane_type(input_type);
  loom_scalar_type_t source_element_type =
      loom_type_element_type(source_scalar_type);
  loom_scalar_type_t result_element_type =
      loom_type_element_type(state->result_scalar_type);
  int32_t source_width = loom_scalar_type_bitwidth(source_element_type);
  int32_t result_width = loom_scalar_type_bitwidth(result_element_type);
  loom_type_t source_integer_type =
      loom_vector_to_scalar_bitcast_integer_type(source_element_type);
  loom_type_t result_integer_type =
      loom_vector_to_scalar_bitcast_integer_type(result_element_type);

  loom_vector_to_scalar_index_term_t result_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_linear_ordinal_term(
      state, state->vector_type, indices, &result_ordinal));
  loom_vector_to_scalar_index_term_t result_base_bit = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitstream_base_term(
      state, result_ordinal, result_width, &result_base_bit));

  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &lower_bound));
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, result_width, &upper_bound));
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, &step));
  loom_value_id_t source_one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, source_integer_type, state->location, 1,
      &source_one));
  loom_value_id_t initial_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, result_integer_type, state->location, 0,
      &initial_accumulator));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, &initial_accumulator, 1, &result_integer_type, 1, NULL, 0,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0, state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_value_id_t result_bit_value =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t accumulator_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_vector_to_scalar_index_term_t result_bit =
      loom_vector_to_scalar_value_term(state, result_bit_value);
  loom_value_id_t yielded_accumulator = LOOM_VALUE_ID_INVALID;
  iree_status_t status = loom_vector_to_scalar_build_dynamic_bitcast_bit(
      state, input, input_type, source_scalar_type, source_integer_type,
      result_integer_type, result_base_bit, result_bit, source_width,
      source_one, accumulator_arg, &yielded_accumulator);
  if (iree_status_is_ok(status)) {
    loom_op_t* yield_op = NULL;
    status =
        loom_scf_yield_build(&state->rewriter->builder, &yielded_accumulator, 1,
                             state->location, &yield_op);
  }
  loom_builder_restore(&state->rewriter->builder, saved);
  IREE_RETURN_IF_ERROR(status);

  loom_value_id_t accumulator = loom_scf_for_results(loop).values[0];
  if (loom_type_equal(result_integer_type, state->result_scalar_type)) {
    *out_lane = accumulator;
    return iree_ok_status();
  }
  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      &state->rewriter->builder, accumulator, result_integer_type,
      state->result_scalar_type, state->location, &bitcast_op));
  *out_lane = loom_scalar_bitcast_result(bitcast_op);
  return iree_ok_status();
}

// Shape-changing bitcasts reinterpret the source vector as a row-major linear
// bitstream, with lane ordinal 0 providing the least-significant source bits.
static iree_status_t
loom_vector_to_scalar_build_static_shape_changing_bitcast_lane(
    loom_vector_to_scalar_state_t* state, const int64_t* result_indices,
    loom_value_id_t* out_lane) {
  loom_value_id_t input = loom_vector_bitcast_input(state->op);
  loom_type_t input_type =
      loom_module_value_type(state->rewriter->module, input);

  loom_type_t source_scalar_type = loom_vector_to_scalar_lane_type(input_type);
  loom_scalar_type_t source_element_type =
      loom_type_element_type(source_scalar_type);
  loom_scalar_type_t result_element_type =
      loom_type_element_type(state->result_scalar_type);
  int32_t source_width = loom_scalar_type_bitwidth(source_element_type);
  int32_t result_width = loom_scalar_type_bitwidth(result_element_type);
  loom_type_t source_integer_type =
      loom_vector_to_scalar_bitcast_integer_type(source_element_type);
  loom_type_t result_integer_type =
      loom_vector_to_scalar_bitcast_integer_type(result_element_type);

  int64_t result_ordinal = loom_vector_to_scalar_linear_ordinal_static(
      state->vector_type, result_indices);
  int64_t result_start = result_ordinal * result_width;
  int64_t result_end = result_start + result_width;
  int64_t first_source_ordinal = result_start / source_width;
  int64_t last_source_ordinal = (result_end - 1) / source_width;

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  uint8_t source_rank = loom_type_rank(input_type);
  int64_t* source_indices = NULL;
  if (source_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena,
                                                   source_rank, sizeof(int64_t),
                                                   (void**)&source_indices));
  }
  for (int64_t source_ordinal = first_source_ordinal;
       source_ordinal <= last_source_ordinal; ++source_ordinal) {
    int64_t source_start = source_ordinal * source_width;
    int64_t source_end = source_start + source_width;
    int64_t overlap_start =
        result_start > source_start ? result_start : source_start;
    int64_t overlap_end = result_end < source_end ? result_end : source_end;
    int64_t bit_count = overlap_end - overlap_start;
    if (bit_count <= 0) continue;

    loom_vector_to_scalar_indices_from_ordinal(input_type, source_ordinal,
                                               source_indices);
    loom_vector_to_scalar_index_list_t source_index_list = {
        .static_indices = source_indices,
        .rank = source_rank,
    };
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, input, source_index_list, &source_lane));
    loom_value_id_t source_integer_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_bitcast_lane_to_integer(
        state, source_lane, source_scalar_type, source_integer_type,
        &source_integer_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitcast_piece(
        state, source_integer_lane, source_integer_type,
        overlap_start - source_start, bit_count, overlap_start - result_start,
        result_integer_type, accumulator, &accumulator));
  }
  IREE_ASSERT_NE(accumulator, LOOM_VALUE_ID_INVALID);

  if (loom_type_equal(result_integer_type, state->result_scalar_type)) {
    *out_lane = accumulator;
    return iree_ok_status();
  }
  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      &state->rewriter->builder, accumulator, result_integer_type,
      state->result_scalar_type, state->location, &bitcast_op));
  *out_lane = loom_scalar_bitcast_result(bitcast_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_bitcast_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t input = loom_vector_bitcast_input(state->op);
  loom_type_t input_type =
      loom_module_value_type(state->rewriter->module, input);
  if (!loom_type_shape_equals(input_type, state->vector_type)) {
    if (loom_vector_to_scalar_indices_are_dynamic(indices) ||
        !loom_type_is_all_static(input_type) ||
        !loom_type_is_all_static(state->vector_type)) {
      return loom_vector_to_scalar_build_dynamic_shape_changing_bitcast_lane(
          state, indices, out_lane);
    }
    return loom_vector_to_scalar_build_static_shape_changing_bitcast_lane(
        state, indices.static_indices, out_lane);
  }
  loom_scalar_type_t input_element_type = loom_type_element_type(input_type);
  loom_scalar_type_t result_element_type =
      loom_type_element_type(state->vector_type);

  loom_value_id_t input_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, input, indices, &input_lane));
  if (input_element_type == result_element_type) {
    *out_lane = input_lane;
    return iree_ok_status();
  }
  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      &state->rewriter->builder, input_lane,
      loom_type_scalar(input_element_type), state->result_scalar_type,
      state->location, &bitcast_op));
  *out_lane = loom_scalar_bitcast_result(bitcast_op);
  return iree_ok_status();
}
