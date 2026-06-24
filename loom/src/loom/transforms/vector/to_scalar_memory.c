// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_memory.h"

#include "loom/analysis/contract.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/transforms/vector/to_scalar_mma.h"

//===----------------------------------------------------------------------===//
// Logical view indices
//===----------------------------------------------------------------------===//

typedef struct loom_vector_to_scalar_view_indices_t {
  // Dynamic logical index operands for INT64_MIN sentinels.
  const loom_value_id_t* dynamic_indices;
  // Number of dynamic logical index operands.
  iree_host_size_t dynamic_index_count;
  // Full-rank static logical indices with INT64_MIN dynamic sentinels.
  const int64_t* static_indices;
  // Number of static logical indices.
  iree_host_size_t static_index_count;
} loom_vector_to_scalar_view_indices_t;

static loom_memory_access_t loom_vector_to_scalar_memory_access(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_cast(state->rewriter->module, state->op);
}

static loom_value_id_t loom_vector_to_scalar_memory_view(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_view(loom_vector_to_scalar_memory_access(state));
}

static loom_attribute_t loom_vector_to_scalar_memory_static_indices(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_static_indices(
      loom_vector_to_scalar_memory_access(state));
}

static loom_value_slice_t loom_vector_to_scalar_memory_dynamic_indices(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_dynamic_indices(
      loom_vector_to_scalar_memory_access(state));
}

static loom_vector_memory_cache_policy_t
loom_vector_to_scalar_memory_cache_policy(
    loom_vector_to_scalar_state_t* state) {
  loom_vector_memory_cache_policy_t policy = {0};
  loom_vector_memory_cache_policy_from_op(state->rewriter->module, state->op,
                                          &policy);
  return policy;
}

static uint8_t loom_vector_to_scalar_enum_attr_value(loom_attribute_t attr) {
  return loom_attr_as_enum(attr);
}

static iree_status_t loom_vector_to_scalar_index_list_to_view_indices(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t index_list,
    loom_vector_to_scalar_view_indices_t* out_indices) {
  if (loom_vector_to_scalar_indices_are_dynamic(index_list)) {
    int64_t* static_indices = NULL;
    if (index_list.rank > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->rewriter->builder.arena, index_list.rank, sizeof(int64_t),
          (void**)&static_indices));
    }
    for (uint8_t i = 0; i < index_list.rank; ++i) {
      static_indices[i] = INT64_MIN;
    }
    *out_indices = (loom_vector_to_scalar_view_indices_t){
        .dynamic_indices = index_list.dynamic_indices,
        .dynamic_index_count = index_list.rank,
        .static_indices = static_indices,
        .static_index_count = index_list.rank,
    };
    return iree_ok_status();
  }

  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_copy_static_indices(
      &state->rewriter->builder, index_list.static_indices, index_list.rank,
      &static_indices));
  *out_indices = (loom_vector_to_scalar_view_indices_t){
      .static_indices = static_indices,
      .static_index_count = index_list.rank,
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_index_lane_to_index(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_value_id_t* out_index) {
  loom_type_t value_type =
      loom_module_value_type(state->rewriter->module, value);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  if (loom_type_equal(value_type, index_type)) {
    *out_index = value;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(&state->rewriter->builder, value,
                                             value_type, index_type,
                                             state->location, &cast_op));
  *out_index = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_memory_terms(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices, bool add_lane_indices,
    loom_value_id_t offset_vector, bool add_last_axis_offset,
    loom_vector_to_scalar_index_term_t last_axis_offset,
    loom_vector_to_scalar_index_term_t** out_terms, uint8_t* out_count) {
  loom_vector_to_scalar_index_term_t* terms = NULL;
  uint8_t term_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_from_explicit_indices(
      state, loom_vector_to_scalar_memory_static_indices(state),
      loom_vector_to_scalar_memory_dynamic_indices(state), &terms,
      &term_count));

  if (add_lane_indices) {
    uint8_t first_vector_axis = (uint8_t)(term_count - lane_indices.rank);
    for (uint8_t axis = 0; axis < lane_indices.rank; ++axis) {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD,
          terms[first_vector_axis + axis],
          loom_vector_to_scalar_lane_term(state, lane_indices, axis),
          &terms[first_vector_axis + axis]));
    }
  }

  if (offset_vector != LOOM_VALUE_ID_INVALID) {
    loom_value_id_t offset_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, offset_vector, lane_indices, &offset_lane));
    loom_value_id_t offset_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_index_lane_to_index(
        state, offset_lane, &offset_index));
    uint8_t last_axis = (uint8_t)(term_count - 1);
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, terms[last_axis],
        loom_vector_to_scalar_value_term(state, offset_index),
        &terms[last_axis]));
  }

  if (add_last_axis_offset) {
    uint8_t last_axis = (uint8_t)(term_count - 1);
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, terms[last_axis],
        last_axis_offset, &terms[last_axis]));
  }

  *out_terms = terms;
  *out_count = term_count;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_view_indices(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices, bool add_lane_indices,
    loom_value_id_t offset_vector, bool add_last_axis_offset,
    loom_vector_to_scalar_index_term_t last_axis_offset,
    loom_vector_to_scalar_view_indices_t* out_indices) {
  loom_vector_to_scalar_index_term_t* terms = NULL;
  uint8_t term_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_memory_terms(
      state, lane_indices, add_lane_indices, offset_vector,
      add_last_axis_offset, last_axis_offset, &terms, &term_count));
  loom_vector_to_scalar_index_list_t index_list = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, terms, term_count, &index_list));
  return loom_vector_to_scalar_index_list_to_view_indices(state, index_list,
                                                          out_indices);
}

//===----------------------------------------------------------------------===//
// Packed active-lane indices
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_build_mask_increment(
    loom_vector_to_scalar_state_t* state, loom_value_id_t mask_vector,
    loom_vector_to_scalar_index_list_t mask_indices,
    loom_value_id_t* out_increment) {
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, mask_vector, mask_indices, &condition));
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, index_type, state->location, 0, &zero));
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, index_type, state->location, 1, &one));
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(&state->rewriter->builder,
                                             condition, one, zero, index_type,
                                             state->location, &select_op));
  *out_increment = loom_scf_select_result(select_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_static_active_prefix(
    loom_vector_to_scalar_state_t* state, loom_value_id_t mask_vector,
    int64_t lane_ordinal, loom_vector_to_scalar_index_term_t* out_prefix) {
  loom_vector_to_scalar_index_term_t prefix =
      loom_vector_to_scalar_static_term(0);
  for (int64_t i = 0; i < lane_ordinal; ++i) {
    int64_t static_index = i;
    loom_vector_to_scalar_index_list_t mask_indices = {
        .static_indices = &static_index,
        .rank = 1,
    };
    loom_value_id_t increment = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_mask_increment(
        state, mask_vector, mask_indices, &increment));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, prefix,
        loom_vector_to_scalar_value_term(state, increment), &prefix));
  }
  *out_prefix = prefix;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_dynamic_active_prefix(
    loom_vector_to_scalar_state_t* state, loom_value_id_t mask_vector,
    loom_value_id_t lane_index,
    loom_vector_to_scalar_index_term_t* out_prefix) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, index_type, state->location, 0, &lower_bound));
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, index_type, state->location, 1, &step));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, lane_index,
      step, &lower_bound, 1, &index_type, 1, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_value_id_t prefix_iv =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t accumulator =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_vector_to_scalar_index_list_t mask_indices = {
      .dynamic_indices = &prefix_iv,
      .rank = 1,
  };
  loom_value_id_t increment = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_mask_increment(
      state, mask_vector, mask_indices, &increment));
  loom_value_id_t next_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, accumulator, increment,
      &next_accumulator));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &next_accumulator, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_prefix = loom_vector_to_scalar_value_term(
      state, loom_scf_for_results(loop).values[0]);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_active_prefix(
    loom_vector_to_scalar_state_t* state, loom_value_id_t mask_vector,
    loom_vector_to_scalar_index_list_t lane_indices,
    loom_vector_to_scalar_index_term_t* out_prefix) {
  if (loom_vector_to_scalar_indices_are_dynamic(lane_indices)) {
    return loom_vector_to_scalar_build_dynamic_active_prefix(
        state, mask_vector, lane_indices.dynamic_indices[0], out_prefix);
  }
  return loom_vector_to_scalar_build_static_active_prefix(
      state, mask_vector, lane_indices.static_indices[0], out_prefix);
}

//===----------------------------------------------------------------------===//
// Load lanes
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_build_view_load_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices, bool add_lane_indices,
    loom_value_id_t offset_vector, loom_value_id_t* out_lane) {
  loom_vector_to_scalar_view_indices_t view_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_indices(
      state, lane_indices, add_lane_indices, offset_vector,
      /*add_last_axis_offset=*/false, (loom_vector_to_scalar_index_term_t){0},
      &view_indices));
  loom_op_t* load_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy =
      loom_vector_to_scalar_memory_cache_policy(state);
  IREE_RETURN_IF_ERROR(loom_view_load_build(
      &state->rewriter->builder, cache_policy.build_flags,
      loom_vector_to_scalar_memory_view(state), view_indices.dynamic_indices,
      view_indices.dynamic_index_count, view_indices.static_indices,
      view_indices.static_index_count, cache_policy.cache_scope,
      cache_policy.cache_temporal, state->result_scalar_type, state->location,
      &load_op));
  *out_lane = loom_view_load_result(load_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_masked_view_load_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices,
    loom_value_id_t mask_vector, loom_value_id_t passthrough_vector,
    bool add_lane_indices, loom_value_id_t offset_vector,
    loom_value_id_t* out_lane) {
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, mask_vector, lane_indices, &condition));

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(&state->rewriter->builder,
                                         LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                                         condition, &state->result_scalar_type,
                                         1, NULL, 0, state->location, &if_op));

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  loom_value_id_t loaded_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_load_lane(
      state, lane_indices, add_lane_indices, offset_vector, &loaded_lane));
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &loaded_lane, 1, state->location,
                                            &then_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  saved = loom_builder_enter_region(&state->rewriter->builder, if_op,
                                    loom_scf_if_else_region(if_op));
  loom_value_id_t passthrough_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, passthrough_vector, lane_indices, &passthrough_lane));
  loom_op_t* else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &passthrough_lane, 1,
                                            state->location, &else_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_lane = loom_scf_if_results(if_op).values[0];
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_load_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_build_view_load_lane(
      state, indices, /*add_lane_indices=*/true, LOOM_VALUE_ID_INVALID,
      out_lane);
}

iree_status_t loom_vector_to_scalar_build_masked_load_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_memory_access_t access = loom_vector_to_scalar_memory_access(state);
  return loom_vector_to_scalar_build_masked_view_load_lane(
      state, indices, loom_memory_access_mask(access),
      loom_memory_access_passthrough(access), /*add_lane_indices=*/true,
      LOOM_VALUE_ID_INVALID, out_lane);
}

iree_status_t loom_vector_to_scalar_build_gather_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_build_view_load_lane(
      state, indices, /*add_lane_indices=*/false,
      loom_memory_access_offsets(loom_vector_to_scalar_memory_access(state)),
      out_lane);
}

iree_status_t loom_vector_to_scalar_build_masked_gather_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_memory_access_t access = loom_vector_to_scalar_memory_access(state);
  return loom_vector_to_scalar_build_masked_view_load_lane(
      state, indices, loom_memory_access_mask(access),
      loom_memory_access_passthrough(access), /*add_lane_indices=*/false,
      loom_memory_access_offsets(access), out_lane);
}

iree_status_t loom_vector_to_scalar_build_load_expand_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_memory_access_t access = loom_vector_to_scalar_memory_access(state);
  loom_value_id_t mask = loom_memory_access_mask(access);
  loom_vector_to_scalar_index_term_t prefix = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_active_prefix(state, mask, indices, &prefix));
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_materialize_lane(state, mask, indices, &condition));

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(&state->rewriter->builder,
                                         LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                                         condition, &state->result_scalar_type,
                                         1, NULL, 0, state->location, &if_op));

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  loom_vector_to_scalar_view_indices_t view_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_indices(
      state, indices, /*add_lane_indices=*/false, LOOM_VALUE_ID_INVALID,
      /*add_last_axis_offset=*/true, prefix, &view_indices));
  loom_op_t* load_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy =
      loom_vector_to_scalar_memory_cache_policy(state);
  IREE_RETURN_IF_ERROR(loom_view_load_build(
      &state->rewriter->builder, cache_policy.build_flags,
      loom_vector_to_scalar_memory_view(state), view_indices.dynamic_indices,
      view_indices.dynamic_index_count, view_indices.static_indices,
      view_indices.static_index_count, cache_policy.cache_scope,
      cache_policy.cache_temporal, state->result_scalar_type, state->location,
      &load_op));
  loom_value_id_t loaded_lane = loom_view_load_result(load_op);
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &loaded_lane, 1, state->location,
                                            &then_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  saved = loom_builder_enter_region(&state->rewriter->builder, if_op,
                                    loom_scf_if_else_region(if_op));
  loom_value_id_t passthrough_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_memory_access_passthrough(access), indices,
      &passthrough_lane));
  loom_op_t* else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &passthrough_lane, 1,
                                            state->location, &else_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_lane = loom_scf_if_results(if_op).values[0];
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Store lanes
//===----------------------------------------------------------------------===//

static bool loom_vector_to_scalar_store_is_masked(
    loom_vector_to_scalar_state_t* state) {
  loom_memory_access_t access = loom_vector_to_scalar_memory_access(state);
  return loom_memory_access_mask(access) != LOOM_VALUE_ID_INVALID;
}

static bool loom_vector_to_scalar_store_is_scatter(
    loom_vector_to_scalar_state_t* state) {
  loom_memory_access_t access = loom_vector_to_scalar_memory_access(state);
  return loom_memory_access_offsets(access) != LOOM_VALUE_ID_INVALID;
}

static loom_value_id_t loom_vector_to_scalar_store_value(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_value(loom_vector_to_scalar_memory_access(state));
}

static loom_value_id_t loom_vector_to_scalar_store_mask(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_mask(loom_vector_to_scalar_memory_access(state));
}

static loom_value_id_t loom_vector_to_scalar_store_offsets(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_offsets(loom_vector_to_scalar_memory_access(state));
}

static iree_status_t loom_vector_to_scalar_emit_view_store_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices) {
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_to_scalar_store_value(state), lane_indices, &lane));
  loom_vector_to_scalar_view_indices_t view_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_indices(
      state, lane_indices, !loom_vector_to_scalar_store_is_scatter(state),
      loom_vector_to_scalar_store_offsets(state),
      /*add_last_axis_offset=*/false, (loom_vector_to_scalar_index_term_t){0},
      &view_indices));
  loom_op_t* store_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy =
      loom_vector_to_scalar_memory_cache_policy(state);
  return loom_view_store_build(
      &state->rewriter->builder, cache_policy.build_flags, lane,
      loom_vector_to_scalar_memory_view(state), view_indices.dynamic_indices,
      view_indices.dynamic_index_count, view_indices.static_indices,
      view_indices.static_index_count, cache_policy.cache_scope,
      cache_policy.cache_temporal, state->location, &store_op);
}

static iree_status_t loom_vector_to_scalar_emit_masked_view_store_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices) {
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_to_scalar_store_mask(state), lane_indices,
      &condition));

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      &state->rewriter->builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
      condition, NULL, 0, NULL, 0, state->location, &if_op));

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_emit_view_store_lane(state, lane_indices));
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &then_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  saved = loom_builder_enter_region(&state->rewriter->builder, if_op,
                                    loom_scf_if_else_region(if_op));
  loom_op_t* else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &else_yield));
  loom_builder_restore(&state->rewriter->builder, saved);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_emit_store_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices) {
  if (loom_vector_to_scalar_store_is_masked(state)) {
    return loom_vector_to_scalar_emit_masked_view_store_lane(state, indices);
  }
  return loom_vector_to_scalar_emit_view_store_lane(state, indices);
}

static iree_status_t loom_vector_to_scalar_emit_store_compress_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices) {
  loom_memory_access_t access = loom_vector_to_scalar_memory_access(state);
  loom_value_id_t mask = loom_memory_access_mask(access);
  loom_vector_to_scalar_index_term_t prefix = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_active_prefix(state, mask, indices, &prefix));
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_materialize_lane(state, mask, indices, &condition));

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      &state->rewriter->builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
      condition, NULL, 0, NULL, 0, state->location, &if_op));

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_memory_access_value(access), indices, &lane));
  loom_vector_to_scalar_view_indices_t view_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_indices(
      state, indices, /*add_lane_indices=*/false, LOOM_VALUE_ID_INVALID,
      /*add_last_axis_offset=*/true, prefix, &view_indices));
  loom_op_t* store_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy =
      loom_vector_to_scalar_memory_cache_policy(state);
  IREE_RETURN_IF_ERROR(loom_view_store_build(
      &state->rewriter->builder, cache_policy.build_flags, lane,
      loom_vector_to_scalar_memory_view(state), view_indices.dynamic_indices,
      view_indices.dynamic_index_count, view_indices.static_indices,
      view_indices.static_index_count, cache_policy.cache_scope,
      cache_policy.cache_temporal, state->location, &store_op));
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &then_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  saved = loom_builder_enter_region(&state->rewriter->builder, if_op,
                                    loom_scf_if_else_region(if_op));
  loom_op_t* else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &else_yield));
  loom_builder_restore(&state->rewriter->builder, saved);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_static_memory_store(
    loom_vector_to_scalar_state_t* state) {
  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      state, state->vector_type, &element_count));
  if (loom_pass_has_error_diagnostics(state->pass)) return iree_ok_status();
  uint8_t rank = loom_type_rank(state->vector_type);
  int64_t* indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, rank, sizeof(int64_t), (void**)&indices));
  }
  for (uint16_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(state->vector_type,
                                               (int64_t)ordinal, indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = indices,
        .rank = rank,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_emit_store_lane(state, index_list));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_memory_store_loop_axis(
    loom_vector_to_scalar_state_t* state, uint8_t axis,
    loom_value_id_t* dynamic_indices) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      state, state->vector_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, NULL, 0, NULL, 0, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  if (axis + 1 == loom_type_rank(state->vector_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->vector_type),
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_emit_store_lane(state, index_list));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_lower_memory_store_loop_axis(
        state, (uint8_t)(axis + 1), dynamic_indices));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_dynamic_memory_store(
    loom_vector_to_scalar_state_t* state) {
  uint8_t rank = loom_type_rank(state->vector_type);
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena, rank,
                                                 sizeof(loom_value_id_t),
                                                 (void**)&dynamic_indices));
  return loom_vector_to_scalar_lower_memory_store_loop_axis(state, 0,
                                                            dynamic_indices);
}

iree_status_t loom_vector_to_scalar_lower_memory_store(
    loom_vector_to_scalar_state_t* state) {
  if (loom_type_is_all_static(state->vector_type)) {
    return loom_vector_to_scalar_lower_static_memory_store(state);
  }
  return loom_vector_to_scalar_lower_dynamic_memory_store(state);
}

//===----------------------------------------------------------------------===//
// Fragment store lanes
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_fragment_store_loop_bounds(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t upper_bound_term,
    loom_value_id_t* out_lower_bound, loom_value_id_t* out_upper_bound,
    loom_value_id_t* out_step) {
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, out_lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_term_value(state, upper_bound_term,
                                                        out_upper_bound));
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, out_step);
}

static bool loom_vector_to_scalar_fragment_store_source_is_supported(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value) {
  loom_type_t type = loom_module_value_type(state->rewriter->module, value);
  if (!loom_type_is_vector(type)) return true;

  loom_op_t* def_op =
      loom_vector_to_scalar_value_def_op(state->rewriter->module, value);
  if (def_op == NULL) {
    return false;
  }
  const loom_trait_flags_t traits =
      loom_op_effective_traits(state->rewriter->module, def_op);
  if (loom_traits_are_value_alias(traits)) {
    IREE_ASSERT(def_op->operand_count >= 1);
    return loom_vector_to_scalar_fragment_store_source_is_supported(
        state, loom_op_const_operands(def_op)[0]);
  }
  if (loom_vector_fragment_load_isa(def_op) || loom_vector_splat_isa(def_op) ||
      loom_vector_constant_isa(def_op) || loom_vector_poison_isa(def_op)) {
    return true;
  }
  if (loom_vector_mma_isa(def_op)) {
    return loom_vector_to_scalar_mma_supports_logical_result_lanes(state,
                                                                   def_op);
  }
  const loom_vector_to_scalar_descriptor_t* descriptor =
      loom_vector_to_scalar_find_descriptor(def_op->kind);
  if (descriptor == NULL ||
      descriptor->lane_kind != LOOM_VECTOR_TO_SCALAR_LANE_GENERIC) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(def_op);
  for (uint8_t i = 0; i < descriptor->lane_operand_count; ++i) {
    if (!loom_vector_to_scalar_fragment_store_source_is_supported(
            state, operands[i])) {
      return false;
    }
  }
  return true;
}

static bool loom_vector_to_scalar_fragment_store_source_is_supported_root(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_fragment_store_source_is_supported(
      state, loom_vector_to_scalar_store_value(state));
}

static uint32_t loom_vector_to_scalar_fragment_store_source_rejection_bits(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value) {
  loom_type_t type = loom_module_value_type(state->rewriter->module, value);
  if (!loom_type_is_vector(type)) return LOOM_CONTRACT_REJECTION_NONE;

  loom_op_t* def_op =
      loom_vector_to_scalar_value_def_op(state->rewriter->module, value);
  if (def_op == NULL) {
    return LOOM_CONTRACT_REJECTION_FRAGMENT;
  }
  const loom_trait_flags_t traits =
      loom_op_effective_traits(state->rewriter->module, def_op);
  if (loom_traits_are_value_alias(traits)) {
    IREE_ASSERT(def_op->operand_count >= 1);
    return loom_vector_to_scalar_fragment_store_source_rejection_bits(
        state, loom_op_const_operands(def_op)[0]);
  }
  if (loom_vector_fragment_load_isa(def_op) || loom_vector_splat_isa(def_op) ||
      loom_vector_constant_isa(def_op) || loom_vector_poison_isa(def_op)) {
    return LOOM_CONTRACT_REJECTION_NONE;
  }
  if (loom_vector_mma_isa(def_op)) {
    loom_vector_to_scalar_state_t mma_state = *state;
    mma_state.op = def_op;
    mma_state.vector_type = loom_module_value_type(
        state->rewriter->module, loom_vector_mma_result(def_op));
    mma_state.result_scalar_type =
        loom_vector_to_scalar_lane_type(mma_state.vector_type);
    return loom_vector_to_scalar_mma_reference_rejection_bits(&mma_state);
  }
  const loom_vector_to_scalar_descriptor_t* descriptor =
      loom_vector_to_scalar_find_descriptor(def_op->kind);
  if (descriptor != NULL &&
      descriptor->lane_kind == LOOM_VECTOR_TO_SCALAR_LANE_GENERIC) {
    const loom_value_id_t* operands = loom_op_const_operands(def_op);
    for (uint8_t i = 0; i < descriptor->lane_operand_count; ++i) {
      uint32_t bits =
          loom_vector_to_scalar_fragment_store_source_rejection_bits(
              state, operands[i]);
      if (bits != LOOM_CONTRACT_REJECTION_NONE) return bits;
    }
    return LOOM_CONTRACT_REJECTION_NONE;
  }
  return LOOM_CONTRACT_REJECTION_CAPABILITY;
}

uint32_t loom_vector_to_scalar_fragment_store_reference_rejection_bits(
    loom_vector_to_scalar_state_t* state) {
  if (!loom_vector_fragment_store_isa(state->op)) {
    return LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
  }
  return loom_vector_to_scalar_fragment_store_source_rejection_bits(
      state, loom_vector_to_scalar_store_value(state));
}

static iree_status_t loom_vector_to_scalar_materialize_fragment_store_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  bool materialized = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_try_materialize_def_lane(
      state, loom_vector_to_scalar_store_value(state), state->vector_type,
      indices, &materialized, out_lane));
  IREE_ASSERT(materialized,
              "vector.fragment.store reference support must match lane "
              "materialization");
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_emit_fragment_store_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices) {
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_fragment_store_lane(
      state, indices, &lane));
  loom_vector_to_scalar_view_indices_t view_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_indices(
      state, indices, /*add_lane_indices=*/true, LOOM_VALUE_ID_INVALID,
      /*add_last_axis_offset=*/false, (loom_vector_to_scalar_index_term_t){0},
      &view_indices));
  loom_op_t* store_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy =
      loom_vector_to_scalar_memory_cache_policy(state);
  return loom_view_store_build(
      &state->rewriter->builder, cache_policy.build_flags, lane,
      loom_vector_to_scalar_memory_view(state), view_indices.dynamic_indices,
      view_indices.dynamic_index_count, view_indices.static_indices,
      view_indices.static_index_count, cache_policy.cache_scope,
      cache_policy.cache_temporal, state->location, &store_op);
}

static iree_status_t loom_vector_to_scalar_lower_fragment_store_columns(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t row) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_fragment_store_loop_bounds(
      state,
      loom_vector_to_scalar_value_term(
          state, loom_vector_fragment_store_columns(state->op)),
      &lower_bound, &upper_bound, &step));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, NULL, 0, NULL, 0, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_vector_to_scalar_index_term_t column =
      loom_vector_to_scalar_dynamic_term(
          loom_region_entry_arg_id(loom_scf_for_body(loop), 0));
  loom_vector_to_scalar_index_term_t terms[2] = {row, column};
  loom_vector_to_scalar_index_list_t indices = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_terms_to_index_list(state, terms, 2, &indices));
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_emit_fragment_store_lane(state, indices));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_lower_fragment_store(
    loom_vector_to_scalar_state_t* state, bool* out_handled) {
  *out_handled = false;
  if (!loom_vector_to_scalar_fragment_store_source_is_supported_root(state)) {
    return iree_ok_status();
  }

  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_fragment_store_loop_bounds(
      state,
      loom_vector_to_scalar_value_term(
          state, loom_vector_fragment_store_rows(state->op)),
      &lower_bound, &upper_bound, &step));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, NULL, 0, NULL, 0, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_vector_to_scalar_index_term_t row = loom_vector_to_scalar_dynamic_term(
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0));
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_fragment_store_columns(state, row));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);
  *out_handled = true;
  return iree_ok_status();
}

static bool loom_vector_to_scalar_physical_result_store_types_match(
    const loom_module_t* module, const loom_op_t* store_op,
    uint16_t register_count, loom_type_t* out_source_lane_type,
    loom_type_t* out_store_lane_type) {
  *out_source_lane_type = loom_type_none();
  *out_store_lane_type = loom_type_none();
  const loom_type_t source_type = loom_module_value_type(
      module, loom_vector_fragment_store_value(store_op));
  const loom_type_t view_type =
      loom_module_value_type(module, loom_vector_fragment_store_view(store_op));
  if (!loom_type_is_vector(source_type) || loom_type_rank(source_type) != 1 ||
      !loom_type_is_all_static(source_type)) {
    return false;
  }
  uint64_t lane_count = 0;
  if (!loom_type_static_element_count(source_type, &lane_count) ||
      lane_count != register_count) {
    return false;
  }
  const loom_type_t source_lane_type =
      loom_vector_to_scalar_lane_type(source_type);
  const loom_type_t store_lane_type =
      loom_type_scalar(loom_type_element_type(view_type));
  if (!loom_type_is_scalar(source_lane_type) ||
      !loom_type_is_scalar(store_lane_type)) {
    return false;
  }
  if (!loom_type_equal(source_lane_type, store_lane_type)) {
    const loom_scalar_type_t source_scalar_type =
        loom_type_element_type(source_lane_type);
    const loom_scalar_type_t store_scalar_type =
        loom_type_element_type(store_lane_type);
    const int32_t source_width = loom_scalar_type_bitwidth(source_scalar_type);
    const int32_t store_width = loom_scalar_type_bitwidth(store_scalar_type);
    if (source_width <= 0 || store_width <= 0) {
      return false;
    }
    const bool float_cast = loom_scalar_type_is_float(source_scalar_type) &&
                            loom_scalar_type_is_float(store_scalar_type) &&
                            source_width != store_width;
    const bool integer_truncation =
        loom_scalar_type_is_integer(source_scalar_type) &&
        loom_scalar_type_is_integer(store_scalar_type) &&
        source_width > store_width;
    if (!float_cast && !integer_truncation) {
      return false;
    }
  }
  *out_source_lane_type = source_lane_type;
  *out_store_lane_type = store_lane_type;
  return true;
}

static iree_status_t loom_vector_to_scalar_build_physical_result_store_indices(
    loom_vector_to_scalar_state_t* state, loom_value_id_t register_index,
    loom_vector_to_scalar_index_term_t lane_id,
    loom_vector_to_scalar_index_list_t* out_indices) {
  const loom_vector_to_scalar_index_term_t register_term =
      loom_vector_to_scalar_dynamic_term(register_index);

  loom_vector_to_scalar_index_term_t logical_terms[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_result_fragment_coordinate_terms(
          state, lane_id, register_term, &logical_terms[0], &logical_terms[1]));

  loom_vector_to_scalar_index_term_t* view_terms = NULL;
  uint8_t view_term_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_from_explicit_indices(
      state, loom_vector_fragment_store_static_indices(state->op),
      loom_vector_fragment_store_indices(state->op), &view_terms,
      &view_term_count));
  if (view_term_count != IREE_ARRAYSIZE(logical_terms)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "physical result fragment store scalarization selected non-rank-2 "
        "indices");
  }
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(logical_terms); ++i) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, view_terms[i],
        logical_terms[i], &view_terms[i]));
  }
  return loom_vector_to_scalar_terms_to_index_list(
      state, view_terms, view_term_count, out_indices);
}

static iree_status_t loom_vector_to_scalar_emit_physical_result_store_lane(
    loom_vector_to_scalar_state_t* state, loom_type_t source_lane_type,
    loom_type_t store_lane_type, loom_value_id_t register_index,
    loom_vector_to_scalar_index_term_t lane_id) {
  const loom_vector_to_scalar_index_list_t physical_indices = {
      .dynamic_indices = &register_index,
      .rank = 1,
  };
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_fragment_store_value(state->op), physical_indices,
      &lane));

  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_numeric_lane(
      state, lane, source_lane_type, store_lane_type,
      /*unsigned_input=*/false, &lane));

  loom_vector_to_scalar_index_list_t logical_indices = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_physical_result_store_indices(
          state, register_index, lane_id, &logical_indices));
  loom_vector_to_scalar_view_indices_t view_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_index_list_to_view_indices(
      state, logical_indices, &view_indices));
  loom_op_t* store_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy =
      loom_vector_to_scalar_memory_cache_policy(state);
  return loom_view_store_build(
      &state->rewriter->builder, cache_policy.build_flags, lane,
      loom_vector_fragment_store_view(state->op), view_indices.dynamic_indices,
      view_indices.dynamic_index_count, view_indices.static_indices,
      view_indices.static_index_count, cache_policy.cache_scope,
      cache_policy.cache_temporal, state->location, &store_op);
}

iree_status_t
loom_vector_fragment_store_to_scalar_physical_result_loop_rewrite_ops(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* const* ops,
    iree_host_size_t op_count,
    const loom_matrix_fragment_layout_t* matrix_fragment_layout,
    uint16_t register_count, bool* out_rewritten) {
  *out_rewritten = false;
  if (op_count == 0 ||
      !loom_vector_to_scalar_result_fragment_layout_is_supported(
          matrix_fragment_layout)) {
    return iree_ok_status();
  }

  loom_type_t local_source_lane_types[8] = {0};
  loom_type_t local_store_lane_types[8] = {0};
  loom_type_t* source_lane_types = local_source_lane_types;
  loom_type_t* store_lane_types = local_store_lane_types;
  if (op_count > IREE_ARRAYSIZE(local_source_lane_types)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->builder.arena, op_count, sizeof(*source_lane_types),
        (void**)&source_lane_types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->builder.arena, op_count, sizeof(*store_lane_types),
        (void**)&store_lane_types));
  }
  for (iree_host_size_t i = 0; i < op_count; ++i) {
    if (!loom_vector_fragment_store_isa(ops[i]) ||
        !loom_vector_to_scalar_physical_result_store_types_match(
            rewriter->module, ops[i], register_count, &source_lane_types[i],
            &store_lane_types[i])) {
      return iree_ok_status();
    }
  }

  loom_op_t* first_op = ops[0];
  loom_builder_set_before(&rewriter->builder, first_op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = first_op,
      .value_checkpoint = value_checkpoint,
      .vector_type = loom_module_value_type(
          rewriter->module, loom_vector_fragment_store_value(first_op)),
      .result_scalar_type = source_lane_types[0],
      .matrix_fragment_layout = matrix_fragment_layout,
      .location = first_op->location,
  };
  loom_vector_to_scalar_state_bind_statistics(&state, pass);

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      first_op->location, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      first_op->location, register_count, &upper_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      first_op->location, 1, &step));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &rewriter->builder, /*build_flags=*/0, zero, upper_bound, step, NULL, 0,
      NULL, 0, NULL, 0, LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0,
      /*unroll_schedule=*/0, first_op->location, &loop));
  loom_vector_to_scalar_record_loop_created(&state);

  loom_builder_ip_t saved = loom_builder_enter_region(&rewriter->builder, loop,
                                                      loom_scf_for_body(loop));
  loom_op_t* lane_id_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_subgroup_lane_id_build(
      &rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      first_op->location, &lane_id_op));
  const loom_vector_to_scalar_index_term_t lane_id =
      loom_vector_to_scalar_dynamic_term(
          loom_kernel_subgroup_lane_id_result(lane_id_op));
  const loom_value_id_t register_index =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  for (iree_host_size_t i = 0; i < op_count; ++i) {
    state.op = ops[i];
    state.vector_type = loom_module_value_type(
        rewriter->module, loom_vector_fragment_store_value(ops[i]));
    state.result_scalar_type = source_lane_types[i];
    state.location = ops[i]->location;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_emit_physical_result_store_lane(
        &state, source_lane_types[i], store_lane_types[i], register_index,
        lane_id));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder, NULL, 0,
                                            state.location, &yield_op));
  loom_builder_restore(&rewriter->builder, saved);

  for (iree_host_size_t i = op_count; i > 0; --i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, ops[i - 1]));
  }
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t
loom_vector_fragment_store_to_scalar_physical_result_loop_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_matrix_fragment_layout_t* matrix_fragment_layout,
    uint16_t register_count, bool* out_rewritten) {
  loom_op_t* ops[1] = {op};
  return loom_vector_fragment_store_to_scalar_physical_result_loop_rewrite_ops(
      pass, rewriter, ops, IREE_ARRAYSIZE(ops), matrix_fragment_layout,
      register_count, out_rewritten);
}

static iree_status_t loom_vector_to_scalar_lower_static_store_compress(
    loom_vector_to_scalar_state_t* state) {
  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      state, state->vector_type, &element_count));
  if (loom_pass_has_error_diagnostics(state->pass)) return iree_ok_status();
  for (uint16_t ordinal = 0; ordinal < element_count; ++ordinal) {
    int64_t index = (int64_t)ordinal;
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = &index,
        .rank = 1,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_emit_store_compress_lane(state, index_list));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_dynamic_store_compress(
    loom_vector_to_scalar_state_t* state) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, index_type, state->location, 0, &lower_bound));
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, index_type, state->location, 1, &step));
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      state, state->vector_type, 0, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, NULL, 0, NULL, 0, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_value_id_t lane_index =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_vector_to_scalar_index_list_t index_list = {
      .dynamic_indices = &lane_index,
      .rank = 1,
  };
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_emit_store_compress_lane(state, index_list));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_lower_memory_store_compress(
    loom_vector_to_scalar_state_t* state) {
  if (loom_type_is_all_static(state->vector_type)) {
    return loom_vector_to_scalar_lower_static_store_compress(state);
  }
  return loom_vector_to_scalar_lower_dynamic_store_compress(state);
}

//===----------------------------------------------------------------------===//
// Atomic reduction lanes
//===----------------------------------------------------------------------===//

static bool loom_vector_to_scalar_atomic_reduce_is_masked(
    loom_vector_to_scalar_state_t* state) {
  loom_memory_access_t access = loom_vector_to_scalar_memory_access(state);
  return loom_memory_access_mask(access) != LOOM_VALUE_ID_INVALID;
}

static loom_value_id_t loom_vector_to_scalar_atomic_reduce_value(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_value(loom_vector_to_scalar_memory_access(state));
}

static loom_value_id_t loom_vector_to_scalar_atomic_reduce_offsets(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_offsets(loom_vector_to_scalar_memory_access(state));
}

static loom_value_id_t loom_vector_to_scalar_atomic_reduce_mask(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_mask(loom_vector_to_scalar_memory_access(state));
}

static uint8_t loom_vector_to_scalar_atomic_reduce_kind(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_enum_attr_value(loom_memory_access_atomic_kind(
      loom_vector_to_scalar_memory_access(state)));
}

static uint8_t loom_vector_to_scalar_atomic_reduce_ordering(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_enum_attr_value(
      loom_memory_access_atomic_ordering(
          loom_vector_to_scalar_memory_access(state)));
}

static uint8_t loom_vector_to_scalar_atomic_reduce_scope(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_enum_attr_value(loom_memory_access_atomic_scope(
      loom_vector_to_scalar_memory_access(state)));
}

static iree_status_t loom_vector_to_scalar_emit_view_atomic_reduce_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices) {
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_to_scalar_atomic_reduce_value(state), lane_indices,
      &lane));
  loom_vector_to_scalar_view_indices_t view_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_indices(
      state, lane_indices, /*add_lane_indices=*/false,
      loom_vector_to_scalar_atomic_reduce_offsets(state),
      /*add_last_axis_offset=*/false, (loom_vector_to_scalar_index_term_t){0},
      &view_indices));
  loom_op_t* atomic_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy =
      loom_vector_to_scalar_memory_cache_policy(state);
  return loom_view_atomic_reduce_build(
      &state->rewriter->builder, cache_policy.build_flags,
      loom_vector_to_scalar_atomic_reduce_kind(state), lane,
      loom_vector_to_scalar_memory_view(state), view_indices.dynamic_indices,
      view_indices.dynamic_index_count, view_indices.static_indices,
      view_indices.static_index_count,
      loom_vector_to_scalar_atomic_reduce_ordering(state),
      loom_vector_to_scalar_atomic_reduce_scope(state),
      cache_policy.cache_scope, cache_policy.cache_temporal, state->location,
      &atomic_op);
}

static iree_status_t loom_vector_to_scalar_emit_masked_view_atomic_reduce_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices) {
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_to_scalar_atomic_reduce_mask(state), lane_indices,
      &condition));

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      &state->rewriter->builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
      condition, NULL, 0, NULL, 0, state->location, &if_op));

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_emit_view_atomic_reduce_lane(state, lane_indices));
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &then_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  saved = loom_builder_enter_region(&state->rewriter->builder, if_op,
                                    loom_scf_if_else_region(if_op));
  loom_op_t* else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &else_yield));
  loom_builder_restore(&state->rewriter->builder, saved);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_emit_atomic_reduce_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices) {
  if (loom_vector_to_scalar_atomic_reduce_is_masked(state)) {
    return loom_vector_to_scalar_emit_masked_view_atomic_reduce_lane(state,
                                                                     indices);
  }
  return loom_vector_to_scalar_emit_view_atomic_reduce_lane(state, indices);
}

static iree_status_t loom_vector_to_scalar_lower_static_atomic_reduce(
    loom_vector_to_scalar_state_t* state) {
  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      state, state->vector_type, &element_count));
  if (loom_pass_has_error_diagnostics(state->pass)) return iree_ok_status();
  uint8_t rank = loom_type_rank(state->vector_type);
  int64_t* indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, rank, sizeof(int64_t), (void**)&indices));
  }
  for (uint16_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(state->vector_type,
                                               (int64_t)ordinal, indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = indices,
        .rank = rank,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_emit_atomic_reduce_lane(state, index_list));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_atomic_reduce_loop_axis(
    loom_vector_to_scalar_state_t* state, uint8_t axis,
    loom_value_id_t* dynamic_indices) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      state, state->vector_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, NULL, 0, NULL, 0, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  if (axis + 1 == loom_type_rank(state->vector_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->vector_type),
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_emit_atomic_reduce_lane(state, index_list));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_lower_atomic_reduce_loop_axis(
        state, (uint8_t)(axis + 1), dynamic_indices));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder, NULL, 0,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_dynamic_atomic_reduce(
    loom_vector_to_scalar_state_t* state) {
  uint8_t rank = loom_type_rank(state->vector_type);
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena, rank,
                                                 sizeof(loom_value_id_t),
                                                 (void**)&dynamic_indices));
  return loom_vector_to_scalar_lower_atomic_reduce_loop_axis(state, 0,
                                                             dynamic_indices);
}

iree_status_t loom_vector_to_scalar_lower_memory_atomic_reduce(
    loom_vector_to_scalar_state_t* state) {
  if (loom_type_is_all_static(state->vector_type)) {
    return loom_vector_to_scalar_lower_static_atomic_reduce(state);
  }
  return loom_vector_to_scalar_lower_dynamic_atomic_reduce(state);
}

//===----------------------------------------------------------------------===//
// Atomic RMW lanes
//===----------------------------------------------------------------------===//

static bool loom_vector_to_scalar_atomic_rmw_is_masked(
    loom_vector_to_scalar_state_t* state) {
  loom_memory_access_t access = loom_vector_to_scalar_memory_access(state);
  return loom_memory_access_mask(access) != LOOM_VALUE_ID_INVALID;
}

static loom_value_id_t loom_vector_to_scalar_atomic_rmw_value(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_value(loom_vector_to_scalar_memory_access(state));
}

static loom_value_id_t loom_vector_to_scalar_atomic_rmw_offsets(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_offsets(loom_vector_to_scalar_memory_access(state));
}

static loom_value_id_t loom_vector_to_scalar_atomic_rmw_mask(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_mask(loom_vector_to_scalar_memory_access(state));
}

static loom_value_id_t loom_vector_to_scalar_atomic_rmw_passthrough(
    loom_vector_to_scalar_state_t* state) {
  return loom_memory_access_passthrough(
      loom_vector_to_scalar_memory_access(state));
}

static uint8_t loom_vector_to_scalar_atomic_rmw_kind(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_enum_attr_value(loom_memory_access_atomic_kind(
      loom_vector_to_scalar_memory_access(state)));
}

static uint8_t loom_vector_to_scalar_atomic_rmw_ordering(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_enum_attr_value(
      loom_memory_access_atomic_ordering(
          loom_vector_to_scalar_memory_access(state)));
}

static uint8_t loom_vector_to_scalar_atomic_rmw_scope(
    loom_vector_to_scalar_state_t* state) {
  return loom_vector_to_scalar_enum_attr_value(loom_memory_access_atomic_scope(
      loom_vector_to_scalar_memory_access(state)));
}

static iree_status_t loom_vector_to_scalar_build_view_atomic_rmw_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices,
    loom_value_id_t* out_lane) {
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_to_scalar_atomic_rmw_value(state), lane_indices,
      &lane));
  loom_vector_to_scalar_view_indices_t view_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_indices(
      state, lane_indices, /*add_lane_indices=*/false,
      loom_vector_to_scalar_atomic_rmw_offsets(state),
      /*add_last_axis_offset=*/false, (loom_vector_to_scalar_index_term_t){0},
      &view_indices));
  loom_op_t* atomic_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy =
      loom_vector_to_scalar_memory_cache_policy(state);
  IREE_RETURN_IF_ERROR(loom_view_atomic_rmw_build(
      &state->rewriter->builder, cache_policy.build_flags,
      loom_vector_to_scalar_atomic_rmw_kind(state), lane,
      loom_vector_to_scalar_memory_view(state), view_indices.dynamic_indices,
      view_indices.dynamic_index_count, view_indices.static_indices,
      view_indices.static_index_count,
      loom_vector_to_scalar_atomic_rmw_ordering(state),
      loom_vector_to_scalar_atomic_rmw_scope(state), cache_policy.cache_scope,
      cache_policy.cache_temporal, state->result_scalar_type, state->location,
      &atomic_op));
  *out_lane = loom_view_atomic_rmw_result(atomic_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_masked_view_atomic_rmw_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices,
    loom_value_id_t* out_lane) {
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_to_scalar_atomic_rmw_mask(state), lane_indices,
      &condition));

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(&state->rewriter->builder,
                                         LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                                         condition, &state->result_scalar_type,
                                         1, NULL, 0, state->location, &if_op));

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  loom_value_id_t old_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_atomic_rmw_lane(
      state, lane_indices, &old_lane));
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->rewriter->builder, &old_lane, 1, state->location, &then_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  saved = loom_builder_enter_region(&state->rewriter->builder, if_op,
                                    loom_scf_if_else_region(if_op));
  loom_value_id_t passthrough_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_to_scalar_atomic_rmw_passthrough(state), lane_indices,
      &passthrough_lane));
  loom_op_t* else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &passthrough_lane, 1,
                                            state->location, &else_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_lane = loom_scf_if_results(if_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_atomic_rmw_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  if (loom_vector_to_scalar_atomic_rmw_is_masked(state)) {
    return loom_vector_to_scalar_build_masked_view_atomic_rmw_lane(
        state, indices, out_lane);
  }
  return loom_vector_to_scalar_build_view_atomic_rmw_lane(state, indices,
                                                          out_lane);
}

static iree_status_t loom_vector_to_scalar_lower_static_atomic_rmw(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      state, state->vector_type, &element_count));
  if (loom_pass_has_error_diagnostics(state->pass)) return iree_ok_status();
  loom_value_id_t* elements = NULL;
  if (element_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->rewriter->arena, element_count,
                                  sizeof(loom_value_id_t), (void**)&elements));
  }
  uint8_t rank = loom_type_rank(state->vector_type);
  int64_t* indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, rank, sizeof(int64_t), (void**)&indices));
  }
  for (uint16_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(state->vector_type,
                                               (int64_t)ordinal, indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = indices,
        .rank = rank,
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_atomic_rmw_lane(
        state, index_list, &elements[ordinal]));
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &state->rewriter->builder, elements, element_count, state->vector_type,
      state->location, &from_elements_op));
  *out_replacement = loom_vector_from_elements_result(from_elements_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_atomic_rmw_dynamic_seed(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_seed) {
  if (loom_vector_to_scalar_atomic_rmw_is_masked(state)) {
    *out_seed = loom_vector_to_scalar_atomic_rmw_passthrough(state);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_vector_zero(state, state->vector_type,
                                                 out_seed);
}

static iree_status_t loom_vector_to_scalar_atomic_rmw_loop_axis(
    loom_vector_to_scalar_state_t* state, uint8_t axis,
    loom_value_id_t current_aggregate, loom_value_id_t* dynamic_indices,
    loom_value_id_t* out_aggregate) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      state, state->vector_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, &current_aggregate, 1, &state->vector_type, 1, NULL, 0,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0, /*unroll_schedule=*/0,
      state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t aggregate_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_aggregate = LOOM_VALUE_ID_INVALID;
  if (axis + 1 == loom_type_rank(state->vector_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->vector_type),
    };
    loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_atomic_rmw_lane(state, index_list, &lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_insert_lane(
        state, lane, aggregate_arg, state->vector_type, index_list,
        &yielded_aggregate));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_atomic_rmw_loop_axis(
        state, (uint8_t)(axis + 1), aggregate_arg, dynamic_indices,
        &yielded_aggregate));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &yielded_aggregate, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_aggregate = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_dynamic_atomic_rmw(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_value_id_t seed = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_atomic_rmw_dynamic_seed(state, &seed));

  uint8_t rank = loom_type_rank(state->vector_type);
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena, rank,
                                                 sizeof(loom_value_id_t),
                                                 (void**)&dynamic_indices));
  return loom_vector_to_scalar_atomic_rmw_loop_axis(
      state, 0, seed, dynamic_indices, out_replacement);
}

iree_status_t loom_vector_to_scalar_lower_memory_atomic_rmw(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  if (loom_type_is_all_static(state->vector_type)) {
    return loom_vector_to_scalar_lower_static_atomic_rmw(state,
                                                         out_replacement);
  }
  return loom_vector_to_scalar_lower_dynamic_atomic_rmw(state, out_replacement);
}

//===----------------------------------------------------------------------===//
// Atomic compare-exchange lanes
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_build_view_atomic_cmpxchg_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t lane_indices,
    loom_value_id_t* out_lane) {
  loom_memory_access_t access = loom_vector_to_scalar_memory_access(state);
  loom_value_id_t expected_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_memory_access_expected(access), lane_indices,
      &expected_lane));
  loom_value_id_t replacement_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_memory_access_replacement(access), lane_indices,
      &replacement_lane));
  loom_vector_to_scalar_view_indices_t view_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_indices(
      state, lane_indices, /*add_lane_indices=*/false,
      loom_memory_access_offsets(access),
      /*add_last_axis_offset=*/false, (loom_vector_to_scalar_index_term_t){0},
      &view_indices));
  loom_op_t* atomic_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy =
      loom_vector_to_scalar_memory_cache_policy(state);
  IREE_RETURN_IF_ERROR(loom_view_atomic_cmpxchg_build(
      &state->rewriter->builder, cache_policy.build_flags, expected_lane,
      replacement_lane, loom_vector_to_scalar_memory_view(state),
      view_indices.dynamic_indices, view_indices.dynamic_index_count,
      view_indices.static_indices, view_indices.static_index_count,
      loom_vector_to_scalar_enum_attr_value(
          loom_memory_access_atomic_success_ordering(access)),
      loom_vector_to_scalar_enum_attr_value(
          loom_memory_access_atomic_failure_ordering(access)),
      loom_vector_to_scalar_enum_attr_value(
          loom_memory_access_atomic_scope(access)),
      cache_policy.cache_scope, cache_policy.cache_temporal,
      state->result_scalar_type, state->location, &atomic_op));
  *out_lane = loom_view_atomic_cmpxchg_old(atomic_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_static_atomic_cmpxchg(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      state, state->vector_type, &element_count));
  if (loom_pass_has_error_diagnostics(state->pass)) return iree_ok_status();
  loom_value_id_t* elements = NULL;
  if (element_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->rewriter->arena, element_count,
                                  sizeof(loom_value_id_t), (void**)&elements));
  }
  uint8_t rank = loom_type_rank(state->vector_type);
  int64_t* indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, rank, sizeof(int64_t), (void**)&indices));
  }
  for (uint16_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(state->vector_type,
                                               (int64_t)ordinal, indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = indices,
        .rank = rank,
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_atomic_cmpxchg_lane(
        state, index_list, &elements[ordinal]));
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &state->rewriter->builder, elements, element_count, state->vector_type,
      state->location, &from_elements_op));
  *out_replacement = loom_vector_from_elements_result(from_elements_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_atomic_cmpxchg_loop_axis(
    loom_vector_to_scalar_state_t* state, uint8_t axis,
    loom_value_id_t current_aggregate, loom_value_id_t* dynamic_indices,
    loom_value_id_t* out_aggregate) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      state, state->vector_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, &current_aggregate, 1, &state->vector_type, 1, NULL, 0,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0, /*unroll_schedule=*/0,
      state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t aggregate_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_aggregate = LOOM_VALUE_ID_INVALID;
  if (axis + 1 == loom_type_rank(state->vector_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->vector_type),
    };
    loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_view_atomic_cmpxchg_lane(
        state, index_list, &lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_insert_lane(
        state, lane, aggregate_arg, state->vector_type, index_list,
        &yielded_aggregate));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_atomic_cmpxchg_loop_axis(
        state, (uint8_t)(axis + 1), aggregate_arg, dynamic_indices,
        &yielded_aggregate));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &yielded_aggregate, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_aggregate = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_dynamic_atomic_cmpxchg(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_value_id_t seed = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_vector_zero(
      state, state->vector_type, &seed));

  uint8_t rank = loom_type_rank(state->vector_type);
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena, rank,
                                                 sizeof(loom_value_id_t),
                                                 (void**)&dynamic_indices));
  return loom_vector_to_scalar_atomic_cmpxchg_loop_axis(
      state, 0, seed, dynamic_indices, out_replacement);
}

iree_status_t loom_vector_to_scalar_lower_memory_atomic_cmpxchg(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  if (loom_type_is_all_static(state->vector_type)) {
    return loom_vector_to_scalar_lower_static_atomic_cmpxchg(state,
                                                             out_replacement);
  }
  return loom_vector_to_scalar_lower_dynamic_atomic_cmpxchg(state,
                                                            out_replacement);
}
