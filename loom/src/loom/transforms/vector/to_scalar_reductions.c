// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_reductions.h"

#include "loom/ir/module.h"
#include "loom/ops/combining.h"
#include "loom/ops/scalar/combining.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"

//===----------------------------------------------------------------------===//
// Reduction lowering
//===----------------------------------------------------------------------===//

typedef struct loom_vector_to_scalar_accumulator_state_t {
  loom_vector_to_scalar_state_t lane_state;
  loom_value_id_t input;
  loom_value_id_t rhs;
  loom_value_id_t init;
  loom_op_kind_t scalar_kind;
  uint8_t instance_flags;
  loom_value_id_t fused_product_lhs;
  loom_value_id_t fused_product_rhs;
  uint8_t product_flags;
  uint8_t fmaf_flags;
  bool use_fmaf;
  bool use_product_fmaf_forest;
} loom_vector_to_scalar_accumulator_state_t;

static iree_status_t loom_vector_to_scalar_build_accumulator_lane(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t accumulator,
    loom_value_id_t* out_next) {
  loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      &state->lane_state, state->input, indices, &lhs_lane));
  if (state->use_fmaf) {
    loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        &state->lane_state, state->rhs, indices, &rhs_lane));
    return loom_vector_to_scalar_build_generic_lane_op(
        &state->lane_state, LOOM_OP_SCALAR_FMAF, state->instance_flags,
        (loom_value_id_t[]){lhs_lane, rhs_lane, accumulator}, 3, NULL, 0,
        state->lane_state.result_scalar_type, out_next);
  }
  return loom_vector_to_scalar_build_generic_lane_op(
      &state->lane_state, state->scalar_kind, state->instance_flags,
      (loom_value_id_t[]){accumulator, lhs_lane}, 2, NULL, 0,
      state->lane_state.result_scalar_type, out_next);
}

static bool loom_vector_to_scalar_accumulator_can_reassociate(
    const loom_vector_to_scalar_accumulator_state_t* state) {
  return !state->use_fmaf &&
         iree_any_bit_set(state->instance_flags,
                          LOOM_SCALAR_FASTMATHFLAGS_REASSOC);
}

static iree_status_t loom_vector_to_scalar_combine_accumulator_terms(
    loom_vector_to_scalar_accumulator_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_generic_lane_op(
      &state->lane_state, state->scalar_kind, state->instance_flags,
      (loom_value_id_t[]){lhs, rhs}, 2, NULL, 0,
      state->lane_state.result_scalar_type, out_result);
}

static iree_status_t loom_vector_to_scalar_product_terms_from_ordinal(
    loom_vector_to_scalar_accumulator_state_t* state, uint16_t ordinal,
    int64_t* indices, loom_value_id_t* out_lhs, loom_value_id_t* out_rhs) {
  loom_vector_to_scalar_indices_from_ordinal(state->lane_state.vector_type,
                                             ordinal, indices);
  loom_vector_to_scalar_index_list_t index_list = {
      .static_indices = indices,
      .rank = loom_type_rank(state->lane_state.vector_type),
  };
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      &state->lane_state, state->fused_product_lhs, index_list, out_lhs));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      &state->lane_state, state->fused_product_rhs, index_list, out_rhs));
  loom_vector_to_scalar_record_lane_materialized(&state->lane_state);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_product_seed(
    loom_vector_to_scalar_accumulator_state_t* state, uint16_t ordinal,
    int64_t* indices, loom_value_id_t* out_product) {
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_product_terms_from_ordinal(
      state, ordinal, indices, &lhs, &rhs));
  return loom_vector_to_scalar_build_generic_lane_op(
      &state->lane_state, LOOM_OP_SCALAR_MULF, state->product_flags,
      (loom_value_id_t[]){lhs, rhs}, 2, NULL, 0,
      state->lane_state.result_scalar_type, out_product);
}

static iree_status_t loom_vector_to_scalar_build_product_fmaf(
    loom_vector_to_scalar_accumulator_state_t* state, uint16_t ordinal,
    int64_t* indices, loom_value_id_t accumulator,
    loom_value_id_t* out_result) {
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_product_terms_from_ordinal(
      state, ordinal, indices, &lhs, &rhs));
  return loom_vector_to_scalar_build_generic_lane_op(
      &state->lane_state, LOOM_OP_SCALAR_FMAF, state->fmaf_flags,
      (loom_value_id_t[]){lhs, rhs, accumulator}, 3, NULL, 0,
      state->lane_state.result_scalar_type, out_result);
}

static iree_status_t loom_vector_to_scalar_combine_accumulator_term_tree(
    loom_vector_to_scalar_accumulator_state_t* state,
    const loom_value_id_t* terms, uint16_t begin_ordinal, uint16_t end_ordinal,
    loom_value_id_t* out_result) {
  if (end_ordinal <= begin_ordinal) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "invalid empty accumulator term range");
  }
  if (end_ordinal - begin_ordinal == 1) {
    *out_result = terms[begin_ordinal];
    return iree_ok_status();
  }

  uint16_t middle_ordinal =
      (uint16_t)(begin_ordinal + (end_ordinal - begin_ordinal) / 2);
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_combine_accumulator_term_tree(
      state, terms, begin_ordinal, middle_ordinal, &lhs));
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_combine_accumulator_term_tree(
      state, terms, middle_ordinal, end_ordinal, &rhs));
  return loom_vector_to_scalar_combine_accumulator_terms(state, lhs, rhs,
                                                         out_result);
}

static iree_status_t loom_vector_to_scalar_lower_static_product_fmaf_forest(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      &state->lane_state, state->lane_state.vector_type, &element_count));
  if (loom_pass_has_error_diagnostics(state->lane_state.pass)) {
    return iree_ok_status();
  }
  if (element_count == 0) {
    *out_replacement = state->init;
    return iree_ok_status();
  }

  uint8_t rank = loom_type_rank(state->lane_state.vector_type);
  int64_t* indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->lane_state.rewriter->arena, rank,
                                  sizeof(int64_t), (void**)&indices));
  }

  uint16_t accumulator_count = element_count < 4 ? element_count : 4;
  loom_value_id_t* accumulators = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->lane_state.rewriter->arena, accumulator_count,
      sizeof(loom_value_id_t), (void**)&accumulators));

  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_product_fmaf(
      state, 0, indices, state->init, &accumulators[0]));
  for (uint16_t accumulator = 1; accumulator < accumulator_count;
       ++accumulator) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_product_seed(
        state, accumulator, indices, &accumulators[accumulator]));
  }

  for (uint16_t ordinal = accumulator_count; ordinal < element_count;
       ++ordinal) {
    uint16_t accumulator = (uint16_t)(ordinal % accumulator_count);
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_product_fmaf(
        state, ordinal, indices, accumulators[accumulator],
        &accumulators[accumulator]));
  }

  return loom_vector_to_scalar_combine_accumulator_term_tree(
      state, accumulators, 0, accumulator_count, out_replacement);
}

static iree_status_t loom_vector_to_scalar_build_static_reduction_tree(
    loom_vector_to_scalar_accumulator_state_t* state, uint16_t begin_ordinal,
    uint16_t end_ordinal, int64_t* indices, loom_value_id_t* out_result) {
  if (end_ordinal <= begin_ordinal) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "invalid empty reduction term range");
  }
  if (end_ordinal - begin_ordinal == 1) {
    loom_vector_to_scalar_indices_from_ordinal(state->lane_state.vector_type,
                                               begin_ordinal, indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = indices,
        .rank = loom_type_rank(state->lane_state.vector_type),
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        &state->lane_state, state->input, index_list, out_result));
    loom_vector_to_scalar_record_lane_materialized(&state->lane_state);
    return iree_ok_status();
  }

  uint16_t middle_ordinal =
      (uint16_t)(begin_ordinal + (end_ordinal - begin_ordinal) / 2);
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_static_reduction_tree(
      state, begin_ordinal, middle_ordinal, indices, &lhs));
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_static_reduction_tree(
      state, middle_ordinal, end_ordinal, indices, &rhs));
  return loom_vector_to_scalar_combine_accumulator_terms(state, lhs, rhs,
                                                         out_result);
}

static iree_status_t
loom_vector_to_scalar_lower_static_reassociated_accumulator(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      &state->lane_state, state->lane_state.vector_type, &element_count));
  if (loom_pass_has_error_diagnostics(state->lane_state.pass)) {
    return iree_ok_status();
  }
  if (element_count == 0) {
    *out_replacement = state->init;
    return iree_ok_status();
  }

  uint8_t rank = loom_type_rank(state->lane_state.vector_type);
  int64_t* indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->lane_state.rewriter->arena, rank,
                                  sizeof(int64_t), (void**)&indices));
  }

  loom_value_id_t tree = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_static_reduction_tree(
      state, 0, element_count, indices, &tree));
  return loom_vector_to_scalar_combine_accumulator_terms(state, state->init,
                                                         tree, out_replacement);
}

static iree_status_t loom_vector_to_scalar_lower_static_accumulator(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  if (state->use_product_fmaf_forest) {
    return loom_vector_to_scalar_lower_static_product_fmaf_forest(
        state, out_replacement);
  }
  if (loom_vector_to_scalar_accumulator_can_reassociate(state)) {
    return loom_vector_to_scalar_lower_static_reassociated_accumulator(
        state, out_replacement);
  }

  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      &state->lane_state, state->lane_state.vector_type, &element_count));
  if (loom_pass_has_error_diagnostics(state->lane_state.pass)) {
    return iree_ok_status();
  }
  loom_value_id_t accumulator = state->init;
  uint8_t rank = loom_type_rank(state->lane_state.vector_type);
  int64_t* indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->lane_state.rewriter->arena, rank,
                                  sizeof(int64_t), (void**)&indices));
  }
  for (uint16_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(state->lane_state.vector_type,
                                               (int64_t)ordinal, indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = indices,
        .rank = rank,
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_accumulator_lane(
        state, index_list, accumulator, &accumulator));
    loom_vector_to_scalar_record_lane_materialized(&state->lane_state);
  }
  *out_replacement = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_accumulator_loop_axis(
    loom_vector_to_scalar_accumulator_state_t* state, uint8_t axis,
    loom_value_id_t current_accumulator, loom_value_id_t* dynamic_indices,
    loom_value_id_t* out_accumulator) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 0,
      &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 1,
      &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      &state->lane_state, state->lane_state.vector_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->lane_state.rewriter->builder, /*build_flags=*/0, lower_bound,
      upper_bound, step, &current_accumulator, 1,
      &state->lane_state.result_scalar_type, 1, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, state->lane_state.location,
      &loop));
  loom_vector_to_scalar_record_loop_created(&state->lane_state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->lane_state.rewriter->builder, loop, loom_scf_for_body(loop));
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t accumulator_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_accumulator = LOOM_VALUE_ID_INVALID;
  if (axis + 1 == loom_type_rank(state->lane_state.vector_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->lane_state.vector_type),
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_accumulator_lane(
        state, index_list, accumulator_arg, &yielded_accumulator));
    loom_vector_to_scalar_record_lane_materialized(&state->lane_state);
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_accumulator_loop_axis(
        state, (uint8_t)(axis + 1), accumulator_arg, dynamic_indices,
        &yielded_accumulator));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->lane_state.rewriter->builder, &yielded_accumulator, 1,
      state->lane_state.location, &yield_op));
  loom_builder_restore(&state->lane_state.rewriter->builder, saved);

  *out_accumulator = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_dynamic_accumulator(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  uint8_t rank = loom_type_rank(state->lane_state.vector_type);
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->lane_state.rewriter->arena, rank, sizeof(loom_value_id_t),
      (void**)&dynamic_indices));
  return loom_vector_to_scalar_accumulator_loop_axis(
      state, 0, state->init, dynamic_indices, out_replacement);
}

static iree_status_t loom_vector_to_scalar_lower_accumulator(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  if (loom_type_is_all_static(state->lane_state.vector_type)) {
    return loom_vector_to_scalar_lower_static_accumulator(state,
                                                          out_replacement);
  }
  return loom_vector_to_scalar_lower_dynamic_accumulator(state,
                                                         out_replacement);
}

static bool loom_vector_to_scalar_try_configure_product_fmaf_forest(
    loom_vector_to_scalar_state_t* state, loom_combining_kind_t reduce_kind,
    uint8_t reduce_flags,
    loom_vector_to_scalar_accumulator_state_t* accumulator_state) {
  if (reduce_kind != LOOM_COMBINING_KIND_ADDF) {
    return false;
  }
  if (!iree_all_bits_set(
          reduce_flags,
          LOOM_VECTOR_FASTMATHFLAGS_REASSOC | LOOM_VECTOR_FASTMATHFLAGS_NNAN |
              LOOM_VECTOR_FASTMATHFLAGS_NINF | LOOM_VECTOR_FASTMATHFLAGS_NSZ |
              LOOM_VECTOR_FASTMATHFLAGS_CONTRACT)) {
    return false;
  }
  if (loom_type_element_type(state->vector_type) != LOOM_SCALAR_TYPE_F32) {
    return false;
  }

  loom_value_id_t input = loom_vector_reduce_input(state->op);
  loom_op_t* input_op =
      loom_vector_to_scalar_value_def_op(state->rewriter->module, input);
  if (!input_op || !loom_vector_mulf_isa(input_op)) {
    return false;
  }

  uint8_t product_flags = loom_vector_mulf_fastmath(input_op);
  if (!iree_any_bit_set(product_flags, LOOM_VECTOR_FASTMATHFLAGS_CONTRACT)) {
    return false;
  }

  accumulator_state->fused_product_lhs = loom_vector_mulf_lhs(input_op);
  accumulator_state->fused_product_rhs = loom_vector_mulf_rhs(input_op);
  accumulator_state->product_flags =
      loom_vector_to_scalar_project_instance_flags(
          LOOM_VECTOR_TO_SCALAR_INSTANCE_FLAG_MODE_FAST_MATH, product_flags);
  accumulator_state->fmaf_flags =
      accumulator_state->product_flags & accumulator_state->instance_flags;
  accumulator_state->use_product_fmaf_forest = true;
  return true;
}

iree_status_t loom_vector_to_scalar_lower_reduce(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_combining_kind_t reduce_kind = loom_vector_reduce_kind(state->op);
  uint8_t reduce_flags = loom_vector_reduce_fastmath(state->op);
  loom_op_kind_t scalar_kind = LOOM_OP_KIND_UNKNOWN;
  const bool has_scalar_kind =
      loom_scalar_combining_kind_op(reduce_kind, &scalar_kind);
  IREE_ASSERT(has_scalar_kind,
              "verified vector.reduce must use a scalar-representable "
              "combining kind");
  loom_vector_to_scalar_accumulator_state_t accumulator_state = {
      .lane_state = *state,
      .input = loom_vector_reduce_input(state->op),
      .init = loom_vector_reduce_init(state->op),
      .scalar_kind = scalar_kind,
      .instance_flags =
          loom_combining_kind_accepts_float(reduce_kind)
              ? loom_vector_to_scalar_project_instance_flags(
                    LOOM_VECTOR_TO_SCALAR_INSTANCE_FLAG_MODE_FAST_MATH,
                    state->op->instance_flags)
              : 0,
  };
  loom_vector_to_scalar_try_configure_product_fmaf_forest(
      state, reduce_kind, reduce_flags, &accumulator_state);
  return loom_vector_to_scalar_lower_accumulator(&accumulator_state,
                                                 out_replacement);
}

//===----------------------------------------------------------------------===//
// Axis-preserving reduction lowering
//===----------------------------------------------------------------------===//

typedef struct loom_vector_to_scalar_reduce_axes_state_t {
  loom_vector_to_scalar_state_t lane_state;
  loom_value_id_t input;
  loom_value_id_t init;
  loom_type_t input_type;
  loom_type_t result_type;
  loom_attribute_t axes;
  loom_op_kind_t scalar_kind;
  uint8_t instance_flags;
} loom_vector_to_scalar_reduce_axes_state_t;

static void loom_vector_to_scalar_reduce_axes_source_terms(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    const loom_vector_to_scalar_index_term_t* reduced_terms,
    loom_vector_to_scalar_index_term_t* source_terms) {
  uint16_t reduced_index = 0;
  uint8_t result_axis = 0;
  uint8_t input_rank = loom_type_rank(state->input_type);
  for (uint8_t input_axis = 0; input_axis < input_rank; ++input_axis) {
    if (reduced_index < state->axes.count &&
        state->axes.i64_array[reduced_index] == input_axis) {
      source_terms[input_axis] = reduced_terms[reduced_index++];
      continue;
    }
    source_terms[input_axis] = loom_vector_to_scalar_lane_term(
        &state->lane_state, result_indices, result_axis++);
  }
}

static iree_status_t loom_vector_to_scalar_reduce_axes_materialize_lane(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    const loom_vector_to_scalar_index_term_t* reduced_terms,
    loom_value_id_t* out_lane) {
  loom_vector_to_scalar_index_term_t source_terms[LOOM_TYPE_MAX_RANK] = {{0}};
  loom_vector_to_scalar_reduce_axes_source_terms(state, result_indices,
                                                 reduced_terms, source_terms);

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      &state->lane_state, source_terms, loom_type_rank(state->input_type),
      &source_indices));
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      &state->lane_state, state->input, source_indices, &lane));
  loom_vector_to_scalar_record_lane_materialized(&state->lane_state);
  *out_lane = lane;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_reduce_axes_build_combiner(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    const loom_vector_to_scalar_index_term_t* reduced_terms,
    loom_value_id_t accumulator, loom_value_id_t* out_accumulator) {
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_reduce_axes_materialize_lane(
      state, result_indices, reduced_terms, &lane));
  return loom_vector_to_scalar_build_generic_lane_op(
      &state->lane_state, state->scalar_kind, state->instance_flags,
      (loom_value_id_t[]){accumulator, lane}, 2, NULL, 0,
      state->lane_state.result_scalar_type, out_accumulator);
}

static iree_status_t loom_vector_to_scalar_reduce_axes_axis(
    loom_vector_to_scalar_reduce_axes_state_t* state, uint16_t axis_index,
    loom_vector_to_scalar_index_list_t result_indices,
    loom_vector_to_scalar_index_term_t* reduced_terms,
    loom_value_id_t current_accumulator, loom_value_id_t* out_accumulator) {
  if (axis_index == state->axes.count) {
    return loom_vector_to_scalar_reduce_axes_build_combiner(
        state, result_indices, reduced_terms, current_accumulator,
        out_accumulator);
  }

  uint8_t input_axis = (uint8_t)state->axes.i64_array[axis_index];
  if (!loom_type_dim_is_dynamic_at(state->input_type, input_axis)) {
    loom_value_id_t accumulator = current_accumulator;
    int64_t extent =
        (int64_t)loom_type_dim_static_size_at(state->input_type, input_axis);
    for (int64_t i = 0; i < extent; ++i) {
      reduced_terms[axis_index] = loom_vector_to_scalar_static_term(i);
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_reduce_axes_axis(
          state, (uint16_t)(axis_index + 1), result_indices, reduced_terms,
          accumulator, &accumulator));
    }
    *out_accumulator = accumulator;
    return iree_ok_status();
  }

  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 0,
      &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 1,
      &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      &state->lane_state, state->input_type, input_axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->lane_state.rewriter->builder, /*build_flags=*/0, lower_bound,
      upper_bound, step, &current_accumulator, 1,
      &state->lane_state.result_scalar_type, 1, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, state->lane_state.location,
      &loop));
  loom_vector_to_scalar_record_loop_created(&state->lane_state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->lane_state.rewriter->builder, loop, loom_scf_for_body(loop));
  reduced_terms[axis_index] = loom_vector_to_scalar_value_term(
      &state->lane_state, loom_region_entry_arg_id(loom_scf_for_body(loop), 0));
  loom_value_id_t accumulator_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_reduce_axes_axis(
      state, (uint16_t)(axis_index + 1), result_indices, reduced_terms,
      accumulator_arg, &yielded_accumulator));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->lane_state.rewriter->builder, &yielded_accumulator, 1,
      state->lane_state.location, &yield_op));
  loom_builder_restore(&state->lane_state.rewriter->builder, saved);

  *out_accumulator = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static bool loom_vector_to_scalar_reduce_axes_can_reassociate(
    const loom_vector_to_scalar_reduce_axes_state_t* state) {
  return iree_any_bit_set(state->instance_flags,
                          LOOM_SCALAR_FASTMATHFLAGS_REASSOC);
}

static bool loom_vector_to_scalar_reduce_axes_static_element_count(
    const loom_vector_to_scalar_reduce_axes_state_t* state,
    uint16_t* out_element_count) {
  uint64_t element_count = 1;
  for (uint16_t axis_index = 0; axis_index < state->axes.count; ++axis_index) {
    uint8_t input_axis = (uint8_t)state->axes.i64_array[axis_index];
    if (loom_type_dim_is_dynamic_at(state->input_type, input_axis)) {
      return false;
    }
    element_count *=
        (uint64_t)loom_type_dim_static_size_at(state->input_type, input_axis);
    if (element_count > UINT16_MAX) return false;
  }
  *out_element_count = (uint16_t)element_count;
  return true;
}

static void loom_vector_to_scalar_reduce_axes_terms_from_ordinal(
    const loom_vector_to_scalar_reduce_axes_state_t* state, uint16_t ordinal,
    loom_vector_to_scalar_index_term_t* reduced_terms) {
  int64_t remaining = ordinal;
  for (uint16_t axis_index = state->axes.count; axis_index > 0; --axis_index) {
    uint16_t reduced_axis = (uint16_t)(axis_index - 1);
    uint8_t input_axis = (uint8_t)state->axes.i64_array[reduced_axis];
    int64_t extent =
        loom_type_dim_static_size_at(state->input_type, input_axis);
    reduced_terms[reduced_axis] =
        loom_vector_to_scalar_static_term(remaining % extent);
    remaining /= extent;
  }
}

static iree_status_t
loom_vector_to_scalar_reduce_axes_build_static_reduction_tree(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices, uint16_t begin_ordinal,
    uint16_t end_ordinal, loom_vector_to_scalar_index_term_t* reduced_terms,
    loom_value_id_t* out_result) {
  if (end_ordinal <= begin_ordinal) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "invalid empty reduce-axes term range");
  }
  if (end_ordinal - begin_ordinal == 1) {
    loom_vector_to_scalar_reduce_axes_terms_from_ordinal(state, begin_ordinal,
                                                         reduced_terms);
    return loom_vector_to_scalar_reduce_axes_materialize_lane(
        state, result_indices, reduced_terms, out_result);
  }

  uint16_t middle_ordinal =
      (uint16_t)(begin_ordinal + (end_ordinal - begin_ordinal) / 2);
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_reduce_axes_build_static_reduction_tree(
          state, result_indices, begin_ordinal, middle_ordinal, reduced_terms,
          &lhs));
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_reduce_axes_build_static_reduction_tree(
          state, result_indices, middle_ordinal, end_ordinal, reduced_terms,
          &rhs));
  return loom_vector_to_scalar_build_generic_lane_op(
      &state->lane_state, state->scalar_kind, state->instance_flags,
      (loom_value_id_t[]){lhs, rhs}, 2, NULL, 0,
      state->lane_state.result_scalar_type, out_result);
}

static iree_status_t loom_vector_to_scalar_reduce_axes_reassociated_lane(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    uint16_t reduced_element_count, loom_value_id_t* out_lane) {
  loom_value_id_t accumulator = state->init;
  if (loom_type_is_vector(state->result_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        &state->lane_state, state->init, result_indices, &accumulator));
  }
  if (reduced_element_count == 0) {
    *out_lane = accumulator;
    return iree_ok_status();
  }

  loom_vector_to_scalar_index_term_t reduced_terms[LOOM_TYPE_MAX_RANK] = {{0}};
  loom_value_id_t tree = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_reduce_axes_build_static_reduction_tree(
          state, result_indices, 0, reduced_element_count, reduced_terms,
          &tree));
  return loom_vector_to_scalar_build_generic_lane_op(
      &state->lane_state, state->scalar_kind, state->instance_flags,
      (loom_value_id_t[]){accumulator, tree}, 2, NULL, 0,
      state->lane_state.result_scalar_type, out_lane);
}

static iree_status_t loom_vector_to_scalar_reduce_axes_lane(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    loom_value_id_t* out_lane) {
  uint16_t reduced_element_count = 0;
  if (loom_vector_to_scalar_reduce_axes_can_reassociate(state) &&
      loom_vector_to_scalar_reduce_axes_static_element_count(
          state, &reduced_element_count)) {
    return loom_vector_to_scalar_reduce_axes_reassociated_lane(
        state, result_indices, reduced_element_count, out_lane);
  }

  loom_value_id_t accumulator = state->init;
  if (loom_type_is_vector(state->result_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        &state->lane_state, state->init, result_indices, &accumulator));
  }

  loom_vector_to_scalar_index_term_t reduced_terms[LOOM_TYPE_MAX_RANK] = {{0}};
  return loom_vector_to_scalar_reduce_axes_axis(
      state, 0, result_indices, reduced_terms, accumulator, out_lane);
}

static iree_status_t loom_vector_to_scalar_reduce_axes_static_result(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_value_id_t* out_replacement) {
  uint16_t element_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_static_element_count(
      &state->lane_state, state->result_type, &element_count));
  if (loom_pass_has_error_diagnostics(state->lane_state.pass)) {
    return iree_ok_status();
  }
  if (element_count == 0) {
    *out_replacement = state->init;
    return iree_ok_status();
  }

  loom_value_id_t* elements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->lane_state.rewriter->arena, element_count, sizeof(loom_value_id_t),
      (void**)&elements));
  uint8_t result_rank = loom_type_rank(state->result_type);
  int64_t* result_indices = NULL;
  if (result_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->lane_state.rewriter->arena, result_rank, sizeof(int64_t),
        (void**)&result_indices));
  }
  for (uint16_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(state->result_type, ordinal,
                                               result_indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = result_indices,
        .rank = result_rank,
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_reduce_axes_lane(
        state, index_list, &elements[ordinal]));
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &state->lane_state.rewriter->builder, elements, element_count,
      state->result_type, state->lane_state.location, &from_elements_op));
  *out_replacement = loom_vector_from_elements_result(from_elements_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_reduce_axes_result_loop_axis(
    loom_vector_to_scalar_reduce_axes_state_t* state, uint8_t axis,
    loom_value_id_t current_aggregate, loom_value_id_t* dynamic_indices,
    loom_value_id_t* out_aggregate) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 0,
      &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 1,
      &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      &state->lane_state, state->result_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->lane_state.rewriter->builder, /*build_flags=*/0, lower_bound,
      upper_bound, step, &current_aggregate, 1, &state->result_type, 1, NULL, 0,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0, /*unroll_schedule=*/0,
      state->lane_state.location, &loop));
  loom_vector_to_scalar_record_loop_created(&state->lane_state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->lane_state.rewriter->builder, loop, loom_scf_for_body(loop));
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t aggregate_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_aggregate = LOOM_VALUE_ID_INVALID;
  if (axis + 1 == loom_type_rank(state->result_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->result_type),
    };
    loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_reduce_axes_lane(state, index_list, &lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_insert_lane(
        &state->lane_state, lane, aggregate_arg, state->result_type, index_list,
        &yielded_aggregate));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_reduce_axes_result_loop_axis(
        state, (uint8_t)(axis + 1), aggregate_arg, dynamic_indices,
        &yielded_aggregate));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->lane_state.rewriter->builder, &yielded_aggregate, 1,
      state->lane_state.location, &yield_op));
  loom_builder_restore(&state->lane_state.rewriter->builder, saved);

  *out_aggregate = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_reduce_axes_dynamic_result(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_value_id_t* out_replacement) {
  uint8_t result_rank = loom_type_rank(state->result_type);
  loom_value_id_t* dynamic_indices = NULL;
  if (result_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->lane_state.rewriter->arena, result_rank, sizeof(loom_value_id_t),
        (void**)&dynamic_indices));
  }
  return loom_vector_to_scalar_reduce_axes_result_loop_axis(
      state, 0, state->init, dynamic_indices, out_replacement);
}

iree_status_t loom_vector_to_scalar_lower_reduce_axes(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_combining_kind_t reduce_kind = loom_vector_reduce_axes_kind(state->op);
  loom_op_kind_t scalar_kind = LOOM_OP_KIND_UNKNOWN;
  const bool has_scalar_kind =
      loom_scalar_combining_kind_op(reduce_kind, &scalar_kind);
  IREE_ASSERT(has_scalar_kind,
              "verified vector.reduce.axes must use a scalar-representable "
              "combining kind");

  loom_type_t result_type = loom_module_value_type(
      state->rewriter->module, loom_vector_reduce_axes_result(state->op));
  loom_vector_to_scalar_reduce_axes_state_t reduce_state = {
      .lane_state = *state,
      .input = loom_vector_reduce_axes_input(state->op),
      .init = loom_vector_reduce_axes_init(state->op),
      .input_type = loom_module_value_type(
          state->rewriter->module, loom_vector_reduce_axes_input(state->op)),
      .result_type = result_type,
      .axes = loom_vector_reduce_axes_axes(state->op),
      .scalar_kind = scalar_kind,
      .instance_flags =
          loom_combining_kind_accepts_float(reduce_kind)
              ? loom_vector_to_scalar_project_instance_flags(
                    LOOM_VECTOR_TO_SCALAR_INSTANCE_FLAG_MODE_FAST_MATH,
                    state->op->instance_flags)
              : 0,
  };
  if (loom_type_is_scalar(result_type)) {
    loom_vector_to_scalar_index_list_t index_list = {0};
    return loom_vector_to_scalar_reduce_axes_lane(&reduce_state, index_list,
                                                  out_replacement);
  }
  if (loom_type_is_all_static(result_type)) {
    return loom_vector_to_scalar_reduce_axes_static_result(&reduce_state,
                                                           out_replacement);
  }
  return loom_vector_to_scalar_reduce_axes_dynamic_result(&reduce_state,
                                                          out_replacement);
}

iree_status_t loom_vector_to_scalar_lower_dotf(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_vector_to_scalar_accumulator_state_t accumulator_state = {
      .lane_state = *state,
      .input = loom_vector_dotf_lhs(state->op),
      .rhs = loom_vector_dotf_rhs(state->op),
      .init = loom_vector_dotf_init(state->op),
      .instance_flags = loom_vector_to_scalar_project_instance_flags(
          LOOM_VECTOR_TO_SCALAR_INSTANCE_FLAG_MODE_FAST_MATH,
          state->op->instance_flags),
      .use_fmaf = true,
  };
  return loom_vector_to_scalar_lower_accumulator(&accumulator_state,
                                                 out_replacement);
}
