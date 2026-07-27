// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_tables.h"

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"

//===----------------------------------------------------------------------===//
// Table lane programs
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_build_zero_lane(
    loom_vector_to_scalar_state_t* state, loom_type_t result_type,
    loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, result_type, state->location, 0, out_lane);
}

static iree_status_t loom_vector_to_scalar_cast_lane_to_index(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_type_t input_type, loom_value_id_t* out_index) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  if (loom_type_equal(input_type, index_type)) {
    *out_index = input;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(&state->rewriter->builder, input,
                                             input_type, index_type,
                                             state->location, &cast_op));
  *out_index = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_table_lookup_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t indices_value = loom_vector_table_lookup_indices(state->op);
  loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, indices_value, indices, &index_lane));
  loom_type_t indices_type =
      loom_module_value_type(state->rewriter->module, indices_value);
  loom_type_t index_lane_type = loom_vector_to_scalar_lane_type(indices_type);
  loom_value_id_t table_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_lane_to_index(
      state, index_lane, index_lane_type, &table_index));

  loom_vector_to_scalar_index_list_t table_indices = {
      .dynamic_indices = &table_index,
      .rank = 1,
  };
  return loom_vector_to_scalar_materialize_lane(
      state, loom_vector_table_lookup_table(state->op), table_indices,
      out_lane);
}

static iree_status_t loom_vector_to_scalar_build_table_quantize_compare(
    loom_vector_to_scalar_state_t* state, loom_value_id_t threshold_lane,
    loom_value_id_t input_lane, loom_type_t operand_type,
    loom_value_id_t* out_condition) {
  uint8_t predicate = loom_vector_table_quantize_tie(state->op) ==
                              LOOM_VECTOR_TABLE_QUANTIZE_TIE_UPPER
                          ? LOOM_SCALAR_CMPF_PREDICATE_OLE
                          : LOOM_SCALAR_CMPF_PREDICATE_OLT;
  loom_op_t* cmp_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(
      &state->rewriter->builder, 0, predicate, threshold_lane, input_lane,
      operand_type, loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location,
      &cmp_op));
  *out_condition = loom_scalar_cmpf_result(cmp_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_table_quantize_step(
    loom_vector_to_scalar_state_t* state, loom_value_id_t accumulator,
    loom_value_id_t threshold_lane, loom_value_id_t input_lane,
    loom_type_t input_scalar_type, loom_value_id_t* out_next) {
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_table_quantize_compare(
      state, threshold_lane, input_lane, input_scalar_type, &condition));
  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, state->result_scalar_type, state->location, 1,
      &one));
  loom_value_id_t incremented = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ADDI, accumulator, one, state->result_scalar_type,
      &incremented));
  return loom_vector_to_scalar_build_select_lane(state, condition, incremented,
                                                 accumulator, out_next);
}

static iree_status_t loom_vector_to_scalar_apply_table_quantize_nan_policy(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input_lane,
    loom_value_id_t quantized_lane,
    loom_vector_to_scalar_index_term_t threshold_count,
    loom_value_id_t* out_lane) {
  if (loom_vector_table_quantize_nan(state->op) ==
      LOOM_VECTOR_TABLE_QUANTIZE_NAN_ZERO) {
    *out_lane = quantized_lane;
    return iree_ok_status();
  }

  loom_value_id_t max_code = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_term_as_scalar(
      state, threshold_count, state->result_scalar_type, &max_code));
  loom_op_t* isnan_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_isnanf_build(
      &state->rewriter->builder, input_lane,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location, &isnan_op));
  return loom_vector_to_scalar_build_select_lane(
      state, loom_scalar_isnanf_result(isnan_op), max_code, quantized_lane,
      out_lane);
}

static iree_status_t loom_vector_to_scalar_build_table_quantize_static_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input_lane,
    loom_type_t input_scalar_type, loom_value_id_t thresholds,
    loom_vector_to_scalar_index_term_t threshold_count,
    loom_value_id_t* out_lane) {
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_zero_lane(
      state, state->result_scalar_type, &accumulator));
  int64_t threshold_index_storage[1] = {0};
  loom_vector_to_scalar_index_list_t threshold_indices = {
      .static_indices = threshold_index_storage,
      .rank = 1,
  };
  for (int64_t threshold_index = 0;
       threshold_index < threshold_count.static_value; ++threshold_index) {
    threshold_index_storage[0] = threshold_index;
    loom_value_id_t threshold_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, thresholds, threshold_indices, &threshold_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_table_quantize_step(
        state, accumulator, threshold_lane, input_lane, input_scalar_type,
        &accumulator));
  }
  return loom_vector_to_scalar_apply_table_quantize_nan_policy(
      state, input_lane, accumulator, threshold_count, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_table_quantize_dynamic_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input_lane,
    loom_type_t input_scalar_type, loom_value_id_t thresholds,
    loom_vector_to_scalar_index_term_t threshold_count,
    loom_value_id_t* out_lane) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, &step));
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, threshold_count, &upper_bound));

  loom_value_id_t initial_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_zero_lane(
      state, state->result_scalar_type, &initial_accumulator));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, &initial_accumulator, 1, &state->result_scalar_type, 1, NULL, 0,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0, /*unroll_schedule=*/0,
      state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_value_id_t threshold_index =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t accumulator_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_vector_to_scalar_index_list_t threshold_indices = {
      .dynamic_indices = &threshold_index,
      .rank = 1,
  };
  loom_value_id_t threshold_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, thresholds, threshold_indices, &threshold_lane));
  loom_value_id_t next_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_table_quantize_step(
      state, accumulator_arg, threshold_lane, input_lane, input_scalar_type,
      &next_accumulator));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &next_accumulator, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  return loom_vector_to_scalar_apply_table_quantize_nan_policy(
      state, input_lane, loom_scf_for_results(loop).values[0], threshold_count,
      out_lane);
}

iree_status_t loom_vector_to_scalar_build_table_quantize_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t input = loom_vector_table_quantize_input(state->op);
  loom_type_t input_type =
      loom_module_value_type(state->rewriter->module, input);
  loom_type_t input_scalar_type = loom_vector_to_scalar_lane_type(input_type);
  loom_value_id_t input_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, input, indices, &input_lane));

  loom_value_id_t thresholds = loom_vector_table_quantize_thresholds(state->op);
  loom_type_t thresholds_type =
      loom_module_value_type(state->rewriter->module, thresholds);
  loom_vector_to_scalar_index_term_t threshold_count =
      loom_vector_to_scalar_dim_bound_term(state, thresholds_type, 0);
  if (threshold_count.is_dynamic) {
    return loom_vector_to_scalar_build_table_quantize_dynamic_lane(
        state, input_lane, input_scalar_type, thresholds, threshold_count,
        out_lane);
  }
  return loom_vector_to_scalar_build_table_quantize_static_lane(
      state, input_lane, input_scalar_type, thresholds, threshold_count,
      out_lane);
}
