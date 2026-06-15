// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_lanes.h"

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector/to_scalar_memory.h"
#include "loom/transforms/vector/to_scalar_mma.h"
#include "loom/transforms/vector/to_scalar_quantized.h"
#include "loom/transforms/vector/to_scalar_structural.h"
#include "loom/transforms/vector/to_scalar_tables.h"
#include "loom/transforms/vector/to_scalar_transforms.h"
#include "loom/util/math.h"

bool loom_vector_to_scalar_indices_are_dynamic(
    loom_vector_to_scalar_index_list_t indices) {
  return indices.dynamic_indices != NULL;
}

loom_type_t loom_vector_to_scalar_lane_type(loom_type_t vector_type) {
  return loom_type_scalar(loom_type_element_type(vector_type));
}

static loom_attribute_t loom_vector_to_scalar_zero_attr(
    loom_scalar_type_t scalar_type) {
  if (scalar_type == LOOM_SCALAR_TYPE_I1) return loom_attr_bool(false);
  if (loom_scalar_type_is_float(scalar_type)) return loom_attr_f64(0.0);
  return loom_attr_i64(0);
}

uint8_t loom_vector_to_scalar_project_instance_flags(
    loom_vector_to_scalar_instance_flag_mode_t mode, uint8_t instance_flags) {
  switch (mode) {
    case LOOM_VECTOR_TO_SCALAR_INSTANCE_FLAG_MODE_DROP:
      return 0;
    case LOOM_VECTOR_TO_SCALAR_INSTANCE_FLAG_MODE_FORWARD:
      return instance_flags;
    case LOOM_VECTOR_TO_SCALAR_INSTANCE_FLAG_MODE_FAST_MATH:
      return instance_flags;
  }
  return 0;
}

iree_status_t loom_vector_to_scalar_build_scalar_constant(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, int64_t integer_value,
    loom_value_id_t* out_value_id) {
  loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
  loom_attribute_t attr = loom_scalar_type_is_float(scalar_type)
                              ? loom_attr_f64((double)integer_value)
                              : loom_attr_i64(integer_value);
  loom_op_t* constant_op = NULL;
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, attr, result_type,
                                                   location, &constant_op));
    *out_value_id = loom_index_constant_result(constant_op);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, attr, result_type,
                                                  location, &constant_op));
  *out_value_id = loom_scalar_constant_result(constant_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_scalar_attr_constant(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, loom_attribute_t value,
    loom_value_id_t* out_value_id) {
  loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
  loom_op_t* constant_op = NULL;
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, value, result_type,
                                                   location, &constant_op));
    *out_value_id = loom_index_constant_result(constant_op);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, value, result_type,
                                                  location, &constant_op));
  *out_value_id = loom_scalar_constant_result(constant_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_vector_zero(
    loom_vector_to_scalar_state_t* state, loom_type_t result_type,
    loom_value_id_t* out_value_id) {
  loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &state->rewriter->builder, loom_vector_to_scalar_zero_attr(scalar_type),
      result_type, state->location, &constant_op));
  *out_value_id = loom_vector_constant_result(constant_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_generic_lane_op(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    uint8_t instance_flags, const loom_value_id_t* operands,
    uint16_t operand_count, const loom_attribute_t* attrs, uint8_t attr_count,
    loom_type_t result_type, loom_value_id_t* out_result) {
  loom_builder_t* builder = &state->rewriter->builder;
  loom_op_t* scalar_op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, kind, operand_count, 1,
                                                0, 0, attr_count,
                                                state->location, &scalar_op));
  scalar_op->instance_flags = instance_flags;
  loom_value_id_t* op_operands = loom_op_operands(scalar_op);
  for (uint16_t i = 0; i < operand_count; ++i) {
    op_operands[i] = operands[i];
  }
  IREE_RETURN_IF_ERROR(loom_builder_define_value(
      builder, result_type, &loom_op_results(scalar_op)[0]));
  loom_attribute_t* op_attrs = loom_op_attrs(scalar_op);
  for (uint8_t i = 0; i < attr_count; ++i) {
    op_attrs[i] = attrs[i];
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, scalar_op));
  *out_result = loom_op_results(scalar_op)[0];
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_scalar_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_generic_lane_op(
      state, kind, 0, (loom_value_id_t[]){lhs, rhs}, 2, NULL, 0, result_type,
      out_result);
}

int64_t loom_vector_to_scalar_integer_mask_value(int32_t bit_width,
                                                 int64_t used_bits) {
  if (used_bits <= 0) return 0;
  if (used_bits >= bit_width) return bit_width == 1 ? 1 : -1;
  return (int64_t)((UINT64_C(1) << (uint64_t)used_bits) - 1);
}

int64_t loom_vector_to_scalar_shifted_integer_mask_value(int32_t bit_width,
                                                         int64_t offset,
                                                         int64_t used_bits) {
  if (used_bits <= 0) return 0;
  uint64_t mask =
      used_bits >= 64 ? UINT64_MAX : ((UINT64_C(1) << (uint64_t)used_bits) - 1);
  mask <<= (uint64_t)offset;
  if (bit_width < 64) mask &= (UINT64_C(1) << (uint32_t)bit_width) - 1;
  if (bit_width == 1) return (mask & 1) ? 1 : 0;
  return (int64_t)mask;
}

iree_status_t loom_vector_to_scalar_build_integer_mask(
    loom_vector_to_scalar_state_t* state, loom_type_t type, int64_t used_bits,
    loom_value_id_t* out_mask) {
  int32_t bit_width = loom_scalar_type_bitwidth(loom_type_element_type(type));
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, type, state->location,
      loom_vector_to_scalar_integer_mask_value(bit_width, used_bits), out_mask);
}

iree_status_t loom_vector_to_scalar_cast_integer_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type, bool signed_extend,
    loom_value_id_t* out_result) {
  if (loom_type_equal(input_type, result_type)) {
    *out_result = input;
    return iree_ok_status();
  }

  int32_t input_width =
      loom_scalar_type_bitwidth(loom_type_element_type(input_type));
  int32_t result_width =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));

  loom_op_t* cast_op = NULL;
  if (input_width < result_width) {
    if (signed_extend) {
      IREE_RETURN_IF_ERROR(
          loom_scalar_extsi_build(&state->rewriter->builder, input, input_type,
                                  result_type, state->location, &cast_op));
      *out_result = loom_scalar_extsi_result(cast_op);
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(&state->rewriter->builder,
                                                 input, input_type, result_type,
                                                 state->location, &cast_op));
    *out_result = loom_scalar_extui_result(cast_op);
    return iree_ok_status();
  }

  if (input_width > result_width) {
    IREE_RETURN_IF_ERROR(
        loom_scalar_trunci_build(&state->rewriter->builder, input, input_type,
                                 result_type, state->location, &cast_op));
    *out_result = loom_scalar_trunci_result(cast_op);
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(&state->rewriter->builder,
                                                 input, input_type, result_type,
                                                 state->location, &cast_op));
  *out_result = loom_scalar_bitcast_result(cast_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_scalar_shift(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t input, loom_type_t type, int64_t amount,
    loom_value_id_t* out_result) {
  if (amount == 0) {
    *out_result = input;
    return iree_ok_status();
  }
  loom_value_id_t amount_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, type, state->location, amount, &amount_value));
  if (kind == LOOM_OP_SCALAR_SHLI) {
    loom_op_t* shift_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(&state->rewriter->builder, 0,
                                                input, amount_value, type,
                                                state->location, &shift_op));
    *out_result = loom_scalar_shli_result(shift_op);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_scalar_binary(
      state, kind, input, amount_value, type, out_result);
}

iree_status_t loom_vector_to_scalar_build_index_term_as_scalar(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t term, loom_type_t result_type,
    loom_value_id_t* out_value) {
  if (!term.is_dynamic) {
    return loom_vector_to_scalar_build_scalar_constant(
        &state->rewriter->builder, result_type, state->location,
        term.static_value, out_value);
  }
  if (loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_INDEX) {
    *out_value = term.dynamic_value;
    return iree_ok_status();
  }
  loom_value_id_t index_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, term, &index_value));
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(&state->rewriter->builder, index_value,
                            loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                            result_type, state->location, &cast_op));
  *out_value = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_scalar_shift_term(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t input, loom_type_t type,
    loom_vector_to_scalar_index_term_t amount, loom_value_id_t* out_result) {
  if (!amount.is_dynamic) {
    return loom_vector_to_scalar_build_scalar_shift(
        state, kind, input, type, amount.static_value, out_result);
  }
  loom_value_id_t amount_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_term_as_scalar(
      state, amount, type, &amount_value));
  if (kind == LOOM_OP_SCALAR_SHLI) {
    loom_op_t* shift_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(&state->rewriter->builder, 0,
                                                input, amount_value, type,
                                                state->location, &shift_op));
    *out_result = loom_scalar_shli_result(shift_op);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_scalar_binary(
      state, kind, input, amount_value, type, out_result);
}

iree_status_t loom_vector_to_scalar_build_bitstream_base_term(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t lane_ordinal, int64_t lane_bit_width,
    loom_vector_to_scalar_index_term_t* out_position) {
  return loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL, lane_ordinal,
      loom_vector_to_scalar_static_term(lane_bit_width), out_position);
}

iree_status_t loom_vector_to_scalar_build_single_bit_extract(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lane,
    loom_type_t lane_type, loom_vector_to_scalar_index_term_t bit_shift,
    loom_value_id_t one_mask, loom_value_id_t* out_bit) {
  loom_value_id_t shifted_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift_term(
      state, LOOM_OP_SCALAR_SHRUI, lane, lane_type, bit_shift, &shifted_lane));
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, shifted_lane, one_mask, lane_type, out_bit);
}

static iree_status_t loom_vector_to_scalar_extract_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t vector_value,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t vector_type =
      loom_module_value_type(state->rewriter->module, vector_value);
  if (!loom_type_is_vector(vector_type)) {
    *out_lane = vector_value;
    return iree_ok_status();
  }
  loom_type_t lane_type = loom_vector_to_scalar_lane_type(vector_type);
  int64_t* static_indices = NULL;
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->rewriter->builder.arena, indices.rank,
                                  sizeof(int64_t), (void**)&static_indices));
    for (uint8_t i = 0; i < indices.rank; ++i) {
      static_indices[i] = INT64_MIN;
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_copy_static_indices(
        &state->rewriter->builder, indices.static_indices, indices.rank,
        &static_indices));
  }
  loom_op_t* extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &state->rewriter->builder, vector_value, indices.dynamic_indices,
      loom_vector_to_scalar_indices_are_dynamic(indices) ? indices.rank : 0,
      static_indices, indices.rank, lane_type, state->location, &extract_op));
  *out_lane = loom_vector_extract_result(extract_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_insert_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lane,
    loom_value_id_t aggregate, loom_type_t aggregate_type,
    loom_vector_to_scalar_index_list_t indices,
    loom_value_id_t* out_aggregate) {
  int64_t* static_indices = NULL;
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->rewriter->builder.arena, indices.rank,
                                  sizeof(int64_t), (void**)&static_indices));
    for (uint8_t i = 0; i < indices.rank; ++i) {
      static_indices[i] = INT64_MIN;
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_copy_static_indices(
        &state->rewriter->builder, indices.static_indices, indices.rank,
        &static_indices));
  }
  loom_op_t* insert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_insert_build(
      &state->rewriter->builder, lane, aggregate, indices.dynamic_indices,
      loom_vector_to_scalar_indices_are_dynamic(indices) ? indices.rank : 0,
      static_indices, indices.rank, aggregate_type, state->location,
      &insert_op));
  *out_aggregate = loom_vector_insert_result(insert_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_dim_bound(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type, uint8_t axis,
    loom_value_id_t* out_bound) {
  if (loom_type_dim_is_dynamic_at(vector_type, axis)) {
    *out_bound = loom_type_dim_value_id_at(vector_type, axis);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, (int64_t)loom_type_dim_static_size_at(vector_type, axis),
      out_bound);
}

iree_status_t loom_vector_to_scalar_build_i1_and(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  loom_op_t* and_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
      &state->rewriter->builder, lhs, rhs,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location, &and_op));
  *out_result = loom_scalar_andi_result(and_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_select_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t condition,
    loom_value_id_t true_lane, loom_value_id_t false_lane,
    loom_value_id_t* out_lane) {
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      &state->rewriter->builder, condition, true_lane, false_lane,
      state->result_scalar_type, state->location, &select_op));
  *out_lane = loom_scf_select_result(select_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_coordinate_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t integer_kind,
    loom_vector_to_scalar_index_binary_t index_binary, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t type, loom_value_id_t* out_result) {
  if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX) {
    return loom_vector_to_scalar_build_index_binary(state, index_binary, lhs,
                                                    rhs, out_result);
  }
  return loom_vector_to_scalar_build_generic_lane_op(
      state, integer_kind, 0, (loom_value_id_t[]){lhs, rhs}, 2, NULL, 0, type,
      out_result);
}

static iree_status_t loom_vector_to_scalar_build_iota_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t base = loom_vector_iota_base(state->op);
  loom_value_id_t step = loom_vector_iota_step(state->op);
  loom_type_t lane_type = state->result_scalar_type;
  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_ordinal_for_lane(
      state, indices, lane_type, &ordinal));
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_MULI, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL,
      ordinal, step, lane_type, &scaled));
  return loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_ADDI, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, base,
      scaled, lane_type, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_mask_range_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t lower_bound = loom_vector_mask_range_lower_bound(state->op);
  loom_value_id_t upper_bound = loom_vector_mask_range_upper_bound(state->op);
  loom_value_id_t step = loom_vector_mask_range_step(state->op);
  loom_type_t coordinate_type =
      loom_module_value_type(state->rewriter->module, lower_bound);

  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_ordinal_for_lane(
      state, indices, coordinate_type, &ordinal));
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_MULI, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL,
      ordinal, step, coordinate_type, &scaled));
  loom_value_id_t coordinate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_ADDI, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD,
      lower_bound, scaled, coordinate_type, &coordinate));

  if (loom_type_element_type(coordinate_type) == LOOM_SCALAR_TYPE_INDEX) {
    loom_op_t* cmp_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        &state->rewriter->builder, LOOM_INDEX_CMP_PREDICATE_SLT, coordinate,
        upper_bound, coordinate_type, loom_type_scalar(LOOM_SCALAR_TYPE_I1),
        state->location, &cmp_op));
    *out_lane = loom_index_cmp_result(cmp_op);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_generic_lane_op(
      state, LOOM_OP_SCALAR_CMPI, 0,
      (loom_value_id_t[]){coordinate, upper_bound}, 2,
      (loom_attribute_t[]){loom_attr_enum(LOOM_SCALAR_CMPI_PREDICATE_SLT)}, 1,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), out_lane);
}

iree_status_t loom_vector_to_scalar_build_constant_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_build_scalar_attr_constant(
      &state->rewriter->builder, state->result_scalar_type, state->location,
      loom_vector_constant_value(state->op), out_lane);
}

iree_status_t loom_vector_to_scalar_build_poison_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_lane) {
  loom_op_t* poison_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_poison_build(&state->rewriter->builder,
                                                state->result_scalar_type,
                                                state->location, &poison_op));
  *out_lane = loom_scalar_poison_result(poison_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_generic_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  const loom_vector_to_scalar_descriptor_t* descriptor = state->descriptor;
  loom_value_id_t lane_operands[4] = {0};
  const loom_value_id_t* operands = loom_op_const_operands(state->op);
  for (uint8_t i = 0; i < descriptor->lane_operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, operands[i], indices, &lane_operands[i]));
  }

  uint8_t instance_flags = loom_vector_to_scalar_project_instance_flags(
      descriptor->instance_flag_mode, state->op->instance_flags);
  const loom_attribute_t* attrs = loom_op_attrs(state->op);
  return loom_vector_to_scalar_build_generic_lane_op(
      state, descriptor->lane_op_kind, instance_flags, lane_operands,
      descriptor->lane_operand_count, attrs, descriptor->copied_attr_count,
      state->result_scalar_type, out_lane);
}

typedef iree_status_t (*loom_vector_to_scalar_lane_lowerer_t)(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

static iree_status_t loom_vector_to_scalar_build_bitfield_extractu_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_build_bitfield_extract_lane(
      state, indices, /*signed_extract=*/false, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitfield_extracts_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_build_bitfield_extract_lane(
      state, indices, /*signed_extract=*/true, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitunpacku_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_build_bitunpack_lane(
      state, indices, /*signed_unpack=*/false, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitunpacks_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_build_bitunpack_lane(
      state, indices, /*signed_unpack=*/true, out_lane);
}

static const loom_vector_to_scalar_lane_lowerer_t
    kVectorToScalarLaneLowerers[] = {
        loom_vector_to_scalar_build_generic_lane,
        loom_vector_to_scalar_build_iota_lane,
        loom_vector_to_scalar_build_mask_range_lane,
        loom_vector_to_scalar_build_broadcast_lane,
        loom_vector_to_scalar_build_extract_lane,
        loom_vector_to_scalar_build_insert_lane,
        loom_vector_to_scalar_build_slice_lane,
        loom_vector_to_scalar_build_concat_lane,
        loom_vector_to_scalar_build_transpose_lane,
        loom_vector_to_scalar_build_shuffle_lane,
        loom_vector_to_scalar_build_interleave_lane,
        loom_vector_to_scalar_build_deinterleave_lane,
        loom_vector_to_scalar_build_bitcast_lane,
        loom_vector_to_scalar_build_bitfield_extractu_lane,
        loom_vector_to_scalar_build_bitfield_extracts_lane,
        loom_vector_to_scalar_build_bitfield_insert_lane,
        loom_vector_to_scalar_build_dot2f_lane,
        loom_vector_to_scalar_build_dot4i_lane,
        loom_vector_to_scalar_build_dot8i4_lane,
        loom_vector_to_scalar_build_dot4f8_lane,
        loom_vector_to_scalar_build_bitpack_lane,
        loom_vector_to_scalar_build_bitunpacku_lane,
        loom_vector_to_scalar_build_bitunpacks_lane,
        loom_vector_to_scalar_build_table_lookup_lane,
        loom_vector_to_scalar_build_table_quantize_lane,
        loom_vector_to_scalar_build_transform_lane,
        loom_vector_to_scalar_build_load_lane,
        loom_vector_to_scalar_build_masked_load_lane,
        loom_vector_to_scalar_build_gather_lane,
        loom_vector_to_scalar_build_masked_gather_lane,
        loom_vector_to_scalar_build_load_expand_lane,
};
static_assert(IREE_ARRAYSIZE(kVectorToScalarLaneLowerers) ==
                  LOOM_VECTOR_TO_SCALAR_LANE_COUNT,
              "vector-to-scalar lane lowerer table must match the lane enum");

iree_status_t loom_vector_to_scalar_build_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_vector_to_scalar_record_lane_materialized(state);
  return kVectorToScalarLaneLowerers[state->descriptor->lane_kind](
      state, indices, out_lane);
}

static bool loom_vector_to_scalar_try_from_elements_lane(
    loom_op_t* def_op, loom_type_t vector_type,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  if (!loom_vector_from_elements_isa(def_op)) return false;
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) return false;
  if (!loom_type_is_all_static(vector_type)) return false;
  int64_t ordinal = loom_vector_to_scalar_linear_ordinal_static(
      vector_type, indices.static_indices);
  if (ordinal < 0) return false;
  loom_value_slice_t elements = loom_vector_from_elements_elements(def_op);
  if (ordinal >= (int64_t)elements.count) return false;
  *out_lane = loom_value_slice_get(elements, (uint16_t)ordinal);
  return true;
}

iree_status_t loom_vector_to_scalar_try_materialize_def_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_type_t vector_type, loom_vector_to_scalar_index_list_t indices,
    bool* out_materialized, loom_value_id_t* out_lane) {
  *out_materialized = false;
  loom_op_t* def_op =
      loom_vector_to_scalar_value_def_op(state->rewriter->module, value);
  if (!def_op) {
    return iree_ok_status();
  }
  const loom_trait_flags_t traits =
      loom_op_effective_traits(state->rewriter->module, def_op);
  if (loom_traits_are_value_alias(traits)) {
    IREE_ASSERT(def_op->operand_count >= 1);
    IREE_ASSERT(def_op->result_count == 1);
    return loom_vector_to_scalar_try_materialize_def_lane(
        state, loom_op_const_operands(def_op)[0], vector_type, indices,
        out_materialized, out_lane);
  }
  if (loom_vector_to_scalar_try_from_elements_lane(def_op, vector_type, indices,
                                                   out_lane)) {
    *out_materialized = true;
    return iree_ok_status();
  }
  if (loom_vector_splat_isa(def_op)) {
    *out_lane = loom_vector_splat_scalar(def_op);
    *out_materialized = true;
    return iree_ok_status();
  }
  if (loom_vector_constant_isa(def_op)) {
    loom_vector_to_scalar_state_t def_state = {
        .pass = state->pass,
        .statistics = state->statistics,
        .rewriter = state->rewriter,
        .op = def_op,
        .value_checkpoint = state->value_checkpoint,
        .vector_type = vector_type,
        .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
        .location = def_op->location,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_constant_lane(&def_state, out_lane));
    *out_materialized = true;
    return iree_ok_status();
  }
  if (loom_vector_poison_isa(def_op)) {
    loom_vector_to_scalar_state_t def_state = {
        .pass = state->pass,
        .statistics = state->statistics,
        .rewriter = state->rewriter,
        .op = def_op,
        .value_checkpoint = state->value_checkpoint,
        .vector_type = vector_type,
        .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
        .location = def_op->location,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_poison_lane(&def_state, out_lane));
    *out_materialized = true;
    return iree_ok_status();
  }
  if (loom_vector_fragment_load_isa(def_op)) {
    if (indices.rank != 2) {
      return iree_ok_status();
    }
    loom_vector_to_scalar_state_t def_state = {
        .pass = state->pass,
        .statistics = state->statistics,
        .rewriter = state->rewriter,
        .op = def_op,
        .value_checkpoint = state->value_checkpoint,
        .vector_type = vector_type,
        .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
        .location = def_op->location,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_load_lane(&def_state, indices, out_lane));
    *out_materialized = true;
    return iree_ok_status();
  }
  if (loom_vector_mma_isa(def_op)) {
    return loom_vector_to_scalar_try_materialize_mma_lane(
        state, def_op, indices, out_materialized, out_lane);
  }

  const loom_vector_to_scalar_descriptor_t* descriptor =
      loom_vector_to_scalar_find_descriptor(def_op->kind);
  if (!descriptor) return iree_ok_status();
  loom_value_t* value_record =
      loom_module_value(state->rewriter->module, value);
  uint16_t result_ordinal = loom_value_def_index(value_record);
  loom_type_t result_type = loom_module_value_type(
      state->rewriter->module, loom_op_results(def_op)[result_ordinal]);
  loom_vector_to_scalar_state_t def_state = {
      .pass = state->pass,
      .statistics = state->statistics,
      .rewriter = state->rewriter,
      .op = def_op,
      .descriptor = descriptor,
      .value_checkpoint = state->value_checkpoint,
      .result_ordinal = result_ordinal,
      .vector_type = result_type,
      .result_scalar_type = descriptor->result_is_i1
                                ? loom_type_scalar(LOOM_SCALAR_TYPE_I1)
                                : loom_vector_to_scalar_lane_type(result_type),
      .location = def_op->location,
  };
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_lane(&def_state, indices, out_lane));
  *out_materialized = true;
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_materialize_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t type = loom_module_value_type(state->rewriter->module, value);
  if (!loom_type_is_vector(type)) {
    *out_lane = value;
    return iree_ok_status();
  }
  bool materialized = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_try_materialize_def_lane(
      state, value, type, indices, &materialized, out_lane));
  if (materialized) return iree_ok_status();
  return loom_vector_to_scalar_extract_lane(state, value, indices, out_lane);
}
