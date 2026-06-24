// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_quantized.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"

//===----------------------------------------------------------------------===//
// Quantized lane programs
//===----------------------------------------------------------------------===//

iree_status_t loom_vector_to_scalar_build_bitfield_extract_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, bool signed_extract,
    loom_value_id_t* out_lane) {
  loom_value_id_t source =
      signed_extract ? loom_vector_bitfield_extracts_source(state->op)
                     : loom_vector_bitfield_extractu_source(state->op);
  int64_t offset = signed_extract
                       ? loom_vector_bitfield_extracts_offset(state->op)
                       : loom_vector_bitfield_extractu_offset(state->op);
  int64_t width = signed_extract
                      ? loom_vector_bitfield_extracts_width(state->op)
                      : loom_vector_bitfield_extractu_width(state->op);
  loom_type_t source_type =
      loom_module_value_type(state->rewriter->module, source);
  loom_type_t source_scalar_type = loom_vector_to_scalar_lane_type(source_type);
  int32_t source_width =
      loom_scalar_type_bitwidth(loom_type_element_type(source_scalar_type));

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, source, indices, &source_lane));

  loom_value_id_t extracted = LOOM_VALUE_ID_INVALID;
  if (signed_extract) {
    loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
        state, LOOM_OP_SCALAR_SHLI, source_lane, source_scalar_type,
        source_width - offset - width, &shifted_left));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
        state, LOOM_OP_SCALAR_SHRSI, shifted_left, source_scalar_type,
        source_width - width, &extracted));
  } else {
    loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
        state, LOOM_OP_SCALAR_SHRUI, source_lane, source_scalar_type, offset,
        &shifted));
    loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
        state, source_scalar_type, width, &mask));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
        state, LOOM_OP_SCALAR_ANDI, shifted, mask, source_scalar_type,
        &extracted));
  }

  return loom_vector_to_scalar_cast_integer_lane(
      state, extracted, source_scalar_type, state->result_scalar_type,
      signed_extract, out_lane);
}

iree_status_t loom_vector_to_scalar_build_bitfield_insert_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t field = loom_vector_bitfield_insert_field(state->op);
  loom_value_id_t base = loom_vector_bitfield_insert_base(state->op);
  int64_t offset = loom_vector_bitfield_insert_offset(state->op);
  int64_t width = loom_vector_bitfield_insert_width(state->op);
  loom_type_t field_type =
      loom_module_value_type(state->rewriter->module, field);
  loom_type_t field_scalar_type = loom_vector_to_scalar_lane_type(field_type);
  loom_type_t base_scalar_type = state->result_scalar_type;
  int32_t base_width =
      loom_scalar_type_bitwidth(loom_type_element_type(base_scalar_type));

  loom_value_id_t field_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, field, indices, &field_lane));
  loom_value_id_t base_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_materialize_lane(state, base, indices, &base_lane));

  loom_value_id_t field_in_base_type = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, field_lane, field_scalar_type, base_scalar_type,
      /*signed_extend=*/false, &field_in_base_type));

  loom_value_id_t field_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, base_scalar_type, width, &field_mask));
  loom_value_id_t field_low_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, field_in_base_type, field_mask,
      base_scalar_type, &field_low_bits));
  loom_value_id_t shifted_field = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHLI, field_low_bits, base_scalar_type, offset,
      &shifted_field));

  loom_value_id_t target_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, base_scalar_type, state->location,
      loom_vector_to_scalar_shifted_integer_mask_value(base_width, offset,
                                                       width),
      &target_mask));
  loom_value_id_t all_ones = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, base_scalar_type, base_width, &all_ones));
  loom_value_id_t clear_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_XORI, target_mask, all_ones, base_scalar_type,
      &clear_mask));
  loom_value_id_t cleared_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, base_lane, clear_mask, base_scalar_type,
      &cleared_base));
  return loom_vector_to_scalar_build_scalar_binary(state, LOOM_OP_SCALAR_ORI,
                                                   cleared_base, shifted_field,
                                                   base_scalar_type, out_lane);
}

static bool loom_vector_to_scalar_dot4i_lhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT4I_KIND_S8S8 ||
         kind == LOOM_VECTOR_DOT4I_KIND_S8U8;
}

static bool loom_vector_to_scalar_dot4i_rhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT4I_KIND_S8S8 ||
         kind == LOOM_VECTOR_DOT4I_KIND_U8S8;
}

static bool loom_vector_to_scalar_dot8i4_lhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT8I4_KIND_S4S4 ||
         kind == LOOM_VECTOR_DOT8I4_KIND_S4U4;
}

static bool loom_vector_to_scalar_dot8i4_rhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT8I4_KIND_S4S4 ||
         kind == LOOM_VECTOR_DOT8I4_KIND_U4S4;
}

static bool loom_vector_to_scalar_dot4f8_lhs_is_fp8(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT4F8_KIND_FP8BF8 ||
         kind == LOOM_VECTOR_DOT4F8_KIND_FP8FP8;
}

static bool loom_vector_to_scalar_dot4f8_rhs_is_fp8(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT4F8_KIND_BF8FP8 ||
         kind == LOOM_VECTOR_DOT4F8_KIND_FP8FP8;
}

static loom_scalar_type_t loom_vector_to_scalar_dot4f8_field_type(bool is_fp8) {
  return is_fp8 ? LOOM_SCALAR_TYPE_F8E4M3 : LOOM_SCALAR_TYPE_F8E5M2;
}

static iree_status_t loom_vector_to_scalar_build_grouped_source_indices(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    loom_vector_to_scalar_index_term_t grouped_axis_base, uint8_t group_lane,
    loom_vector_to_scalar_index_list_t* out_source_indices) {
  uint8_t rank = result_indices.rank;
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->rewriter->arena, rank, sizeof(loom_vector_to_scalar_index_term_t),
      (void**)&source_terms));
  for (uint8_t axis = 0; axis < rank; ++axis) {
    source_terms[axis] =
        loom_vector_to_scalar_lane_term(state, result_indices, axis);
  }

  uint8_t grouped_axis = (uint8_t)(rank - 1);
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, grouped_axis_base,
      loom_vector_to_scalar_static_term(group_lane),
      &source_terms[grouped_axis]));
  return loom_vector_to_scalar_terms_to_index_list(state, source_terms, rank,
                                                   out_source_indices);
}

iree_status_t loom_vector_to_scalar_build_dot2f_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_dot2f_acc(state->op), indices, &accumulator));
  uint8_t grouped_axis = (uint8_t)(indices.rank - 1);
  loom_vector_to_scalar_index_term_t grouped_axis_base = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL,
      loom_vector_to_scalar_lane_term(state, indices, grouped_axis),
      loom_vector_to_scalar_static_term(2), &grouped_axis_base));

  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  for (uint8_t group_lane = 0; group_lane < 2; ++group_lane) {
    loom_vector_to_scalar_index_list_t source_indices = {0};
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_grouped_source_indices(
        state, indices, grouped_axis_base, group_lane, &source_indices));

    loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, loom_vector_dot2f_lhs(state->op), source_indices, &lhs_lane));
    loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, loom_vector_dot2f_rhs(state->op), source_indices, &rhs_lane));

    loom_value_id_t lhs_f32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_EXTF, 0, (loom_value_id_t[]){lhs_lane}, 1, NULL,
        0, f32_type, &lhs_f32));
    loom_value_id_t rhs_f32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_EXTF, 0, (loom_value_id_t[]){rhs_lane}, 1, NULL,
        0, f32_type, &rhs_f32));

    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_FMAF, 0,
        (loom_value_id_t[]){lhs_f32, rhs_f32, accumulator}, 3, NULL, 0,
        f32_type, &accumulator));
  }
  *out_lane = accumulator;
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_dot4i_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  uint8_t kind = loom_vector_dot4i_kind(state->op);

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_dot4i_acc(state->op), indices, &accumulator));
  uint8_t grouped_axis = (uint8_t)(indices.rank - 1);
  loom_vector_to_scalar_index_term_t grouped_axis_base = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL,
      loom_vector_to_scalar_lane_term(state, indices, grouped_axis),
      loom_vector_to_scalar_static_term(4), &grouped_axis_base));

  loom_type_t i8_type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  for (uint8_t group_lane = 0; group_lane < 4; ++group_lane) {
    loom_vector_to_scalar_index_list_t source_indices = {0};
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_grouped_source_indices(
        state, indices, grouped_axis_base, group_lane, &source_indices));

    loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, loom_vector_dot4i_lhs(state->op), source_indices, &lhs_lane));
    loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, loom_vector_dot4i_rhs(state->op), source_indices, &rhs_lane));

    loom_value_id_t lhs_i32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
        state, lhs_lane, i8_type, i32_type,
        loom_vector_to_scalar_dot4i_lhs_is_signed(kind), &lhs_i32));
    loom_value_id_t rhs_i32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
        state, rhs_lane, i8_type, i32_type,
        loom_vector_to_scalar_dot4i_rhs_is_signed(kind), &rhs_i32));

    loom_value_id_t product = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_MULI, 0, (loom_value_id_t[]){lhs_i32, rhs_i32}, 2,
        NULL, 0, i32_type, &product));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_ADDI, 0,
        (loom_value_id_t[]){accumulator, product}, 2, NULL, 0, i32_type,
        &accumulator));
  }
  *out_lane = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_packed_i4_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t storage_lane,
    loom_value_id_t nibble_mask, loom_value_id_t field_shift,
    loom_value_id_t sign_extend_shift, bool signed_field,
    loom_value_id_t* out_lane) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t shifted_storage = storage_lane;
  if (field_shift != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_SHRUI, 0,
        (loom_value_id_t[]){storage_lane, field_shift}, 2, NULL, 0, i32_type,
        &shifted_storage));
  }
  loom_value_id_t unsigned_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, shifted_storage, nibble_mask, i32_type,
      &unsigned_lane));
  if (!signed_field) {
    *out_lane = unsigned_lane;
    return iree_ok_status();
  }

  loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
      state, LOOM_OP_SCALAR_SHLI, 0,
      (loom_value_id_t[]){unsigned_lane, sign_extend_shift}, 2, NULL, 0,
      i32_type, &shifted_left));
  return loom_vector_to_scalar_build_generic_lane_op(
      state, LOOM_OP_SCALAR_SHRSI, 0,
      (loom_value_id_t[]){shifted_left, sign_extend_shift}, 2, NULL, 0,
      i32_type, out_lane);
}

iree_status_t loom_vector_to_scalar_build_dot8i4_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  uint8_t kind = loom_vector_dot8i4_kind(state->op);

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t lhs_storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_dot8i4_lhs(state->op), indices, &lhs_storage));
  loom_value_id_t rhs_storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_dot8i4_rhs(state->op), indices, &rhs_storage));
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_dot8i4_acc(state->op), indices, &accumulator));

  loom_value_id_t nibble_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, i32_type, 4, &nibble_mask));
  bool lhs_is_signed = loom_vector_to_scalar_dot8i4_lhs_is_signed(kind);
  bool rhs_is_signed = loom_vector_to_scalar_dot8i4_rhs_is_signed(kind);
  loom_value_id_t field_shifts[8] = {
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  for (uint8_t field_ordinal = 1; field_ordinal < 8; ++field_ordinal) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
        &state->rewriter->builder, i32_type, state->location,
        4 * (int64_t)field_ordinal, &field_shifts[field_ordinal]));
  }
  loom_value_id_t sign_extend_shift =
      lhs_is_signed || rhs_is_signed ? field_shifts[7] : LOOM_VALUE_ID_INVALID;
  for (uint8_t field_ordinal = 0; field_ordinal < 8; ++field_ordinal) {
    loom_value_id_t lhs_i32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_packed_i4_lane(
        state, lhs_storage, nibble_mask, field_shifts[field_ordinal],
        sign_extend_shift, lhs_is_signed, &lhs_i32));
    loom_value_id_t rhs_i32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_packed_i4_lane(
        state, rhs_storage, nibble_mask, field_shifts[field_ordinal],
        sign_extend_shift, rhs_is_signed, &rhs_i32));

    loom_value_id_t product = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_MULI, 0, (loom_value_id_t[]){lhs_i32, rhs_i32}, 2,
        NULL, 0, i32_type, &product));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_ADDI, 0,
        (loom_value_id_t[]){accumulator, product}, 2, NULL, 0, i32_type,
        &accumulator));
  }
  *out_lane = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_packed_f8_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t storage_lane,
    loom_value_id_t byte_mask, loom_value_id_t field_shift,
    loom_scalar_type_t float_type, loom_value_id_t* out_lane) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t i8_type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  loom_type_t f8_type = loom_type_scalar(float_type);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_value_id_t shifted_storage = storage_lane;
  if (field_shift != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_SHRUI, 0,
        (loom_value_id_t[]){storage_lane, field_shift}, 2, NULL, 0, i32_type,
        &shifted_storage));
  }

  loom_value_id_t unsigned_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, shifted_storage, byte_mask, i32_type,
      &unsigned_byte));

  loom_value_id_t byte_i8 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, unsigned_byte, i32_type, i8_type, /*signed_extend=*/false,
      &byte_i8));

  loom_value_id_t byte_f8 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
      state, LOOM_OP_SCALAR_BITCAST, 0, (loom_value_id_t[]){byte_i8}, 1, NULL,
      0, f8_type, &byte_f8));

  return loom_vector_to_scalar_build_generic_lane_op(
      state, LOOM_OP_SCALAR_EXTF, 0, (loom_value_id_t[]){byte_f8}, 1, NULL, 0,
      f32_type, out_lane);
}

iree_status_t loom_vector_to_scalar_build_dot4f8_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  uint8_t kind = loom_vector_dot4f8_kind(state->op);

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t lhs_storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_dot4f8_lhs(state->op), indices, &lhs_storage));
  loom_value_id_t rhs_storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_dot4f8_rhs(state->op), indices, &rhs_storage));
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_dot4f8_acc(state->op), indices, &accumulator));

  loom_value_id_t byte_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_integer_mask(state, i32_type, 8, &byte_mask));
  loom_scalar_type_t lhs_float_type = loom_vector_to_scalar_dot4f8_field_type(
      loom_vector_to_scalar_dot4f8_lhs_is_fp8(kind));
  loom_scalar_type_t rhs_float_type = loom_vector_to_scalar_dot4f8_field_type(
      loom_vector_to_scalar_dot4f8_rhs_is_fp8(kind));
  loom_value_id_t field_shifts[4] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint8_t field_ordinal = 1; field_ordinal < 4; ++field_ordinal) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
        &state->rewriter->builder, i32_type, state->location,
        8 * (int64_t)field_ordinal, &field_shifts[field_ordinal]));
  }
  for (uint8_t field_ordinal = 0; field_ordinal < 4; ++field_ordinal) {
    loom_value_id_t lhs_f32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_packed_f8_lane(
        state, lhs_storage, byte_mask, field_shifts[field_ordinal],
        lhs_float_type, &lhs_f32));
    loom_value_id_t rhs_f32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_packed_f8_lane(
        state, rhs_storage, byte_mask, field_shifts[field_ordinal],
        rhs_float_type, &rhs_f32));

    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_FMAF, 0,
        (loom_value_id_t[]){lhs_f32, rhs_f32, accumulator}, 3, NULL, 0,
        f32_type, &accumulator));
  }
  *out_lane = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_zero_lane(
    loom_vector_to_scalar_state_t* state, loom_type_t type,
    loom_value_id_t* out_value) {
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, type, state->location, 0, out_value);
}

static loom_type_t loom_vector_to_scalar_integer_work_type(
    loom_type_t scalar_type) {
  int32_t bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(scalar_type));
  return loom_type_scalar(bit_width <= 32 ? LOOM_SCALAR_TYPE_I32
                                          : LOOM_SCALAR_TYPE_I64);
}

static iree_status_t loom_vector_to_scalar_build_bitpack_piece(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source_work_lane,
    loom_type_t source_work_type, loom_type_t storage_work_type,
    int64_t source_shift, int64_t bit_count, int64_t dest_shift,
    loom_value_id_t accumulator, loom_value_id_t* out_accumulator) {
  loom_value_id_t shifted_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHRUI, source_work_lane, source_work_type,
      source_shift, &shifted_source));
  loom_value_id_t source_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, source_work_type, bit_count, &source_mask));
  loom_value_id_t source_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, shifted_source, source_mask, source_work_type,
      &source_bits));
  loom_value_id_t storage_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, source_bits, source_work_type, storage_work_type,
      /*signed_extend=*/false, &storage_bits));
  loom_value_id_t shifted_storage_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHLI, storage_bits, storage_work_type, dest_shift,
      &shifted_storage_bits));
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ORI, accumulator, shifted_storage_bits,
      storage_work_type, out_accumulator);
}

static iree_status_t loom_vector_to_scalar_build_bitpack_static_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, int64_t width,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t source_scalar_type = loom_vector_to_scalar_lane_type(source_type);
  loom_type_t storage_scalar_type = state->result_scalar_type;
  loom_type_t source_work_type =
      loom_vector_to_scalar_integer_work_type(source_scalar_type);
  loom_type_t storage_work_type =
      loom_vector_to_scalar_integer_work_type(storage_scalar_type);
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));
  int64_t storage_ordinal = loom_vector_to_scalar_linear_ordinal_static(
      state->vector_type, indices.static_indices);
  int64_t storage_start = storage_ordinal * storage_width;
  int64_t storage_end = storage_start + storage_width;
  int64_t first_source_ordinal = storage_start / width;
  int64_t last_source_ordinal = (storage_end - 1) / width;

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_zero_lane(
      state, storage_work_type, &accumulator));
  for (int64_t source_ordinal = first_source_ordinal;
       source_ordinal <= last_source_ordinal; ++source_ordinal) {
    int64_t source_start = source_ordinal * width;
    int64_t source_end = source_start + width;
    int64_t overlap_start =
        storage_start > source_start ? storage_start : source_start;
    int64_t overlap_end = storage_end < source_end ? storage_end : source_end;
    int64_t bit_count = overlap_end - overlap_start;
    if (bit_count <= 0) continue;

    loom_value_id_t source_work_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_materialize_cached_static_integer_lane(
            state, source, source_type, source_ordinal, source_work_type,
            LOOM_VECTOR_TO_SCALAR_INTEGER_EXTENSION_ZERO, &source_work_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitpack_piece(
        state, source_work_lane, source_work_type, storage_work_type,
        overlap_start - source_start, bit_count, overlap_start - storage_start,
        accumulator, &accumulator));
  }
  return loom_vector_to_scalar_cast_integer_lane(
      state, accumulator, storage_work_type, storage_scalar_type,
      /*signed_extend=*/false, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitpack_divisible_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, int64_t width,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t source_scalar_type = loom_vector_to_scalar_lane_type(source_type);
  loom_type_t storage_scalar_type = state->result_scalar_type;
  loom_type_t source_work_type =
      loom_vector_to_scalar_integer_work_type(source_scalar_type);
  loom_type_t storage_work_type =
      loom_vector_to_scalar_integer_work_type(storage_scalar_type);
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));
  int64_t fields_per_storage = storage_width / width;

  loom_vector_to_scalar_index_term_t storage_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_linear_ordinal_term(
      state, state->vector_type, indices, &storage_ordinal));
  loom_vector_to_scalar_index_term_t source_base = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL, storage_ordinal,
      loom_vector_to_scalar_static_term(fields_per_storage), &source_base));

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_zero_lane(
      state, storage_work_type, &accumulator));
  for (int64_t field_ordinal = 0; field_ordinal < fields_per_storage;
       ++field_ordinal) {
    loom_vector_to_scalar_index_term_t source_ordinal = {0};
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, source_base,
        loom_vector_to_scalar_static_term(field_ordinal), &source_ordinal));
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_linear_lane(
        state, source, source_type, source_ordinal, &source_lane));
    loom_value_id_t source_work_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
        state, source_lane, source_scalar_type, source_work_type,
        /*signed_extend=*/false, &source_work_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitpack_piece(
        state, source_work_lane, source_work_type, storage_work_type,
        /*source_shift=*/0, width, field_ordinal * width, accumulator,
        &accumulator));
  }
  return loom_vector_to_scalar_cast_integer_lane(
      state, accumulator, storage_work_type, storage_scalar_type,
      /*signed_extend=*/false, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitpack_dynamic_bit(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, loom_type_t source_scalar_type,
    loom_type_t source_work_type, loom_type_t storage_work_type,
    loom_vector_to_scalar_index_term_t storage_base_bit,
    loom_vector_to_scalar_index_term_t storage_bit, int64_t width,
    loom_value_id_t source_one, loom_value_id_t accumulator,
    loom_value_id_t* out_accumulator) {
  loom_vector_to_scalar_index_term_t global_bit = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, storage_base_bit,
      storage_bit, &global_bit));
  loom_vector_to_scalar_index_term_t source_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV, global_bit,
      loom_vector_to_scalar_static_term(width), &source_ordinal));
  loom_vector_to_scalar_index_term_t source_shift = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM, global_bit,
      loom_vector_to_scalar_static_term(width), &source_shift));

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_linear_lane(
      state, source, source_type, source_ordinal, &source_lane));
  loom_value_id_t source_work_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, source_lane, source_scalar_type, source_work_type,
      /*signed_extend=*/false, &source_work_lane));
  loom_value_id_t source_bit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_single_bit_extract(
      state, source_work_lane, source_work_type, source_shift, source_one,
      &source_bit));
  loom_value_id_t storage_bit_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, source_bit, source_work_type, storage_work_type,
      /*signed_extend=*/false, &storage_bit_value));
  loom_value_id_t shifted_storage_bit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift_term(
      state, LOOM_OP_SCALAR_SHLI, storage_bit_value, storage_work_type,
      storage_bit, &shifted_storage_bit));
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ORI, accumulator, shifted_storage_bit,
      storage_work_type, out_accumulator);
}

static iree_status_t loom_vector_to_scalar_build_bitpack_bitstream_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, int64_t width,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t source_scalar_type = loom_vector_to_scalar_lane_type(source_type);
  loom_type_t storage_scalar_type = state->result_scalar_type;
  loom_type_t source_work_type =
      loom_vector_to_scalar_integer_work_type(source_scalar_type);
  loom_type_t storage_work_type =
      loom_vector_to_scalar_integer_work_type(storage_scalar_type);
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));

  loom_vector_to_scalar_index_term_t storage_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_linear_ordinal_term(
      state, state->vector_type, indices, &storage_ordinal));
  loom_vector_to_scalar_index_term_t storage_base_bit = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitstream_base_term(
      state, storage_ordinal, storage_width, &storage_base_bit));

  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &lower_bound));
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, storage_width, &upper_bound));
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, &step));
  loom_value_id_t source_one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, source_work_type, state->location, 1,
      &source_one));
  loom_value_id_t initial_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_zero_lane(
      state, storage_work_type, &initial_accumulator));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, &initial_accumulator, 1, &storage_work_type, 1, NULL, 0,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0, /*unroll_schedule=*/0,
      state->location, &loop));
  loom_vector_to_scalar_record_loop_created(state);

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_value_id_t storage_bit_value =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t accumulator_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_vector_to_scalar_index_term_t storage_bit =
      loom_vector_to_scalar_value_term(state, storage_bit_value);
  loom_value_id_t yielded_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitpack_dynamic_bit(
      state, source, source_type, source_scalar_type, source_work_type,
      storage_work_type, storage_base_bit, storage_bit, width, source_one,
      accumulator_arg, &yielded_accumulator));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &yielded_accumulator, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  return loom_vector_to_scalar_cast_integer_lane(
      state, loom_scf_for_results(loop).values[0], storage_work_type,
      storage_scalar_type, /*signed_extend=*/false, out_lane);
}

iree_status_t loom_vector_to_scalar_build_bitpack_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t source = loom_vector_bitpack_source(state->op);
  loom_type_t source_type =
      loom_module_value_type(state->rewriter->module, source);
  int64_t width = loom_vector_bitpack_width(state->op);
  int32_t storage_width = loom_scalar_type_bitwidth(
      loom_type_element_type(state->result_scalar_type));

  if (!loom_vector_to_scalar_indices_are_dynamic(indices) &&
      loom_type_is_all_static(source_type)) {
    return loom_vector_to_scalar_build_bitpack_static_lane(
        state, source, source_type, width, indices, out_lane);
  }
  if ((storage_width % width) == 0) {
    return loom_vector_to_scalar_build_bitpack_divisible_lane(
        state, source, source_type, width, indices, out_lane);
  }
  return loom_vector_to_scalar_build_bitpack_bitstream_lane(
      state, source, source_type, width, indices, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitunpack_piece(
    loom_vector_to_scalar_state_t* state, loom_value_id_t storage_work_lane,
    loom_type_t storage_work_type, loom_type_t result_work_type,
    int64_t source_shift, int64_t bit_count, int64_t dest_shift,
    loom_value_id_t accumulator, loom_value_id_t* out_accumulator) {
  loom_value_id_t shifted_storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHRUI, storage_work_lane, storage_work_type,
      source_shift, &shifted_storage));
  loom_value_id_t storage_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, storage_work_type, bit_count, &storage_mask));
  loom_value_id_t storage_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, shifted_storage, storage_mask,
      storage_work_type, &storage_bits));
  loom_value_id_t result_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, storage_bits, storage_work_type, result_work_type,
      /*signed_extend=*/false, &result_bits));
  loom_value_id_t shifted_result_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHLI, result_bits, result_work_type, dest_shift,
      &shifted_result_bits));
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ORI, accumulator, shifted_result_bits,
      result_work_type, out_accumulator);
}

static iree_status_t loom_vector_to_scalar_finish_bitunpack_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t unsigned_lane,
    loom_type_t result_work_type, int64_t width, bool signed_unpack,
    loom_value_id_t* out_lane) {
  loom_value_id_t result_lane = unsigned_lane;
  if (signed_unpack) {
    int32_t work_width =
        loom_scalar_type_bitwidth(loom_type_element_type(result_work_type));
    loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
        state, LOOM_OP_SCALAR_SHLI, unsigned_lane, result_work_type,
        work_width - width, &shifted_left));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
        state, LOOM_OP_SCALAR_SHRSI, shifted_left, result_work_type,
        work_width - width, &result_lane));
  }
  return loom_vector_to_scalar_cast_integer_lane(
      state, result_lane, result_work_type, state->result_scalar_type,
      signed_unpack, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitunpack_static_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, int64_t width, bool signed_unpack,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t storage_scalar_type =
      loom_vector_to_scalar_lane_type(source_type);
  loom_type_t result_scalar_type = state->result_scalar_type;
  loom_type_t storage_work_type =
      loom_vector_to_scalar_integer_work_type(storage_scalar_type);
  loom_type_t result_work_type =
      loom_vector_to_scalar_integer_work_type(result_scalar_type);
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));
  int64_t result_ordinal = loom_vector_to_scalar_linear_ordinal_static(
      state->vector_type, indices.static_indices);
  int64_t result_start = result_ordinal * width;
  int64_t result_end = result_start + width;
  int64_t first_storage_ordinal = result_start / storage_width;
  int64_t last_storage_ordinal = (result_end - 1) / storage_width;

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_zero_lane(
      state, result_work_type, &accumulator));
  for (int64_t storage_ordinal = first_storage_ordinal;
       storage_ordinal <= last_storage_ordinal; ++storage_ordinal) {
    int64_t storage_start = storage_ordinal * storage_width;
    int64_t storage_end = storage_start + storage_width;
    int64_t overlap_start =
        result_start > storage_start ? result_start : storage_start;
    int64_t overlap_end = result_end < storage_end ? result_end : storage_end;
    int64_t bit_count = overlap_end - overlap_start;
    if (bit_count <= 0) continue;

    loom_value_id_t storage_work_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_materialize_cached_static_integer_lane(
            state, source, source_type, storage_ordinal, storage_work_type,
            LOOM_VECTOR_TO_SCALAR_INTEGER_EXTENSION_ZERO, &storage_work_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitunpack_piece(
        state, storage_work_lane, storage_work_type, result_work_type,
        overlap_start - storage_start, bit_count, overlap_start - result_start,
        accumulator, &accumulator));
  }
  return loom_vector_to_scalar_finish_bitunpack_lane(
      state, accumulator, result_work_type, width, signed_unpack, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitunpack_divisible_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, int64_t width, bool signed_unpack,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t storage_scalar_type =
      loom_vector_to_scalar_lane_type(source_type);
  loom_type_t result_scalar_type = state->result_scalar_type;
  loom_type_t storage_work_type =
      loom_vector_to_scalar_integer_work_type(storage_scalar_type);
  loom_type_t result_work_type =
      loom_vector_to_scalar_integer_work_type(result_scalar_type);
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));
  int64_t fields_per_storage = storage_width / width;
  loom_vector_to_scalar_index_term_t result_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_linear_ordinal_term(
      state, state->vector_type, indices, &result_ordinal));
  loom_vector_to_scalar_index_term_t storage_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV, result_ordinal,
      loom_vector_to_scalar_static_term(fields_per_storage), &storage_ordinal));
  loom_vector_to_scalar_index_term_t field_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM, result_ordinal,
      loom_vector_to_scalar_static_term(fields_per_storage), &field_ordinal));
  loom_value_id_t storage_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_linear_lane(
      state, source, source_type, storage_ordinal, &storage_lane));
  loom_value_id_t storage_work_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, storage_lane, storage_scalar_type, storage_work_type,
      /*signed_extend=*/false, &storage_work_lane));
  loom_vector_to_scalar_index_term_t shift = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL, field_ordinal,
      loom_vector_to_scalar_static_term(width), &shift));
  loom_value_id_t shifted_storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift_term(
      state, LOOM_OP_SCALAR_SHRUI, storage_work_lane, storage_work_type, shift,
      &shifted_storage));
  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, storage_work_type, width, &mask));
  loom_value_id_t storage_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, shifted_storage, mask, storage_work_type,
      &storage_bits));
  loom_value_id_t result_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, storage_bits, storage_work_type, result_work_type,
      /*signed_extend=*/false, &result_bits));
  return loom_vector_to_scalar_finish_bitunpack_lane(
      state, result_bits, result_work_type, width, signed_unpack, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitunpack_dynamic_bit(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, loom_type_t storage_scalar_type,
    loom_type_t storage_work_type, loom_type_t result_work_type,
    loom_vector_to_scalar_index_term_t result_base_bit,
    loom_vector_to_scalar_index_term_t result_bit, int32_t storage_width,
    loom_value_id_t storage_one, loom_value_id_t accumulator,
    loom_value_id_t* out_accumulator) {
  loom_vector_to_scalar_index_term_t global_bit = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, result_base_bit,
      result_bit, &global_bit));
  loom_vector_to_scalar_index_term_t storage_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV, global_bit,
      loom_vector_to_scalar_static_term(storage_width), &storage_ordinal));
  loom_vector_to_scalar_index_term_t storage_shift = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM, global_bit,
      loom_vector_to_scalar_static_term(storage_width), &storage_shift));

  loom_value_id_t storage_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_linear_lane(
      state, source, source_type, storage_ordinal, &storage_lane));
  loom_value_id_t storage_work_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, storage_lane, storage_scalar_type, storage_work_type,
      /*signed_extend=*/false, &storage_work_lane));
  loom_value_id_t storage_bit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_single_bit_extract(
      state, storage_work_lane, storage_work_type, storage_shift, storage_one,
      &storage_bit));
  loom_value_id_t result_bit_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, storage_bit, storage_work_type, result_work_type,
      /*signed_extend=*/false, &result_bit_value));
  loom_value_id_t shifted_result_bit = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift_term(
      state, LOOM_OP_SCALAR_SHLI, result_bit_value, result_work_type,
      result_bit, &shifted_result_bit));
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ORI, accumulator, shifted_result_bit,
      result_work_type, out_accumulator);
}

static iree_status_t loom_vector_to_scalar_build_bitunpack_bitstream_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, int64_t width, bool signed_unpack,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t storage_scalar_type =
      loom_vector_to_scalar_lane_type(source_type);
  loom_type_t result_scalar_type = state->result_scalar_type;
  loom_type_t storage_work_type =
      loom_vector_to_scalar_integer_work_type(storage_scalar_type);
  loom_type_t result_work_type =
      loom_vector_to_scalar_integer_work_type(result_scalar_type);
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));

  loom_vector_to_scalar_index_term_t result_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_linear_ordinal_term(
      state, state->vector_type, indices, &result_ordinal));
  loom_vector_to_scalar_index_term_t result_base_bit = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitstream_base_term(
      state, result_ordinal, width, &result_base_bit));

  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &lower_bound));
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, width, &upper_bound));
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, &step));
  loom_value_id_t storage_one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, storage_work_type, state->location, 1,
      &storage_one));
  loom_value_id_t initial_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_zero_lane(
      state, result_work_type, &initial_accumulator));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, /*build_flags=*/0, lower_bound, upper_bound,
      step, &initial_accumulator, 1, &result_work_type, 1, NULL, 0,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0, /*unroll_schedule=*/0,
      state->location, &loop));
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
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitunpack_dynamic_bit(
      state, source, source_type, storage_scalar_type, storage_work_type,
      result_work_type, result_base_bit, result_bit, storage_width, storage_one,
      accumulator_arg, &yielded_accumulator));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &yielded_accumulator, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  return loom_vector_to_scalar_finish_bitunpack_lane(
      state, loom_scf_for_results(loop).values[0], result_work_type, width,
      signed_unpack, out_lane);
}

iree_status_t loom_vector_to_scalar_build_bitunpack_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, bool signed_unpack,
    loom_value_id_t* out_lane) {
  loom_value_id_t source = signed_unpack
                               ? loom_vector_bitunpacks_source(state->op)
                               : loom_vector_bitunpacku_source(state->op);
  loom_type_t source_type =
      loom_module_value_type(state->rewriter->module, source);
  int64_t width = signed_unpack ? loom_vector_bitunpacks_width(state->op)
                                : loom_vector_bitunpacku_width(state->op);
  loom_type_t storage_scalar_type =
      loom_vector_to_scalar_lane_type(source_type);
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));

  if (!loom_vector_to_scalar_indices_are_dynamic(indices) &&
      loom_type_is_all_static(source_type)) {
    return loom_vector_to_scalar_build_bitunpack_static_lane(
        state, source, source_type, width, signed_unpack, indices, out_lane);
  }
  if ((storage_width % width) == 0) {
    return loom_vector_to_scalar_build_bitunpack_divisible_lane(
        state, source, source_type, width, signed_unpack, indices, out_lane);
  }
  return loom_vector_to_scalar_build_bitunpack_bitstream_lane(
      state, source, source_type, width, signed_unpack, indices, out_lane);
}
