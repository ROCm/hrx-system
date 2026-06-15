// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared lane materialization and scalar helper emission.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_LANES_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_LANES_H_

#include "loom/transforms/vector/to_scalar_descriptors.h"
#include "loom/transforms/vector/to_scalar_terms.h"

#ifdef __cplusplus
extern "C" {
#endif

loom_type_t loom_vector_to_scalar_lane_type(loom_type_t vector_type);

uint8_t loom_vector_to_scalar_project_instance_flags(
    loom_vector_to_scalar_instance_flag_mode_t mode, uint8_t instance_flags);

iree_status_t loom_vector_to_scalar_build_scalar_constant(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, int64_t integer_value,
    loom_value_id_t* out_value_id);

// Builds a scalar or index constant using |value| exactly as the constant
// attribute, preserving non-integer floating-point constants such as transform
// normalization scales.
iree_status_t loom_vector_to_scalar_build_scalar_attr_constant(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, loom_attribute_t value,
    loom_value_id_t* out_value_id);

iree_status_t loom_vector_to_scalar_build_vector_zero(
    loom_vector_to_scalar_state_t* state, loom_type_t result_type,
    loom_value_id_t* out_value_id);

iree_status_t loom_vector_to_scalar_build_generic_lane_op(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    uint8_t instance_flags, const loom_value_id_t* operands,
    uint16_t operand_count, const loom_attribute_t* attrs, uint8_t attr_count,
    loom_type_t result_type, loom_value_id_t* out_result);

iree_status_t loom_vector_to_scalar_build_scalar_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    loom_value_id_t* out_result);

int64_t loom_vector_to_scalar_integer_mask_value(int32_t bit_width,
                                                 int64_t used_bits);

int64_t loom_vector_to_scalar_shifted_integer_mask_value(int32_t bit_width,
                                                         int64_t offset,
                                                         int64_t used_bits);

iree_status_t loom_vector_to_scalar_build_integer_mask(
    loom_vector_to_scalar_state_t* state, loom_type_t type, int64_t used_bits,
    loom_value_id_t* out_mask);

iree_status_t loom_vector_to_scalar_cast_integer_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type, bool signed_extend,
    loom_value_id_t* out_result);

iree_status_t loom_vector_to_scalar_build_scalar_shift(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t input, loom_type_t type, int64_t amount,
    loom_value_id_t* out_result);

iree_status_t loom_vector_to_scalar_build_index_term_as_scalar(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t term, loom_type_t result_type,
    loom_value_id_t* out_value);

iree_status_t loom_vector_to_scalar_build_scalar_shift_term(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t input, loom_type_t type,
    loom_vector_to_scalar_index_term_t amount, loom_value_id_t* out_result);

iree_status_t loom_vector_to_scalar_build_bitstream_base_term(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t lane_ordinal, int64_t lane_bit_width,
    loom_vector_to_scalar_index_term_t* out_position);

iree_status_t loom_vector_to_scalar_build_single_bit_extract(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lane,
    loom_type_t lane_type, loom_vector_to_scalar_index_term_t bit_shift,
    loom_value_id_t one_mask, loom_value_id_t* out_bit);

iree_status_t loom_vector_to_scalar_build_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_materialize_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_materialize_linear_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t vector_value,
    loom_type_t vector_type, loom_vector_to_scalar_index_term_t ordinal,
    loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_insert_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lane,
    loom_value_id_t aggregate, loom_type_t aggregate_type,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_aggregate);

iree_status_t loom_vector_to_scalar_dim_bound(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type, uint8_t axis,
    loom_value_id_t* out_bound);

iree_status_t loom_vector_to_scalar_build_i1_and(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result);

iree_status_t loom_vector_to_scalar_build_select_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t condition,
    loom_value_id_t true_lane, loom_value_id_t false_lane,
    loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_try_materialize_def_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_type_t vector_type, loom_vector_to_scalar_index_list_t indices,
    bool* out_materialized, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_constant_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_poison_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_lane);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_LANES_H_
