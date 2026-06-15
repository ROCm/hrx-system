// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Quantized and packed lane programs.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_QUANTIZED_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_QUANTIZED_H_

#include "loom/transforms/vector/to_scalar_lanes.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_vector_to_scalar_build_bitfield_extract_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, bool signed_extract,
    loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_bitfield_insert_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_dot2f_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_dot4i_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_dot8i4_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_dot4f8_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_bitpack_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_bitunpack_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, bool signed_unpack,
    loom_value_id_t* out_lane);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_QUANTIZED_H_
