// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector memory lane and side-effect lowering.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_MEMORY_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_MEMORY_H_

#include "loom/transforms/vector/to_scalar_lanes.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_vector_to_scalar_build_load_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_masked_load_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_gather_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_masked_gather_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_build_load_expand_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

iree_status_t loom_vector_to_scalar_lower_memory_store(
    loom_vector_to_scalar_state_t* state);

iree_status_t loom_vector_to_scalar_lower_fragment_store(
    loom_vector_to_scalar_state_t* state, bool* out_handled);

// Returns contract rejection bits for the vector.fragment.store forms this
// reference lowering cannot scalarize.
uint32_t loom_vector_to_scalar_fragment_store_reference_rejection_bits(
    loom_vector_to_scalar_state_t* state);

iree_status_t loom_vector_to_scalar_lower_memory_store_compress(
    loom_vector_to_scalar_state_t* state);

iree_status_t loom_vector_to_scalar_lower_memory_atomic_reduce(
    loom_vector_to_scalar_state_t* state);

iree_status_t loom_vector_to_scalar_lower_memory_atomic_rmw(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

iree_status_t loom_vector_to_scalar_lower_memory_atomic_cmpxchg(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_MEMORY_H_
