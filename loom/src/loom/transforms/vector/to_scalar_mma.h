// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Matrix contraction reference lowering.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_MMA_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_MMA_H_

#include "loom/transforms/vector/to_scalar_lanes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lowers a vector.mma using scalar reference semantics. Dense logical fragments
// lower directly; target-shaped physical fragments require selected matrix
// fragment layout state and permission to emit subgroup communication ops.
// Returns out_handled=false when the selected lowering state does not cover the
// op's numeric, shape, payload, or placement form.
iree_status_t loom_vector_to_scalar_lower_mma(
    loom_vector_to_scalar_state_t* state, bool* out_handled,
    loom_value_id_t* out_replacement);

// Returns true when result fragment physical lanes can be mapped back to
// logical row/column terms by the shared scalarization helpers.
bool loom_vector_to_scalar_result_fragment_layout_is_supported(
    const loom_matrix_fragment_layout_t* layout);

// Builds logical result row/column terms for a physical result payload register
// addressed by |register_index| in the current subgroup lane.
iree_status_t loom_vector_to_scalar_build_result_fragment_coordinate_terms(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t lane_id,
    loom_vector_to_scalar_index_term_t register_index,
    loom_vector_to_scalar_index_term_t* out_row,
    loom_vector_to_scalar_index_term_t* out_column);

// Materializes one result lane of a vector.mma. Dense full-logical result
// vectors accept their normal vector indices; target-shaped matrix-fragment
// consumers may pass logical rank-2 matrix indices.
// Returns out_materialized=false when the op is not an MMA or uses a shape that
// the target-independent reference decomposition does not cover.
iree_status_t loom_vector_to_scalar_try_materialize_mma_lane(
    loom_vector_to_scalar_state_t* state, loom_op_t* op,
    loom_vector_to_scalar_index_list_t indices, bool* out_materialized,
    loom_value_id_t* out_lane);

// Returns true when every operand needed to materialize logical result lanes is
// present in value facts and covered by the target-independent reference
// semantics. This is a no-IR-emission predicate used before building fragment
// store loops.
bool loom_vector_to_scalar_mma_supports_logical_result_lanes(
    loom_vector_to_scalar_state_t* state, loom_op_t* op);

// Returns contract rejection bits for the vector.mma forms this reference
// lowering cannot scalarize under the selected state.
uint32_t loom_vector_to_scalar_mma_reference_rejection_bits(
    loom_vector_to_scalar_state_t* state);

// Returns the first role-local contract rejection detail for the vector.mma
// forms this reference lowering cannot scalarize under the selected state.
uint32_t loom_vector_to_scalar_mma_reference_rejection_detail(
    loom_vector_to_scalar_state_t* state);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_MMA_H_
