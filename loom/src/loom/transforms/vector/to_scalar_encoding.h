// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Encoded vector operand lane reference semantics.
//
// These helpers consume compact encoded-operand schema facts plus ordinary SSA
// auxiliary operands and produce explicit scalar lane programs. They are the
// portable semantic floor for target-independent reference legalizers; native
// target legalizers should select encoded fragments before this path runs.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_ENCODING_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_ENCODING_H_

#include "iree/base/api.h"
#include "loom/ops/vector/encoding_auxiliary.h"
#include "loom/transforms/vector/to_scalar_lanes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vector_to_scalar_encoded_matrix_operand_t {
  // Target-independent interpretation facts for the physical payload.
  loom_value_fact_encoded_operand_schema_t schema;

  // Explicit auxiliary SSA operands keyed by vector auxiliary enum bits.
  loom_vector_encoding_auxiliary_view_t auxiliary;

  // Logical row count for row-indexed auxiliary topologies.
  loom_vector_to_scalar_index_term_t rows;

  // Logical column count for column and block-indexed auxiliary topologies.
  loom_vector_to_scalar_index_term_t columns;
} loom_vector_to_scalar_encoded_matrix_operand_t;

// Returns true when |operand| can be decoded into |result_type| by the generic
// reference lane builder. Unsupported target-fragment, packed-bitstream,
// hierarchical-scale, sparse, or static-codebook schemas return false so a
// later legality diagnostic can report one final unsupported op.
bool loom_vector_to_scalar_encoded_matrix_operand_is_supported(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_type_t raw_lane_type, loom_type_t result_type);

// Returns contract rejection bits for encoded matrix operand forms the generic
// scalar lane builder cannot decode.
uint32_t loom_vector_to_scalar_encoded_matrix_operand_rejection_bits(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_type_t raw_lane_type, loom_type_t result_type);

// Builds one decoded logical lane from |raw_lane|. The caller provides the
// logical matrix coordinates and row-major logical ordinal so scale/table
// topologies remain explicit SSA computations in the generated reference IR.
iree_status_t loom_vector_to_scalar_build_encoded_matrix_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_matrix_operand_t* operand,
    loom_value_id_t raw_lane, loom_type_t raw_lane_type,
    loom_type_t result_type, loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column,
    loom_vector_to_scalar_index_term_t ordinal, loom_value_id_t* out_lane);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_ENCODING_H_
