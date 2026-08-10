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
#include "loom/ops/encoding/auxiliary.h"
#include "loom/transforms/vector/to_scalar_lanes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_vector_to_scalar_encoding_direction_t;
enum loom_vector_to_scalar_encoding_direction_e {
  // Convert physical encoded lanes into logical numeric lanes.
  LOOM_VECTOR_TO_SCALAR_ENCODING_DIRECTION_DECODE = 0,
  // Convert logical numeric lanes into physical encoded lanes.
  LOOM_VECTOR_TO_SCALAR_ENCODING_DIRECTION_ENCODE = 1,
};

typedef struct loom_vector_to_scalar_encoded_operand_t {
  // Target-independent interpretation facts for the physical payload.
  loom_value_fact_encoded_operand_schema_t schema;

  // Explicit auxiliary SSA operands keyed by vector auxiliary enum bits.
  loom_encoding_auxiliary_view_t auxiliary;

  // Independent logical block count for block-indexed auxiliary topologies.
  loom_vector_to_scalar_index_term_t blocks;

  // Logical row count for row-indexed auxiliary topologies.
  loom_vector_to_scalar_index_term_t rows;

  // Logical column count for column and block-indexed auxiliary topologies.
  loom_vector_to_scalar_index_term_t columns;

  // Logical numeric lane type before encoding or after decoding.
  loom_type_t logical_lane_type;

  // Physical payload lane type after encoding or before decoding.
  loom_type_t physical_lane_type;

  // Semantic conversion direction.
  loom_vector_to_scalar_encoding_direction_t direction;
} loom_vector_to_scalar_encoded_operand_t;

// Returns true when |operand| has a generic reference lane implementation.
// Unsupported target-fragment, packed-bitstream, hierarchical-scale, sparse,
// or noninvertible schemas return false so a later legality diagnostic can
// report one final unsupported op.
bool loom_vector_to_scalar_encoded_operand_is_supported(
    const loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_operand_t* operand);

// Returns contract rejection bits for standalone vector.encode/decode
// scalarization.
uint32_t loom_vector_to_scalar_encoding_rejection_bits(
    loom_vector_to_scalar_state_t* state);

// Returns contract rejection bits for forms the generic scalar lane builder
// cannot implement in the requested direction.
uint32_t loom_vector_to_scalar_encoded_operand_rejection_bits(
    const loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_operand_t* operand);

// Builds one decoded logical lane from |physical_lane|. The caller provides the
// logical matrix block/row/column coordinates and row-major logical ordinal so
// scale/table topologies remain explicit SSA computations in the generated
// reference IR.
iree_status_t loom_vector_to_scalar_build_decoded_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_encoded_operand_t* operand,
    loom_value_id_t physical_lane, loom_vector_to_scalar_index_term_t block,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column,
    loom_vector_to_scalar_index_term_t ordinal, loom_value_id_t* out_lane);

// Builds one lane of a supported standalone vector.decode op.
iree_status_t loom_vector_to_scalar_build_decode_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

// Builds one lane of a supported standalone vector.encode op.
iree_status_t loom_vector_to_scalar_build_encode_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_ENCODING_H_
