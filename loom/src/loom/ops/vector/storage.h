// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector storage shape helpers.
//
// These helpers describe source-level vector storage properties without naming
// a target register file. Targets can apply their own lane/register caps on top
// of these facts while sharing the same type-shape interpretation.

#ifndef LOOM_OPS_VECTOR_STORAGE_H_
#define LOOM_OPS_VECTOR_STORAGE_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packed integer storage shape for a rank-1 vector type.
typedef struct loom_vector_packed_integer_storage_shape_t {
  // Source vector type being described.
  loom_type_t type;
  // Integer element type carried by each logical source lane.
  loom_scalar_type_t element_type;
  // Number of logical lanes in the static rank-1 vector.
  uint32_t lane_count;
  // Bit count of each logical source lane.
  uint32_t element_bit_count;
  // Total payload bits occupied by all logical lanes.
  uint32_t payload_bit_count;
  // Bit count of one target-selected storage unit.
  uint32_t storage_unit_bit_count;
  // Number of storage units needed to hold payload_bit_count bits.
  uint32_t storage_unit_count;
} loom_vector_packed_integer_storage_shape_t;

// Storage relationship for logical integer lanes packed into payload storage.
typedef struct loom_vector_packed_integer_payload_from_lanes_match_t {
  // Shape of the logical source lanes before packing.
  loom_vector_packed_integer_storage_shape_t source_shape;
  // Shape of the integer storage lanes after packing.
  loom_vector_packed_integer_storage_shape_t result_shape;
  // Bit width packed from each source lane.
  uint32_t width;
} loom_vector_packed_integer_payload_from_lanes_match_t;

// Storage relationship for payload storage unpacked into logical integer lanes.
typedef struct loom_vector_packed_integer_lanes_from_payload_match_t {
  // Shape of the integer storage lanes before unpacking.
  loom_vector_packed_integer_storage_shape_t source_shape;
  // Shape of the logical result lanes after unpacking.
  loom_vector_packed_integer_storage_shape_t result_shape;
  // Bit width unpacked from each storage field.
  uint32_t width;
  // Number of logical result lanes produced from the source payload.
  uint32_t lane_count;
} loom_vector_packed_integer_lanes_from_payload_match_t;

// Returns the rank-1 static lane count for |type| when it has |element_type|
// and is within |maximum_lane_count|. Returns zero for dynamic, non-rank-1,
// wrong-element, empty, or over-limit vectors.
uint32_t loom_vector_static_rank1_lane_count(loom_type_t type,
                                             loom_scalar_type_t element_type,
                                             uint32_t maximum_lane_count);

// Describes integer bitstream storage for a rank-1 static vector in units of
// |storage_unit_bit_count| bits. Returns false for non-integer vectors,
// dynamic/non-rank-1 vectors, arithmetic overflow, empty payloads, or shapes
// requiring more than |maximum_storage_unit_count| units.
bool loom_vector_packed_integer_storage_shape(
    loom_type_t type, uint32_t storage_unit_bit_count,
    uint32_t maximum_storage_unit_count,
    loom_vector_packed_integer_storage_shape_t* out_shape);

// Returns true when |source_type| logical integer lanes can produce
// |result_type| packed integer payload storage under the given storage unit
// size and result storage limit. The caller owns target-specific type patterns
// and any additional payload multiple constraints.
bool loom_vector_packed_integer_payload_from_lanes_match(
    loom_type_t source_type, loom_type_t result_type, uint32_t width,
    uint32_t storage_unit_bit_count, uint32_t maximum_result_storage_unit_count,
    loom_vector_packed_integer_payload_from_lanes_match_t* out_match);

// Returns true when |source_type| packed integer payload storage can produce
// |result_type| logical integer lanes under the given storage unit size, source
// storage limit, and lane limit. The caller owns target-specific type patterns.
bool loom_vector_packed_integer_lanes_from_payload_match(
    loom_type_t source_type, loom_type_t result_type, uint32_t width,
    uint32_t storage_unit_bit_count, uint32_t maximum_source_storage_unit_count,
    uint32_t maximum_lane_count,
    loom_vector_packed_integer_lanes_from_payload_match_t* out_match);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_VECTOR_STORAGE_H_
