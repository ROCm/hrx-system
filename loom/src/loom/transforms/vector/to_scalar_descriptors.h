// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lane-program descriptors for vector-to-scalar lowering.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_DESCRIPTORS_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_DESCRIPTORS_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_vector_to_scalar_lane_kind_t;
enum loom_vector_to_scalar_lane_kind_e {
  LOOM_VECTOR_TO_SCALAR_LANE_GENERIC = 0,
  LOOM_VECTOR_TO_SCALAR_LANE_IOTA,
  LOOM_VECTOR_TO_SCALAR_LANE_MASK_RANGE,
  LOOM_VECTOR_TO_SCALAR_LANE_BROADCAST,
  LOOM_VECTOR_TO_SCALAR_LANE_EXTRACT,
  LOOM_VECTOR_TO_SCALAR_LANE_INSERT,
  LOOM_VECTOR_TO_SCALAR_LANE_SLICE,
  LOOM_VECTOR_TO_SCALAR_LANE_CONCAT,
  LOOM_VECTOR_TO_SCALAR_LANE_TRANSPOSE,
  LOOM_VECTOR_TO_SCALAR_LANE_SHUFFLE,
  LOOM_VECTOR_TO_SCALAR_LANE_INTERLEAVE,
  LOOM_VECTOR_TO_SCALAR_LANE_DEINTERLEAVE,
  LOOM_VECTOR_TO_SCALAR_LANE_BITCAST,
  LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_EXTRACTU,
  LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_EXTRACTS,
  LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_INSERT,
  LOOM_VECTOR_TO_SCALAR_LANE_DOT2F,
  LOOM_VECTOR_TO_SCALAR_LANE_DOT4I,
  LOOM_VECTOR_TO_SCALAR_LANE_DOT8I4,
  LOOM_VECTOR_TO_SCALAR_LANE_DOT4F8,
  LOOM_VECTOR_TO_SCALAR_LANE_BITPACK,
  LOOM_VECTOR_TO_SCALAR_LANE_BITUNPACKU,
  LOOM_VECTOR_TO_SCALAR_LANE_BITUNPACKS,
  LOOM_VECTOR_TO_SCALAR_LANE_TABLE_LOOKUP,
  LOOM_VECTOR_TO_SCALAR_LANE_TABLE_QUANTIZE,
  LOOM_VECTOR_TO_SCALAR_LANE_DECODE,
  LOOM_VECTOR_TO_SCALAR_LANE_ENCODE,
  LOOM_VECTOR_TO_SCALAR_LANE_LOAD,
  LOOM_VECTOR_TO_SCALAR_LANE_LOAD_MASK,
  LOOM_VECTOR_TO_SCALAR_LANE_GATHER,
  LOOM_VECTOR_TO_SCALAR_LANE_GATHER_MASK,
  LOOM_VECTOR_TO_SCALAR_LANE_LOAD_EXPAND,
  LOOM_VECTOR_TO_SCALAR_LANE_COUNT,
};
static_assert(LOOM_VECTOR_TO_SCALAR_LANE_COUNT <= UINT8_MAX,
              "vector-to-scalar lane kinds must fit in uint8_t");

typedef struct loom_vector_to_scalar_descriptor_t {
  // Op kind emitted per lane for generic mechanical lowering.
  loom_op_kind_t lane_op_kind;
  // Lane program family.
  loom_vector_to_scalar_lane_kind_t lane_kind;
  // Preferred operand for seeding a dynamic aggregate loop, or UINT8_MAX.
  uint8_t seed_operand_index;
} loom_vector_to_scalar_descriptor_t;

static_assert(sizeof(loom_vector_to_scalar_descriptor_t) == 4,
              "vector-to-scalar descriptors must remain compact");

// Resolves the generated mechanical scalarization relation or an explicit
// reference lane program for |kind|.
bool loom_vector_to_scalar_resolve_descriptor(
    loom_op_kind_t kind, loom_vector_to_scalar_descriptor_t* out_descriptor);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_DESCRIPTORS_H_
