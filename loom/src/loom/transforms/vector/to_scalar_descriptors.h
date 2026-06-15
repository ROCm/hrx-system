// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor rows for mechanical vector-to-scalar lane lowering.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_DESCRIPTORS_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_DESCRIPTORS_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_vector_to_scalar_lane_kind_e {
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
  LOOM_VECTOR_TO_SCALAR_LANE_TRANSFORM,
  LOOM_VECTOR_TO_SCALAR_LANE_LOAD,
  LOOM_VECTOR_TO_SCALAR_LANE_LOAD_MASK,
  LOOM_VECTOR_TO_SCALAR_LANE_GATHER,
  LOOM_VECTOR_TO_SCALAR_LANE_GATHER_MASK,
  LOOM_VECTOR_TO_SCALAR_LANE_LOAD_EXPAND,
  LOOM_VECTOR_TO_SCALAR_LANE_COUNT,
} loom_vector_to_scalar_lane_kind_t;

typedef enum loom_vector_to_scalar_instance_flag_mode_e {
  LOOM_VECTOR_TO_SCALAR_INSTANCE_FLAG_MODE_DROP = 0,
  LOOM_VECTOR_TO_SCALAR_INSTANCE_FLAG_MODE_FORWARD,
  LOOM_VECTOR_TO_SCALAR_INSTANCE_FLAG_MODE_FAST_MATH,
} loom_vector_to_scalar_instance_flag_mode_t;

typedef struct loom_vector_to_scalar_descriptor_t {
  // Vector op kind matched by this descriptor.
  loom_op_kind_t vector_kind;
  // Op kind emitted per lane for generic mechanical lowering.
  loom_op_kind_t lane_op_kind;
  // Lane program family.
  loom_vector_to_scalar_lane_kind_t lane_kind;
  // Number of vector operands consumed as lane inputs.
  uint8_t lane_operand_count;
  // Number of leading attrs copied from the vector op to the scalar op.
  uint8_t copied_attr_count;
  // Projection applied to op->instance_flags before emitting scalar ops.
  loom_vector_to_scalar_instance_flag_mode_t instance_flag_mode;
  // Whether the scalar result type is i1 instead of the vector element type.
  bool result_is_i1;
  // Operand that can seed a dynamic aggregate loop, or UINT8_MAX.
  uint8_t seed_operand_index;
} loom_vector_to_scalar_descriptor_t;

const loom_vector_to_scalar_descriptor_t* loom_vector_to_scalar_find_descriptor(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_DESCRIPTORS_H_
