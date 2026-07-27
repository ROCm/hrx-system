// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scalar lane relations for elementwise vector operations.

#ifndef LOOM_OPS_VECTOR_SCALARIZATION_H_
#define LOOM_OPS_VECTOR_SCALARIZATION_H_

#include "loom/ops/vector/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_vector_scalarization_flag_bits_e {
  LOOM_VECTOR_SCALARIZATION_FLAG_NONE = 0u,
  // The scalar lane op has the same fixed operand, result, attribute, and
  // instance-flag shape and can be cloned mechanically per vector lane.
  LOOM_VECTOR_SCALARIZATION_FLAG_MECHANICAL = 1u << 0,
} loom_vector_scalarization_flag_bits_t;
typedef uint8_t loom_vector_scalarization_flags_t;

typedef struct loom_vector_scalarization_t {
  // Scalar operation with the same per-lane numeric semantics.
  loom_op_kind_t lane_op_kind;
  // Bitfield of LOOM_VECTOR_SCALARIZATION_FLAG_* values.
  loom_vector_scalarization_flags_t flags;
  // Preferred result-compatible operand for dynamic reconstruction, or
  // UINT8_MAX when reconstruction should start from an empty aggregate.
  uint8_t seed_operand_index;
} loom_vector_scalarization_t;

static_assert(sizeof(loom_vector_scalarization_t) == 4,
              "vector scalarization rows must remain compact");

// Generated dense scalarization rows indexed by vector dialect op ordinal.
extern const loom_vector_scalarization_t
    loom_vector_scalarization_rows[LOOM_OP_VECTOR_COUNT_];

// Returns generated scalar lane metadata for |kind|, or NULL when no relation
// is recorded for the vector operation.
static inline const loom_vector_scalarization_t*
loom_vector_scalarization_lookup(loom_op_kind_t kind) {
  if (loom_op_dialect_id(kind) != LOOM_DIALECT_VECTOR) return NULL;
  const uint8_t op_index = loom_op_dialect_index(kind);
  if (op_index >= LOOM_OP_VECTOR_COUNT_) return NULL;
  const loom_vector_scalarization_t* scalarization =
      &loom_vector_scalarization_rows[op_index];
  return scalarization->lane_op_kind != LOOM_OP_KIND_UNKNOWN ? scalarization
                                                             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_VECTOR_SCALARIZATION_H_
