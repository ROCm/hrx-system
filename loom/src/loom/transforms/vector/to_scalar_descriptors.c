// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/to_scalar_descriptors.h"

#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/scalarization.h"

// Explicit lane programs for vector operations that cannot be cloned as one
// scalar counterpart per lane. Mechanical elementwise relations come from the
// generated vector scalarization table.
// clang-format off
#define LOOM_VECTOR_TO_SCALAR_VECTOR_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))
#define LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(op, lane_op_kind_, lane_kind_, seed_operand_index_) \
  [LOOM_VECTOR_TO_SCALAR_VECTOR_OP_INDEX(LOOM_OP_VECTOR_##op)] = {                               \
      .lane_op_kind = (lane_op_kind_),                                                            \
      .lane_kind = (lane_kind_),                                                                  \
      .seed_operand_index = (seed_operand_index_),                                                \
  }

static_assert(LOOM_OP_VECTOR_COUNT_ <= UINT8_MAX,
              "vector op indexes must fit in the descriptor lookup key");

static const loom_vector_to_scalar_descriptor_t
    kVectorToScalarExplicitDescriptors[LOOM_OP_VECTOR_COUNT_] = {
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(SELECT, LOOM_OP_SCF_SELECT,
     LOOM_VECTOR_TO_SCALAR_LANE_GENERIC, 1),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(DECODE, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_DECODE, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(ENCODE, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_ENCODE, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(IOTA, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_IOTA, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(MASK_RANGE, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_MASK_RANGE, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(BROADCAST, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_BROADCAST, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(EXTRACT, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_EXTRACT, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(INSERT, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_INSERT, 1),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(SLICE, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_SLICE, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(CONCAT, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_CONCAT, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(TRANSPOSE, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_TRANSPOSE, 0),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(SHUFFLE, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_SHUFFLE, 0),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(INTERLEAVE, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_INTERLEAVE, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(DEINTERLEAVE, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_DEINTERLEAVE, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(BITCAST, LOOM_OP_SCALAR_BITCAST,
     LOOM_VECTOR_TO_SCALAR_LANE_BITCAST, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(BITFIELD_EXTRACTU, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_EXTRACTU, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(BITFIELD_EXTRACTS, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_EXTRACTS, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(BITFIELD_INSERT, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_INSERT, 1),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(DOT2F, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_DOT2F, 2),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(DOT4I, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_DOT4I, 2),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(DOT8I4, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_DOT8I4, 2),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(DOT4F8, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_DOT4F8, 2),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(BITPACK, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_BITPACK, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(BITUNPACKU, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_BITUNPACKU, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(BITUNPACKS, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_BITUNPACKS, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(TABLE_LOOKUP, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_TABLE_LOOKUP, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(TABLE_QUANTIZE, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_TABLE_QUANTIZE, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(LOAD, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_LOAD, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(LOAD_MASK, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_LOAD_MASK, 2),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(LOAD_EXPAND, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_LOAD_EXPAND, 2),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(GATHER, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_GATHER, UINT8_MAX),
    LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW(GATHER_MASK, LOOM_OP_KIND_UNKNOWN,
     LOOM_VECTOR_TO_SCALAR_LANE_GATHER_MASK, 3),
};

#undef LOOM_VECTOR_TO_SCALAR_DESCRIPTOR_ROW
#undef LOOM_VECTOR_TO_SCALAR_VECTOR_OP_INDEX
// clang-format on

static bool loom_vector_to_scalar_descriptor_is_empty(
    const loom_vector_to_scalar_descriptor_t* descriptor) {
  return descriptor->lane_kind == LOOM_VECTOR_TO_SCALAR_LANE_GENERIC &&
         descriptor->lane_op_kind == LOOM_OP_KIND_UNKNOWN;
}

bool loom_vector_to_scalar_resolve_descriptor(
    loom_op_kind_t kind, loom_vector_to_scalar_descriptor_t* out_descriptor) {
  const loom_vector_scalarization_t* scalarization =
      loom_vector_scalarization_lookup(kind);
  if (scalarization != NULL &&
      iree_all_bits_set(scalarization->flags,
                        LOOM_VECTOR_SCALARIZATION_FLAG_MECHANICAL)) {
    *out_descriptor = (loom_vector_to_scalar_descriptor_t){
        .lane_op_kind = scalarization->lane_op_kind,
        .lane_kind = LOOM_VECTOR_TO_SCALAR_LANE_GENERIC,
        .seed_operand_index = scalarization->seed_operand_index,
    };
    return true;
  }

  if (loom_op_dialect_id(kind) != LOOM_DIALECT_VECTOR) return false;
  const uint8_t op_index = loom_op_dialect_index(kind);
  if (op_index >= IREE_ARRAYSIZE(kVectorToScalarExplicitDescriptors)) {
    return false;
  }
  const loom_vector_to_scalar_descriptor_t* descriptor =
      &kVectorToScalarExplicitDescriptors[op_index];
  if (loom_vector_to_scalar_descriptor_is_empty(descriptor)) return false;
  *out_descriptor = *descriptor;
  return true;
}
