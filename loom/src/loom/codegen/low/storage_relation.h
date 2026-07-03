// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Storage relations introduced by structural operations.
//
// This helper is the shared storage-shape query for compiler decisions over
// lowered register values. Exact op implementations may use concrete low op
// accessors directly, but passes asking whether values may share storage should
// consume this relation contract instead of rediscovering structural op
// families.

#ifndef LOOM_CODEGEN_LOW_STORAGE_RELATION_H_
#define LOOM_CODEGEN_LOW_STORAGE_RELATION_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_storage_relation_kind_e {
  // Unknown or uninitialized storage relation kind.
  LOOM_LOW_STORAGE_RELATION_UNKNOWN = 0,
  // Destination and source unit ranges occupy identical storage units.
  LOOM_LOW_STORAGE_RELATION_SAME_STORAGE = 1,
  // Destination units occupy a contiguous subrange of the source storage units.
  LOOM_LOW_STORAGE_RELATION_SUBRANGE = 2,
  // Destination units occupy a contiguous packed range of source values.
  LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART = 3,
} loom_low_storage_relation_kind_t;

typedef enum loom_low_storage_relation_cause_e {
  // Unknown or uninitialized storage relation cause.
  LOOM_LOW_STORAGE_RELATION_CAUSE_UNKNOWN = 0,
  // Descriptor tied result requiring source/result storage identity.
  LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT = 1,
  // low.copy source/result storage affinity.
  LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_COPY = 2,
  // low.slice source/result subrange affinity.
  LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SLICE = 3,
  // low.concat source/result contiguous packing affinity.
  LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_CONCAT = 4,
  // low.br edge payload source/block-argument affinity.
  LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH = 5,
  // low.scf.for initial iter_arg/body-argument affinity.
  LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_FOR = 6,
  // low.scf.yield payload/result or backedge affinity.
  LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD = 7,
} loom_low_storage_relation_cause_t;

enum loom_low_storage_relation_flag_bits_e {
  // The relation is required for the operation semantics.
  LOOM_LOW_STORAGE_RELATION_FLAG_HARD = 1u << 0,
  // The relation removes a move when allocation can satisfy it.
  LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED = 1u << 1,
};

// Bitset of loom_low_storage_relation_flag_bits_e values.
typedef uint16_t loom_low_storage_relation_flags_t;

// One structural storage relation between destination and source ranges.
typedef struct loom_low_storage_relation_t {
  // Operation that introduced this relation.
  const loom_op_t* op;
  // Result, block argument, or other destination value receiving storage.
  loom_value_id_t destination_value_id;
  // Source value providing storage.
  loom_value_id_t source_value_id;
  // Unit offset inside the destination value.
  uint32_t destination_unit_offset;
  // Unit offset inside the source value.
  uint32_t source_unit_offset;
  // Number of storage units covered by this relation.
  uint32_t unit_count;
  // Structural relation shape.
  loom_low_storage_relation_kind_t kind;
  // IR feature that created the relation.
  loom_low_storage_relation_cause_t cause;
  // Hard/soft relation behavior.
  loom_low_storage_relation_flags_t flags;
} loom_low_storage_relation_t;

// Returns the number of storage relations introduced by |op|.
//
// Verified low IR is assumed. Invalid structural metadata is asserted instead
// of reported through status because it is a compiler/source-of-truth bug, not
// user-controlled runtime data.
uint16_t loom_low_storage_relation_count(const loom_module_t* module,
                                         const loom_op_t* op);

// Returns storage relation |relation_index| for |op|.
//
// |relation_index| must be less than loom_low_storage_relation_count(module,
// op). Verified low IR is assumed.
void loom_low_storage_relation_get(const loom_module_t* module,
                                   const loom_op_t* op, uint16_t relation_index,
                                   loom_low_storage_relation_t* out_relation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_STORAGE_RELATION_H_
