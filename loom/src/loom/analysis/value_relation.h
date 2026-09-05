// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned structural relations between SSA values.
//
// Relations describe one local semantic transport edge introduced by an
// operation. They do not prescribe whether a consumer must share storage,
// physical representation, facts, or any other property across the edge.
// Consumers select the relation kinds meaningful to their contract.
//
// Enumeration is allocation-free and operation-local. It may inspect the
// current operation's direct parent to decode a verified region terminator,
// but it never walks source IR or follows def/use chains.

#ifndef LOOM_ANALYSIS_VALUE_RELATION_H_
#define LOOM_ANALYSIS_VALUE_RELATION_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_value_relation_kind_e {
  // Unknown or uninitialized relation kind.
  LOOM_VALUE_RELATION_UNKNOWN = 0,
  // An operation result explicitly tied to one operand.
  LOOM_VALUE_RELATION_TIED_RESULT = 1,
  // An ordinal operand/result pair on a FactIdentity operation.
  LOOM_VALUE_RELATION_FACT_IDENTITY = 2,
  // Operand zero and result zero on a ValueAlias operation.
  LOOM_VALUE_RELATION_VALUE_ALIAS = 3,
  // One selectable payload and the result column it may produce.
  LOOM_VALUE_RELATION_SELECT_PAYLOAD = 4,
  // One data operand and result of an elementwise operation.
  LOOM_VALUE_RELATION_ELEMENTWISE = 5,
  // A terminator payload and its destination block argument.
  LOOM_VALUE_RELATION_CFG_ARGUMENT = 6,
  // One edge in a verified loop-carried state cycle.
  LOOM_VALUE_RELATION_LOOP_CARRIED = 7,
  // A counted-loop initial value and its zero-trip result.
  LOOM_VALUE_RELATION_LOOP_BYPASS = 8,
  // A mutually exclusive region yield and its parent result.
  LOOM_VALUE_RELATION_REGION_RESULT = 9,
  LOOM_VALUE_RELATION_COUNT_ = 10,
};
typedef uint8_t loom_value_relation_kind_t;

enum loom_value_relation_flag_bits_e {
  // Source and destination have different types despite sharing one tied
  // storage identity.
  LOOM_VALUE_RELATION_FLAG_TYPE_CHANGE = 1u << 0,
};
typedef uint8_t loom_value_relation_flags_t;

// Bitset of loom_value_relation_kind_e values.
typedef uint16_t loom_value_relation_mask_t;

#define LOOM_VALUE_RELATION_MASK(kind) \
  ((loom_value_relation_mask_t)1u << ((kind) - 1u))
#define LOOM_VALUE_RELATION_MASK_ALL \
  ((loom_value_relation_mask_t)((1u << (LOOM_VALUE_RELATION_COUNT_ - 1u)) - 1u))

// Sentinel used when a relation source is not an operand of the observed op.
#define LOOM_VALUE_RELATION_OPERAND_INDEX_NONE UINT16_MAX

// One directed structural relation from a source value to a destination.
typedef struct loom_value_relation_t {
  // Value transported by the structural edge.
  loom_value_id_t source_value_id;
  // Result or block argument receiving the transported value.
  loom_value_id_t destination_value_id;
  // Flat operand index of |source_value_id| on the observed operation.
  uint16_t source_operand_index;
  // Semantic kind of transport edge.
  loom_value_relation_kind_t kind;
  // Relation details such as tied-result type changes.
  loom_value_relation_flags_t flags;
} loom_value_relation_t;

static_assert(sizeof(loom_value_relation_t) == 12,
              "value relations must stay compact");

// Allocation-free iterator over selected relations introduced by one op.
//
// Verified IR is assumed. Invalid tied-result, interface, and control-flow
// shapes are asserted because they indicate a compiler/source-of-truth bug.
typedef struct loom_value_relation_iterator_t {
  // Module providing generated operation interface tables.
  const loom_module_t* module;
  // Current operation being observed.
  const loom_op_t* op;
  // Generated descriptor for |op|, or NULL for an unregistered operation.
  const loom_op_vtable_t* vtable;
  // Remaining structurally possible relation-family phases.
  uint16_t phase_bits;
  // Primary ordinal within the active relation family.
  uint16_t outer_index;
  // Secondary ordinal within the active relation family.
  uint16_t inner_index;
  // Number of select payloads preceding |outer_index|.
  uint16_t select_payload_ordinal;
} loom_value_relation_iterator_t;

// Initializes |out_iterator| to enumerate the selected local relations of
// |op|. Unknown mask bits are ignored.
void loom_value_relation_iterator_initialize(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_relation_mask_t relation_mask,
    loom_value_relation_iterator_t* out_iterator);

// Returns the next selected relation or false when exhausted.
bool loom_value_relation_iterator_next(loom_value_relation_iterator_t* iterator,
                                       loom_value_relation_t* out_relation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_VALUE_RELATION_H_
