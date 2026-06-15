// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Matrix-fragment interpretation facts for vector values.
//
// vector.fragment keeps the physical vector type intact and records how that
// vector should be interpreted by matrix-contract lowering. The operation is a
// fact boundary: dense fragments carry only role and logical matrix shape,
// while encoded fragments additionally point at a schema SSA value and ordinary
// auxiliary SSA operands for scales, tables, sparse metadata, and online state.

#ifndef LOOM_OPS_VECTOR_FRAGMENT_H_
#define LOOM_OPS_VECTOR_FRAGMENT_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/vector/encoding_auxiliary.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Raw extension payload tag used for vector fragment facts.
enum loom_vector_fragment_fact_payload_tag_e {
  LOOM_VECTOR_FRAGMENT_FACT_PAYLOAD_TAG_FRAGMENT = 1,
};

// Fragment role bitset values. Keep these in the same order as the generated
// vector.fragment role enum so role -> bit conversion is a shift.
typedef enum loom_vector_fragment_role_flag_bits_e {
  LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS = 1u << LOOM_VECTOR_ROLE_LHS,
  LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS = 1u << LOOM_VECTOR_ROLE_RHS,
  LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT = 1u << LOOM_VECTOR_ROLE_INIT,
  LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT = 1u << LOOM_VECTOR_ROLE_RESULT,
} loom_vector_fragment_role_flag_bits_t;

typedef uint32_t loom_vector_fragment_role_flags_t;

// Fragment fact bitset values.
typedef enum loom_vector_fragment_fact_flag_bits_e {
  // The fragment carries an explicit schema SSA value.
  LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_SCHEMA = 1u << 0,
  // The schema was resolved to an exact static storage-schema encoding.
  LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_STATIC_SCHEMA = 1u << 1,
} loom_vector_fragment_fact_flag_bits_t;

typedef uint32_t loom_vector_fragment_fact_flags_t;

// Compact fragment facts attached to the result of vector.fragment. The record
// is intentionally fixed-width and byte-comparable so fact joins can preserve
// equal fragments and drop mismatches without custom deep comparison.
typedef struct loom_vector_fragment_fact_t {
  // Bitset of loom_vector_fragment_fact_flag_bits_t values.
  loom_vector_fragment_fact_flags_t flags;

  // Bitset of loom_vector_fragment_role_flag_bits_t values.
  loom_vector_fragment_role_flags_t role_flags;

  // Schema SSA value when HAS_SCHEMA is set.
  loom_value_id_t schema_value_id;

  // One-based static schema encoding ID when HAS_STATIC_SCHEMA is set.
  uint16_t static_schema_encoding_id;

  // Number of logical matrix shape values stored in shape_value_ids.
  uint16_t shape_rank;

  // Logical row and column count SSA value IDs for the fragment role.
  loom_value_id_t shape_value_ids[2];

  // Explicit auxiliary SSA values keyed by vector auxiliary enum bits.
  loom_vector_encoding_auxiliary_view_t auxiliary;

  // Static target-independent encoded operand facts when the schema is known.
  loom_value_fact_encoded_operand_schema_t encoded_operand;
} loom_vector_fragment_fact_t;
static_assert(sizeof(loom_vector_fragment_fact_t) <=
                  LOOM_VALUE_FACT_RAW_PAYLOAD_LENGTH_LIMIT,
              "fragment facts must fit in a raw fact payload");

// Resolved vector.fragment parameter dictionary.
typedef struct loom_vector_fragment_parameter_view_t {
  // True when the using dictionary contains a schema key.
  bool has_schema;

  // Schema SSA value associated with the schema key.
  loom_value_id_t schema_value_id;

  // Ordinal of the schema operand within parameter_values.
  uint16_t schema_parameter_ordinal;

  // Auxiliary SSA values keyed by vector auxiliary enum bits.
  loom_vector_encoding_auxiliary_view_t auxiliary;
} loom_vector_fragment_parameter_view_t;

// Returns the flag bit corresponding to |role|, or zero for invalid values.
loom_vector_fragment_role_flags_t loom_vector_fragment_role_flag(
    loom_vector_role_t role);

// Returns the fact role flags corresponding to |role|. Initial and result
// accumulators share the same physical fragment interpretation, so fact
// propagation canonicalizes both roles to the combined accumulator role set.
loom_vector_fragment_role_flags_t loom_vector_fragment_fact_role_flags(
    loom_vector_role_t role);

// Initializes |out_fact| to the unknown all-zero fragment fact.
void loom_vector_fragment_fact_initialize(
    loom_vector_fragment_fact_t* out_fact);

// Returns true when no fragment facts are known.
bool loom_vector_fragment_fact_is_unknown(loom_vector_fragment_fact_t fact);

// Returns true when fragment facts are byte-identical.
bool loom_vector_fragment_fact_equal(loom_vector_fragment_fact_t lhs,
                                     loom_vector_fragment_fact_t rhs);

// Creates value facts carrying |fact| as a compact raw payload.
iree_status_t loom_vector_fragment_fact_make_value_facts(
    loom_fact_context_t* context, loom_vector_fragment_fact_t fact,
    loom_value_facts_t* out_facts);

// Queries compact fragment facts from |facts|.
bool loom_vector_fragment_fact_query_value_facts(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_vector_fragment_fact_t* out_fact);

// Resolves vector.fragment using-dictionary metadata into schema and dense
// auxiliary slots. Returns false and stores the unknown key spelling in
// |out_unknown_key| when a key is neither schema nor vector auxiliary data.
bool loom_vector_fragment_parameter_view_resolve(
    const loom_module_t* module, loom_value_slice_t parameter_values,
    loom_named_attr_slice_t parameter_names,
    loom_vector_fragment_parameter_view_t* out_view,
    iree_string_view_t* out_unknown_key);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_VECTOR_FRAGMENT_H_
