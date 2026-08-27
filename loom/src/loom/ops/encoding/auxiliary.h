// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Auxiliary SSA operands attached to encoded values.
//
// vector.encode/vector.decode keep large scale tables, codebooks, sparse
// metadata, and online statistics as ordinary SSA values while a compact schema
// value describes how to interpret them. Encoding families declare the shared
// key vocabulary, while this helper folds OperandDict metadata into dense enum
// slots and bitsets for verification and lowering.

#ifndef LOOM_OPS_ENCODING_AUXILIARY_H_
#define LOOM_OPS_ENCODING_AUXILIARY_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

static_assert(LOOM_ENCODING_AUXILIARY_KEY_COUNT_ <= 64,
              "auxiliary key flags must fit in one 64-bit word");

// A resolved auxiliary operand dictionary. Missing entries have invalid value
// IDs and absent bits in present_keys.
typedef struct loom_encoding_auxiliary_view_t {
  // Bitset of loom_encoding_auxiliary_key_t values.
  loom_encoding_auxiliary_key_flags_t present_keys;

  // SSA value IDs indexed by loom_encoding_auxiliary_key_t.
  loom_value_id_t values[LOOM_ENCODING_AUXILIARY_KEY_COUNT_];
} loom_encoding_auxiliary_view_t;

// Returns the stable text spelling for |key|, or an empty view for invalid
// values. The returned view has static storage duration.
iree_string_view_t loom_encoding_auxiliary_key_name(
    loom_encoding_auxiliary_key_t key);

// Resolves a text key name to a dense auxiliary key enum. This is intended for
// parser/verifier boundary code; pass and lowering code should consume resolved
// views and key flags instead of spelling strings.
bool loom_encoding_auxiliary_key_lookup(iree_string_view_t name,
                                        loom_encoding_auxiliary_key_t* out_key);

// Resolves a stable key ID to a dense auxiliary key enum. This is the preferred
// path once text has crossed into IR because it avoids comparing spellings in
// analysis and verification loops.
bool loom_encoding_auxiliary_key_lookup_stable_id(
    uint64_t stable_id, loom_encoding_auxiliary_key_t* out_key);

// Resolves the conventional key for an explicit scale-like schema operand.
// Returns false when |index| exceeds the fixed auxiliary scale vocabulary.
bool loom_encoding_auxiliary_scale_key(uint16_t index,
                                       loom_encoding_auxiliary_key_t* out_key);

// Returns the flag bit corresponding to |key|, or zero for invalid values.
static inline loom_encoding_auxiliary_key_flags_t
loom_encoding_auxiliary_key_flag(loom_encoding_auxiliary_key_t key) {
  if (key >= LOOM_ENCODING_AUXILIARY_KEY_COUNT_) {
    return 0;
  }
  return 1ull << key;
}

// Initializes |out_view| to all auxiliary values absent.
void loom_encoding_auxiliary_view_initialize(
    loom_encoding_auxiliary_view_t* out_view);

// Resolves OperandDict metadata and variadic operands into dense auxiliary key
// slots. Returns false and stores the unknown key spelling in |out_unknown_key|
// when a dictionary key is outside the encoding auxiliary vocabulary.
bool loom_encoding_auxiliary_view_resolve(
    const loom_module_t* module, loom_value_slice_t auxiliary_values,
    loom_named_attr_slice_t auxiliary_names,
    loom_encoding_auxiliary_view_t* out_view,
    iree_string_view_t* out_unknown_key);

// Computes the auxiliary key bits required by a fully known encoded operand
// schema. Returns false when the schema requires more explicit scale-like
// operands than the fixed vocabulary can name.
bool loom_encoding_auxiliary_required_keys_from_schema(
    loom_value_fact_encoded_operand_schema_t schema,
    loom_encoding_auxiliary_key_flags_t* out_required_keys,
    uint16_t* out_unsupported_scale_index);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_AUXILIARY_H_
