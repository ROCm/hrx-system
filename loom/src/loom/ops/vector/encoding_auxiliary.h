// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Auxiliary SSA operands attached to encoded vector values.
//
// vector.encode/vector.decode keep large scale tables, codebooks, sparse
// metadata, and online statistics as ordinary SSA values while a compact schema
// value describes how to interpret them. This helper is the vector-owned
// vocabulary for those keyed auxiliary operands: text and bytecode keep stable
// names for readability, while verification and lowering can immediately fold
// the dictionary into dense enum slots and bitsets.

#ifndef LOOM_OPS_VECTOR_ENCODING_AUXILIARY_H_
#define LOOM_OPS_VECTOR_ENCODING_AUXILIARY_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dense auxiliary operand key slots. Keep this enum in the same order as the
// bit enum below so key -> bit conversion is a shift, not a lookup.
typedef enum loom_vector_encoding_auxiliary_key_e {
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE = 0,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE = 1,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE2 = 2,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE3 = 3,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE4 = 4,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE5 = 5,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE6 = 6,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE7 = 7,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_ZERO_POINT = 8,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_MINIMUM = 9,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIAS = 10,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SUM_CORRECTION = 11,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_CODEBOOK = 12,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SPARSITY = 13,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_METADATA = 14,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_INDICES = 15,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_OFFSETS = 16,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_MASK = 17,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SIGNS = 18,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_RESIDUAL = 19,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_AMAX = 20,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_THRESHOLDS = 21,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_CENTROIDS = 22,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_OUTLIERS = 23,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_ = 24,
} loom_vector_encoding_auxiliary_key_t;

// Auxiliary key bitset values.
typedef enum loom_vector_encoding_auxiliary_key_bits_e {
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SECONDARY_SCALE =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE2 =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE2,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE3 =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE3,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE4 =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE4,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE5 =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE5,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE6 =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE6,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE7 =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE7,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_ZERO_POINT =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_ZERO_POINT,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_MINIMUM =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_MINIMUM,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_BIAS =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIAS,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SUM_CORRECTION =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SUM_CORRECTION,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_CODEBOOK =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_CODEBOOK,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SPARSITY =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SPARSITY,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_METADATA =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_METADATA,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_INDICES =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_INDICES,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_OFFSETS =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_OFFSETS,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_MASK =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_MASK,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SIGNS =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SIGNS,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_RESIDUAL =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_RESIDUAL,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_AMAX =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_AMAX,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_THRESHOLDS =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_THRESHOLDS,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_CENTROIDS =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_CENTROIDS,
  LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_OUTLIERS =
      1ull << LOOM_VECTOR_ENCODING_AUXILIARY_KEY_OUTLIERS,
} loom_vector_encoding_auxiliary_key_bits_t;

typedef uint64_t loom_vector_encoding_auxiliary_key_flags_t;

static_assert(LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_ <= 64,
              "auxiliary key flags must fit in one 64-bit word");

// A resolved auxiliary operand dictionary. Missing entries have invalid value
// IDs and absent bits in present_keys.
typedef struct loom_vector_encoding_auxiliary_view_t {
  // Bitset of loom_vector_encoding_auxiliary_key_bits_t values.
  loom_vector_encoding_auxiliary_key_flags_t present_keys;

  // SSA value IDs indexed by loom_vector_encoding_auxiliary_key_t.
  loom_value_id_t values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_];
} loom_vector_encoding_auxiliary_view_t;

// Returns the stable text spelling for |key|, or an empty view for invalid
// values. The returned view has static storage duration.
iree_string_view_t loom_vector_encoding_auxiliary_key_name(
    loom_vector_encoding_auxiliary_key_t key);

// Resolves a text key name to a dense auxiliary key enum. This is intended for
// parser/verifier boundary code; pass and lowering code should consume resolved
// views and key flags instead of spelling strings.
bool loom_vector_encoding_auxiliary_key_lookup(
    iree_string_view_t name, loom_vector_encoding_auxiliary_key_t* out_key);

// Resolves a stable key ID to a dense auxiliary key enum. This is the preferred
// path once text has crossed into IR because it avoids comparing spellings in
// analysis and verification loops.
bool loom_vector_encoding_auxiliary_key_lookup_stable_id(
    uint64_t stable_id, loom_vector_encoding_auxiliary_key_t* out_key);

// Resolves the conventional key for an explicit scale-like schema operand.
// Returns false when |index| exceeds the fixed auxiliary scale vocabulary.
bool loom_vector_encoding_auxiliary_scale_key(
    uint16_t index, loom_vector_encoding_auxiliary_key_t* out_key);

// Returns the flag bit corresponding to |key|, or zero for invalid values.
static inline loom_vector_encoding_auxiliary_key_flags_t
loom_vector_encoding_auxiliary_key_flag(
    loom_vector_encoding_auxiliary_key_t key) {
  if (key >= LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_) {
    return 0;
  }
  return 1ull << key;
}

// Initializes |out_view| to all auxiliary values absent.
void loom_vector_encoding_auxiliary_view_initialize(
    loom_vector_encoding_auxiliary_view_t* out_view);

// Resolves OperandDict metadata and variadic operands into dense auxiliary key
// slots. Returns false and stores the unknown key spelling in |out_unknown_key|
// when a dictionary key is outside the vector auxiliary vocabulary.
bool loom_vector_encoding_auxiliary_view_resolve(
    const loom_module_t* module, loom_value_slice_t auxiliary_values,
    loom_named_attr_slice_t auxiliary_names,
    loom_vector_encoding_auxiliary_view_t* out_view,
    iree_string_view_t* out_unknown_key);

// Computes the auxiliary key bits required by a fully known encoded operand
// schema. Returns false when the schema requires more explicit scale-like
// operands than the fixed vocabulary can name.
bool loom_vector_encoding_auxiliary_required_keys_from_schema(
    loom_value_fact_encoded_operand_schema_t schema,
    loom_vector_encoding_auxiliary_key_flags_t* out_required_keys,
    uint16_t* out_unsupported_scale_index);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_VECTOR_ENCODING_AUXILIARY_H_
