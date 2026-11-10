// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact signed and unsigned interpretations of fixed-width integer payloads.

#ifndef LOOM_ANALYSIS_SYMBOLIC_INTEGER_CAST_H_
#define LOOM_ANALYSIS_SYMBOLIC_INTEGER_CAST_H_

#include "iree/base/api.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Numeric interpretation applied after retaining the low payload bits.
typedef enum loom_symbolic_integer_cast_kind_e {
  // Sign-extend the retained payload bits.
  LOOM_SYMBOLIC_INTEGER_CAST_SIGNED = 0,
  // Zero-extend the retained payload bits.
  LOOM_SYMBOLIC_INTEGER_CAST_UNSIGNED = 1,
  // Sign-extend the payload, then interpret the wider result unsigned.
  LOOM_SYMBOLIC_INTEGER_CAST_SIGNED_THEN_UNSIGNED = 2,
} loom_symbolic_integer_cast_kind_t;

// Exact numeric function represented by an integer cast result. The result is
// the signed or unsigned interpretation of the low |bit_count| bits of
// |source_value_id|, optionally followed by an unsigned interpretation at a
// larger width. This form is closed under fixed-width integer casts. The
// producer retains the composition so consumers never walk cast producers.
typedef struct loom_symbolic_integer_cast_t {
  // SSA value supplying the original payload bits.
  loom_value_id_t source_value_id;
  // Number of low payload bits interpreted by the cast.
  uint8_t bit_count;
  // loom_symbolic_integer_cast_kind_t selecting the payload interpretation.
  uint8_t kind;
  // Wider unsigned interpretation width, or zero for a simple interpretation.
  uint8_t unsigned_bit_count;
} loom_symbolic_integer_cast_t;

static_assert(sizeof(loom_symbolic_integer_cast_t) == 8,
              "integer cast summaries must fit in two words");

// Describes the exact fixed-width payload interpretation performed by |op|.
// Returns false when the op is not a supported integer/address cast.
bool loom_symbolic_integer_cast_from_op(const loom_module_t* module,
                                        const loom_op_t* op,
                                        loom_symbolic_integer_cast_t* out_cast);

// Composes a signed or unsigned |bit_count|-bit interpretation after |input|.
// |kind| is SIGNED or UNSIGNED; |input| may itself be a composition.
loom_symbolic_integer_cast_t loom_symbolic_integer_cast_compose(
    loom_symbolic_integer_cast_t input, loom_symbolic_integer_cast_kind_t kind,
    uint8_t bit_count);

// Applies the exact interpretation to the source's numeric facts. Scoped
// callers may use the returned facts without strengthening the source value.
loom_value_facts_t loom_symbolic_integer_cast_apply(
    const loom_symbolic_integer_cast_t* cast, loom_value_facts_t source_facts);

// Returns true when the source facts prove that applying |cast| preserves the
// source's mathematical integer value.
bool loom_symbolic_integer_cast_is_identity(
    const loom_symbolic_integer_cast_t* cast, loom_value_facts_t source_facts);

// Returns true when both casts apply the same interpretation and width. Source
// value identity is deliberately left to the owning symbolic analysis.
bool loom_symbolic_integer_cast_same_interpretation(
    const loom_symbolic_integer_cast_t* left,
    const loom_symbolic_integer_cast_t* right);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_INTEGER_CAST_H_
