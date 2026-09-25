// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-phase conversion composition shared by scalar and vector patterns.

#ifndef LOOM_TRANSFORMS_CONVERSION_CHAIN_H_
#define LOOM_TRANSFORMS_CONVERSION_CHAIN_H_

#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Elementwise conversion semantics, independent of the containing dialect.
typedef uint8_t loom_conversion_kind_t;
enum loom_conversion_kind_e {
  // An unrecognized producer or a chain that must remain sequenced.
  LOOM_CONVERSION_NONE = 0,
  LOOM_CONVERSION_EXTF,
  LOOM_CONVERSION_FPTRUNC,
  LOOM_CONVERSION_EXTSI,
  LOOM_CONVERSION_EXTUI,
  LOOM_CONVERSION_TRUNCI,
  // Composition returns the original input without a conversion.
  LOOM_CONVERSION_IDENTITY,
  LOOM_CONVERSION_COUNT_,
};

typedef uint8_t loom_conversion_chain_flags_t;
enum loom_conversion_chain_flag_bits_e {
  // The intermediate result must be proven non-negative before composition.
  // Signed extension of a true i1 is negative despite the Boolean input facts.
  LOOM_CONVERSION_CHAIN_FLAG_NON_NEGATIVE = 1u << 0,
};

// Semantic decision for two consecutive elementwise conversions.
typedef struct loom_conversion_chain_match_t {
  // Candidate to resolve against endpoint types, or NONE to retain the chain.
  loom_conversion_kind_t candidate;
  // Facts the caller must prove about the intermediate conversion result.
  loom_conversion_chain_flags_t flags;
} loom_conversion_chain_match_t;

// Matches outer(inner(value)). Registered patterns supply the known outer kind
// and classify only its immediate producer. NONE preserves the chain without
// querying endpoint types or facts. This source-phase query does not authorize
// bypassing target-selected conversion sequences during universal cleanup.
loom_conversion_chain_match_t loom_conversion_chain_match(
    loom_conversion_kind_t outer_kind, loom_conversion_kind_t inner_kind);

// Resolves a matched candidate against the original input and final result
// types. The caller has discharged the match's fact obligations. Verified
// conversions preserve shape; complete type identity is required for bypass.
// Equal-width distinct float formats retain their intermediate conversion.
// This query has no IR, fact, allocation, or mutation ownership.
loom_conversion_kind_t loom_conversion_chain_resolve(
    loom_conversion_kind_t candidate, loom_type_t input_type,
    loom_type_t result_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_CONVERSION_CHAIN_H_
