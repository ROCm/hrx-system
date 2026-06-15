// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/predicate.h"

typedef struct loom_pass_trait_name_t {
  // User-facing trait spelling accepted by pass.where trait predicates.
  iree_string_view_t name;
  // Trait bit represented by name.
  loom_trait_flags_t flag;
} loom_pass_trait_name_t;

static const loom_pass_trait_name_t kLoomPassTraitNames[] = {
    {IREE_SVL("Pure"), LOOM_TRAIT_PURE},
    {IREE_SVL("pure"), LOOM_TRAIT_PURE},
    {IREE_SVL("Commutative"), LOOM_TRAIT_COMMUTATIVE},
    {IREE_SVL("commutative"), LOOM_TRAIT_COMMUTATIVE},
    {IREE_SVL("Idempotent"), LOOM_TRAIT_IDEMPOTENT},
    {IREE_SVL("idempotent"), LOOM_TRAIT_IDEMPOTENT},
    {IREE_SVL("Involution"), LOOM_TRAIT_INVOLUTION},
    {IREE_SVL("involution"), LOOM_TRAIT_INVOLUTION},
    {IREE_SVL("Terminator"), LOOM_TRAIT_TERMINATOR},
    {IREE_SVL("terminator"), LOOM_TRAIT_TERMINATOR},
    {IREE_SVL("ConstantLike"), LOOM_TRAIT_CONSTANT_LIKE},
    {IREE_SVL("constant-like"), LOOM_TRAIT_CONSTANT_LIKE},
    {IREE_SVL("Elementwise"), LOOM_TRAIT_ELEMENTWISE},
    {IREE_SVL("elementwise"), LOOM_TRAIT_ELEMENTWISE},
    {IREE_SVL("Decomposable"), LOOM_TRAIT_DECOMPOSABLE},
    {IREE_SVL("decomposable"), LOOM_TRAIT_DECOMPOSABLE},
    {IREE_SVL("SymbolDefine"), LOOM_TRAIT_SYMBOL_DEFINE},
    {IREE_SVL("symbol-define"), LOOM_TRAIT_SYMBOL_DEFINE},
    {IREE_SVL("ReadsMemory"), LOOM_TRAIT_READS_MEMORY},
    {IREE_SVL("reads-memory"), LOOM_TRAIT_READS_MEMORY},
    {IREE_SVL("WritesMemory"), LOOM_TRAIT_WRITES_MEMORY},
    {IREE_SVL("writes-memory"), LOOM_TRAIT_WRITES_MEMORY},
    {IREE_SVL("NonDeterministic"), LOOM_TRAIT_NON_DETERMINISTIC},
    {IREE_SVL("non-deterministic"), LOOM_TRAIT_NON_DETERMINISTIC},
    {IREE_SVL("UnknownEffects"), LOOM_TRAIT_UNKNOWN_EFFECTS},
    {IREE_SVL("unknown-effects"), LOOM_TRAIT_UNKNOWN_EFFECTS},
    {IREE_SVL("IsolatedFromAbove"), LOOM_TRAIT_ISOLATED_FROM_ABOVE},
    {IREE_SVL("isolated-from-above"), LOOM_TRAIT_ISOLATED_FROM_ABOVE},
    {IREE_SVL("UniqueIdentity"), LOOM_TRAIT_UNIQUE_IDENTITY},
    {IREE_SVL("unique-identity"), LOOM_TRAIT_UNIQUE_IDENTITY},
    {IREE_SVL("Hint"), LOOM_TRAIT_HINT},
    {IREE_SVL("hint"), LOOM_TRAIT_HINT},
    {IREE_SVL("Convergent"), LOOM_TRAIT_CONVERGENT},
    {IREE_SVL("convergent"), LOOM_TRAIT_CONVERGENT},
    {IREE_SVL("SafeToSpeculate"), LOOM_TRAIT_SAFE_TO_SPECULATE},
    {IREE_SVL("safe-to-speculate"), LOOM_TRAIT_SAFE_TO_SPECULATE},
    {IREE_SVL("RefinableResultTypeRefs"),
     LOOM_TRAIT_REFINABLE_RESULT_TYPE_REFS},
    {IREE_SVL("refinable-result-type-refs"),
     LOOM_TRAIT_REFINABLE_RESULT_TYPE_REFS},
    {IREE_SVL("FactIdentity"), LOOM_TRAIT_FACT_IDENTITY},
    {IREE_SVL("fact-identity"), LOOM_TRAIT_FACT_IDENTITY},
};

bool loom_pass_predicate_lookup_trait(iree_string_view_t name,
                                      loom_trait_flags_t* out_trait_flags) {
  *out_trait_flags = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kLoomPassTraitNames); ++i) {
    if (iree_string_view_equal(kLoomPassTraitNames[i].name, name)) {
      *out_trait_flags = kLoomPassTraitNames[i].flag;
      return true;
    }
  }
  return false;
}
