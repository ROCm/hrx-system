// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_CSE_H_
#define LOOM_TRANSFORMS_CSE_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for the common subexpression elimination pass.
const loom_pass_info_t* loom_cse_pass_info(void);

// Common subexpression elimination pass.
//
// Iterative DFS over the region nesting tree with a scope chain of
// hash tables. For each eligible op (has results, no regions, no
// writes/unknown-effects/non-determinism), computes a content-aware
// hash and looks up the scope chain for a structurally equivalent op.
// If found, replaces all uses and erases the duplicate.
//
// Key properties:
//   - Iterative (no recursion): bounded stack usage via an explicit
//     DFS frame stack, safe on arbitrarily nested input.
//   - Tombstone-based write barriers: PURE ops survive writes, non-PURE
//     ops (reads) are invalidated without breaking probe chains.
//   - Entry block dominance: in multi-block regions, the entry block's
//     CSE candidates are visible to all successor blocks.
//   - Deep attribute comparison: pointer-valued attribute kinds
//     (I64_ARRAY, PREDICATE_LIST, DICT) are compared by content via
//     loom_attribute_equal, not by pointer identity.
//   - Split arena strategy: scope tables live in a dedicated arena
//     reset between top-level blocks; the DFS stack lives in the pass
//     arena. Peak memory is bounded by the largest single subtree.
iree_status_t loom_cse_run(loom_pass_t* pass, loom_module_t* module,
                           loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CSE_H_
