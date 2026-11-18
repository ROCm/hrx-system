// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Costed selection of legal native-call frontiers.

#ifndef LOOM_TRANSFORMS_SYMBOL_INLINE_FRONTIER_H_
#define LOOM_TRANSFORMS_SYMBOL_INLINE_FRONTIER_H_

#include "loom/transforms/symbol/inline_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Measures the authored body in materialization units. Calls and terminators
// each cost one unit; other operations scale with their static value payloads.
iree_status_t loom_inline_compute_direct_body_cost(
    loom_inline_callables_plan_t* plan, loom_symbol_id_t symbol_id,
    uint32_t* out_cost);

// Selects optional expansions that move a leaf-only native-call frontier into
// a reused parent definition instead of cloning that parent through required
// ancestors. Returns one byte per plan entry in the pass arena; selected bytes
// are nonzero. The plan itself is not mutated.
iree_status_t loom_inline_select_native_frontier(
    loom_inline_callables_plan_t* plan, uint8_t** out_selected_entries);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_INLINE_FRONTIER_H_
