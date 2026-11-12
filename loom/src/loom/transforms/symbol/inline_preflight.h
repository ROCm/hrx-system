// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Call-site and callee-body legality for one immutable inline plan.

#ifndef LOOM_TRANSFORMS_SYMBOL_INLINE_PREFLIGHT_H_
#define LOOM_TRANSFORMS_SYMBOL_INLINE_PREFLIGHT_H_

#include "loom/transforms/symbol/inline_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Checks whether the call site can absorb the callee's planned control flow.
loom_inline_blocker_t loom_inline_preflight_site(
    const loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry);

// Inventories each Low callee at most once in the plan and returns its cached
// body legality and resource/storage requirements in the symbol summary.
loom_inline_blocker_t loom_inline_preflight_body(
    loom_inline_callables_plan_t* state, loom_symbol_id_t symbol_id);

// Applies call-site and cached callee-body preflight together.
loom_inline_blocker_t loom_inline_preflight_entry(
    loom_inline_callables_plan_t* state, const loom_inline_plan_entry_t* entry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_INLINE_PREFLIGHT_H_
