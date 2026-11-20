// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured entry guards for invariant loop preheaders.

#ifndef LOOM_TRANSFORMS_LOOP_ENTRY_H_
#define LOOM_TRANSFORMS_LOOP_ENTRY_H_

#include "loom/analysis/motion.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Whether loop entry can be tested outside the loop without changing effects.
// Counted loops use their verified positive-step range. Condition loops require
// an effect-free leaf header, whose initial evaluation can be duplicated.
bool loom_loop_entry_can_guard(const loom_module_t* module,
                               loom_loop_like_t loop);

// Moves a completed shared motion plan into a preheader executed exactly when
// the first body iteration executes. The original loop, including its condition
// and recurrence semantics, is retained. Empty loops yield their original
// initial/false-condition tuple. Structural mutation invalidates analyses, so
// callers build all plans before emitting guards. A plan with proven loop entry
// moves directly before the loop without constructing an additional guard.
iree_status_t loom_loop_entry_hoist(loom_rewriter_t* rewriter,
                                    loom_loop_like_t loop,
                                    const loom_motion_loop_entry_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_LOOP_ENTRY_H_
