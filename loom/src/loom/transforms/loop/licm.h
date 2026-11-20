// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_LICM_H_
#define LOOM_TRANSFORMS_LICM_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for the loop-invariant code motion pass.
const loom_pass_info_t* loom_licm_pass_info(void);

// Loop-invariant code motion pass.
//
// Processes LoopLike operations from inner loops to outer loops. An enclosing
// loop immediately sees work hoisted from its children. Hoists speculatable
// pure work and ordinary reads with proven execution and memory legality.
// Every captured SSA value must be available at the insertion point or defined
// inside the candidate subtree that moves with it.
// Invariant reads in possibly empty loops use a preheader guarded by the first
// iteration's entry condition. The original recurrence and zero-trip results
// are preserved; condition-loop headers must be effect-free to duplicate.
//
// Shared legality analysis is acquired at the first loop, before any mutation,
// and retained across hoists. Loop-free functions require only traversal.
iree_status_t loom_licm_run(loom_pass_t* pass, loom_module_t* module,
                            loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_LICM_H_
