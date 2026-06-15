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
// Walks function bodies, finds ops implementing the LoopLike interface, and
// hoists speculatable pure work out of loop bodies. A candidate moves before
// the loop when every operand and SSA value referenced by result types is
// either defined outside the loop or inside the candidate subtree that moves
// with it.
iree_status_t loom_licm_run(loom_pass_t* pass, loom_module_t* module,
                            loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_LICM_H_
