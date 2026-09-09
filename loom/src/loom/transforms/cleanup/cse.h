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

// Semantic expression elimination for source function-like definitions.
// Equivalent deterministic expressions are reused within their dominance scope,
// respecting memory/convergence epochs and linear ownership. Low definitions
// are left unchanged: descriptor-backed machine identity belongs to low-cse.
// Each root region has independent SSA availability.
iree_status_t loom_cse_run(loom_pass_t* pass, loom_module_t* module,
                           loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CSE_H_
