// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_CSE_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_CSE_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Static metadata for descriptor-backed machine expression elimination.
const loom_pass_info_t* loom_low_cse_pass_info(void);

// Commoning for Low function and kernel definitions. Descriptor state writes
// remain distinct; implicit reads require an unchanged architectural state
// epoch and straight-line control flow. Constants and arithmetic use the same
// dominance-scoped SSA identity; register allocation owns physical lifetimes.
// Source definitions are handled by the cse pass.
iree_status_t loom_low_cse_run(loom_pass_t* pass, loom_module_t* module,
                               loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_CSE_H_
