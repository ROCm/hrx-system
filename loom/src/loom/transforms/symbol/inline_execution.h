// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Executes one fully selected callable composition plan.

#ifndef LOOM_TRANSFORMS_SYMBOL_INLINE_EXECUTION_H_
#define LOOM_TRANSFORMS_SYMBOL_INLINE_EXECUTION_H_

#include "loom/transforms/symbol/inline_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns whether the private symbol definition can disappear after the given
// number of direct calls are expanded. Non-call references, retained function
// versions, and demanded template families keep the definition alive.
bool loom_inline_symbol_can_transfer_after_removing(
    const loom_inline_callables_plan_t* plan, loom_symbol_id_t symbol_id,
    uint32_t planned_call_removals);

iree_status_t loom_inline_execute_plan(loom_inline_callables_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_INLINE_EXECUTION_H_
