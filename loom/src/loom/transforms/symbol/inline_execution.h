// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Executes one fully selected callable composition plan.

#ifndef LOOM_TRANSFORMS_SYMBOL_INLINE_EXECUTION_H_
#define LOOM_TRANSFORMS_SYMBOL_INLINE_EXECUTION_H_

#include "loom/transforms/symbol/inline_callables.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_inline_execute_plan(loom_inline_callables_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_INLINE_EXECUTION_H_
