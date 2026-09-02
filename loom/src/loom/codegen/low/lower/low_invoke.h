// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-Low materialization for low.invoke register fragments.

#ifndef LOOM_CODEGEN_LOW_LOWER_LOW_INVOKE_H_
#define LOOM_CODEGEN_LOW_LOWER_LOW_INVOKE_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Required-inlines |source_op|'s Low helper into the active Low function.
// Source operands and results cross the boundary through the target-selected
// register mapping; the helper's target, representation, value, and scheduling
// contracts must be preservable in the caller.
iree_status_t loom_low_lower_invoke(loom_low_lower_context_t* context,
                                    const loom_op_t* source_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_LOW_INVOKE_H_
