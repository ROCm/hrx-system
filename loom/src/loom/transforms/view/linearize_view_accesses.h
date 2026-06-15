// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Linearizes dense multi-dimensional view accesses into rank-1 source IR.
//
// The pass rewrites scalar view.load/store operations that index static dense
// views formed by buffer.view into equivalent accesses through a rank-1 view of
// the same buffer and byte offset. The resulting index.madd expressions are
// ordinary source SSA, so canonicalize and CSE can share address math before
// source-to-low without target-specific emission knowledge.

#ifndef LOOM_TRANSFORMS_LINEARIZE_VIEW_ACCESSES_H_
#define LOOM_TRANSFORMS_LINEARIZE_VIEW_ACCESSES_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_linearize_view_accesses_pass_info(void);

iree_status_t loom_linearize_view_accesses_run(loom_pass_t* pass,
                                               loom_module_t* module,
                                               loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_LINEARIZE_VIEW_ACCESSES_H_
