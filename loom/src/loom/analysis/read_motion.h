// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared legality for source-order motion of ordinary memory reads.

#ifndef LOOM_ANALYSIS_READ_MOTION_H_
#define LOOM_ANALYSIS_READ_MOTION_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when sinking an ordinary read across |op| preserves source
// ordering. This is a deliberately alias-independent predicate: any write,
// fence, unknown effect, nested region, convergence, or retained semantic
// boundary blocks motion.
bool loom_motion_read_can_cross_op(const loom_module_t* module,
                                   const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_READ_MOTION_H_
