// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent dead-code elimination for descriptor-backed low functions.
//
// Low DCE removes ordinary trivially dead ops and descriptor packets whose
// target descriptor explicitly marks them dead-removable. Generic redundancy
// elimination remains the normal cse pass; this pass only owns the low-specific
// deadness query needed to interpret descriptor-backed packets.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_DCE_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_DCE_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Static pass metadata for the low-dce function pass.
const loom_pass_info_t* loom_low_dce_pass_info(void);

// Runs low-dce as a function pass.
iree_status_t loom_low_dce_run(loom_pass_t* pass, loom_module_t* module,
                               loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_DCE_H_
