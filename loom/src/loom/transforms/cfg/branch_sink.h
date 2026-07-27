// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_BRANCH_SINK_H_
#define LOOM_TRANSFORMS_BRANCH_SINK_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for the branch code sinking pass.
const loom_pass_info_t* loom_branch_sink_pass_info(void);

// Sinks pure/effect-free producers into mutually-exclusive branch regions when
// every result use is contained in exactly one branch. This reduces work on
// paths that do not observe the value without cloning or speculating code.
iree_status_t loom_branch_sink_run(loom_pass_t* pass, loom_module_t* module,
                                   loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_BRANCH_SINK_H_
