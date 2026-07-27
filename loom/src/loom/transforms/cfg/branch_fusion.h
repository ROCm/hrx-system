// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Conservative fusion for adjacent same-selector branch regions.
//
// The pass consumes compatible adjacent structured branch operations with the
// same selector and merges their per-branch bodies into one wider branch. V1 is
// intentionally narrow and only rebuilds scf.if, but the legality and body
// movement are phrased around the RegionBranch interface so scf.switch support
// can use the same path once its selector/case equivalence is modeled.

#ifndef LOOM_TRANSFORMS_BRANCH_FUSION_H_
#define LOOM_TRANSFORMS_BRANCH_FUSION_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_branch_fusion_pass_info(void);

iree_status_t loom_branch_fusion_run(loom_pass_t* pass, loom_module_t* module,
                                     loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_BRANCH_FUSION_H_
