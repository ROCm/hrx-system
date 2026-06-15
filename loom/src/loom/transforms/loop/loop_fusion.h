// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Conservative fusion for adjacent counted loops.
//
// The pass consumes compatible adjacent scf.for loops with identical domains
// and moves their bodies into one fused loop. It is intentionally narrow: body
// interleaving is only allowed when region effects and lane-local dependencies
// prove the transformation preserves the source loop ordering.

#ifndef LOOM_TRANSFORMS_LOOP_FUSION_H_
#define LOOM_TRANSFORMS_LOOP_FUSION_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_loop_fusion_pass_info(void);

iree_status_t loom_loop_fusion_run(loom_pass_t* pass, loom_module_t* module,
                                   loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_LOOP_FUSION_H_
