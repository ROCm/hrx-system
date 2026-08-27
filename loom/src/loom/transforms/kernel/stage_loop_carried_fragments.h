// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stages loop-carried accumulator fragments through workgroup memory.
//
// The pass recognizes scf.for carried values that are matrix accumulator
// fragments and rewrites them into explicit vector.fragment.store/load traffic
// against a dense workgroup view. Invoking the pass explicitly requests this
// storage strategy; profitability belongs to the pipeline selecting the pass,
// not to source-shape thresholds inside the transform.

#ifndef LOOM_TRANSFORMS_STAGE_LOOP_CARRIED_FRAGMENTS_H_
#define LOOM_TRANSFORMS_STAGE_LOOP_CARRIED_FRAGMENTS_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_stage_loop_carried_fragments_pass_info(void);

iree_status_t loom_stage_loop_carried_fragments_run(loom_pass_t* pass,
                                                    loom_module_t* module,
                                                    loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_STAGE_LOOP_CARRIED_FRAGMENTS_H_
