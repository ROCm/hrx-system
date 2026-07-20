// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-informed loop-invariant placement.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_PLACE_LOOP_INVARIANTS_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_PLACE_LOOP_INVARIANTS_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for the target-informed placement pass.
const loom_pass_info_t* loom_place_loop_invariants_pass_info(void);

// Places source loop invariants without crossing a selected target residency
// cliff and consumes authored scf.for residency policies before unrolling.
iree_status_t loom_place_loop_invariants_run(loom_pass_t* pass,
                                             loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_PLACE_LOOP_INVARIANTS_H_
