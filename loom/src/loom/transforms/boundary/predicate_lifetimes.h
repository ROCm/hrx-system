// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_BOUNDARY_PREDICATE_LIFETIMES_H_
#define LOOM_TRANSFORMS_BOUNDARY_PREDICATE_LIFETIMES_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects CFG loop live-outs after source canonicalization and convergence.
// Source-to-Low consumes this representation without another source cleanup:
// CSE may legitimately re-expose a loop-local predicate at an escaping use.
const loom_pass_info_t* loom_project_loop_predicate_lifetimes_pass_info(void);
iree_status_t loom_project_loop_predicate_lifetimes_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_BOUNDARY_PREDICATE_LIFETIMES_H_
