// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Whole-program boundary fact and type refinement.
//
// This module pass propagates stable value facts across direct function
// boundaries, uses those facts to refine private boundary types, and drives the
// reusable canonicalizer until those boundary summaries stop changing. It is
// intentionally a boundary driver, not a local pattern collection: local
// simplification still happens through canonicalize.

#ifndef LOOM_TRANSFORMS_REFINE_BOUNDARIES_H_
#define LOOM_TRANSFORMS_REFINE_BOUNDARIES_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for the refine-boundaries pass.
const loom_pass_info_t* loom_refine_boundaries_pass_info(void);

typedef struct loom_refine_boundaries_options_t {
  // Maximum number of outer boundary fixed-point iterations. Zero selects the
  // pass default.
  uint32_t max_iterations;
} loom_refine_boundaries_options_t;

// Creates refine-boundaries pass state from a textual option dictionary.
iree_status_t loom_refine_boundaries_create(loom_pass_t* pass,
                                            iree_string_view_t options);

// Propagates direct-call boundary facts and type refinements through the
// module.
iree_status_t loom_refine_boundaries_run(loom_pass_t* pass,
                                         loom_module_t* module);

// Propagates direct-call boundary facts and type refinements through the module
// using explicit driver options.
iree_status_t loom_refine_boundaries_run_with_options(
    loom_pass_t* pass, loom_module_t* module,
    const loom_refine_boundaries_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_REFINE_BOUNDARIES_H_
