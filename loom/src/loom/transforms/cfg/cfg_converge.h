// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_CFG_CONVERGE_H_
#define LOOM_TRANSFORMS_CFG_CONVERGE_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for final source CFG convergence.
const loom_pass_info_t* loom_cfg_converge_pass_info(void);

// Factors overlapping decisions through ordinary predicate joins and gives
// natural loops exclusive exit frontiers when enclosing decisions bypass the
// same continuation. Downstream shared bodies also converge with early bypasses
// before executing; their original argument tuples cross a predicate join.
// Effects stay in their original blocks; scalar live-ins that lose dominance
// travel through join arguments. Unobservable incoming values are defined zero.
// Inside natural loops, factoring stays within one iteration and preserves the
// header and exit boundaries. Irreducible cycles and captures requiring
// type/attribute or ownership changes retain their original CFG. Run after
// branch threading and before source lowering.
iree_status_t loom_cfg_converge_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CFG_CONVERGE_H_
