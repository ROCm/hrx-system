// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SCF_LOOKUP_H_
#define LOOM_TRANSFORMS_SCF_LOOKUP_H_

#include "loom/rewrite/rewriter.h"
#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects the verified lookup to selects in its original block. Used for
// small tables, lane-varying selectors, and region-local dataflow.
iree_status_t loom_scf_lookup_project_selects(loom_rewriter_t* rewriter,
                                              iree_arena_allocator_t* arena,
                                              loom_op_t* lookup);

// Partitions live keyed SSA rows on targets that can take one branch for the
// entire subgroup. Both CFG and structured targets consume ordinary scf.if.
const loom_target_legalizer_provider_t* loom_scf_lookup_legalizer_provider(
    void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SCF_LOOKUP_H_
