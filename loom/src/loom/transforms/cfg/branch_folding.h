// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Folds proved CFG branches in a single structural edit.

#ifndef LOOM_TRANSFORMS_CFG_BRANCH_FOLDING_H_
#define LOOM_TRANSFORMS_CFG_BRANCH_FOLDING_H_

#include "loom/analysis/cfg_condition_facts.h"
#include "loom/analysis/condition_facts.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Folds conditional terminators with equal destinations or exact conditions
// from the rewriter's value facts. All decisions precede mutation; scratch
// storage grows only for branches that fold. Block and SSA identities remain
// unchanged. On success, a nonzero |out_folded_count| completes one structural
// edit. The caller refreshes CFG facts before querying analyses or draining
// the rewriter worklist again.
iree_status_t loom_cfg_fold_constant_branches(loom_rewriter_t* rewriter,
                                              loom_region_t* region,
                                              iree_arena_allocator_t* arena,
                                              uint16_t* out_folded_count);

// Folds conditional terminators proved by the retained block facts in |table|.
// |graph|, |table|, and |query| describe the same unchanged region snapshot and
// stay live throughout selection. Applying the selected folds has the same
// refresh contract as loom_cfg_fold_constant_branches.
iree_status_t loom_cfg_fold_path_sensitive_branches(
    loom_rewriter_t* rewriter, const loom_cfg_graph_t* graph,
    const loom_cfg_condition_relation_table_t* table,
    loom_condition_query_t* query, iree_arena_allocator_t* arena,
    uint16_t* out_folded_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_CFG_BRANCH_FOLDING_H_
