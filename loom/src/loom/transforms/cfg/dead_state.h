// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Removal of unobserved CFG state, including closed recurrence cycles.

#ifndef LOOM_TRANSFORMS_CFG_DEAD_STATE_H_
#define LOOM_TRANSFORMS_CFG_DEAD_STATE_H_

#include "loom/ir/local_value_domain.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Traces observable uses backward through operations and incoming block
// arguments, then removes the closed unused state without changing control
// flow. Borrows the current CFG and acquired function-local value domain;
// the caller refreshes CFG facts after a changed result. Entry arguments,
// effects, branch decisions, and nested region captures retain their meaning.
iree_status_t loom_cfg_eliminate_dead_state(
    loom_rewriter_t* rewriter, const loom_cfg_graph_t* graph,
    const loom_local_value_domain_t* domain,
    iree_host_size_t* out_removed_arguments,
    iree_host_size_t* out_removed_operations);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CFG_DEAD_STATE_H_
