// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_CFG_RETURN_FRONTIER_H_
#define LOOM_TRANSFORMS_CFG_RETURN_FRONTIER_H_

#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

// Joins reachable normal returns when their execution may be
// subgroup-divergent. The function's declared body terminator owns the result
// tuple. Branches carry that tuple to one common return, giving every returning
// lane a real CFG continuation. Uniform returns and abnormal exits remain
// unchanged. Consumes the current function's CFG/control facts; a changed
// frontier invalidates them.
iree_status_t loom_cfg_converge_returns(
    loom_rewriter_t* rewriter, loom_func_like_t function,
    const loom_value_fact_cfg_region_t* facts, bool* out_changed);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CFG_RETURN_FRONTIER_H_
