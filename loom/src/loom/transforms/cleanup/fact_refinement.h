// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sparse dominance-scoped preservation of relations carried by exact values.

#ifndef LOOM_TRANSFORMS_CLEANUP_FACT_REFINEMENT_H_
#define LOOM_TRANSFORMS_CLEANUP_FACT_REFINEMENT_H_

#include "iree/base/api.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/cleanup/fact_refinement_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

// Consumes exact identities retained by the rewriter's fact table, including
// their inherited counterpart relations. Relations that still constrain dynamic
// values become path-local SSA aliases, and indexed dominated references are
// retargeted to those aliases.
// An empty journal returns without allocating or inspecting IR. Mutations are
// published through the rewriter's ordinary flags and worklist notifications.
iree_status_t loom_fact_refinement_preserve_pending(
    loom_rewriter_t* rewriter,
    const loom_fact_refinement_policy_t* refinement_policy);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_CLEANUP_FACT_REFINEMENT_H_
