// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_DESTRUCTIVE_REUSE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_DESTRUCTIVE_REUSE_H_

#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

// Refines optional structural alias permissions before physical allocation.
// A borrowed unit whose storage remains observable across a destructive use
// of the result requires a materialized transfer. Required ties and identity
// aliases retain their equality; CFG handoffs retain their edge semantics.
//
// Consumes initialized per-unit and sparse semantic liveness before storage
// propagation. Sparse segments distinguish observations on mutually exclusive
// paths without rediscovering CFG structure. Scratch is released on return;
// the placement permission bits retain the result for every allocation order
// and strategy.
iree_status_t loom_low_allocation_refine_destructive_reuse(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    loom_low_placement_table_t* placement, iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_DESTRUCTIVE_REUSE_H_
