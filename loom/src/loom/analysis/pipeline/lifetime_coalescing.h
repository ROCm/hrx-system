// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_LIFETIME_COALESCING_H_
#define LOOM_ANALYSIS_PIPELINE_LIFETIME_COALESCING_H_

#include "loom/analysis/pipeline/lifetimes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Assigns each construction to itself or an enclosing construction that can
// retain its payload between visits. owner_lengths contains independently
// aligned units per construction. Their sum is representable in uint64_t.
// Concurrent subtrees reserve disjoint budgets. Serial alternatives may
// occupy the same interval; loop depth, then retained length, prioritizes
// which alternatives move to the enclosing owner. An initial peak above
// capacity leaves every owner independent. out_owners is indexed by original
// owner ordinal and contains direct final owners, including empty payloads.
iree_status_t loom_pipeline_lifetimes_coalesce(
    const loom_pipeline_lifetimes_t* lifetimes, const uint64_t* owner_lengths,
    uint64_t capacity, iree_arena_allocator_t* scratch_arena,
    iree_host_size_t* out_owners);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_LIFETIME_COALESCING_H_
