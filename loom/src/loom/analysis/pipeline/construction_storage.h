// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_CONSTRUCTION_STORAGE_H_
#define LOOM_ANALYSIS_PIPELINE_CONSTRUCTION_STORAGE_H_

#include "loom/analysis/pipeline/resources.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_pipeline_invocations_t;

typedef struct loom_pipeline_storage_allocation_t {
  // Allocation requirement in its constructing occurrence's source domain.
  const loom_pipeline_resource_allocation_t* source;
  // Immutable byte offset within the source's selected backing store.
  uint64_t byte_offset;
} loom_pipeline_storage_allocation_t;

// Places construction-owned storage from the retained lifetime tree. Concurrent
// branches occupy distinct ranges; serial branches share a range large enough
// for their maximum extent and strongest alignment. Parent-owned allocations
// remain disjoint from every descendant. The canonical packing excludes fixed
// reservations and retains the resulting footprint for later worker/service
// allocations. No allocation or invocation identity is encoded as an SSA value.
//
// Returns one immutable array per invocation, indexed like its source
// allocation table. Placement is completed before the invocation graph is
// published. Source discovery does not reserve storage; independently placed
// graphs sharing a canonical backing remain disjoint. Status carries allocation
// or authored extent arithmetic failure.
iree_status_t loom_pipeline_construction_storage_build(
    const struct loom_pipeline_invocations_t* invocations,
    iree_arena_allocator_t* arena,
    const loom_pipeline_storage_allocation_t*** out_allocations);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_CONSTRUCTION_STORAGE_H_
