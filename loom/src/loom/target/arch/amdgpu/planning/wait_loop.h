// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical loop-interval analysis for AMDGPU wait placement.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_LOOP_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_LOOP_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_wait_loop_t loom_amdgpu_wait_loop_t;

// Transient AMDGPU eligibility and ancestor index over the schedule's preserved
// canonical loop forest. Unsupported target-low edge shapes remain valid but
// produce no relocation target.
typedef struct loom_amdgpu_wait_loop_analysis_t {
  // Schedule whose CFG and node table are analyzed.
  const loom_low_schedule_table_t* schedule;
  // Canonical loop records in header order.
  const loom_amdgpu_wait_loop_t* loops;
  // Binary-lifted valid ancestor loop indices.
  const uint32_t* valid_ancestor_loops;
  // Number of canonical loop records.
  iree_host_size_t loop_count;
} loom_amdgpu_wait_loop_analysis_t;

// Builds target eligibility and a bounded ancestor index in |arena| from the
// function model's preserved loop forest. Construction takes O(16L) time and
// storage for L candidate loops. Acyclic schedules perform no allocations and
// return an empty analysis.
iree_status_t loom_amdgpu_wait_loop_analysis_initialize(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_loop_analysis_t* out_analysis);

// Returns the dedicated preheader that safely dominates |consumer_node| and
// follows |producer_node|, choosing the outermost eligible loop. Returns
// UINT16_MAX when the dependency must remain at its exact consumer. Queries
// perform at most 16 ancestor probes independent of loop nesting depth.
uint16_t loom_amdgpu_wait_loop_analysis_preheader(
    const loom_amdgpu_wait_loop_analysis_t* analysis, uint32_t producer_node,
    uint32_t consumer_node);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_LOOP_H_
