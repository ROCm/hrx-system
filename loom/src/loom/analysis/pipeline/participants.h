// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_PARTICIPANTS_H_
#define LOOM_ANALYSIS_PIPELINE_PARTICIPANTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable engine-to-worker associations. Empty sets are NULL. Radix branches
// share unchanged subtrees between invocation and serial-scope summaries; leaf
// masks pack only present workers within a 64-engine neighborhood. The full
// host-size ordinal domain is supported. Queries never follow source edges.
typedef struct loom_pipeline_participants_t loom_pipeline_participants_t;

typedef struct loom_pipeline_participant_t {
  // Physical progress engine selected by the realization.
  iree_host_size_t engine;
  // Representative worker for this engine in the selected lifetime.
  iree_host_size_t worker;
} loom_pipeline_participant_t;

typedef struct loom_pipeline_execution_conflict_t {
  // One worker requiring progress concurrently with second_worker.
  iree_host_size_t first_worker;
  // Other worker assigned to the same engine; SIZE_MAX when valid.
  iree_host_size_t second_worker;
} loom_pipeline_execution_conflict_t;

typedef struct loom_pipeline_participant_location_t {
  // Zero-based position among present engines, or SIZE_MAX when absent.
  iree_host_size_t index;
  // Representative worker, or SIZE_MAX when absent.
  iree_host_size_t worker;
} loom_pipeline_participant_location_t;

// Builds one local assignment. Entries are sorted in place by engine and worker
// ordinal. Duplicate engines report a conflict; allocation is the only status
// failure. Result storage belongs to arena, independently of the input array.
iree_status_t loom_pipeline_participants_build(
    iree_arena_allocator_t* arena, loom_pipeline_participant_t* entries,
    iree_host_size_t count,
    const loom_pipeline_participants_t** out_participants,
    loom_pipeline_execution_conflict_t* out_conflict);

// Concurrent unions reject duplicate engines; ordered unions retain the first
// representative. Only changed radix paths allocate. Sharing an unchanged child
// or wrapping a complete set costs no additional membership storage. conflict
// is required only for a concurrent union and arrives with no prior conflict.
iree_status_t loom_pipeline_participants_merge(
    iree_arena_allocator_t* arena, const loom_pipeline_participants_t* first,
    const loom_pipeline_participants_t* second, bool concurrent,
    loom_pipeline_execution_conflict_t* conflict,
    const loom_pipeline_participants_t** out_participants);

// Adds a selected controller, preserving an already selected representative.
iree_status_t loom_pipeline_participants_insert(
    iree_arena_allocator_t* arena, loom_pipeline_participant_t participant,
    const loom_pipeline_participants_t** participants);

iree_host_size_t loom_pipeline_participants_count(
    const loom_pipeline_participants_t* participants);

// Indexed membership and compact position in O(log(engine domain)) time.
loom_pipeline_participant_location_t loom_pipeline_participants_find(
    const loom_pipeline_participants_t* participants, iree_host_size_t engine);

static inline iree_host_size_t loom_pipeline_participants_worker(
    const loom_pipeline_participants_t* participants, iree_host_size_t engine) {
  return loom_pipeline_participants_find(participants, engine).worker;
}

typedef struct loom_pipeline_participant_iterator_t {
  // Unvisited right branches; radix height is bounded by the ordinal bit width.
  const loom_pipeline_participants_t* pending[sizeof(iree_host_size_t) * 8];
  // Number of initialized pending entries.
  iree_host_size_t pending_count;
  // Current leaf while remaining is nonzero.
  const loom_pipeline_participants_t* leaf;
  // Unvisited engine bits in leaf, in ascending order.
  uint64_t remaining;
  // Next packed worker position within leaf.
  iree_host_size_t worker_index;
} loom_pipeline_participant_iterator_t;

// Iteration visits only present engines, in ascending order, without
// allocation.
void loom_pipeline_participants_begin(
    const loom_pipeline_participants_t* participants,
    loom_pipeline_participant_iterator_t* iterator);
bool loom_pipeline_participants_next(
    loom_pipeline_participant_iterator_t* iterator,
    loom_pipeline_participant_t* out_participant);

// Lexicographic comparison of membership bits, ignoring representative workers.
// This preserves the ordering of a dense absent/present row without storing it.
int loom_pipeline_participants_compare_members(
    const loom_pipeline_participants_t* first,
    const loom_pipeline_participants_t* second);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_PARTICIPANTS_H_
