// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_EXECUTION_H_
#define LOOM_ANALYSIS_PIPELINE_EXECUTION_H_

#include "loom/analysis/pipeline/invocations.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independent progress engines selected by a physical realization. Two workers
// can use one engine only when their complete invocation lifetimes are ordered.
// Coordinates alone do not identify an engine: a processor and a DMA controller
// at the same location can have independent progress.
typedef struct loom_pipeline_execution_schedule_t {
  // Bound serial scope, including its occurrence-specific child bindings.
  const loom_pipeline_invocation_schedule_t* source;
  // Representative worker per engine, or IREE_HOST_SIZE_MAX when absent.
  const iree_host_size_t* workers;
} loom_pipeline_execution_schedule_t;

typedef struct loom_pipeline_execution_t {
  // Number of independently progressing engines in the selected assignment.
  iree_host_size_t engine_count;
  // Invocation-major representative workers, engine_count entries per row.
  // Each row includes its concurrent children and all ordered phase visits.
  const iree_host_size_t* workers;
  // Serial participant rows, sorted by bound source scope address.
  const loom_pipeline_execution_schedule_t* schedules;
  // Number of bound serial scopes, including nested scopes.
  iree_host_size_t schedule_count;
} loom_pipeline_execution_t;

typedef struct loom_pipeline_execution_conflict_t {
  // One worker requiring progress concurrently with second_worker.
  iree_host_size_t first_worker;
  // Other worker assigned to the same engine; IREE_HOST_SIZE_MAX when valid.
  iree_host_size_t second_worker;
} loom_pipeline_execution_conflict_t;

// Physical resource correspondence requiring two worker occurrences to use the
// same progress engine. This is a proposed assignment, not an ordering proof.
typedef struct loom_pipeline_execution_sharing_t {
  // One worker borrowing the selected physical resource.
  iree_host_size_t first_worker;
  // Other worker borrowing that same resource.
  iree_host_size_t second_worker;
} loom_pipeline_execution_sharing_t;

// Resolves transitive physical sharing to the lowest worker ordinal in each
// component. Unconnected workers retain their own ordinal. Inputs are trusted
// worker indices; the common lifetime check below still admits or rejects the
// resulting assignment. No source traversal, placement, or scheduling occurs.
// Returned representatives and temporary construction state belong to arena.
iree_status_t loom_pipeline_execution_coalesce(
    iree_host_size_t worker_count,
    const loom_pipeline_execution_sharing_t* sharing,
    iree_host_size_t sharing_count, iree_arena_allocator_t* arena,
    iree_host_size_t** out_representatives);

// Checks a selected engine assignment against the retained fork/join graph and
// retains participant sets for control projection. worker_engines has one entry
// per inventory worker; IREE_HOST_SIZE_MAX excludes an independently realized
// worker from this assignment. Other entries are trusted indices below
// engine_count. An excluded worker still belongs to its source completion
// boundary; this result does not discharge its transport retirement.
//
// Independent strands, compositions, and outer serial scopes overlap. Children
// of one serial scope do not, including nested scopes, branches, and loop
// revisits. Runtime trip counts never expand the inventory. The owning analysis
// visits retained control once; physical consumers query the participant rows
// instead of rediscovering transitive worker membership or execution ordering.
//
// A conflict identifies incompatible authored placement without mutating IR.
// Status carries allocation failure only. All returned storage belongs to
// arena.
iree_status_t loom_pipeline_execution_build(
    const loom_pipeline_invocations_t* invocations,
    const iree_host_size_t* worker_engines, iree_host_size_t engine_count,
    iree_arena_allocator_t* arena, loom_pipeline_execution_t* out_execution,
    loom_pipeline_execution_conflict_t* out_conflict);

// Returns the retained physical participant record for a source scope. The
// scope belongs to the graph used to build execution; the lookup neither
// traverses nor replans it.
const loom_pipeline_execution_schedule_t* loom_pipeline_execution_schedule(
    const loom_pipeline_execution_t* execution,
    const loom_pipeline_invocation_schedule_t* schedule);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_EXECUTION_H_
