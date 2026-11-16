// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_LIFETIME_LAYOUT_H_
#define LOOM_ANALYSIS_PIPELINE_LIFETIME_LAYOUT_H_

#include "loom/analysis/pipeline/lifetimes.h"

#ifdef __cplusplus
extern "C" {
#endif

// A contiguous resource range owned by one construction. Units are selected
// by the caller and uniform within a pool: bytes, semaphore pairs, or other
// interchangeable resource slots. Separate pools never alias.
typedef struct loom_pipeline_lifetime_request_t {
  // OWNER node in the retained complete lifetime tree, even for projections.
  iree_host_size_t lifetime;
  // Dense resource pool index.
  iree_host_size_t pool;
  // Required number of resource units.
  uint64_t length;
  // Required power-of-two alignment in those units.
  uint64_t alignment;
  // Assigned offset within the pool, populated by layout.
  uint64_t offset;
} loom_pipeline_lifetime_request_t;

typedef struct loom_pipeline_lifetime_extent_t {
  // Complete resource extent, including alignment gaps.
  uint64_t length;
  // Strongest request alignment in the pool, or one when empty.
  uint64_t alignment;
} loom_pipeline_lifetime_extent_t;

// One pool's sparse lifetime topology and reusable numeric workspace. The
// indexed source lifetime tree outlives this arena-owned projection. Evaluation
// mutates its workspace; independent evaluations use independent projections.
typedef struct loom_pipeline_lifetime_projection_t
    loom_pipeline_lifetime_projection_t;

// Retains the relative placement of requests in one resource pool. Only owner
// identities and input order participate in preparation; pool fields are not
// used. The input request array is not borrowed. Sizes and alignments may vary
// between evaluations without rebuilding ancestry or sorting requests again.
// Storage and construction work depend on requested owners, not tree depth.
iree_status_t loom_pipeline_lifetime_projection_prepare(
    const loom_pipeline_lifetimes_t* lifetimes,
    const loom_pipeline_lifetime_request_t* requests,
    iree_host_size_t request_count, iree_arena_allocator_t* arena,
    loom_pipeline_lifetime_projection_t** out_projection);

// Places the same ordered request identities with current lengths/alignments.
// Writes all offsets and the pool extent without allocation, sorting, or
// ancestry queries. Status reports an extent exceeding INT64_MAX; another
// evaluation may reuse the projection after that failure with representable
// request sizes.
iree_status_t loom_pipeline_lifetime_projection_evaluate(
    loom_pipeline_lifetime_projection_t* projection,
    loom_pipeline_lifetime_request_t* requests,
    loom_pipeline_lifetime_extent_t* out_extent);

// Places owner-local requests in input order, orders concurrent subtrees by
// their first request, and overlays serial lifetimes. This preserves the
// caller's resource preferences independently of source enumeration order.
// Parent-owned resources coexist with descendants.
// Requests may cover only a projection of the original tree. Unrequested
// lifetimes occupy no space. Offsets are relative to each pool; the caller
// reserves the returned extents in its physical backing. Status reports arena
// allocation or an extent exceeding the compiler's INT64_MAX resource bound.
// Scratch storage is linear in the lifetime, request, and pool inventories.
// Sparse demand projects only its relevant ancestry; dense demand shares a
// bounded group of columns instead of retaining a lifetime-by-pool matrix.
iree_status_t loom_pipeline_lifetime_layout(
    const loom_pipeline_lifetimes_t* lifetimes,
    loom_pipeline_lifetime_request_t* requests, iree_host_size_t request_count,
    iree_host_size_t pool_count, iree_arena_allocator_t* arena,
    const loom_pipeline_lifetime_extent_t** out_extents);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_LIFETIME_LAYOUT_H_
