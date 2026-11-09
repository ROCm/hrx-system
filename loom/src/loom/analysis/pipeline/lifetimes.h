// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_LIFETIMES_H_
#define LOOM_ANALYSIS_PIPELINE_LIFETIMES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_pipeline_invocations_t;

typedef enum loom_pipeline_lifetime_kind_t {
  // State owned by one construction remains live throughout its descendants.
  LOOM_PIPELINE_LIFETIME_OWNER,
  // Children may coexist, regardless of their structural enumeration order.
  LOOM_PIPELINE_LIFETIME_CONCURRENT,
  // At most one child is live. Runtime loops revisit these static lifetimes.
  LOOM_PIPELINE_LIFETIME_SERIAL,
} loom_pipeline_lifetime_kind_t;

typedef struct loom_pipeline_lifetime_t {
  // Relation between this node's children, or one construction-owned leaf.
  loom_pipeline_lifetime_kind_t kind;
  // Owning invocation ordinal for OWNER; unused for composition nodes.
  iree_host_size_t owner;
  // Parent index, or IREE_HOST_SIZE_MAX for the root.
  iree_host_size_t parent;
  // First child index, or IREE_HOST_SIZE_MAX for an empty node.
  iree_host_size_t first_child;
  // Next sibling index, or IREE_HOST_SIZE_MAX at the end of a child list.
  iree_host_size_t next_sibling;
} loom_pipeline_lifetime_t;

typedef struct loom_pipeline_lifetimes_t {
  // Parent-before-child tree derived once from retained construction control.
  const loom_pipeline_lifetime_t* nodes;
  // Number of nodes, including the root at index zero.
  iree_host_size_t count;
  // Whether any serial scope can establish reusable construction lifetimes.
  bool has_serial;
} loom_pipeline_lifetimes_t;

// Retains the coexistence structure of the bound construction graph.
// Independent compositions and independent serial scopes may coexist with each
// other and their owner's state. Children inside one serial scope cannot
// coexist, even across nested control or runtime loop iterations. No source IR
// is walked. The complete graph owns this result alongside its storage
// placements.
iree_status_t loom_pipeline_lifetimes_build(
    const struct loom_pipeline_invocations_t* invocations,
    iree_arena_allocator_t* arena, loom_pipeline_lifetimes_t* out_lifetimes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_LIFETIMES_H_
