// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_FACT_LOOP_READS_H_
#define LOOM_UTIL_FACT_LOOP_READS_H_

#include "loom/util/fact_range.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packed generation and nearest-loop ordinal for one indexed definition.
typedef uint64_t loom_value_fact_loop_owner_t;

static inline loom_value_fact_loop_owner_t loom_value_fact_loop_owner_make(
    uint32_t generation, uint32_t ordinal) {
  return ((uint64_t)generation << 32) | ordinal;
}

static inline uint32_t loom_value_fact_loop_owner_generation(
    loom_value_fact_loop_owner_t owner) {
  return (uint32_t)(owner >> 32);
}

static inline uint32_t loom_value_fact_loop_owner_ordinal(
    loom_value_fact_loop_owner_t owner) {
  return (uint32_t)owner;
}

// Ownership assigned once by the outer solve's immutable definition walk.
typedef struct loom_value_fact_loop_owners_t {
  // Scope-owned packed ownership entries indexed through storage_range.
  const loom_value_fact_loop_owner_t* entries;
  // Scope-wide value-ID range covering entries.
  loom_value_fact_range_t storage_range;
  // Current root definitions; definitions outside this range cannot change.
  loom_value_fact_range_t root_range;
  // Nonzero generation selecting entries written for the current root.
  uint32_t generation;
} loom_value_fact_loop_owners_t;

// Contiguous preorder interval containing a loop and all of its descendants.
typedef struct loom_value_fact_loop_subtree_t {
  // One-based ordinal of the loop whose inputs are being observed.
  uint32_t ordinal;
  // Exclusive end of the descendant-owner interval.
  uint32_t end;
} loom_value_fact_loop_subtree_t;

// A solve observes only definitions inside its root but outside its subtree.
// Querying facts requires this immutable ownership view, not the loop solver.
// Buffers are prepared before any infallible query and reused by sibling loops.
typedef struct loom_value_fact_loop_reads_t {
  // Borrowed ownership view, frozen before the first collection begins.
  const loom_value_fact_loop_owners_t* owners;
  // Borrowed active subtree, or NULL while the buffers are idle.
  const loom_value_fact_loop_subtree_t* subtree;
  // Value-ID membership bits covering the current root range.
  uint64_t* bits;
  // Nonzero words, so clearing and publication visit only observed inputs.
  uint32_t* words;
  // Number of nonzero words in words.
  iree_host_size_t word_count;
  // Reusable buffers at the next nesting depth, shared by sibling loops.
  struct loom_value_fact_loop_reads_t* child;
} loom_value_fact_loop_reads_t;

// Prepares an idle collector before entering an infallible query. The owner
// view is already bound; the arena retains buffers until the outer solve ends.
// Failure leaves the collector idle and its arena responsible for cleanup.
iree_status_t loom_value_fact_loop_reads_begin(
    loom_value_fact_loop_reads_t* reads,
    const loom_value_fact_loop_subtree_t* subtree,
    iree_arena_allocator_t* arena);

// Records an input once, excluding immutable captures and subtree definitions.
IREE_ATTRIBUTE_NOINLINE void loom_value_fact_loop_record_read(
    loom_value_fact_loop_reads_t* reads, loom_value_id_t value_id);

// Clears observed words and retires the subtree without releasing capacity.
void loom_value_fact_loop_reads_reset(loom_value_fact_loop_reads_t* reads);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_LOOP_READS_H_
