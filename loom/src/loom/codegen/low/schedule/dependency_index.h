// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact producer/consumer indexing for target-low schedule dependencies.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_DEPENDENCY_INDEX_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_DEPENDENCY_INDEX_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/schedule/dependencies.h"
#include "loom/util/segmented_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE UINT32_MAX

// Number of grouped dependency rows stored in each stable segment.
#define LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_CAPACITY 4096u

// Shift mapping a group index to its segment index.
#define LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_SHIFT 12u

// Mask mapping a group index to its row within a segment.
#define LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_MASK \
  (LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_CAPACITY - 1u)

// Number of temporary raw dependency indices stored in each segment.
#define LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_CAPACITY 16384u

// Shift mapping a raw detail index to its segment index.
#define LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_SHIFT 14u

// Mask mapping a raw detail index to its row within a segment.
#define LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_MASK \
  (LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_CAPACITY - 1u)

static_assert((1u << LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_SHIFT) ==
                  LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_CAPACITY,
              "group segment capacity must match its index shift");
static_assert((1u << LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_SHIFT) ==
                  LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_CAPACITY,
              "detail segment capacity must match its index shift");

// One distinct producer-to-consumer relation. Multiple dependency rows between
// the same nodes contribute to dependency_count instead of duplicating dynamic
// scheduler work.
typedef struct loom_low_schedule_dependency_group_t {
  // Consumer node index.
  uint32_t consumer_node;
  // Number of raw dependency rows represented by this group.
  uint32_t dependency_count;
  // Strongest signed issue-separation requirement in this relation.
  int32_t minimum_issue_separation_cycles;
  // Reserved for future grouped dependency timing facts.
  uint32_t reserved;
} loom_low_schedule_dependency_group_t;

// Stable storage for one segment of producer/consumer groups.
typedef iree_alignas(64) struct loom_low_schedule_dependency_group_segment_t {
  // Group rows indexed by the low bits of a group index.
  loom_low_schedule_dependency_group_t
      rows[LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_CAPACITY];
} loom_low_schedule_dependency_group_segment_t;

static_assert(sizeof(loom_low_schedule_dependency_group_segment_t) == 64 * 1024,
              "group segment must fit in a compiler workspace block");

// Stable scratch storage for one segment of raw dependency indices.
typedef iree_alignas(64) struct loom_low_schedule_dependency_detail_segment_t {
  // Raw dependency indices indexed by the low bits of a detail index.
  uint32_t rows[LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_CAPACITY];
} loom_low_schedule_dependency_detail_segment_t;

static_assert(sizeof(loom_low_schedule_dependency_detail_segment_t) ==
                  64 * 1024,
              "detail segment must fit in a compiler workspace block");

// Compact outgoing dependency groups for one scheduling run.
typedef struct loom_low_schedule_dependency_index_t {
  // Group range starts indexed by producer node, with a terminal sentinel.
  uint32_t* producer_group_starts;
  // Producer-contiguous segmented dependency groups.
  loom_segmented_storage_t groups;
  // Bitset identifying groups containing at least one SSA dependency.
  uint8_t* ssa_group_bits;
  // Bitset identifying groups containing at least one memory-effect dependency.
  uint8_t* effect_group_bits;
  // Number of represented nodes.
  uint32_t node_count;
  // Number of populated groups.
  uint32_t group_count;
} loom_low_schedule_dependency_index_t;

// Temporary producer-contiguous raw dependency detail used by analyses that
// need operand indices or dependency kinds. Storage is owned by the scratch
// arena passed to loom_low_schedule_dependency_index_initialize and remains
// valid until that arena is reset or deinitialized.
typedef struct loom_low_schedule_dependency_detail_index_t {
  // Raw dependency range starts indexed by producer, with a terminal sentinel.
  uint32_t* producer_dependency_starts;
  // Raw dependency indices grouped by producer in scratch segments.
  loom_segmented_storage_t dependency_indices;
  // Number of raw dependency indices.
  uint32_t dependency_count;
} loom_low_schedule_dependency_detail_index_t;

// Mutable distinct-producer frontier for one grouped dependency index.
typedef struct loom_low_schedule_dependency_frontier_t {
  // Remaining producer-group count indexed by consumer node.
  uint32_t* remaining_producer_counts;
  // XOR of remaining producer node indices indexed by consumer node.
  uint32_t* remaining_producer_xors;
  // Number of represented consumer nodes.
  uint32_t node_count;
  // Number of producer groups consumed since initialization.
  uint64_t consumed_group_count;
} loom_low_schedule_dependency_frontier_t;

// Builds an exact grouped index and initial raw indegrees. Retained index
// storage is allocated from |arena|. Temporary raw detail and construction
// storage are allocated from |scratch_arena|.
iree_status_t loom_low_schedule_dependency_index_initialize(
    const loom_low_schedule_dependency_graph_t* graph, uint32_t node_count,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* arena,
    uint32_t* out_indegrees, loom_low_schedule_dependency_index_t* out_index,
    loom_low_schedule_dependency_detail_index_t* out_detail_index);

// Returns the first group index for |producer_node|.
static inline uint32_t loom_low_schedule_dependency_index_group_begin(
    const loom_low_schedule_dependency_index_t* index, uint32_t producer_node) {
  IREE_ASSERT_LT(producer_node, index->node_count);
  return index->producer_group_starts[producer_node];
}

// Returns the exclusive group end for |producer_node|.
static inline uint32_t loom_low_schedule_dependency_index_group_end(
    const loom_low_schedule_dependency_index_t* index, uint32_t producer_node) {
  IREE_ASSERT_LT(producer_node, index->node_count);
  return index->producer_group_starts[producer_node + 1];
}

// Returns true when |group_index| contains an SSA dependency.
static inline bool loom_low_schedule_dependency_index_group_has_ssa(
    const loom_low_schedule_dependency_index_t* index, uint32_t group_index) {
  IREE_ASSERT_LT(group_index, index->group_count);
  return (index->ssa_group_bits[group_index >> 3] &
          (uint8_t)(1u << (group_index & 7u))) != 0;
}

// Returns true when |group_index| contains a memory-effect dependency.
static inline bool loom_low_schedule_dependency_index_group_has_effect(
    const loom_low_schedule_dependency_index_t* index, uint32_t group_index) {
  IREE_ASSERT_LT(group_index, index->group_count);
  return (index->effect_group_bits[group_index >> 3] &
          (uint8_t)(1u << (group_index & 7u))) != 0;
}

// Returns a dependency group by its stable zero-based index.
static inline const loom_low_schedule_dependency_group_t*
loom_low_schedule_dependency_index_group_at(
    const loom_low_schedule_dependency_index_t* index, uint32_t group_index) {
  IREE_ASSERT_LT(group_index, index->group_count);
  const loom_low_schedule_dependency_group_segment_t* segment =
      (const loom_low_schedule_dependency_group_segment_t*)
          loom_segmented_storage_const_segment(
              &index->groups,
              group_index >> LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_SHIFT);
  return &segment->rows[group_index &
                        LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_MASK];
}

// Returns one producer-contiguous raw dependency index from scratch detail.
static inline uint32_t loom_low_schedule_dependency_detail_index_at(
    const loom_low_schedule_dependency_detail_index_t* detail_index,
    uint32_t detail_ordinal) {
  IREE_ASSERT_LT(detail_ordinal, detail_index->dependency_count);
  const loom_low_schedule_dependency_detail_segment_t* segment =
      (const loom_low_schedule_dependency_detail_segment_t*)
          loom_segmented_storage_const_segment(
              &detail_index->dependency_indices,
              detail_ordinal >>
                  LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_SHIFT);
  return segment
      ->rows[detail_ordinal & LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_MASK];
}

// Initializes the mutable remaining-producer frontier for |index|.
iree_status_t loom_low_schedule_dependency_frontier_initialize(
    const loom_low_schedule_dependency_index_t* index,
    iree_arena_allocator_t* arena,
    loom_low_schedule_dependency_frontier_t* out_frontier);

// Returns the sole remaining producer for |consumer_node|, or GROUP_NONE when
// the consumer has zero or multiple remaining producers.
static inline uint32_t loom_low_schedule_dependency_frontier_remaining_producer(
    const loom_low_schedule_dependency_frontier_t* frontier,
    uint32_t consumer_node) {
  IREE_ASSERT_LT(consumer_node, frontier->node_count);
  return frontier->remaining_producer_counts[consumer_node] == 1
             ? frontier->remaining_producer_xors[consumer_node]
             : LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE;
}

// Removes one producer group after its producer is scheduled. Returns the sole
// remaining producer when this removal transitions the consumer from two
// producers to one, or GROUP_NONE otherwise.
uint32_t loom_low_schedule_dependency_frontier_consume_group(
    loom_low_schedule_dependency_frontier_t* frontier, uint32_t producer_node,
    const loom_low_schedule_dependency_group_t* group);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_DEPENDENCY_INDEX_H_
