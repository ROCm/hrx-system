// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stable indexed storage for target-low schedule dependencies.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_DEPENDENCIES_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_DEPENDENCIES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/util/segmented_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

// Number of dependency rows stored in each stable segment.
#define LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_CAPACITY 4096u

// Shift mapping a dependency index to its segment index.
#define LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_SHIFT 12u

// Mask mapping a dependency index to its row within a segment.
#define LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_MASK \
  (LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_CAPACITY - 1u)

static_assert((1u << LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_SHIFT) ==
                  LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_CAPACITY,
              "dependency segment capacity must match its index shift");

typedef enum loom_low_schedule_dependency_kind_e {
  // Unknown or uninitialized dependency kind.
  LOOM_LOW_SCHEDULE_DEPENDENCY_UNKNOWN = 0,
  // SSA producer-to-consumer dependency.
  LOOM_LOW_SCHEDULE_DEPENDENCY_SSA = 1,
  // Conservative side-effect ordering dependency.
  LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT = 2,
  // Block-control dependency keeping terminators after block contents.
  LOOM_LOW_SCHEDULE_DEPENDENCY_CONTROL = 3,
  // Structural anchoring dependency keeping fixed-position packets in place.
  LOOM_LOW_SCHEDULE_DEPENDENCY_ANCHOR = 4,
  // Target architectural state dependency such as flags or special registers.
  LOOM_LOW_SCHEDULE_DEPENDENCY_STATE = 5,
  // Tied-result storage dependency keeping older readers before an overwrite.
  LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE = 6,
} loom_low_schedule_dependency_kind_t;

// One dependency edge between two schedule nodes.
typedef struct loom_low_schedule_dependency_t {
  // Producer node index.
  uint32_t producer_node;
  // Consumer node index.
  uint32_t consumer_node;
  // Dependency kind.
  loom_low_schedule_dependency_kind_t kind;
  // Operand index for SSA dependencies, or UINT32_MAX.
  uint32_t operand_index;
} loom_low_schedule_dependency_t;

// One memory-visibility edge between two schedule nodes. Visibility edges are
// consumed by target overlays such as wait insertion; they do not constrain the
// block-local list scheduler.
typedef struct loom_low_schedule_visibility_dependency_t {
  // Producer node whose memory effect must be visible first.
  uint32_t producer_node;
  // Consumer node that observes or overwrites the producer's memory effect.
  uint32_t consumer_node;
  // Dependency kind. Currently always EFFECT.
  loom_low_schedule_dependency_kind_t kind;
} loom_low_schedule_visibility_dependency_t;

// Stable storage for one range of indexed scheduler dependencies.
typedef iree_alignas(64) struct loom_low_schedule_dependency_segment_t {
  // Dependency rows indexed by the low bits of a dependency index.
  loom_low_schedule_dependency_t
      rows[LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_CAPACITY];
} loom_low_schedule_dependency_segment_t;

// Stable storage for one range of indexed visibility dependencies.
typedef iree_alignas(
    64) struct loom_low_schedule_visibility_dependency_segment_t {
  // Visibility rows indexed by the low bits of a dependency index.
  loom_low_schedule_visibility_dependency_t
      rows[LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_CAPACITY];
} loom_low_schedule_visibility_dependency_segment_t;

static_assert(sizeof(loom_low_schedule_dependency_segment_t) == 64 * 1024,
              "dependency segment must fit in a compiler workspace block");
static_assert(sizeof(loom_low_schedule_visibility_dependency_segment_t) ==
                  48 * 1024,
              "visibility segment must fit in a compiler workspace block");

// One stable segmented row sequence.
typedef struct loom_low_schedule_dependency_storage_t {
  // Number of populated rows.
  iree_host_size_t count;
  // Arena-backed stable segment directory.
  loom_segmented_storage_t segments;
} loom_low_schedule_dependency_storage_t;

// Complete dependency graph storage for one scheduled function.
typedef struct loom_low_schedule_dependency_graph_t {
  // Edges that constrain the block-local list scheduler.
  loom_low_schedule_dependency_storage_t ordering;
  // Cross-block edges consumed by target visibility planning.
  loom_low_schedule_dependency_storage_t visibility;
} loom_low_schedule_dependency_graph_t;

// Initializes an empty dependency graph. Payload segments are allocated lazily.
void loom_low_schedule_dependency_graph_initialize(
    loom_low_schedule_dependency_graph_t* out_graph);

// Moves |source| into uninitialized |out_graph| without moving dependency
// payloads. |source| is left uninitialized and must be initialized again before
// reuse.
void loom_low_schedule_dependency_graph_move(
    loom_low_schedule_dependency_graph_t* source,
    loom_low_schedule_dependency_graph_t* out_graph);

// Appends one ordering dependency.
iree_status_t loom_low_schedule_dependency_graph_append_ordering(
    loom_low_schedule_dependency_graph_t* graph,
    loom_low_schedule_dependency_t dependency, iree_arena_allocator_t* arena);

// Appends one visibility dependency.
iree_status_t loom_low_schedule_dependency_graph_append_visibility(
    loom_low_schedule_dependency_graph_t* graph,
    loom_low_schedule_visibility_dependency_t dependency,
    iree_arena_allocator_t* arena);

// Returns an ordering dependency by its stable zero-based index.
static inline const loom_low_schedule_dependency_t*
loom_low_schedule_dependency_graph_ordering_at(
    const loom_low_schedule_dependency_graph_t* graph,
    uint32_t dependency_index) {
  IREE_ASSERT(dependency_index < graph->ordering.count);
  const loom_low_schedule_dependency_segment_t* segment =
      (const loom_low_schedule_dependency_segment_t*)
          loom_segmented_storage_const_segment(
              &graph->ordering.segments,
              dependency_index >> LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_SHIFT);
  return &segment->rows[dependency_index &
                        LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_MASK];
}

// Returns a visibility dependency by its stable zero-based index.
static inline const loom_low_schedule_visibility_dependency_t*
loom_low_schedule_dependency_graph_visibility_at(
    const loom_low_schedule_dependency_graph_t* graph,
    uint32_t dependency_index) {
  IREE_ASSERT(dependency_index < graph->visibility.count);
  const loom_low_schedule_visibility_dependency_segment_t* segment =
      (const loom_low_schedule_visibility_dependency_segment_t*)
          loom_segmented_storage_const_segment(
              &graph->visibility.segments,
              dependency_index >> LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_SHIFT);
  return &segment->rows[dependency_index &
                        LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_MASK];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_DEPENDENCIES_H_
