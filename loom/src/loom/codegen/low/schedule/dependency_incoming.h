// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Same-block consumer indexing for target-low schedule dependencies.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_DEPENDENCY_INCOMING_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_DEPENDENCY_INCOMING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/schedule/dependency_index.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reverse view of a dependency index used by analyses that walk a consumer's
// same-block scheduling prerequisites. Group rows remain owned by the forward
// index; this view stores only their stable indices.
typedef struct loom_low_schedule_dependency_incoming_index_t {
  // Group range starts indexed by consumer node, with a terminal sentinel.
  uint32_t* consumer_group_starts;
  // Forward-index group indices contiguous by consumer node.
  uint32_t* group_indices;
  // Number of represented nodes.
  uint32_t node_count;
  // Number of represented same-block dependency groups.
  uint32_t group_count;
} loom_low_schedule_dependency_incoming_index_t;

// Builds the same-block consumer-indexed view of |index|. Storage is allocated
// once from |arena| and remains immutable for the scheduling run.
iree_status_t loom_low_schedule_dependency_incoming_index_initialize(
    const loom_low_schedule_dependency_index_t* index,
    const loom_low_schedule_node_t* nodes, iree_arena_allocator_t* arena,
    loom_low_schedule_dependency_incoming_index_t* out_incoming_index);

// Returns the first incoming group ordinal for |consumer_node|.
static inline uint32_t loom_low_schedule_dependency_incoming_index_group_begin(
    const loom_low_schedule_dependency_incoming_index_t* incoming_index,
    uint32_t consumer_node) {
  IREE_ASSERT_LT(consumer_node, incoming_index->node_count);
  return incoming_index->consumer_group_starts[consumer_node];
}

// Returns the exclusive incoming group ordinal end for |consumer_node|.
static inline uint32_t loom_low_schedule_dependency_incoming_index_group_end(
    const loom_low_schedule_dependency_incoming_index_t* incoming_index,
    uint32_t consumer_node) {
  IREE_ASSERT_LT(consumer_node, incoming_index->node_count);
  return incoming_index->consumer_group_starts[consumer_node + 1];
}

// Returns the forward dependency-group index at |incoming_group_ordinal|.
static inline uint32_t loom_low_schedule_dependency_incoming_index_group_at(
    const loom_low_schedule_dependency_incoming_index_t* incoming_index,
    uint32_t incoming_group_ordinal) {
  IREE_ASSERT_LT(incoming_group_ordinal, incoming_index->group_count);
  return incoming_index->group_indices[incoming_group_ordinal];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_DEPENDENCY_INCOMING_H_
