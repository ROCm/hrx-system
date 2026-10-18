// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact execution facts projected from common control-condition proofs.

#ifndef LOOM_ANALYSIS_CONTROL_PARTICIPANTS_H_
#define LOOM_ANALYSIS_CONTROL_PARTICIPANTS_H_

#include "loom/analysis/cfg_condition_derivation.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/util/segmented_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_control_participant_zeros_t;
enum loom_control_participant_zero_bits_e {
  LOOM_CONTROL_PARTICIPANT_ZERO_WORKITEM_X = 1u
                                             << LOOM_VALUE_FACT_TOPOLOGY_AXIS_X,
  LOOM_CONTROL_PARTICIPANT_ZERO_WORKITEM_Y = 1u
                                             << LOOM_VALUE_FACT_TOPOLOGY_AXIS_Y,
  LOOM_CONTROL_PARTICIPANT_ZERO_WORKITEM_Z = 1u
                                             << LOOM_VALUE_FACT_TOPOLOGY_AXIS_Z,
  LOOM_CONTROL_PARTICIPANT_ZERO_SUBGROUP_LANE =
      1u << LOOM_VALUE_FACT_TOPOLOGY_AXIS_LANE,
  LOOM_CONTROL_PARTICIPANT_ZERO_ALL =
      (1u << LOOM_VALUE_FACT_TOPOLOGY_AXIS_COUNT_) - 1,
};

typedef struct loom_control_participant_table_t {
  // Arena owning sparse block records through immutable source planning.
  iree_arena_allocator_t* arena;
  // Fixed-size segments of records, sorted by block address on publication.
  loom_segmented_storage_t records;
  // Number of initialized block records.
  uint32_t count;
} loom_control_participant_table_t;

// Initializes empty retained storage without allocating.
void loom_control_participant_table_initialize(
    iree_arena_allocator_t* arena, loom_control_participant_table_t* out_table);

// Records a block's complete inherited and local guarantees once. Zero masks
// need no record. The producer finishes all appends before publishing.
iree_status_t loom_control_participant_table_append(
    loom_control_participant_table_t* table, const loom_block_t* block,
    loom_control_participant_zeros_t zeros);

// Orders sparse records in place with O(n log n) worst-case work and no
// scratch.
void loom_control_participant_table_publish(
    loom_control_participant_table_t* table);

// Reads the published block summary without traversing IR or allocating.
// Missing blocks have no proven zero coordinate. Source edits invalidate the
// complete table, whose lifetime is bounded by its owning planning arena.
loom_control_participant_zeros_t loom_control_participant_table_lookup(
    const loom_control_participant_table_t* table, const loom_block_t* block);

// Projects shared condition relations into topology-coordinate guarantees.
// Expressions are expanded by the existing shared symbolic owner; no caller
// follows SSA producers or reconstructs condition expressions.
iree_status_t loom_control_participant_zeros_from_conditions(
    loom_symbolic_expr_context_t* expressions,
    const loom_condition_integer_relation_t* relations,
    iree_host_size_t relation_count,
    loom_control_participant_zeros_t* out_zeros);

// Produces local block guarantees from already-derived edge conditions.
// Each of four bits reaches each block at most once, bounding propagation by
// O(blocks + edges). All returned and worklist storage is scratch-owned.
iree_status_t loom_control_participant_zeros_for_cfg(
    const loom_cfg_graph_t* graph, const loom_cfg_condition_edge_facts_t* edges,
    loom_symbolic_expr_context_t* expressions, iree_arena_allocator_t* scratch,
    loom_control_participant_zeros_t** out_blocks);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CONTROL_PARTICIPANTS_H_
