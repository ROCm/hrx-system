// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-owned finalized structural move planning.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_MOVE_PLAN_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_MOVE_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/move.h"
#include "loom/codegen/low/allocation/move_sequence.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_low_schedule_table_t;

// Allocation-wide facts needed to resolve structural move groups.
typedef struct loom_low_allocation_move_plan_context_t {
  // Descriptor set defining target storage aliasing.
  const loom_low_descriptor_set_t* descriptor_set;
  // Mutable target storage constraints and diagnostic state.
  loom_low_allocation_target_constraints_t* target_constraints;
  // Per-allocation-unit live end points.
  const loom_low_allocation_unit_liveness_t* unit_liveness;
  // Completed assignment lookup map.
  loom_low_allocation_assignment_map_t assignment_map;
  // Accepted schedule used by assignment liveness, or NULL for source order.
  const struct loom_low_schedule_table_t* schedule;
} loom_low_allocation_move_plan_context_t;

// Source-preorder traversal position for one structural move producer. A
// zero-initialized cursor starts at the function's first operation. Top-level
// operations use the accepted schedule permutation; their nested operations
// follow contiguous liveness rows in source order.
typedef struct loom_low_allocation_move_cursor_t {
  // Next top-level source node when consuming a schedule permutation.
  uint32_t source_node_index;
  // Next accepted liveness row within the current source-ordered subtree.
  uint32_t operation_index;
} loom_low_allocation_move_cursor_t;

// Allocation-wide move rows and reusable sequencing scratch.
typedef struct loom_low_allocation_move_plan_t {
  // Immutable facts shared by every move group.
  loom_low_allocation_move_plan_context_t context;
  // Arena retaining final move rows and cycle-scratch write indices.
  iree_arena_allocator_t* output_arena;
  // Temporary liveness-row indices by top-level schedule node. NULL when no
  // schedule is supplied. Never retained in allocation tables.
  uint32_t* operation_indices_by_source_node;
  // Final sequential move rows.
  loom_low_move_t* moves;
  // Number of initialized records in |moves|.
  iree_host_size_t move_count;
  // Number of records available in |moves|.
  iree_host_size_t move_capacity;
  // First final row writing each distinct cycle-scratch location.
  iree_host_size_t* scratch_move_indices;
  // Number of initialized records in |scratch_move_indices|.
  iree_host_size_t scratch_move_index_count;
  // Maximum scratch records across the plan; storage is allocated only when
  // the first cycle needs a temporary and never grows afterward.
  iree_host_size_t scratch_move_index_capacity;
  // Reusable caller-populated and solver scratch.
  loom_low_move_sequence_scratch_t sequence_scratch;
} loom_low_allocation_move_plan_t;

// Initializes a plan for at most |move_input_capacity| input rows across all
// groups and |raw_group_capacity| rows in any one group. Final storage reserves
// one extra row per possible two-row cycle. |output_arena| retains the result;
// |scratch_arena| owns the source-order index and sequencing workspace and may
// be released after the last group is appended.
iree_status_t loom_low_allocation_move_plan_initialize(
    const loom_low_allocation_move_plan_context_t* context,
    iree_host_size_t move_input_capacity, iree_host_size_t raw_group_capacity,
    iree_arena_allocator_t* output_arena, iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_move_plan_t* out_plan);

// Advances a source-preorder cursor over |op| and returns its accepted liveness
// row in constant time. The producer visits every operation included in
// liveness, not only operations that emit moves. Separate producers use
// separate zero-initialized cursors over the same immutable index.
const loom_liveness_operation_point_t*
loom_low_allocation_move_plan_next_operation(
    const loom_low_allocation_move_plan_t* plan, const loom_op_t* op,
    loom_low_allocation_move_cursor_t* cursor);

// Returns reusable storage for constructing one raw parallel move group.
loom_low_move_t* loom_low_allocation_move_plan_raw_moves(
    loom_low_allocation_move_plan_t* plan);

// Sequences |raw_move_count| rows from the reusable raw storage and appends
// the final rows to |plan|. Cycle scratch is resolved and indexed only when
// required.
iree_status_t loom_low_allocation_move_plan_append_group(
    loom_low_allocation_move_plan_t* plan,
    const loom_liveness_operation_point_t* operation_point,
    iree_host_size_t raw_move_count, loom_low_move_group_t* out_group);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_MOVE_PLAN_H_
