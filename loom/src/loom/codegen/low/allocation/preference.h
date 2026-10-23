// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-local snapshots and evaluation of bound placement preferences.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_PREFERENCE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_PREFERENCE_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_preference_location_t {
  // Borrowed published/fixed assignment, including spills which block
  // prediction.
  const loom_low_allocation_assignment_t* assignment;
  // Representative slot for mandatory identity, local to the prepared use.
  uint32_t representative;
  // Zero for a peer, one for the primary candidate, two for its aggregate.
  uint32_t candidate_index;
} loom_low_allocation_preference_location_t;

typedef struct loom_low_allocation_preference_memo_entry_t
    loom_low_allocation_preference_memo_entry_t;

// Reused by every query in one assignment attempt. No candidate allocates.
typedef struct loom_low_allocation_preference_workspace_t {
  // Stable merged incident-use indexes, sized from the placement producer.
  uint32_t* use_indices;
  // Peer locations in prepared use/slot order.
  loom_low_allocation_preference_location_t* locations;
  // Reusable periodic-score entries, sized from the placement producer.
  loom_low_allocation_preference_memo_entry_t* memo_entries;
} loom_low_allocation_preference_workspace_t;

// Borrows workspace until another preparation or assignment publication.
typedef struct loom_low_allocation_preference_query_t {
  // Producer-owned uses and real value bindings.
  const loom_low_placement_preference_index_t* index;
  // Prepared distinct incident uses in stable collection order.
  const uint32_t* use_indices;
  // Prepared peer locations in use/slot order.
  const loom_low_allocation_preference_location_t* locations;
  // Number of incident uses, or zero for an inert query.
  uint32_t use_count;
  // Exact periodic scores retained only for this immutable query snapshot.
  struct {
    // Borrowed workspace cells, or NULL for a directly evaluated objective.
    loom_low_allocation_preference_memo_entry_t* entries;
    // All candidate base bits affecting the score, including offset carries.
    uint32_t location_mask;
    // Low-bit index into the power-of-two entry span; collisions compare tags.
    uint32_t index_mask;
  } memo;
  // Structural preferences and predicted hard-ordinal peers keep their
  // existing relation owner and indexes. The same objective includes both
  // tentative assembly values.
  struct {
    // Relation table, or NULL when no structural relation is actionable.
    const loom_low_placement_table_t* placement;
    // Published assignment lookup, borrowed for the query lifetime.
    const loom_low_allocation_assignment_map_t* assignments;
    // Future fixed assignments, borrowed for the query lifetime.
    const loom_low_allocation_target_constraints_t* constraints;
    // Candidate-local relation ranges. The second value may be INVALID.
    struct {
      // Actual candidate ordinal, not its tied-storage origin.
      loom_value_ordinal_t ordinal;
      // Relations whose result is this candidate.
      loom_low_placement_relation_range_t results;
      // Relations whose source is this candidate.
      loom_low_placement_relation_range_t sources;
    } candidates[2];
  } structural;
} loom_low_allocation_preference_query_t;

// Allocates once per assignment attempt. Empty indexes allocate no storage.
iree_status_t loom_low_allocation_preference_workspace_initialize(
    const loom_low_placement_preference_index_t* index,
    iree_arena_allocator_t* arena,
    loom_low_allocation_preference_workspace_t* out_workspace);

// Queries retained incidence without preparing or invalidating a snapshot.
bool loom_low_allocation_preference_has_uses(
    const loom_low_placement_preference_index_t* index,
    const loom_low_placement_table_t* placement, loom_value_ordinal_t ordinal);

// Snapshots the union of uses incident to one or two tentative values. The
// secondary ordinal is INVALID for an ordinary allocation. Concrete and fixed
// members precede mandatory origins and at most one permitted defining copy;
// optional copy prediction never changes the mandatory-identity partition.
// Named candidates remain tentative when the assignment map already contains
// them so callers can compare a bounded reassignment against its current cost.
loom_low_allocation_preference_query_t loom_low_allocation_preference_prepare(
    const loom_low_placement_preference_index_t* index,
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_target_constraints_t* constraints,
    loom_value_ordinal_t primary_ordinal,
    loom_value_ordinal_t secondary_ordinal,
    loom_low_allocation_preference_workspace_t* workspace);

// Returns the sum of established weighted clause violations. Unknown locations
// contribute only facts proved by mandatory identity. |secondary| is NULL for
// an ordinary candidate and otherwise describes the same tentative assembly.
// Each present candidate keeps the same value, class, representation and
// footprint throughout a query; only location bases and presence may vary.
uint32_t loom_low_allocation_preference_penalty(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_preference_query_t* query,
    const loom_low_allocation_assignment_t* primary,
    const loom_low_allocation_assignment_t* secondary);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_PREFERENCE_H_
