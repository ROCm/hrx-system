// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Footprint-preserving physical register numbering of a completed allocation.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_NUMBERING_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_NUMBERING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/interval_assignment.h"
#include "loom/codegen/low/allocation/move.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

// External physical coordinates retained by boundary-move construction before
// identity elimination and cycle sequencing lose the ABI endpoint identity.
typedef struct loom_low_allocation_numbering_anchor_t {
  // Fixed ABI coordinate, independent of the value's local assignment.
  loom_low_allocation_abi_location_t location;
  // Number of allocation units in the boundary value.
  uint32_t unit_count;
} loom_low_allocation_numbering_anchor_t;

// Mutable allocation facts before publishing the final table. Storage
// assignment and move sequencing are complete; numbering changes only
// physical coordinates, never interference, move topology or resource bounds.
typedef struct loom_low_allocation_numbering_context_t {
  // Producer-retained operand requirements and hard storage relations.
  const loom_low_placement_table_t* placement;
  // Bound instruction and scheduled-pair preferences.
  const loom_low_placement_preference_index_t* preferences;
  // Original class extents and fixed/reserved locations, unchanged by
  // numbering.
  const loom_low_allocation_target_constraints_t* target_constraints;
  // External source coordinates indexed by formal-argument ordinal. Numbering
  // preserves these even when entry transport was an elided identity.
  const loom_low_allocation_abi_location_t* entry_locations;
  // Number of entries in |entry_locations|.
  iree_host_size_t entry_location_count;
  // Exact outgoing/incoming call and return coordinates, including identities.
  const loom_low_allocation_numbering_anchor_t* boundary_anchors;
  // Number of producer-retained ranges in |boundary_anchors|.
  iree_host_size_t boundary_anchor_count;
  // Retained implicit physical uses anchoring architectural locations.
  const loom_low_allocation_unit_liveness_t* unit_liveness;
  // Completed assignments and ordinal lookup, updated in place.
  loom_low_allocation_interval_assignment_result_t* interval_assignment;
  // Completed leases, whose physical index is rebuilt in its existing storage.
  loom_low_allocation_storage_lease_state_t* storage_leases;
  // Final physical moves, including cycle-scratch endpoints.
  loom_low_move_t* moves;
  // Number of initialized physical move rows.
  iree_host_size_t move_count;
} loom_low_allocation_numbering_context_t;

// Costs that the completed assignment presents to physical numbering.
// Instruction and scheduled-pair costs remain separate because only
// instruction costs justify replacing an otherwise successful coloring.
typedef struct loom_low_allocation_numbering_analysis_t {
  // Cost of instruction preferences at the assignment's current locations.
  uint64_t instruction_cost;
  // Cost of scheduled-pair preferences at the current locations.
  uint64_t scheduled_pair_cost;
  // Instruction cost fixed by overlap-connected physical ranges.
  uint64_t unavoidable_instruction_cost;
  // Scheduled-pair cost fixed by overlap-connected physical ranges.
  uint64_t unavoidable_scheduled_pair_cost;
} loom_low_allocation_numbering_analysis_t;

// Analyzes the preference costs exposed by one completed assignment without
// changing any physical coordinates. Unavoidable costs include only clauses
// whose violations are invariant under every rigid block permutation, making
// them a conservative lower bound for final numbering. When
// |unavoidable_instruction_reg_classes| is nonempty, the analysis resets it
// and marks every register class sharing physical storage with an unavoidable
// instruction predicate. Scratch storage is released before return.
iree_status_t loom_low_allocation_numbering_analyze(
    const loom_low_allocation_numbering_context_t* context,
    iree_bitmap_t unavoidable_instruction_reg_classes,
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_numbering_analysis_t* out_analysis);

// Improves retained instruction costs with a bounded search over bijections
// within the existing physical extent. Every assignment/lease remains rigid,
// every class stays within its original extent, and no preference exceeds its
// original cost. Explicit physical-register views and target address-state
// locations retain their original numbering. No instruction uses means no
// work or scratch allocation. Only scratch allocation can fail.
iree_status_t loom_low_allocation_number_registers(
    const loom_low_allocation_numbering_context_t* context,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_NUMBERING_H_
