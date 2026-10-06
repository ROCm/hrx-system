// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical writes constrained by retained instruction reads.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/move.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/local_value_domain.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Arena-owned retained-read events and indexed physical-write constraints.
// Unit-liveness owns operand collection; fixed-binding resolution completes
// the table before any assignment. Queries never inspect IR or rescan events.
typedef struct loom_low_allocation_write_interference_t
    loom_low_allocation_write_interference_t;

// Creates arena-owned collection state for targets with an active
// read-retention rule. Other targets leave |out_interference| NULL and allocate
// no state. Unit-liveness has already bounded the sum of register interval
// widths to UINT32_MAX; retained and copy-unit domains are subsets of it.
iree_status_t loom_low_allocation_write_interference_create(
    const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_write_interference_t** out_interference);

// Records one operand during the existing unit-liveness descriptor traversal.
// |point| is the instruction's write point, after its ordinary operand uses.
iree_status_t loom_low_allocation_write_interference_note_operand(
    loom_low_allocation_write_interference_t* interference,
    const loom_local_value_domain_t* value_domain,
    const loom_low_descriptor_set_t* descriptors,
    const loom_low_descriptor_t* descriptor, const loom_op_t* op,
    uint16_t operand_index, uint32_t point, iree_arena_allocator_t* arena);

// Retains the resolved fixed location of a value in the tracked bank. Required
// tied aliases canonicalize to one origin; other banks contribute no facts.
void loom_low_allocation_write_interference_note_fixed(
    loom_low_allocation_write_interference_t* interference,
    loom_value_ordinal_t ordinal,
    const loom_low_allocation_assignment_t* assignment);

// Finalizes path-sensitive constraints and per-point temporary exclusions.
// This consumes canonical liveness order, root CFG and structural placements;
// the same operation covers scheduled and source-order allocation.
// Persistent indexes use |arena|; construction scratch borrows and restores
// |scratch_arena|'s tail. The two arenas must be distinct.
iree_status_t loom_low_allocation_write_interference_finalize(
    loom_low_allocation_write_interference_t* interference,
    const loom_liveness_analysis_t* liveness, const loom_cfg_graph_t* cfg_graph,
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena);

// Returns the retained origin rejecting a candidate, or INVALID. Future
// fixed assignments participate. Producer-proved zero-copy equations propagate
// in both directions using reusable inference scratch, without assigning their
// endpoints. Provisional spills end propagation: spill plans rebuild allocation
// before emission. Required aliases query their canonical storage origin.
loom_value_ordinal_t loom_low_allocation_write_interference_conflicting_read(
    loom_low_allocation_write_interference_t* interference,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_assignment_t* candidate);

// A complete simultaneous location proposal, indexed by local value ordinal.
// UINT32_MAX entries preserve the current assignment. Affected origins are
// listed once so transaction checks visit only their indexed constraints.
typedef struct loom_low_allocation_write_proposal_t {
  // Proposed base per canonical origin, or UINT32_MAX when unchanged.
  uint32_t* bases;
  // Canonical origins whose locations change together.
  loom_value_ordinal_t* origins;
  // Number of initialized entries in |origins|.
  iree_host_size_t count;
} loom_low_allocation_write_proposal_t;

// Allocates reusable transaction scratch only when a rule is active.
iree_status_t loom_low_allocation_write_proposal_initialize(
    const loom_low_allocation_write_interference_t* interference,
    iree_arena_allocator_t* arena,
    loom_low_allocation_write_proposal_t* out_proposal);

// Clears the preceding proposal without walking unaffected values.
void loom_low_allocation_write_proposal_reset(
    loom_low_allocation_write_proposal_t* proposal);

// Records one member of a complete coalesced relocation group or an eviction.
void loom_low_allocation_write_proposal_add(
    const loom_low_allocation_write_interference_t* interference,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_assignment_t* assignment, uint32_t base,
    loom_low_allocation_write_proposal_t* proposal);

// Validates all endpoints against the simultaneous proposal before mutation.
bool loom_low_allocation_write_proposal_conflicts(
    loom_low_allocation_write_interference_t* interference,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_write_proposal_t* proposal);

// Returns whether a copy-cycle temporary would overwrite a retained unit at
// |write_point|. Such a temporary has no SSA lifetime of its own. Move planning
// supplies final spill-free assignments, with physical locations for every
// retained origin in the tracked bank.
bool loom_low_allocation_write_interference_temporary_conflicts(
    const loom_low_allocation_write_interference_t* interference,
    const loom_low_allocation_assignment_map_t* assignments,
    uint32_t write_point, const loom_low_move_location_t* temporary);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_H_
