// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent sequencing for parallel target-low storage moves.

#ifndef LOOM_CODEGEN_LOW_MOVE_SEQUENCE_H_
#define LOOM_CODEGEN_LOW_MOVE_SEQUENCE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"

#ifdef __cplusplus
extern "C" {
#endif

// One target-visible allocation unit.
typedef struct loom_low_move_location_t {
  // Target-visible storage kind.
  loom_low_allocation_location_kind_t location_kind;
  // Storage class for the unit.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // Physical register, target ID, or spill slot ordinal.
  uint32_t location;
} loom_low_move_location_t;

// One parallel move from an old source unit to a destination unit.
typedef struct loom_low_move_t {
  // Unit overwritten by the move.
  loom_low_move_location_t destination;
  // Unit read by the move.
  loom_low_move_location_t source;
} loom_low_move_t;

typedef struct loom_low_move_sequence_node_t loom_low_move_sequence_node_t;
typedef struct loom_low_move_sequence_location_entry_t
    loom_low_move_sequence_location_entry_t;

// Reusable arena-backed scratch for parallel move sequencing.
//
// Callers reserve move and temporary rows, populate the returned arrays, and
// call loom_low_move_sequence_emit. The scratch may grow by allocating larger
// arrays from |arena|, abandoning earlier scratch arrays to be reclaimed when
// the arena is reset. No individual allocation survives beyond the arena
// lifetime.
typedef struct loom_low_move_sequence_scratch_t {
  // Arena that owns all scratch arrays.
  iree_arena_allocator_t* arena;
  // Caller-populated parallel move rows consumed by emit.
  loom_low_move_t* moves;
  // Number of entries available in |moves|.
  iree_host_size_t move_capacity;
  // Caller-populated temporary locations used to break cycles.
  loom_low_move_location_t* temporary_locations;
  // Number of entries available in |temporary_locations|.
  iree_host_size_t temporary_location_capacity;
  // Per-move solver rows indexed by move ordinal.
  loom_low_move_sequence_node_t* nodes;
  // Number of entries available in |nodes|.
  iree_host_size_t node_capacity;
  // Ready queue storage indexed by queue ordinal.
  iree_host_size_t* ready_queue;
  // Number of entries available in |ready_queue|.
  iree_host_size_t ready_queue_capacity;
  // Open-addressed location table used for destination and source-use lookup.
  loom_low_move_sequence_location_entry_t* location_entries;
  // Power-of-two entry count available in |location_entries|.
  iree_host_size_t location_entry_capacity;
} loom_low_move_sequence_scratch_t;

typedef iree_status_t (*loom_low_move_sequence_emit_fn_t)(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source);

// Callback used to emit one already-sequenced scalar storage move.
typedef struct loom_low_move_sequence_emit_callback_t {
  // Emits |destination| <- |source|.
  loom_low_move_sequence_emit_fn_t fn;
  // Opaque user data passed to |fn|.
  void* user_data;
} loom_low_move_sequence_emit_callback_t;

// Options for lowering one parallel move set to a linear sequence.
typedef struct loom_low_move_sequence_options_t {
  // Descriptor set defining register-class alias contracts. When absent,
  // move sequencing treats descriptor register classes as independent storage.
  const loom_low_descriptor_set_t* descriptor_set;
  // Scratch units available to break cycles.
  const loom_low_move_location_t* temporary_locations;
  // Number of entries in |temporary_locations|.
  iree_host_size_t temporary_location_count;
  // Required sequenced move emitter.
  loom_low_move_sequence_emit_callback_t emit_move;
} loom_low_move_sequence_options_t;

// Returns true when two move locations name the same target-visible unit.
bool loom_low_move_locations_equal(const loom_low_move_location_t* lhs,
                                   const loom_low_move_location_t* rhs);

// Initializes reusable move-sequencing scratch backed by |arena|.
void loom_low_move_sequence_scratch_initialize(
    iree_arena_allocator_t* arena,
    loom_low_move_sequence_scratch_t* out_scratch);

// Reserves caller-populated move rows in |scratch|.
iree_status_t loom_low_move_sequence_scratch_reserve_moves(
    loom_low_move_sequence_scratch_t* scratch, iree_host_size_t move_count,
    loom_low_move_t** out_moves);

// Reserves caller-populated temporary rows in |scratch|.
iree_status_t loom_low_move_sequence_scratch_reserve_temporaries(
    loom_low_move_sequence_scratch_t* scratch,
    iree_host_size_t temporary_location_count,
    loom_low_move_location_t** out_temporary_locations);

// Returns the location for one assignment unit.
iree_status_t loom_low_move_location_from_assignment_unit(
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index,
    loom_low_move_location_t* out_location);

// Counts scalar unit moves required by one allocation edge-copy group.
iree_status_t loom_low_move_sequence_count_edge_copy_units(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group,
    iree_host_size_t* out_move_count);

// Populates |moves| from one allocation edge-copy group. |move_count| must
// match loom_low_move_sequence_count_edge_copy_units.
iree_status_t loom_low_move_sequence_populate_edge_copy_units(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group, loom_low_move_t* moves,
    iree_host_size_t move_count);

// Populates |temporary_locations| from the scratch units planned for one
// allocation edge-copy group. |temporary_location_count| must match
// group->temporary_count.
iree_status_t loom_low_move_sequence_populate_edge_copy_temporaries(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group,
    loom_low_move_location_t* temporary_locations,
    iree_host_size_t temporary_location_count);

// Populates |temporary_locations| from the scratch units planned for one
// packet-local move group. |temporary_location_count| must match
// group->temporary_count.
iree_status_t loom_low_move_sequence_populate_packet_move_temporaries(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_packet_move_temporary_group_t* group,
    loom_low_move_location_t* temporary_locations,
    iree_host_size_t temporary_location_count);

// Counts scalar unit moves required to materialize one low.slice. Coalesced
// structural slices require zero moves.
iree_status_t loom_low_move_sequence_count_slice_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    iree_host_size_t* out_move_count);

// Populates |moves| from one low.slice. |move_count| must match
// loom_low_move_sequence_count_slice_units.
iree_status_t loom_low_move_sequence_populate_slice_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    loom_low_move_t* moves, iree_host_size_t move_count);

// Counts scalar unit moves required to materialize one low.concat. Coalesced
// structural concats require zero moves.
iree_status_t loom_low_move_sequence_count_concat_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    iree_host_size_t* out_move_count);

// Populates |moves| from one low.concat. |move_count| must match
// loom_low_move_sequence_count_concat_units.
iree_status_t loom_low_move_sequence_populate_concat_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    loom_low_move_t* moves, iree_host_size_t move_count);

// Emits |moves| as a sequential move list that preserves parallel-copy
// semantics. Identity moves are elided. Acyclic overlap is handled by choosing
// a safe order. Cycles use a matching storage-class entry in
// |temporary_locations| and otherwise fail loud with
// IREE_STATUS_FAILED_PRECONDITION.
//
// The first |move_count| rows in |scratch->moves| are mutated during
// sequencing.
iree_status_t loom_low_move_sequence_emit(
    loom_low_move_sequence_scratch_t* scratch, iree_host_size_t move_count,
    const loom_low_move_sequence_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_MOVE_SEQUENCE_H_
