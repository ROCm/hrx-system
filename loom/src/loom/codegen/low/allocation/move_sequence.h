// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-time sequencing for parallel target-low storage moves.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_MOVE_SEQUENCE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_MOVE_SEQUENCE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/move.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_move_sequence_node_t loom_low_move_sequence_node_t;
typedef struct loom_low_move_sequence_location_entry_t
    loom_low_move_sequence_location_entry_t;
typedef struct loom_low_move_sequence_location_set_t
    loom_low_move_sequence_location_set_t;
typedef struct loom_low_move_sequence_temporary_t
    loom_low_move_sequence_temporary_t;

// Reusable arena-backed state for sequencing one parallel move group at a
// time. |moves| is caller-populated. Solver arrays are allocated once at the
// declared group bound when first needed, then reused without growth. Trivial
// groups need no solver storage; acyclic groups need no cycle temporaries.
typedef struct loom_low_move_sequence_scratch_t {
  // Arena that owns all scratch arrays.
  iree_arena_allocator_t* arena;
  // Caller-populated parallel move rows.
  loom_low_move_t* moves;
  // Number of entries available in |moves|.
  iree_host_size_t move_capacity;
  // Per-move solver rows with |move_capacity| entries.
  loom_low_move_sequence_node_t* nodes;
  // Ready queue storage with |move_capacity| entries.
  iree_host_size_t* ready_queue;
  // Retained traversal order for one cycle, allocated on the first cycle and
  // reused for later groups. Has |move_capacity| entries.
  iree_host_size_t* cycle_indices;
  // Open-addressed location table used for destination and source-use lookup.
  loom_low_move_sequence_location_entry_t* location_entries;
  // Power-of-two entry count available in |location_entries|.
  iree_host_size_t location_entry_capacity;
  // Per-atomic-unit occupancy epochs for explicit physical registers. Lazily
  // allocated when a group first uses explicit storage.
  uint32_t* explicit_atomic_unit_epochs;
  // Nonzero occupancy epoch assigned to the current move group.
  uint32_t explicit_atomic_unit_epoch;
  // Cycle temporaries resolved for the current move group. Storage is bounded
  // by the smaller of half the move capacity and the descriptor class count.
  loom_low_move_sequence_temporary_t* temporaries;
  // Number of initialized entries in |temporaries| for the current group.
  iree_host_size_t temporary_count;
} loom_low_move_sequence_scratch_t;

// Resolves one scratch unit for a cyclic storage class. A false |out_resolved|
// reports that the callback emitted a user-facing allocation diagnostic.
typedef iree_status_t (*loom_low_move_sequence_resolve_temporary_fn_t)(
    void* user_data, const loom_low_move_location_t* storage_class,
    const loom_low_move_sequence_location_set_t* occupied_locations,
    loom_low_move_location_t* out_temporary, bool* out_resolved);

typedef struct loom_low_move_sequence_resolve_temporary_callback_t {
  // Resolves one target-visible scratch location.
  loom_low_move_sequence_resolve_temporary_fn_t fn;
  // Opaque data passed to |fn|.
  void* user_data;
} loom_low_move_sequence_resolve_temporary_callback_t;

// Records the first output row that writes a resolved cycle-scratch location.
typedef iree_status_t (*loom_low_move_sequence_record_scratch_fn_t)(
    void* user_data, iree_host_size_t move_index);

typedef struct loom_low_move_sequence_record_scratch_callback_t {
  // Records a cycle-scratch write by output-row index.
  loom_low_move_sequence_record_scratch_fn_t fn;
  // Opaque data passed to |fn|.
  void* user_data;
} loom_low_move_sequence_record_scratch_callback_t;

// Options for lowering one parallel move set to a linear sequence.
typedef struct loom_low_move_sequence_options_t {
  // Descriptor set defining register-class alias contracts.
  const loom_low_descriptor_set_t* descriptor_set;
  // Required resolver invoked when a cycle cannot use full-unit exchanges.
  loom_low_move_sequence_resolve_temporary_callback_t resolve_temporary;
  // Optional cycle-scratch row recorder.
  loom_low_move_sequence_record_scratch_callback_t record_scratch;
} loom_low_move_sequence_options_t;

enum loom_low_move_sequence_input_flag_bits_e {
  // At least one input row names the same source and destination storage.
  LOOM_LOW_MOVE_SEQUENCE_INPUT_FLAG_IDENTITY = 1u << 0,
  // At least one input row requires a physical move.
  LOOM_LOW_MOVE_SEQUENCE_INPUT_FLAG_ACTIVE = 1u << 1,
};
typedef uint8_t loom_low_move_sequence_input_flags_t;

// Returns true when |location| overlaps a source or destination in the current
// parallel move group, including identity moves that emit no instruction.
// |occupied_locations| is valid only during the temporary resolver callback
// that received it.
bool loom_low_move_sequence_location_set_contains(
    const loom_low_move_sequence_location_set_t* occupied_locations,
    const loom_low_move_location_t* location);

// Initializes |out_scratch| with the maximum caller-populated move count of any
// group in the surrounding plan. Later groups must fit this immutable bound.
iree_status_t loom_low_move_sequence_scratch_initialize(
    iree_arena_allocator_t* arena, iree_host_size_t move_capacity,
    loom_low_move_sequence_scratch_t* out_scratch);

// Resolves the first |move_count| caller-populated rows in |scratch->moves| to
// an ordered move list in |out_moves|. Identity moves are elided. Homogeneous
// exchange-capable cycles use one fewer exchange than their input move count;
// other cycles add one scratch-save move. Transfers retain their source
// and destination register views; aliased views need not have the same width.
// Scratch saves use the source view whose contents must survive the cycle.
// When non-NULL, |out_input_flags| reports the identity and active rows already
// classified while preparing the move set.
// |out_complete| is false only when the temporary resolver emitted an
// allocation diagnostic.
// All groups sharing |scratch| use the same descriptor set and its resolved
// storage classes.
iree_status_t loom_low_move_sequence_resolve(
    loom_low_move_sequence_scratch_t* scratch, iree_host_size_t move_count,
    const loom_low_move_sequence_options_t* options,
    iree_host_size_t out_move_capacity, loom_low_move_t* out_moves,
    iree_host_size_t* out_move_count,
    loom_low_move_sequence_input_flags_t* out_input_flags, bool* out_complete);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_MOVE_SEQUENCE_H_
