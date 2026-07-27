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

// Reusable arena-backed state for sequencing one parallel move group at a
// time. |moves| is caller-populated and every other array is solver scratch
// retained at the largest group observed.
typedef struct loom_low_move_sequence_scratch_t {
  // Arena that owns all scratch arrays.
  iree_arena_allocator_t* arena;
  // Caller-populated parallel move rows.
  loom_low_move_t* moves;
  // Number of entries available in |moves|.
  iree_host_size_t move_capacity;
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
  // Cycle temporaries resolved for the current move group.
  loom_low_move_location_t* temporaries;
  // Number of entries available in |temporaries|.
  iree_host_size_t temporary_capacity;
  // Number of initialized entries in |temporaries| for the current group.
  iree_host_size_t temporary_count;
} loom_low_move_sequence_scratch_t;

// Resolves one scratch unit for a cyclic storage class. A false |out_resolved|
// reports that the callback emitted a user-facing allocation diagnostic.
typedef iree_status_t (*loom_low_move_sequence_resolve_temporary_fn_t)(
    void* user_data, const loom_low_move_location_t* storage_class,
    const loom_low_move_t* moves, iree_host_size_t move_count,
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
  // Required resolver invoked only when a cycle is encountered.
  loom_low_move_sequence_resolve_temporary_callback_t resolve_temporary;
  // Optional cycle-scratch row recorder.
  loom_low_move_sequence_record_scratch_callback_t record_scratch;
} loom_low_move_sequence_options_t;

// Initializes |out_scratch| with exact caller-populated move capacity.
iree_status_t loom_low_move_sequence_scratch_initialize(
    iree_arena_allocator_t* arena, iree_host_size_t move_capacity,
    loom_low_move_sequence_scratch_t* out_scratch);

// Resolves the first |move_count| caller-populated rows in |scratch->moves| to
// an ordered move list in |out_moves|. Identity moves are elided and each
// cycle adds one scratch-save move. |out_complete| is false only when the
// temporary resolver emitted an allocation diagnostic.
iree_status_t loom_low_move_sequence_resolve(
    loom_low_move_sequence_scratch_t* scratch, iree_host_size_t move_count,
    const loom_low_move_sequence_options_t* options,
    iree_host_size_t out_move_capacity, loom_low_move_t* out_moves,
    iree_host_size_t* out_move_count, bool* out_complete);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_MOVE_SEQUENCE_H_
