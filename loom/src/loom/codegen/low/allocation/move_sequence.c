// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/move_sequence.h"

#include <string.h>

#include "loom/codegen/low/allocation/storage.h"
#include "loom/target/registers.h"

static bool loom_low_move_locations_share_target_storage(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_move_location_t* lhs, const loom_low_move_location_t* rhs) {
  if (lhs->location_kind != rhs->location_kind ||
      lhs->location != rhs->location) {
    return false;
  }
  return loom_low_allocation_storage_reg_classes_share(
      descriptor_set, lhs->descriptor_reg_class_id,
      rhs->descriptor_reg_class_id);
}

static bool loom_low_move_locations_share_storage_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_move_location_t* lhs, const loom_low_move_location_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         loom_low_allocation_storage_reg_classes_share(
             descriptor_set, lhs->descriptor_reg_class_id,
             rhs->descriptor_reg_class_id) &&
         loom_liveness_value_class_equal(lhs->value_class, rhs->value_class);
}

#define LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE IREE_HOST_SIZE_MAX

enum loom_low_move_sequence_node_flag_bits_e {
  LOOM_LOW_MOVE_SEQUENCE_NODE_FLAG_ACTIVE = 1u << 0,
};
typedef uint8_t loom_low_move_sequence_node_flags_t;

struct loom_low_move_sequence_node_t {
  // Active-state flags for the corresponding move row.
  loom_low_move_sequence_node_flags_t flags;
};

enum loom_low_move_sequence_location_flag_bits_e {
  LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_OCCUPIED = 1u << 0,
};
typedef uint8_t loom_low_move_sequence_location_flags_t;

struct loom_low_move_sequence_location_entry_t {
  // Target-visible unit keyed by this hash entry.
  loom_low_move_location_t location;
  // Move row that writes |location|, or INDEX_NONE when none does.
  iree_host_size_t destination_move_index;
  // Number of active moves that read |location|.
  iree_host_size_t source_use_count;
  // Occupancy flags.
  loom_low_move_sequence_location_flags_t flags;
};

static iree_status_t loom_low_move_sequence_reserve_array(
    loom_low_move_sequence_scratch_t* scratch,
    iree_host_size_t minimum_capacity, iree_host_size_t element_size,
    iree_host_size_t* inout_capacity, void** inout_ptr) {
  if (*inout_capacity >= minimum_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = minimum_capacity;
  if (*inout_capacity != 0) {
    iree_host_size_t doubled_capacity = 0;
    if (iree_host_size_checked_mul(*inout_capacity, 2, &doubled_capacity) &&
        doubled_capacity > new_capacity) {
      new_capacity = doubled_capacity;
    }
  }
  void* new_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch->arena, new_capacity,
                                                 element_size, &new_ptr));
  *inout_capacity = new_capacity;
  *inout_ptr = new_ptr;
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_scratch_initialize(
    iree_arena_allocator_t* arena, iree_host_size_t move_capacity,
    loom_low_move_sequence_scratch_t* out_scratch) {
  *out_scratch = (loom_low_move_sequence_scratch_t){
      .arena = arena,
  };
  if (move_capacity == 0) {
    return iree_ok_status();
  }
  out_scratch->move_capacity = move_capacity;
  return iree_arena_allocate_array(arena, move_capacity,
                                   sizeof(*out_scratch->moves),
                                   (void**)&out_scratch->moves);
}

typedef struct loom_low_move_sequence_state_t {
  // Scratch storage owned by the allocation arena.
  loom_low_move_sequence_scratch_t* scratch;
  // Options controlling target storage aliasing and cycle scratch resolution.
  const loom_low_move_sequence_options_t* options;
  // Number of active, non-identity moves in |scratch->moves|.
  iree_host_size_t move_count;
  // Number of moves that have not been emitted yet.
  iree_host_size_t active_count;
  // Power-of-two location-table entries used for this move set.
  iree_host_size_t location_entry_count;
  // Current read position in |scratch->ready_queue|.
  iree_host_size_t ready_head;
  // Current write position in |scratch->ready_queue|.
  iree_host_size_t ready_tail;
  // Cursor used to find the next active move when only cycles remain.
  iree_host_size_t active_cursor;
  // Caller-owned destination for ordered moves.
  loom_low_move_t* out_moves;
  // Number of records available in |out_moves|.
  iree_host_size_t out_move_capacity;
  // Number of records written to |out_moves|.
  iree_host_size_t out_move_count;
  // False when scratch resolution emitted an allocation diagnostic.
  bool complete;
} loom_low_move_sequence_state_t;

static uint64_t loom_low_move_sequence_mix64(uint64_t value) {
  value ^= value >> 33;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33;
  value *= UINT64_C(0xc4ceb9fe1a85ec53);
  value ^= value >> 33;
  return value;
}

static uint64_t loom_low_move_sequence_location_hash(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_move_location_t* location) {
  uint64_t value = (uint64_t)location->location;
  value ^= (uint64_t)location->location_kind << 32;
  value ^= (uint64_t)loom_low_reg_class_storage_key(
               descriptor_set, location->descriptor_reg_class_id)
           << 40;
  return loom_low_move_sequence_mix64(value);
}

static iree_status_t loom_low_move_sequence_next_power_of_two(
    iree_host_size_t minimum_capacity, iree_host_size_t* out_capacity) {
  iree_host_size_t capacity = 16;
  while (capacity < minimum_capacity) {
    if (capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parallel move location table is too large");
    }
    capacity *= 2;
  }
  *out_capacity = capacity;
  return iree_ok_status();
}

static iree_status_t loom_low_move_sequence_prepare_solver_scratch(
    loom_low_move_sequence_state_t* state) {
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_reserve_array(
      scratch, state->move_count, sizeof(*scratch->nodes),
      &scratch->node_capacity, (void**)&scratch->nodes));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_reserve_array(
      scratch, state->move_count, sizeof(*scratch->ready_queue),
      &scratch->ready_queue_capacity, (void**)&scratch->ready_queue));
  iree_host_size_t minimum_location_count = 0;
  if (!iree_host_size_checked_mul(state->move_count, 4,
                                  &minimum_location_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parallel move location table exceeds host size");
  }
  iree_host_size_t location_entry_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_next_power_of_two(
      minimum_location_count, &location_entry_capacity));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_reserve_array(
      scratch, location_entry_capacity, sizeof(*scratch->location_entries),
      &scratch->location_entry_capacity, (void**)&scratch->location_entries));
  state->location_entry_count = location_entry_capacity;
  return iree_ok_status();
}

static loom_low_move_sequence_location_entry_t*
loom_low_move_sequence_lookup_location(
    loom_low_move_sequence_state_t* state,
    const loom_low_move_location_t* location) {
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  const iree_host_size_t mask = state->location_entry_count - 1;
  iree_host_size_t index =
      (iree_host_size_t)loom_low_move_sequence_location_hash(
          state->options->descriptor_set, location) &
      mask;
  for (iree_host_size_t probe = 0; probe < state->location_entry_count;
       ++probe) {
    loom_low_move_sequence_location_entry_t* entry =
        &scratch->location_entries[index];
    if (!iree_any_bit_set(entry->flags,
                          LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_OCCUPIED)) {
      return NULL;
    }
    if (loom_low_move_locations_share_target_storage(
            state->options->descriptor_set, &entry->location, location)) {
      return entry;
    }
    index = (index + 1) & mask;
  }
  return NULL;
}

static loom_low_move_sequence_location_entry_t*
loom_low_move_sequence_insert_location(
    loom_low_move_sequence_state_t* state,
    const loom_low_move_location_t* location) {
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  const iree_host_size_t mask = state->location_entry_count - 1;
  iree_host_size_t index =
      (iree_host_size_t)loom_low_move_sequence_location_hash(
          state->options->descriptor_set, location) &
      mask;
  for (iree_host_size_t probe = 0; probe < state->location_entry_count;
       ++probe) {
    loom_low_move_sequence_location_entry_t* entry =
        &scratch->location_entries[index];
    if (!iree_any_bit_set(entry->flags,
                          LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_OCCUPIED)) {
      *entry = (loom_low_move_sequence_location_entry_t){
          .location = *location,
          .destination_move_index = LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE,
          .flags = LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_OCCUPIED,
      };
      return entry;
    }
    if (loom_low_move_locations_share_target_storage(
            state->options->descriptor_set, &entry->location, location)) {
      return entry;
    }
    index = (index + 1) & mask;
  }
  IREE_ASSERT_UNREACHABLE(
      "parallel move location table exhausted after reserve");
  return NULL;
}

static loom_low_move_sequence_location_entry_t*
loom_low_move_sequence_require_location(
    loom_low_move_sequence_state_t* state,
    const loom_low_move_location_t* location) {
  loom_low_move_sequence_location_entry_t* entry =
      loom_low_move_sequence_lookup_location(state, location);
  IREE_ASSERT(entry != NULL, "parallel move location must be present");
  return entry;
}

static void loom_low_move_sequence_enqueue_ready(
    loom_low_move_sequence_state_t* state, iree_host_size_t move_index) {
  IREE_ASSERT_LT(state->ready_tail, state->move_count,
                 "ready queue must have one entry per move");
  state->scratch->ready_queue[state->ready_tail++] = move_index;
}

static bool loom_low_move_sequence_node_is_active(
    const loom_low_move_sequence_state_t* state, iree_host_size_t move_index) {
  return iree_any_bit_set(state->scratch->nodes[move_index].flags,
                          LOOM_LOW_MOVE_SEQUENCE_NODE_FLAG_ACTIVE);
}

static void loom_low_move_sequence_deactivate_move(
    loom_low_move_sequence_state_t* state, iree_host_size_t move_index) {
  IREE_ASSERT(loom_low_move_sequence_node_is_active(state, move_index),
              "parallel move row must be active before deactivation");
  state->scratch->nodes[move_index].flags = 0;
  --state->active_count;
  loom_low_move_sequence_location_entry_t* source_entry =
      loom_low_move_sequence_require_location(
          state, &state->scratch->moves[move_index].source);
  IREE_ASSERT_NE(source_entry->source_use_count, 0,
                 "parallel move source-use count must be positive");
  --source_entry->source_use_count;
  if (source_entry->source_use_count == 0 &&
      source_entry->destination_move_index !=
          LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE &&
      loom_low_move_sequence_node_is_active(
          state, source_entry->destination_move_index)) {
    loom_low_move_sequence_enqueue_ready(state,
                                         source_entry->destination_move_index);
  }
}

static iree_status_t loom_low_move_sequence_resolve_temporary(
    loom_low_move_sequence_state_t* state,
    const loom_low_move_location_t* storage_class,
    const loom_low_move_location_t** out_temporary, bool* out_first_use) {
  *out_temporary = NULL;
  *out_first_use = false;
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  for (iree_host_size_t i = 0; i < scratch->temporary_count; ++i) {
    if (loom_low_move_locations_share_storage_class(
            state->options->descriptor_set, &scratch->temporaries[i],
            storage_class)) {
      *out_temporary = &scratch->temporaries[i];
      return iree_ok_status();
    }
  }

  if (scratch->temporary_count == 0) {
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_reserve_array(
        scratch, state->move_count, sizeof(*scratch->temporaries),
        &scratch->temporary_capacity, (void**)&scratch->temporaries));
  }
  IREE_ASSERT_LT(scratch->temporary_count, scratch->temporary_capacity);
  loom_low_move_location_t* temporary =
      &scratch->temporaries[scratch->temporary_count];
  bool resolved = false;
  IREE_RETURN_IF_ERROR(state->options->resolve_temporary.fn(
      state->options->resolve_temporary.user_data, storage_class,
      scratch->moves, state->move_count, temporary, &resolved));
  if (!resolved) {
    state->complete = false;
    return iree_ok_status();
  }
  ++scratch->temporary_count;
  *out_temporary = temporary;
  *out_first_use = true;
  return iree_ok_status();
}

static iree_status_t loom_low_move_sequence_prepare(
    loom_low_move_sequence_state_t* state) {
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  IREE_ASSERT_LE(state->move_count, scratch->move_capacity);
  iree_host_size_t active_move_count = 0;
  for (iree_host_size_t i = 0; i < state->move_count; ++i) {
    const loom_low_move_t move = scratch->moves[i];
    if (loom_low_move_locations_share_target_storage(
            state->options->descriptor_set, &move.destination, &move.source)) {
      continue;
    }
    scratch->moves[active_move_count++] = move;
  }
  state->move_count = active_move_count;
  state->active_count = active_move_count;
  scratch->temporary_count = 0;
  if (active_move_count == 0) {
    return iree_ok_status();
  }
  if (active_move_count == 1) {
    state->out_moves[0] = scratch->moves[0];
    state->out_move_count = 1;
    state->active_count = 0;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_low_move_sequence_prepare_solver_scratch(state));
  memset(scratch->nodes, 0, active_move_count * sizeof(*scratch->nodes));
  memset(scratch->location_entries, 0,
         state->location_entry_count * sizeof(*scratch->location_entries));

  for (iree_host_size_t i = 0; i < active_move_count; ++i) {
    scratch->nodes[i].flags = LOOM_LOW_MOVE_SEQUENCE_NODE_FLAG_ACTIVE;
    loom_low_move_sequence_location_entry_t* destination_entry =
        loom_low_move_sequence_insert_location(state,
                                               &scratch->moves[i].destination);
    IREE_ASSERT_EQ(destination_entry->destination_move_index,
                   LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE,
                   "allocation move destinations must be unique");
    destination_entry->destination_move_index = i;
    loom_low_move_sequence_location_entry_t* source_entry =
        loom_low_move_sequence_insert_location(state,
                                               &scratch->moves[i].source);
    ++source_entry->source_use_count;
  }

  for (iree_host_size_t i = 0; i < active_move_count; ++i) {
    loom_low_move_sequence_location_entry_t* destination_entry =
        loom_low_move_sequence_require_location(state,
                                                &scratch->moves[i].destination);
    if (destination_entry->source_use_count == 0) {
      loom_low_move_sequence_enqueue_ready(state, i);
    }
  }
  return iree_ok_status();
}

static void loom_low_move_sequence_append(
    loom_low_move_sequence_state_t* state,
    const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  if (loom_low_move_locations_share_target_storage(
          state->options->descriptor_set, destination, source)) {
    return;
  }
  IREE_ASSERT_LT(state->out_move_count, state->out_move_capacity,
                 "allocation move output capacity must cover sequencing");
  state->out_moves[state->out_move_count++] = (loom_low_move_t){
      .destination = *destination,
      .source = *source,
  };
}

static void loom_low_move_sequence_emit_ready_move(
    loom_low_move_sequence_state_t* state, iree_host_size_t move_index) {
  const loom_low_move_t move = state->scratch->moves[move_index];
  loom_low_move_sequence_append(state, &move.destination, &move.source);
  loom_low_move_sequence_deactivate_move(state, move_index);
}

static iree_host_size_t loom_low_move_sequence_next_active_move(
    loom_low_move_sequence_state_t* state) {
  while (state->active_cursor < state->move_count) {
    const iree_host_size_t move_index = state->active_cursor++;
    if (loom_low_move_sequence_node_is_active(state, move_index)) {
      return move_index;
    }
  }
  IREE_ASSERT_UNREACHABLE("parallel move active count must name an active row");
  return 0;
}

static iree_status_t loom_low_move_sequence_emit_cycle(
    loom_low_move_sequence_state_t* state, iree_host_size_t first_move_index) {
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  const loom_low_move_location_t saved_location =
      scratch->moves[first_move_index].destination;
  const loom_low_move_location_t* temporary_location = NULL;
  bool first_temporary_use = false;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_resolve_temporary(
      state, &saved_location, &temporary_location, &first_temporary_use));
  if (!state->complete) {
    return iree_ok_status();
  }
  const iree_host_size_t scratch_move_index = state->out_move_count;
  loom_low_move_sequence_append(state, temporary_location, &saved_location);
  if (first_temporary_use && state->options->record_scratch.fn != NULL) {
    IREE_RETURN_IF_ERROR(state->options->record_scratch.fn(
        state->options->record_scratch.user_data, scratch_move_index));
  }

  loom_low_move_location_t destination =
      scratch->moves[first_move_index].destination;
  loom_low_move_location_t source = scratch->moves[first_move_index].source;
  for (;;) {
    loom_low_move_sequence_location_entry_t* destination_entry =
        loom_low_move_sequence_require_location(state, &destination);
    IREE_ASSERT_NE(destination_entry->destination_move_index,
                   LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE);
    IREE_ASSERT(loom_low_move_sequence_node_is_active(
        state, destination_entry->destination_move_index));
    const iree_host_size_t move_index =
        destination_entry->destination_move_index;
    if (loom_low_move_locations_share_target_storage(
            state->options->descriptor_set, &source, &saved_location)) {
      loom_low_move_sequence_append(state, &destination, temporary_location);
      loom_low_move_sequence_deactivate_move(state, move_index);
      return iree_ok_status();
    }

    loom_low_move_sequence_append(state, &destination, &source);
    loom_low_move_sequence_deactivate_move(state, move_index);
    destination = source;
    destination_entry =
        loom_low_move_sequence_require_location(state, &destination);
    IREE_ASSERT_NE(destination_entry->destination_move_index,
                   LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE);
    source = scratch->moves[destination_entry->destination_move_index].source;
  }
}

iree_status_t loom_low_move_sequence_resolve(
    loom_low_move_sequence_scratch_t* scratch, iree_host_size_t move_count,
    const loom_low_move_sequence_options_t* options,
    iree_host_size_t out_move_capacity, loom_low_move_t* out_moves,
    iree_host_size_t* out_move_count, bool* out_complete) {
  *out_move_count = 0;
  *out_complete = true;
  loom_low_move_sequence_state_t state = {
      .scratch = scratch,
      .options = options,
      .move_count = move_count,
      .out_moves = out_moves,
      .out_move_capacity = out_move_capacity,
      .complete = true,
  };
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_prepare(&state));
  while (state.active_count != 0 && state.complete) {
    while (state.ready_head != state.ready_tail) {
      const iree_host_size_t move_index =
          scratch->ready_queue[state.ready_head++];
      if (!loom_low_move_sequence_node_is_active(&state, move_index)) {
        continue;
      }
      loom_low_move_sequence_emit_ready_move(&state, move_index);
    }
    if (state.active_count == 0) {
      break;
    }
    const iree_host_size_t first_move_index =
        loom_low_move_sequence_next_active_move(&state);
    IREE_RETURN_IF_ERROR(
        loom_low_move_sequence_emit_cycle(&state, first_move_index));
  }
  *out_move_count = state.out_move_count;
  *out_complete = state.complete;
  return iree_ok_status();
}
