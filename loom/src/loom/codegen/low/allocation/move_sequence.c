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

struct loom_low_move_sequence_temporary_t {
  // Kind requested by the cycle, which may differ from the selected storage.
  uint16_t requested_kind;
  // Resolved unit in the requested register class.
  loom_low_move_location_t location;
};

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
  // Number of input moves, including identities whose locations stay occupied.
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

struct loom_low_move_sequence_location_set_t {
  // Sequencing scratch retaining the current group's location membership.
  const loom_low_move_sequence_scratch_t* scratch;
  // Descriptor set defining storage alias and atomic-unit identities.
  const loom_low_descriptor_set_t* descriptor_set;
  // Initialized power-of-two prefix of the scratch location table.
  iree_host_size_t location_entry_count;
};

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
  if (!scratch->nodes) {
    iree_host_size_t minimum_location_count = 0;
    if (!iree_host_size_checked_mul(scratch->move_capacity, 4,
                                    &minimum_location_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parallel move location table exceeds host size");
    }
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_next_power_of_two(
        minimum_location_count, &scratch->location_entry_capacity));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch->arena, scratch->move_capacity, sizeof(*scratch->nodes),
        (void**)&scratch->nodes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch->arena, scratch->move_capacity, sizeof(*scratch->ready_queue),
        (void**)&scratch->ready_queue));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch->arena, scratch->location_entry_capacity,
        sizeof(*scratch->location_entries),
        (void**)&scratch->location_entries));
  }
  // Only touch the prefix needed by this group, even though a later group can
  // use the complete fixed allocation. The product is bounded above by the
  // checked capacity used to allocate that storage.
  return loom_low_move_sequence_next_power_of_two(state->move_count * 4,
                                                  &state->location_entry_count);
}

static loom_low_move_sequence_location_entry_t*
loom_low_move_sequence_find_location(
    loom_low_move_sequence_location_entry_t* location_entries,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t location_entry_count,
    const loom_low_move_location_t* location) {
  const iree_host_size_t mask = location_entry_count - 1;
  iree_host_size_t index =
      (iree_host_size_t)loom_low_move_sequence_location_hash(descriptor_set,
                                                             location) &
      mask;
  for (iree_host_size_t probe = 0; probe < location_entry_count; ++probe) {
    loom_low_move_sequence_location_entry_t* entry = &location_entries[index];
    if (!iree_any_bit_set(entry->flags,
                          LOOM_LOW_MOVE_SEQUENCE_LOCATION_FLAG_OCCUPIED)) {
      return NULL;
    }
    if (loom_low_move_locations_share_target_storage(
            descriptor_set, &entry->location, location)) {
      return entry;
    }
    index = (index + 1) & mask;
  }
  return NULL;
}

static loom_low_move_sequence_location_entry_t*
loom_low_move_sequence_lookup_location(
    loom_low_move_sequence_state_t* state,
    const loom_low_move_location_t* location) {
  return loom_low_move_sequence_find_location(
      state->scratch->location_entries, state->options->descriptor_set,
      state->location_entry_count, location);
}

bool loom_low_move_sequence_location_set_contains(
    const loom_low_move_sequence_location_set_t* occupied_locations,
    const loom_low_move_location_t* location) {
  const loom_low_move_sequence_scratch_t* scratch = occupied_locations->scratch;
  const loom_low_descriptor_set_t* descriptor_set =
      occupied_locations->descriptor_set;
  const loom_low_allocation_assignment_t assignment = {
      .descriptor_reg_class_id = location->descriptor_reg_class_id,
      .location_kind = location->location_kind,
      .location_base = location->location,
      .location_count = 1,
  };
  if (!loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, &assignment)) {
    return loom_low_move_sequence_find_location(
               scratch->location_entries, descriptor_set,
               occupied_locations->location_entry_count, location) != NULL;
  }
  if (scratch->explicit_atomic_unit_epochs == NULL) {
    return false;
  }
  const uint32_t atomic_unit_count =
      loom_low_allocation_storage_assignment_atomic_unit_count(descriptor_set,
                                                               &assignment);
  for (uint32_t atomic_unit = 0; atomic_unit < atomic_unit_count;
       ++atomic_unit) {
    uint32_t ignored_storage_key = 0;
    uint32_t atomic_location = 0;
    loom_low_allocation_storage_assignment_atomic_unit(
        descriptor_set, &assignment, atomic_unit, &ignored_storage_key,
        &atomic_location);
    if (scratch->explicit_atomic_unit_epochs[atomic_location] ==
        scratch->explicit_atomic_unit_epoch) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_move_sequence_mark_explicit_location(
    loom_low_move_sequence_state_t* state,
    const loom_low_move_location_t* location) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->options->descriptor_set;
  const loom_low_allocation_assignment_t assignment = {
      .descriptor_reg_class_id = location->descriptor_reg_class_id,
      .location_kind = location->location_kind,
      .location_base = location->location,
      .location_count = 1,
  };
  if (!loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          descriptor_set, &assignment)) {
    return iree_ok_status();
  }
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  if (scratch->explicit_atomic_unit_epochs == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch->arena, descriptor_set->physical_register_unit_count,
        sizeof(*scratch->explicit_atomic_unit_epochs),
        (void**)&scratch->explicit_atomic_unit_epochs));
    memset(scratch->explicit_atomic_unit_epochs, 0,
           descriptor_set->physical_register_unit_count *
               sizeof(*scratch->explicit_atomic_unit_epochs));
  }
  const uint32_t atomic_unit_count =
      loom_low_allocation_storage_assignment_atomic_unit_count(descriptor_set,
                                                               &assignment);
  for (uint32_t atomic_unit = 0; atomic_unit < atomic_unit_count;
       ++atomic_unit) {
    uint32_t ignored_storage_key = 0;
    uint32_t atomic_location = 0;
    loom_low_allocation_storage_assignment_atomic_unit(
        descriptor_set, &assignment, atomic_unit, &ignored_storage_key,
        &atomic_location);
    scratch->explicit_atomic_unit_epochs[atomic_location] =
        scratch->explicit_atomic_unit_epoch;
  }
  return iree_ok_status();
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
    const loom_low_move_sequence_temporary_t* temporary =
        &scratch->temporaries[i];
    if (temporary->requested_kind == storage_class->location_kind &&
        temporary->location.descriptor_reg_class_id ==
            storage_class->descriptor_reg_class_id) {
      *out_temporary = &temporary->location;
      return iree_ok_status();
    }
  }

  if (!scratch->temporaries) {
    // Each nontrivial cycle needs at least two input rows. A register class
    // reuses its temporary across all its cycles within a group.
    const iree_host_size_t temporary_capacity =
        iree_min(scratch->move_capacity / 2,
                 state->options->descriptor_set->reg_class_count);
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch->arena, temporary_capacity, sizeof(*scratch->temporaries),
        (void**)&scratch->temporaries));
  }
  loom_low_move_sequence_temporary_t* temporary =
      &scratch->temporaries[scratch->temporary_count];
  const loom_low_move_sequence_location_set_t occupied_locations = {
      .scratch = scratch,
      .descriptor_set = state->options->descriptor_set,
      .location_entry_count = state->location_entry_count,
  };
  bool resolved = false;
  IREE_RETURN_IF_ERROR(state->options->resolve_temporary.fn(
      state->options->resolve_temporary.user_data, storage_class,
      &occupied_locations, &temporary->location, &resolved));
  if (!resolved) {
    state->complete = false;
    return iree_ok_status();
  }
  ++scratch->temporary_count;
  temporary->requested_kind = storage_class->location_kind;
  *out_temporary = &temporary->location;
  *out_first_use = true;
  return iree_ok_status();
}

static iree_status_t loom_low_move_sequence_prepare(
    loom_low_move_sequence_state_t* state,
    loom_low_move_sequence_input_flags_t* out_input_flags) {
  loom_low_move_sequence_scratch_t* scratch = state->scratch;
  IREE_ASSERT_LE(state->move_count, scratch->move_capacity);
  iree_host_size_t active_move_count = 0;
  iree_host_size_t active_move_index = 0;
  for (iree_host_size_t i = 0; i < state->move_count; ++i) {
    const loom_low_move_t move = scratch->moves[i];
    if (loom_low_move_locations_share_target_storage(
            state->options->descriptor_set, &move.destination, &move.source)) {
      continue;
    }
    active_move_index = i;
    ++active_move_count;
  }
  *out_input_flags = 0;
  if (active_move_count != state->move_count) {
    *out_input_flags |= LOOM_LOW_MOVE_SEQUENCE_INPUT_FLAG_IDENTITY;
  }
  if (active_move_count != 0) {
    *out_input_flags |= LOOM_LOW_MOVE_SEQUENCE_INPUT_FLAG_ACTIVE;
  }
  state->active_count = active_move_count;
  scratch->temporary_count = 0;
  if (active_move_count == 0) {
    return iree_ok_status();
  }
  if (active_move_count == 1) {
    state->out_moves[0] = scratch->moves[active_move_index];
    state->out_move_count = 1;
    state->active_count = 0;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_low_move_sequence_prepare_solver_scratch(state));
  memset(scratch->nodes, 0, state->move_count * sizeof(*scratch->nodes));
  memset(scratch->location_entries, 0,
         state->location_entry_count * sizeof(*scratch->location_entries));
  if (++scratch->explicit_atomic_unit_epoch == 0) {
    if (scratch->explicit_atomic_unit_epochs != NULL) {
      memset(scratch->explicit_atomic_unit_epochs, 0,
             state->options->descriptor_set->physical_register_unit_count *
                 sizeof(*scratch->explicit_atomic_unit_epochs));
    }
    scratch->explicit_atomic_unit_epoch = 1;
  }

  // Identity destinations still own their contents throughout the parallel
  // transfer. Keep them in the occupied set even though they emit no move,
  // so a cycle cannot borrow their storage as a temporary.
  for (iree_host_size_t i = 0; i < state->move_count; ++i) {
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
    if (!loom_low_move_locations_share_target_storage(
            state->options->descriptor_set, &scratch->moves[i].destination,
            &scratch->moves[i].source)) {
      scratch->nodes[i].flags = LOOM_LOW_MOVE_SEQUENCE_NODE_FLAG_ACTIVE;
      ++source_entry->source_use_count;
    }
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_mark_explicit_location(
        state, &scratch->moves[i].destination));
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_mark_explicit_location(
        state, &scratch->moves[i].source));
  }

  for (iree_host_size_t i = 0; i < state->move_count; ++i) {
    if (!loom_low_move_sequence_node_is_active(state, i)) {
      continue;
    }
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
    const loom_low_move_location_t* source, loom_low_move_kind_t kind) {
  if (loom_low_move_locations_share_target_storage(
          state->options->descriptor_set, destination, source)) {
    return;
  }
  IREE_ASSERT_LT(state->out_move_count, state->out_move_capacity,
                 "allocation move output capacity must cover sequencing");
  state->out_moves[state->out_move_count++] = (loom_low_move_t){
      .destination = *destination,
      .source = *source,
      .kind = kind,
  };
}

static void loom_low_move_sequence_emit_ready_move(
    loom_low_move_sequence_state_t* state, iree_host_size_t move_index) {
  const loom_low_move_t move = state->scratch->moves[move_index];
  loom_low_move_sequence_append(state, &move.destination, &move.source,
                                LOOM_LOW_MOVE_KIND_COPY);
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
  // Save a source using the view that reads its original contents. The view
  // about to overwrite this storage may be narrower (for example EDX vs RDX).
  const loom_low_move_location_t saved_location =
      scratch->moves[first_move_index].source;

  // Retain the cycle order while classifying its complete storage contract.
  // Emission consumes these indices instead of walking the location graph
  // again. Mixed register views keep their original scratch-save semantics.
  if (scratch->cycle_indices == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch->arena, scratch->move_capacity, sizeof(*scratch->cycle_indices),
        (void**)&scratch->cycle_indices));
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->options->descriptor_set;
  bool can_exchange =
      saved_location.location_kind ==
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
      iree_any_bit_set(
          descriptor_set->reg_classes[saved_location.descriptor_reg_class_id]
              .flags,
          LOOM_LOW_REG_CLASS_FLAG_EXCHANGE);
  iree_host_size_t cycle_count = 0;
  const loom_low_move_sequence_location_entry_t* destination_entry =
      loom_low_move_sequence_require_location(state, &saved_location);
  for (;;) {
    IREE_ASSERT_NE(destination_entry->destination_move_index,
                   LOOM_LOW_MOVE_SEQUENCE_INDEX_NONE);
    const iree_host_size_t move_index =
        destination_entry->destination_move_index;
    IREE_ASSERT(loom_low_move_sequence_node_is_active(state, move_index));
    const loom_low_move_t move = scratch->moves[move_index];
    scratch->cycle_indices[cycle_count++] = move_index;
    can_exchange &=
        move.destination.location_kind == saved_location.location_kind &&
        move.source.location_kind == saved_location.location_kind &&
        move.destination.descriptor_reg_class_id ==
            saved_location.descriptor_reg_class_id &&
        move.source.descriptor_reg_class_id ==
            saved_location.descriptor_reg_class_id;
    if (loom_low_move_locations_share_target_storage(
            descriptor_set, &move.source, &saved_location)) {
      break;
    }
    destination_entry =
        loom_low_move_sequence_require_location(state, &move.source);
  }

  const loom_low_move_location_t* temporary_location = NULL;
  if (!can_exchange) {
    bool first_temporary_use = false;
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_resolve_temporary(
        state, &saved_location, &temporary_location, &first_temporary_use));
    if (!state->complete) {
      return iree_ok_status();
    }
    const iree_host_size_t scratch_move_index = state->out_move_count;
    loom_low_move_sequence_append(state, temporary_location, &saved_location,
                                  LOOM_LOW_MOVE_KIND_COPY);
    if (first_temporary_use && state->options->record_scratch.fn != NULL) {
      IREE_RETURN_IF_ERROR(state->options->record_scratch.fn(
          state->options->record_scratch.user_data, scratch_move_index));
    }
  }

  for (iree_host_size_t i = 0; i < cycle_count; ++i) {
    const iree_host_size_t move_index = scratch->cycle_indices[i];
    const loom_low_move_t move = scratch->moves[move_index];
    const bool is_last = i + 1 == cycle_count;
    if (can_exchange) {
      // Each exchange places one result and carries the displaced value to
      // the next destination. The final destination already has its result.
      if (!is_last) {
        loom_low_move_sequence_append(state, &move.destination, &move.source,
                                      LOOM_LOW_MOVE_KIND_EXCHANGE);
      }
    } else {
      loom_low_move_sequence_append(state, &move.destination,
                                    is_last ? temporary_location : &move.source,
                                    LOOM_LOW_MOVE_KIND_COPY);
    }
    loom_low_move_sequence_deactivate_move(state, move_index);
  }
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_resolve(
    loom_low_move_sequence_scratch_t* scratch, iree_host_size_t move_count,
    const loom_low_move_sequence_options_t* options,
    iree_host_size_t out_move_capacity, loom_low_move_t* out_moves,
    iree_host_size_t* out_move_count,
    loom_low_move_sequence_input_flags_t* out_input_flags, bool* out_complete) {
  *out_move_count = 0;
  if (out_input_flags != NULL) {
    *out_input_flags = 0;
  }
  *out_complete = true;
  loom_low_move_sequence_state_t state = {
      .scratch = scratch,
      .options = options,
      .move_count = move_count,
      .out_moves = out_moves,
      .out_move_capacity = out_move_capacity,
      .complete = true,
  };
  loom_low_move_sequence_input_flags_t input_flags = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_prepare(&state, &input_flags));
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
  if (out_input_flags != NULL) {
    *out_input_flags = input_flags;
  }
  *out_complete = state.complete;
  return iree_ok_status();
}
