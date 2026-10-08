// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/placement.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/unit_location.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/codegen/low/storage_relation.h"
#include "loom/ops/low/ops.h"

// Bounded collection payload fitting even a 4 KiB arena block.
#define LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK 64u

// Collection preserves operation order until the final endpoint-indexed
// scatter. Every chunk preceding the tail is full; only the total row count
// is needed to determine the tail's initialized prefix.
typedef struct loom_low_placement_relation_chunk_t {
  // Next chunk in collection order, or NULL at the tail.
  struct loom_low_placement_relation_chunk_t* next;
  // Complete relation rows in the initialized prefix.
  loom_low_placement_relation_t values[LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK];
} loom_low_placement_relation_chunk_t;

typedef struct loom_low_placement_build_state_t {
  // Module containing the analyzed low region.
  loom_module_t* module;
  // Allocation owner of validated physical bindings and diagnostics.
  loom_low_allocation_target_constraints_t* target_constraints;
  // Borrowed fixed-location requests normalized during construction.
  const loom_low_allocation_fixed_value_t* fixed_values;
  // Number of borrowed fixed-location requests.
  iree_host_size_t fixed_value_count;
  // Low function body region being analyzed.
  const loom_region_t* region;
  // Descriptor set defining target packet constraints.
  const loom_low_descriptor_set_t* descriptor_set;
  // Acquired local value domain for |region|.
  const loom_local_value_domain_t* value_domain;
  // Liveness analysis over |value_domain|.
  const loom_liveness_analysis_t* liveness;
  // Concrete scheduled pair opportunities to convert into relations.
  loom_low_placement_pair_use_list_t pair_uses;
  // Borrowed instruction recipes indexed by the selected descriptor set.
  loom_low_placement_instruction_preferences_t instruction_preferences;
  // Allocation-owned working arena, never retained with the placement table.
  iree_arena_allocator_t* preference_arena;
  // Collected uses and bindings, temporarily grown in scratch_arena.
  struct {
    // Collected real instruction/pair uses.
    loom_low_placement_preference_use_t* uses;
    // Binding slots in use order.
    loom_low_placement_preference_binding_t* bindings;
    // Number of collected uses.
    iree_host_size_t use_count;
    // Leading instruction uses, before collecting scheduled pairs.
    uint32_t instruction_use_count;
    // Number of collected binding slots.
    iree_host_size_t binding_count;
    // Capacity of uses.
    iree_host_size_t use_capacity;
    // Capacity of bindings.
    iree_host_size_t binding_capacity;
  } preferences;
  // Arena owning placement table storage.
  iree_arena_allocator_t* arena;
  // Resettable arena owning transient collected relations.
  iree_arena_allocator_t* scratch_arena;
  // Transient relations collected during the single IR walk.
  struct {
    // First chunk, or NULL before collecting any relation.
    loom_low_placement_relation_chunk_t* head;
    // Append destination, with its initialized prefix given by relation_count.
    loom_low_placement_relation_chunk_t* tail;
  } collected_relations;
  // Mutable relation records being populated.
  loom_low_placement_relation_t* relations;
  // Final whole-value edge relation indices in liveness operation order.
  uint32_t* edge_relation_indices;
  // Storage summary populated during collection and indexed during append.
  struct {
    // Final mandatory writing relation indices in liveness operation order.
    uint32_t* write_relation_indices;
    // Number of writing relations counted during collection.
    uint32_t write_relation_count;
    // Number of write indices populated at their final relation locations.
    uint32_t appended_write_relation_count;
    // Alias families present before optional storage refinement.
    loom_low_placement_storage_flags_t flags;
  } storage;
  // Relation ranges indexed by result ordinal or hard-location storage owner.
  // Before prefixing, start marks a reserved defining-transfer first slot.
  loom_low_placement_relation_range_t* ranges_by_result_ordinal;
  // Relation indices grouped by source ordinal or hard-location storage owner.
  uint32_t* relation_indices_by_source_ordinal;
  // Relation ranges indexed by source ordinal or hard-location storage owner.
  // Before prefixing, start records a unique edge destination plus one;
  // UINT32_MAX means destinations disagree, and zero means none.
  loom_low_placement_relation_range_t* ranges_by_source_ordinal;
  // Users-before-sources order for structural SSA storage relations.
  loom_value_ordinal_t* storage_value_order;
  // Number of populated entries in |storage_value_order|.
  loom_value_ordinal_t storage_value_order_count;
  // Canonical physical-storage identities borrowed from function preparation.
  const loom_value_ordinal_t* tied_storage_origins_by_value_ordinal;
  // Combined operand requirements, dense by liveness interval when needed.
  loom_low_placement_operand_constraints_t* operand_constraints_by_interval;
  // Number of relation records counted or populated.
  uint32_t relation_count;
  // Number of whole-value edge relations counted during collection.
  uint32_t edge_relation_count;
  // Number of collected concrete-location relations.
  iree_host_size_t location_relation_count;
  // Number of collected hard concrete-location relations.
  uint32_t hard_location_relation_count;
  // Number of low.copy/move/slice/concat operations that may require packet
  // moves.
  uint32_t packet_move_group_count;
  // Total units covered by low.copy/move/slice/concat relations.
  iree_host_size_t packet_move_unit_count;
  // Number of low.br operations that may require edge copies.
  uint32_t edge_copy_group_count;
  // Total units covered by low.br relations.
  iree_host_size_t branch_unit_count;
  // Maximum raw move units contributed by one operation during collection.
  iree_host_size_t max_move_group_unit_count;
  // Number of relation records appended after range prefixing.
  iree_host_size_t appended_relation_count;
  // Number of source relation indices appended after range prefixing.
  iree_host_size_t appended_source_relation_count;
  // Number of edge relation indices appended after range prefixing.
  uint32_t appended_edge_relation_count;
} loom_low_placement_build_state_t;

enum loom_low_placement_move_group_flag_bits_e {
  LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET = 1u << 0,
  LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_EDGE = 1u << 1,
};
typedef uint8_t loom_low_placement_move_group_flags_t;

static bool loom_low_placement_cause_can_alias(
    loom_low_placement_cause_t cause) {
  switch (cause) {
    case LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY:
      return true;
    default:
      return false;
  }
}

static bool loom_low_placement_relation_is_edge_payload(
    const loom_low_placement_relation_t* relation) {
  return loom_low_placement_cause_is_edge(relation->cause) &&
         relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
}

static bool loom_low_placement_cause_is_defining_transfer(
    loom_low_placement_cause_t cause) {
  return cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY ||
         cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE;
}

static bool loom_low_placement_cause_is_direct_source(
    loom_low_placement_cause_t cause) {
  return cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT ||
         loom_low_placement_cause_is_defining_transfer(cause);
}

static loom_value_ordinal_t loom_low_placement_value_ordinal(
    const loom_low_placement_build_state_t* state, loom_value_id_t value_id) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_try_ordinal(state->value_domain, value_id);
  IREE_ASSERT(value_ordinal != LOOM_VALUE_ORDINAL_INVALID,
              "verified low placement value must be inside the local value "
              "domain");
  return value_ordinal;
}

static const loom_liveness_interval_t* loom_low_placement_interval_for_ordinal(
    const loom_low_placement_build_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(state->liveness, value_ordinal);
  IREE_ASSERT(interval != NULL,
              "verified low placement value must have a liveness interval");
  return interval;
}

static iree_status_t loom_low_placement_collect_relation(
    loom_low_placement_build_state_t* state,
    const loom_low_placement_relation_t* relation) {
  IREE_ASSERT_LT(state->relation_count, UINT32_MAX);
  const uint32_t chunk_index =
      state->relation_count % LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK;
  if (chunk_index == 0) {
    loom_low_placement_relation_chunk_t* chunk = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(state->scratch_arena,
                                             sizeof(*chunk), (void**)&chunk));
    chunk->next = NULL;
    if (state->collected_relations.tail != NULL) {
      state->collected_relations.tail->next = chunk;
    } else {
      state->collected_relations.head = chunk;
    }
    state->collected_relations.tail = chunk;
  }
  loom_low_placement_relation_t* collected_relation =
      &state->collected_relations.tail->values[chunk_index];
  *collected_relation = *relation;
  if (relation->kind >= LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE &&
      relation->kind <= LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART &&
      loom_low_placement_cause_can_alias(relation->cause)) {
    collected_relation->flags |=
        LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
  }
  if (relation->kind == LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION ||
      relation->kind == LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE ||
      relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL) {
    ++state->location_relation_count;
    if (iree_any_bit_set(relation->flags,
                         LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD)) {
      ++state->hard_location_relation_count;
    }
  }
  if (loom_low_placement_relation_is_edge_payload(relation)) {
    IREE_ASSERT_LT(state->edge_relation_count, UINT32_MAX);
    ++state->edge_relation_count;
    if (relation->source_ordinal == relation->result_ordinal &&
        relation->source_unit_offset == relation->result_unit_offset) {
      collected_relation->flags |=
          LOOM_LOW_PLACEMENT_RELATION_FLAG_IDENTITY_EDGE;
    }
  }
  if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
    if (iree_any_bit_set(relation->flags,
                         LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE)) {
      ++state->storage.write_relation_count;
    } else {
      state->storage.flags |= LOOM_LOW_PLACEMENT_STORAGE_FLAG_IDENTITY_ALIASES;
    }
  } else if (relation->cause >= LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY &&
             relation->cause <= LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT &&
             loom_low_placement_relation_can_alias(collected_relation)) {
    state->storage.flags |= LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES;
    if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      state->storage.flags |= LOOM_LOW_PLACEMENT_STORAGE_FLAG_CONCAT;
    }
  }
  ++state->relation_count;
  return iree_ok_status();
}

static loom_value_ordinal_t loom_low_placement_relation_index_ordinal(
    const loom_low_placement_build_state_t* state,
    const loom_low_placement_relation_t* relation,
    loom_value_ordinal_t ordinal) {
  if (state->tied_storage_origins_by_value_ordinal != NULL &&
      relation->kind == LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL &&
      iree_any_bit_set(relation->flags,
                       LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD)) {
    return state->tied_storage_origins_by_value_ordinal[ordinal];
  }
  return ordinal;
}

static void loom_low_placement_prefix_ranges(
    loom_low_placement_build_state_t* state) {
  // Hard location obligations belong to the storage component, including
  // obligations introduced by aliases that have not been assigned yet.
  // Storage relations retain their selected SSA keys and endpoints.
  for (const loom_low_placement_relation_chunk_t* chunk =
           state->collected_relations.head;
       chunk != NULL; chunk = chunk->next) {
    const uint32_t count = chunk->next != NULL
                               ? LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK
                               : 1 + (state->relation_count - 1) %
                                         LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK;
    for (uint32_t i = 0; i < count; ++i) {
      const loom_low_placement_relation_t* relation = &chunk->values[i];
      const loom_value_ordinal_t result_ordinal =
          loom_low_placement_relation_index_ordinal(state, relation,
                                                    relation->result_ordinal);
      const loom_value_ordinal_t source_ordinal =
          loom_low_placement_relation_index_ordinal(state, relation,
                                                    relation->source_ordinal);
      if (loom_low_placement_cause_is_direct_source(relation->cause)) {
        loom_low_placement_relation_range_t* result =
            &state->ranges_by_result_ordinal[result_ordinal];
        IREE_ASSERT_EQ(result->start, 0,
                       "verified SSA result must have one direct source");
        result->start = 1;
      }
      ++state->ranges_by_result_ordinal[result_ordinal].count;
      ++state->ranges_by_source_ordinal[source_ordinal].count;
    }
  }
  uint32_t result_start = 0;
  uint32_t source_start = 0;
  for (loom_value_ordinal_t i = 0; i < state->value_domain->value_count; ++i) {
    loom_low_placement_relation_range_t* result =
        &state->ranges_by_result_ordinal[i];
    const uint32_t reserved_count = result->start;
    result_start += result->count;
    result->start = result_start - result->count;
    result->count = reserved_count;
    loom_low_placement_relation_range_t* source =
        &state->ranges_by_source_ordinal[i];
    source_start += source->count;
    source->start = source_start - source->count;
    source->count = 0;
  }
}

static void loom_low_placement_append_relation(
    loom_low_placement_build_state_t* state,
    const loom_low_placement_relation_t* relation) {
  const loom_value_ordinal_t result_ordinal =
      loom_low_placement_relation_index_ordinal(state, relation,
                                                relation->result_ordinal);
  const loom_value_ordinal_t source_ordinal =
      loom_low_placement_relation_index_ordinal(state, relation,
                                                relation->source_ordinal);
  loom_low_placement_relation_range_t* result_range =
      &state->ranges_by_result_ordinal[result_ordinal];
  // SSA gives each tied or copy/move result exactly one direct source. Its
  // slot was reserved during prefixing, even when block layout visited a use
  // of the result first. Publish reverse and edge indexes at the final
  // location.
  const iree_host_size_t relation_index =
      (iree_host_size_t)result_range->start +
      (loom_low_placement_cause_is_direct_source(relation->cause)
           ? 0
           : result_range->count++);
  IREE_ASSERT_LT(relation_index, state->relation_count);
  state->relations[relation_index] = *relation;

  loom_low_placement_relation_range_t* source_range =
      &state->ranges_by_source_ordinal[source_ordinal];
  const iree_host_size_t source_index =
      (iree_host_size_t)source_range->start + source_range->count;
  IREE_ASSERT_LT(source_index, state->relation_count);
  state->relation_indices_by_source_ordinal[source_index] =
      (uint32_t)relation_index;
  ++source_range->count;
  if (loom_low_placement_relation_is_edge_payload(relation)) {
    IREE_ASSERT_LT(state->appended_edge_relation_count,
                   state->edge_relation_count);
    state->edge_relation_indices[state->appended_edge_relation_count++] =
        (uint32_t)relation_index;
  }
  if (iree_any_bit_set(relation->flags,
                       LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE)) {
    const uint32_t write_index = state->storage.appended_write_relation_count++;
    // Required writes belong to region-free packets. Liveness collection
    // order therefore preserves their write points even across nested regions.
    IREE_ASSERT(
        write_index == 0 ||
            state->relations[state->storage
                                 .write_relation_indices[write_index - 1]]
                    .write_point <= relation->write_point,
        "required writes must retain liveness point order");
    state->storage.write_relation_indices[write_index] =
        (uint32_t)relation_index;
  }
  ++state->appended_relation_count;
  ++state->appended_source_relation_count;
}

static loom_low_placement_relation_kind_t
loom_low_placement_kind_from_storage_relation(
    loom_low_storage_relation_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_STORAGE_RELATION_SAME_STORAGE:
      return LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE;
    case LOOM_LOW_STORAGE_RELATION_SUBRANGE:
      return LOOM_LOW_PLACEMENT_RELATION_SUBRANGE;
    case LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART:
      return LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART;
    case LOOM_LOW_STORAGE_RELATION_DISJOINT_STORAGE:
      return LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE;
    case LOOM_LOW_STORAGE_RELATION_UNKNOWN:
      return LOOM_LOW_PLACEMENT_RELATION_UNKNOWN;
  }
  return LOOM_LOW_PLACEMENT_RELATION_UNKNOWN;
}

static loom_low_placement_cause_t
loom_low_placement_collect_storage_relation_cause(
    loom_low_placement_build_state_t* state,
    loom_low_placement_move_group_flags_t* move_group_flags,
    loom_low_storage_relation_cause_t cause, uint32_t unit_count) {
  switch (cause) {
    case LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT:
      return LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_COPY:
      if ((*move_group_flags & LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET) ==
          0) {
        *move_group_flags |= LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET;
        ++state->packet_move_group_count;
      }
      state->packet_move_unit_count += unit_count;
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_MOVE:
      if ((*move_group_flags & LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET) ==
          0) {
        *move_group_flags |= LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET;
        ++state->packet_move_group_count;
      }
      state->packet_move_unit_count += unit_count;
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SLICE:
      if ((*move_group_flags & LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET) ==
          0) {
        *move_group_flags |= LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET;
        ++state->packet_move_group_count;
      }
      state->packet_move_unit_count += unit_count;
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_CONCAT:
      if ((*move_group_flags & LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET) ==
          0) {
        *move_group_flags |= LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_PACKET;
        ++state->packet_move_group_count;
      }
      state->packet_move_unit_count += unit_count;
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH:
      if ((*move_group_flags & LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_EDGE) == 0) {
        *move_group_flags |= LOOM_LOW_PLACEMENT_MOVE_GROUP_FLAG_EDGE;
        ++state->edge_copy_group_count;
      }
      state->branch_unit_count += unit_count;
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_LOOP_ENTRY:
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD:
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_CONDITION:
      return LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_CONDITION;
    case LOOM_LOW_STORAGE_RELATION_CAUSE_UNKNOWN:
      return LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN;
  }
  return LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN;
}

static loom_low_placement_relation_flags_t
loom_low_placement_flags_from_storage_relation(
    loom_low_storage_relation_flags_t flags) {
  loom_low_placement_relation_flags_t placement_flags = 0;
  if (iree_any_bit_set(flags, LOOM_LOW_STORAGE_RELATION_FLAG_HARD)) {
    placement_flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD;
  }
  if (iree_any_bit_set(flags, LOOM_LOW_STORAGE_RELATION_FLAG_PREFERRED)) {
    placement_flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED;
  }
  if (iree_any_bit_set(flags, LOOM_LOW_STORAGE_RELATION_FLAG_WRITES_STORAGE)) {
    placement_flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE;
  }
  return placement_flags;
}

static void loom_low_placement_assert_storage_relation_units(
    const loom_low_storage_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    const loom_liveness_interval_t* source_interval) {
  IREE_ASSERT(
      relation->destination_unit_offset <= result_interval->unit_count &&
          relation->unit_count <=
              result_interval->unit_count - relation->destination_unit_offset,
      "verified low storage destination range must fit liveness "
      "units");
  IREE_ASSERT(relation->source_unit_offset <= source_interval->unit_count &&
                  relation->unit_count <= source_interval->unit_count -
                                              relation->source_unit_offset,
              "verified low storage source range must fit liveness units");
}

static loom_value_id_t loom_low_placement_descriptor_operand_value_id(
    const loom_op_t* op, const loom_low_operand_t* descriptor_operand) {
  if (descriptor_operand->role == LOOM_LOW_OPERAND_ROLE_RESULT) {
    return loom_op_const_results(op)[descriptor_operand->source_value_index];
  }
  return loom_op_const_operands(op)[descriptor_operand->source_value_index];
}

static iree_status_t loom_low_placement_collect_operand_constraints(
    loom_low_placement_build_state_t* state, const loom_low_operand_t* operand,
    loom_value_id_t value_id) {
  const loom_value_ordinal_t ordinal =
      loom_low_placement_value_ordinal(state, value_id);
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(state->liveness, ordinal);
  if (interval == NULL ||
      interval->value_class.type_kind != LOOM_TYPE_REGISTER) {
    return iree_ok_status();
  }
  const uint16_t reg_class_id = interval->value_class.register_class_id;
  uint8_t unit_alignment_log2 = 0;
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const loom_low_reg_class_alt_t* alternative =
        &state->descriptor_set
             ->reg_class_alts[operand->reg_class_alt_start + i];
    if (alternative->reg_class_id == reg_class_id) {
      unit_alignment_log2 =
          iree_max(unit_alignment_log2, alternative->unit_alignment_log2);
    }
  }
  const uint16_t addressable_unit_count =
      operand->address_map_kind == LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET
          ? operand->addressable_unit_count
          : 0;
  const bool has_target_address_state =
      operand->address_map_kind == LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE;
  if (unit_alignment_log2 == 0 && addressable_unit_count == 0 &&
      !has_target_address_state) {
    return iree_ok_status();
  }
  if (state->operand_constraints_by_interval == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->liveness->interval_count,
        sizeof(*state->operand_constraints_by_interval),
        (void**)&state->operand_constraints_by_interval));
    memset(state->operand_constraints_by_interval, 0,
           state->liveness->interval_count *
               sizeof(*state->operand_constraints_by_interval));
  }
  loom_low_placement_operand_constraints_t* constraints =
      &state->operand_constraints_by_interval[interval -
                                              state->liveness->intervals];
  constraints->unit_alignment_log2 =
      iree_max(constraints->unit_alignment_log2, unit_alignment_log2);
  if (addressable_unit_count != 0 &&
      (constraints->addressable_unit_count == 0 ||
       addressable_unit_count < constraints->addressable_unit_count)) {
    constraints->addressable_unit_count = addressable_unit_count;
  }
  constraints->has_target_address_state |= has_target_address_state;
  return iree_ok_status();
}

static iree_status_t loom_low_placement_collect_preference(
    loom_low_placement_build_state_t* state,
    const loom_low_placement_preference_t* preference,
    const loom_op_t* const* operations, uint16_t priority) {
  const iree_host_size_t binding_start = state->preferences.binding_count;
  uint32_t memo_capacity = UINT32_MAX;
  for (uint16_t i = 0; i < preference->value_count; ++i) {
    const loom_value_ordinal_t ordinal = loom_low_placement_value_ordinal(
        state, loom_low_placement_operation_value_id(operations,
                                                     &preference->values[i]));
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(state->liveness, ordinal);
    if (interval == NULL ||
        interval->value_class.type_kind != LOOM_TYPE_REGISTER) {
      return iree_ok_status();
    }
    memo_capacity =
        iree_min(memo_capacity,
                 state->descriptor_set
                     ->reg_classes[interval->value_class.register_class_id]
                     .allocatable_count);
  }
  // Masked locations describe linear register coordinates, not physical IDs.
  // Applicability is established once here, outside candidate enumeration.
  uint32_t dependency_mask = 0;
  for (uint16_t i = 0; i < preference->clause_count; ++i) {
    const loom_low_placement_clause_t* clause = &preference->clauses[i];
    for (uint16_t j = 0; j < clause->predicate_count; ++j) {
      const loom_low_placement_predicate_t* predicate =
          &preference->predicates[clause->predicate_start + j];
      if (predicate->kind !=
          LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION) {
        memo_capacity = 0;
        continue;
      }
      dependency_mask |= predicate->location_mask;
      const uint16_t result_class =
          loom_low_placement_interval_for_ordinal(
              state, loom_low_placement_value_ordinal(
                         state, loom_low_placement_operation_value_id(
                                    operations,
                                    &preference->values[predicate->result])))
              ->value_class.register_class_id;
      const uint16_t source_class =
          loom_low_placement_interval_for_ordinal(
              state, loom_low_placement_value_ordinal(
                         state, loom_low_placement_operation_value_id(
                                    operations,
                                    &preference->values[predicate->source])))
              ->value_class.register_class_id;
      if (loom_low_reg_class_uses_explicit_physical_registers(
              &state->descriptor_set->reg_classes[result_class]) ||
          loom_low_reg_class_uses_explicit_physical_registers(
              &state->descriptor_set->reg_classes[source_class]) ||
          loom_low_reg_class_storage_key(state->descriptor_set, result_class) !=
              loom_low_reg_class_storage_key(state->descriptor_set,
                                             source_class)) {
        return iree_ok_status();
      }
    }
  }
  // Adding a fixed unit offset can carry through every lower location bit.
  // Keep that dependency here instead of rescanning recipes at query time.
  const uint8_t location_bit_count =
      (uint8_t)(32 - iree_math_count_leading_zeros_u32(dependency_mask));
  const uint8_t index_bit_count_plus_one =
      location_bit_count == 0 || memo_capacity == 0
          ? 0
          : 1 + (uint8_t)iree_min(
                    location_bit_count,
                    31 - iree_math_count_leading_zeros_u32(memo_capacity));
  if (binding_start + preference->value_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many placement preference bindings");
  }
  if (binding_start + preference->value_count >
      state->preferences.binding_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(state->scratch_arena, binding_start,
                              binding_start + preference->value_count,
                              sizeof(*state->preferences.bindings),
                              &state->preferences.binding_capacity,
                              (void**)&state->preferences.bindings));
  }
  for (uint16_t i = 0; i < preference->value_count; ++i) {
    state->preferences.bindings[binding_start + i].value_ordinal =
        loom_low_placement_value_ordinal(
            state, loom_low_placement_operation_value_id(
                       operations, &preference->values[i]));
    state->preferences.bindings[binding_start + i].representative = i;
  }
  if (state->preferences.use_count == state->preferences.use_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->scratch_arena, state->preferences.use_count,
        state->preferences.use_count + 1, sizeof(*state->preferences.uses),
        &state->preferences.use_capacity, (void**)&state->preferences.uses));
  }
  state->preferences.uses[state->preferences.use_count++] =
      (loom_low_placement_preference_use_t){
          .preference = preference,
          .binding_start = (uint32_t)binding_start,
          .priority = iree_max(priority, 1u),
          .memo = {.location_bit_count = location_bit_count,
                   .index_bit_count_plus_one = index_bit_count_plus_one},
      };
  state->preferences.binding_count += preference->value_count;
  return iree_ok_status();
}

static bool loom_low_placement_concat_has_packet_uses(
    const loom_low_placement_build_state_t* state, loom_value_id_t value_id) {
  const loom_value_t* result = loom_module_value(state->module, value_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(result, use) {
    if (!loom_low_br_isa(loom_use_user_op(*use))) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_placement_collect_op_relations(
    loom_low_placement_build_state_t* state,
    const loom_liveness_operation_point_t* operation_point) {
  const loom_op_t* op = operation_point->op;
  bool materialize_concat = false;
  const iree_host_size_t move_unit_start =
      state->packet_move_unit_count + state->branch_unit_count;
  loom_low_placement_move_group_flags_t move_group_flags = 0;
  loom_low_storage_relation_iterator_t iterator;
  loom_low_storage_relation_iterator_initialize(state->module, op, &iterator);
  loom_low_storage_relation_t storage_relation;
  while (
      loom_low_storage_relation_iterator_next(&iterator, &storage_relation)) {
    const loom_value_ordinal_t result_ordinal =
        loom_low_placement_value_ordinal(state,
                                         storage_relation.destination_value_id);
    // Payload-bearing branches retain a group even if every destination is
    // unused, so consumers can advance their edge-copy cursor once per branch.
    const loom_low_placement_cause_t cause =
        loom_low_placement_collect_storage_relation_cause(
            state, &move_group_flags, storage_relation.cause,
            storage_relation.unit_count);
    const loom_liveness_interval_t* result_interval =
        loom_liveness_interval_for_value_ordinal(state->liveness,
                                                 result_ordinal);
    if (result_interval == NULL) {
      continue;
    }
    const loom_value_ordinal_t source_ordinal =
        loom_low_placement_value_ordinal(state,
                                         storage_relation.source_value_id);
    const loom_liveness_interval_t* source_interval =
        loom_low_placement_interval_for_ordinal(state, source_ordinal);
    loom_low_placement_assert_storage_relation_units(
        &storage_relation, result_interval, source_interval);
    loom_low_placement_relation_t placement_relation = {
        .op = storage_relation.op,
        .result_ordinal = result_ordinal,
        .source_ordinal = source_ordinal,
        .result_unit_offset = storage_relation.destination_unit_offset,
        .source_unit_offset = storage_relation.source_unit_offset,
        .unit_count = storage_relation.unit_count,
        .kind = loom_low_placement_kind_from_storage_relation(
            storage_relation.kind),
        .cause = cause,
        .flags = loom_low_placement_flags_from_storage_relation(
            storage_relation.flags),
        .write_point = cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY
                           ? operation_point->start_point + 1
                           : operation_point->end_point,
        .priority = 1,
        .source_operand_index = storage_relation.source_operand_index,
    };
    if (placement_relation.cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      if (placement_relation.result_unit_offset == 0) {
        materialize_concat = loom_low_placement_concat_has_packet_uses(
            state, storage_relation.destination_value_id);
      }
      if (materialize_concat) {
        placement_relation.flags |=
            LOOM_LOW_PLACEMENT_RELATION_FLAG_MATERIALIZE_PART;
      }
    }
    if (placement_relation.cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD ||
        placement_relation.cause ==
            LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_CONDITION) {
      // A structured edge can forward a capture that remains observable after
      // this handoff, including the next iteration of an enclosing loop.
      // Only a consumed source permits ignoring its storage interference.
      if (source_interval->end_point <= operation_point->end_point) {
        placement_relation.flags |=
            LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_low_placement_collect_relation(state, &placement_relation));
  }

  const iree_host_size_t move_unit_count = state->packet_move_unit_count +
                                           state->branch_unit_count -
                                           move_unit_start;
  state->max_move_group_unit_count =
      iree_max(state->max_move_group_unit_count, move_unit_count);

  loom_low_descriptor_packet_t packet;
  loom_low_descriptor_packet_initialize(state->descriptor_set, op, &packet);
  if (packet.descriptor == NULL) {
    return iree_ok_status();
  }
  if (state->instruction_preferences.indices_by_descriptor != NULL) {
    const uint16_t preference_index =
        state->instruction_preferences
            .indices_by_descriptor[packet.descriptor_ordinal];
    if (preference_index != 0) {
      IREE_RETURN_IF_ERROR(loom_low_placement_collect_preference(
          state,
          &state->instruction_preferences.preferences[preference_index - 1],
          &op, 1));
    }
  }
  const loom_low_descriptor_t* descriptor = packet.descriptor;
  const loom_low_operand_t* descriptor_operands =
      &state->descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand = &descriptor_operands[i];
    if (operand->source_value_index == LOOM_LOW_ID_NONE) {
      continue;
    }
    if (iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_VARIADIC)) {
      for (uint16_t j = operand->source_value_index; j < op->operand_count;
           ++j) {
        IREE_RETURN_IF_ERROR(loom_low_placement_collect_operand_constraints(
            state, operand, loom_op_const_operands(op)[j]));
      }
    } else {
      IREE_RETURN_IF_ERROR(loom_low_placement_collect_operand_constraints(
          state, operand,
          loom_low_placement_descriptor_operand_value_id(op, operand)));
    }
  }
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &state->descriptor_set
             ->constraints[descriptor->constraint_start + (uint32_t)i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_SAME_REGISTER_ORDINAL) {
      continue;
    }
    const loom_value_id_t result_value_id =
        loom_low_placement_descriptor_operand_value_id(
            op, &descriptor_operands[constraint->lhs_operand_index]);
    const loom_value_id_t source_value_id =
        loom_low_placement_descriptor_operand_value_id(
            op, &descriptor_operands[constraint->rhs_operand_index]);
    if (result_value_id == source_value_id) {
      continue;
    }
    const loom_low_placement_relation_t placement_relation = {
        .op = op,
        .result_ordinal =
            loom_low_placement_value_ordinal(state, result_value_id),
        .source_ordinal =
            loom_low_placement_value_ordinal(state, source_value_id),
        .result_unit_offset = 0,
        .source_unit_offset = 0,
        .unit_count = 1,
        .kind = LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL,
        .cause = LOOM_LOW_PLACEMENT_CAUSE_DESCRIPTOR_CONSTRAINT,
        .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD,
        .priority = 1,
        .source_operand_index = LOOM_LOW_PLACEMENT_SOURCE_OPERAND_NONE,
    };
    IREE_RETURN_IF_ERROR(
        loom_low_placement_collect_relation(state, &placement_relation));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_placement_collect_pair_relations(
    loom_low_placement_build_state_t* state,
    const loom_low_placement_pair_use_t* use) {
  IREE_ASSERT(use->first_op != NULL);
  IREE_ASSERT(use->second_op != NULL);
  IREE_ASSERT(use->placement_recipe_index !=
              LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE);
  const uint16_t recipe_index = (uint16_t)(use->placement_recipe_index - 1u);
  IREE_ASSERT_LT(recipe_index, state->pair_uses.placement_recipe_count);
  const loom_low_placement_pair_recipe_t* recipe =
      &state->pair_uses.placement_recipes[recipe_index];
  const loom_low_placement_preference_t* const* selected_preferences =
      loom_low_placement_select_pair_alternative(use, recipe);
  const loom_op_t* operations[] = {use->first_op, use->second_op};
  for (uint16_t i = 0; i < recipe->preference_count; ++i) {
    const loom_low_placement_preference_t* preference = selected_preferences[i];
    const loom_low_placement_predicate_t* recipe_relation =
        preference->predicates;
    const loom_value_id_t result_value_id = loom_low_placement_pair_value_id(
        use, &preference->values[recipe_relation->result]);
    const loom_value_id_t source_value_id = loom_low_placement_pair_value_id(
        use, &preference->values[recipe_relation->source]);
    if (result_value_id == source_value_id) {
      continue;
    }
    const iree_host_size_t use_count = state->preferences.use_count;
    IREE_RETURN_IF_ERROR(loom_low_placement_collect_preference(
        state, preference, operations, use->priority));
    if (state->preferences.use_count == use_count) {
      continue;
    }
    const loom_value_ordinal_t result_ordinal =
        loom_low_placement_value_ordinal(state, result_value_id);
    const loom_value_ordinal_t source_ordinal =
        loom_low_placement_value_ordinal(state, source_value_id);
    const loom_liveness_interval_t* result_interval =
        loom_low_placement_interval_for_ordinal(state, result_ordinal);
    const loom_liveness_interval_t* source_interval =
        loom_low_placement_interval_for_ordinal(state, source_ordinal);
    IREE_ASSERT_LE(recipe_relation->result_unit_offset,
                   result_interval->unit_count);
    IREE_ASSERT_LE(
        recipe_relation->unit_count,
        result_interval->unit_count - recipe_relation->result_unit_offset);
    IREE_ASSERT_LE(recipe_relation->source_unit_offset,
                   source_interval->unit_count);
    IREE_ASSERT_LE(
        recipe_relation->unit_count,
        source_interval->unit_count - recipe_relation->source_unit_offset);
    const loom_low_placement_relation_t relation = {
        .op = use->second_op,
        .result_ordinal = result_ordinal,
        .source_ordinal = source_ordinal,
        .result_unit_offset = recipe_relation->result_unit_offset,
        .source_unit_offset = recipe_relation->source_unit_offset,
        .unit_count = recipe_relation->unit_count,
        .location_mask = recipe_relation->location_mask,
        .kind = recipe_relation->kind,
        .cause = LOOM_LOW_PLACEMENT_CAUSE_SCHEDULE_PAIR_AFFINITY,
        .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED,
        .priority = use->priority,
        .source_operand_index = LOOM_LOW_PLACEMENT_SOURCE_OPERAND_NONE,
    };
    IREE_RETURN_IF_ERROR(loom_low_placement_collect_relation(state, &relation));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_placement_visit_ops(
    loom_low_placement_build_state_t* state) {
  for (uint32_t index = 0; index < state->liveness->operation_count;) {
    const loom_liveness_operation_span_t span = loom_liveness_operation_span(
        state->liveness, index, (uint32_t)state->liveness->operation_count);
    for (uint32_t i = 0; i < span.count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_low_placement_collect_op_relations(state, &span.rows[i]));
    }
    index += span.count;
  }
  return iree_ok_status();
}

static bool loom_low_placement_relation_orders_storage(
    const loom_low_placement_relation_t* relation) {
  return loom_low_placement_relation_can_alias(relation) &&
         relation->cause >= LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT &&
         relation->cause <= LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT;
}

static iree_status_t loom_low_placement_build_storage_value_order(
    loom_low_placement_build_state_t* state) {
  bool has_storage_relation = false;
  for (iree_host_size_t i = 0; i < state->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &state->relations[i];
    has_storage_relation |=
        loom_low_placement_relation_orders_storage(relation);
  }
  if (!has_storage_relation) {
    return iree_ok_status();
  }

  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  uint32_t* pending_users = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, value_count, sizeof(*pending_users),
      (void**)&pending_users));
  memset(pending_users, 0, value_count * sizeof(*pending_users));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, value_count, sizeof(*state->storage_value_order),
      (void**)&state->storage_value_order));
  for (iree_host_size_t i = 0; i < state->relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &state->relations[i];
    if (loom_low_placement_relation_orders_storage(relation)) {
      ++pending_users[relation->source_ordinal];
    }
  }
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    if (pending_users[i] == 0) {
      state->storage_value_order[state->storage_value_order_count++] = i;
    }
  }
  for (loom_value_ordinal_t cursor = 0;
       cursor < state->storage_value_order_count; ++cursor) {
    const loom_value_ordinal_t result_ordinal =
        state->storage_value_order[cursor];
    const loom_low_placement_relation_range_t range =
        state->ranges_by_result_ordinal[result_ordinal];
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_low_placement_relation_t* relation =
          &state->relations[range.start + i];
      if (!loom_low_placement_relation_orders_storage(relation)) {
        continue;
      }
      IREE_ASSERT_GT(pending_users[relation->source_ordinal], 0);
      if (--pending_users[relation->source_ordinal] == 0) {
        state->storage_value_order[state->storage_value_order_count++] =
            relation->source_ordinal;
      }
    }
  }
  IREE_ASSERT_EQ(state->storage_value_order_count, value_count,
                 "structural SSA storage relations must be acyclic");
  return iree_ok_status();
}

static iree_status_t loom_low_placement_index_preferences(
    loom_low_placement_build_state_t* state,
    loom_low_placement_preference_index_t* out_index) {
  const uint32_t use_count = (uint32_t)state->preferences.use_count;
  if (use_count == 0) {
    return iree_ok_status();
  }
  const uint32_t binding_count = (uint32_t)state->preferences.binding_count;
  const uint32_t instruction_use_count =
      state->preferences.instruction_use_count;
  loom_low_placement_preference_use_t* uses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->preference_arena, use_count, sizeof(*uses), (void**)&uses));
  memcpy(uses, state->preferences.uses, use_count * sizeof(*uses));
  loom_low_placement_preference_binding_t* bindings = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->preference_arena, binding_count,
                                sizeof(*bindings), (void**)&bindings));
  memcpy(bindings, state->preferences.bindings,
         binding_count * sizeof(*bindings));
  *out_index = (loom_low_placement_preference_index_t){
      .uses = uses,
      .bindings = bindings,
      .use_count = use_count,
      .instruction_use_count = instruction_use_count,
      .binding_count = binding_count,
  };
  if (instruction_use_count == use_count) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  const loom_value_ordinal_t* origins =
      state->tied_storage_origins_by_value_ordinal;
  uint32_t* offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->preference_arena, (iree_host_size_t)value_count + 1,
      sizeof(*offsets), (void**)&offsets));
  memset(offsets, 0, ((iree_host_size_t)value_count + 1) * sizeof(*offsets));
  uint32_t* last_bindings = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, value_count, sizeof(*last_bindings),
      (void**)&last_bindings));
  memset(last_bindings, 0, value_count * sizeof(*last_bindings));
  uint32_t max_memo_entry_count = 0;
  for (uint32_t i = instruction_use_count; i < use_count; ++i) {
    const loom_low_placement_preference_use_t* use = &uses[i];
    if (use->memo.index_bit_count_plus_one != 0) {
      max_memo_entry_count =
          iree_max(max_memo_entry_count,
                   UINT32_C(1) << (use->memo.index_bit_count_plus_one - 1));
    }
    for (uint16_t slot = 0; slot < use->preference->value_count; ++slot) {
      loom_low_placement_preference_binding_t* binding =
          &bindings[use->binding_start + slot];
      const loom_value_ordinal_t origin =
          origins ? origins[binding->value_ordinal] : binding->value_ordinal;
      if (last_bindings[origin] > use->binding_start) {
        binding->representative =
            last_bindings[origin] - use->binding_start - 1;
      } else {
        binding->representative = slot;
        last_bindings[origin] = use->binding_start + slot + 1;
        ++offsets[origin + 1];
      }
    }
  }
  uint32_t max_incident_use_count = 0;
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    max_incident_use_count = iree_max(max_incident_use_count, offsets[i + 1]);
    offsets[i + 1] += offsets[i];
    // The marker array becomes incident binding counts after representatives
    // have been retained. It is never cleared per use or per query.
    last_bindings[i] = 0;
  }
  const uint32_t incidence_count = offsets[value_count];
  uint32_t* use_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->preference_arena, incidence_count,
                                sizeof(*use_indices), (void**)&use_indices));
  uint32_t max_incident_binding_count = 0;
  for (uint32_t i = use_count; i > instruction_use_count; --i) {
    const loom_low_placement_preference_use_t* use = &uses[i - 1];
    for (uint16_t slot = 0; slot < use->preference->value_count; ++slot) {
      const loom_low_placement_preference_binding_t* binding =
          &bindings[use->binding_start + slot];
      if (binding->representative != slot) {
        continue;
      }
      const loom_value_ordinal_t origin =
          origins ? origins[binding->value_ordinal] : binding->value_ordinal;
      use_indices[--offsets[origin + 1]] = i - 1;
      last_bindings[origin] += use->preference->value_count;
      max_incident_binding_count =
          iree_max(max_incident_binding_count, last_bindings[origin]);
    }
  }
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    offsets[i] = offsets[i + 1];
  }
  offsets[value_count] = incidence_count;
  out_index->use_indices = use_indices;
  out_index->offsets_by_origin = offsets;
  out_index->max_incident_use_count = max_incident_use_count;
  out_index->max_incident_binding_count = max_incident_binding_count;
  out_index->max_memo_entry_count = max_memo_entry_count;
  return iree_ok_status();
}

// Resolves the identity forest and compresses every visited link. Units that
// never participate in an edge or available-copy query need no canonical root.
static uint32_t loom_low_placement_unit_origin(uint32_t* origins,
                                               uint32_t unit) {
  uint32_t origin = unit;
  while (origins[origin] != origin) {
    origin = origins[origin];
  }
  while (origins[unit] != unit) {
    const uint32_t next = origins[unit];
    origins[unit] = origin;
    unit = next;
  }
  return origin;
}

// Inverts the value-prefix index only for a candidate provider. The per-unit
// availability map stores flat offsets instead of another owner/offset pair.
static loom_value_ordinal_t loom_low_placement_unit_value(
    const uint32_t* starts, loom_value_ordinal_t value_count, uint32_t unit) {
  loom_value_ordinal_t first = 0;
  loom_value_ordinal_t count = value_count;
  while (count != 0) {
    const loom_value_ordinal_t half = count / 2;
    const loom_value_ordinal_t middle = first + half;
    if (starts[middle] <= unit) {
      first = middle + 1;
      count -= half + 1;
    } else {
      count = half;
    }
  }
  return first - 1;
}

// Storage equality and bit identity are different facts: a tied result writes
// new bits, while slice/concat/copy paths forward existing bits. Compose those
// paths once, then retain only edge identity and selected transport sources.
// No per-unit identity or availability table survives construction.
static iree_status_t loom_low_placement_select_copy_sources(
    loom_low_placement_build_state_t* state) {
  if (state->edge_relation_count == 0 ||
      !iree_any_bit_set(state->storage.flags,
                        LOOM_LOW_PLACEMENT_STORAGE_FLAG_OPTIONAL_ALIASES |
                            LOOM_LOW_PLACEMENT_STORAGE_FLAG_IDENTITY_ALIASES)) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  const bool select_sources =
      state->target_constraints->fixed_value_count != 0 &&
      iree_any_bit_set(state->storage.flags,
                       LOOM_LOW_PLACEMENT_STORAGE_FLAG_CONCAT);
  const bool track_writes =
      select_sources && state->storage.write_relation_count != 0;
  // Value offsets and optional write markers share one construction lifetime.
  uint32_t* starts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, (iree_host_size_t)value_count + 1,
      sizeof(*starts) + (track_writes ? sizeof(uint8_t) : 0), (void**)&starts));
  uint8_t* written_origins =
      track_writes ? (uint8_t*)(starts + value_count + 1) : NULL;
  if (written_origins != NULL) {
    memset(written_origins, 0, value_count);
  }
  uint64_t total_unit_count = 0;
  for (loom_value_ordinal_t v = 0; v < value_count; ++v) {
    starts[v] = (uint32_t)total_unit_count;
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(state->liveness, v);
    total_unit_count += interval != NULL ? interval->unit_count : 0;
    if (total_unit_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "placement storage units exceed u32 range");
    }
  }
  const uint32_t unit_count = (uint32_t)total_unit_count;
  starts[value_count] = unit_count;
  // Roots and optional available providers occupy disjoint halves of one
  // allocation. Identity-only construction needs just the first half.
  uint32_t* origins = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, unit_count,
      sizeof(*origins) * (select_sources ? 2u : 1u), (void**)&origins));
  for (uint32_t u = 0; u < unit_count; ++u) {
    origins[u] = u;
  }
  bool has_fixed_destination = false;
  for (const loom_low_placement_relation_chunk_t* chunk =
           state->collected_relations.head;
       chunk != NULL; chunk = chunk->next) {
    const uint32_t count = chunk->next != NULL
                               ? LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK
                               : 1 + (state->relation_count - 1) %
                                         LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK;
    for (uint32_t i = 0; i < count; ++i) {
      const loom_low_placement_relation_t* relation = &chunk->values[i];
      if (select_sources &&
          loom_low_placement_relation_is_edge_payload(relation)) {
        has_fixed_destination |=
            state->target_constraints
                ->fixed_value_indices_by_ordinal[relation->result_ordinal] != 0;
        const uint32_t destination = relation->result_ordinal + 1;
        uint32_t* selected =
            &state->ranges_by_source_ordinal[relation->source_ordinal].start;
        if (*selected == 0) {
          *selected = destination;
        } else if (*selected != destination) {
          *selected = UINT32_MAX;
        }
      }
      if (written_origins != NULL &&
          iree_any_bit_set(relation->flags,
                           LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE)) {
        written_origins[state->tied_storage_origins_by_value_ordinal
                            [relation->source_ordinal]] = 1;
      }
      if (relation->cause < LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT ||
          relation->cause > LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
          !loom_low_placement_relation_can_alias(relation) ||
          iree_any_bit_set(relation->flags,
                           LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE)) {
        continue;
      }
      for (uint32_t u = 0; u < relation->unit_count; ++u) {
        origins[starts[relation->result_ordinal] +
                relation->result_unit_offset + u] =
            starts[relation->source_ordinal] + relation->source_unit_offset + u;
      }
    }
  }
  uint32_t* available = has_fixed_destination ? origins + unit_count : NULL;
  if (available != NULL) {
    memset(available, 0xFF, unit_count * sizeof(*available));
  }
  for (loom_low_placement_relation_chunk_t* chunk =
           state->collected_relations.head;
       chunk != NULL; chunk = chunk->next) {
    const uint32_t count = chunk->next != NULL
                               ? LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK
                               : 1 + (state->relation_count - 1) %
                                         LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK;
    for (uint32_t i = 0; i < count; ++i) {
      loom_low_placement_relation_t* relation = &chunk->values[i];
      const uint32_t source_start =
          starts[relation->source_ordinal] + relation->source_unit_offset;
      const uint32_t result_start =
          starts[relation->result_ordinal] + relation->result_unit_offset;
      if (loom_low_placement_relation_is_edge_payload(relation)) {
        bool identity = true;
        for (uint32_t u = 0; u < relation->unit_count && identity; ++u) {
          identity =
              loom_low_placement_unit_origin(origins, result_start + u) ==
              loom_low_placement_unit_origin(origins, source_start + u);
        }
        if (identity) {
          relation->flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_IDENTITY_EDGE;
        }
      }
      if (available == NULL || relation->unit_count == 0 ||
          relation->cause < LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY ||
          relation->cause > LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
          !loom_low_placement_relation_can_alias(relation)) {
        continue;
      }
      const bool materialized =
          relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
          iree_any_bit_set(relation->flags,
                           LOOM_LOW_PLACEMENT_RELATION_FLAG_MATERIALIZE_PART);
      if (materialized) {
        const loom_value_ordinal_t origin =
            state->tied_storage_origins_by_value_ordinal != NULL
                ? state->tied_storage_origins_by_value_ordinal
                      [relation->result_ordinal]
                : relation->result_ordinal;
        if (written_origins != NULL && written_origins[origin]) {
          continue;
        }
        for (uint32_t u = 0; u < relation->unit_count; ++u) {
          available[loom_low_placement_unit_origin(origins, result_start + u)] =
              result_start + u;
        }
        continue;
      }
      const uint32_t destination =
          state->ranges_by_source_ordinal[relation->result_ordinal].start;
      if (destination == 0 || destination == UINT32_MAX) {
        continue;
      }
      const uint32_t* fixed_indices =
          state->target_constraints->fixed_value_indices_by_ordinal;
      const uint32_t source_index = fixed_indices[relation->source_ordinal];
      const uint32_t destination_index = fixed_indices[destination - 1];
      if (source_index == 0 || destination_index == 0) {
        continue;
      }
      const loom_low_allocation_resolved_fixed_value_t* fixed_source =
          &state->target_constraints->fixed_values[source_index - 1];
      const loom_low_allocation_resolved_fixed_value_t* fixed_destination =
          &state->target_constraints->fixed_values[destination_index - 1];
      // An already-aligned unit needs no transfer. Keep the authored component
      // unless every unit moves, including units in non-linear physical views.
      bool requires_moves = true;
      for (uint32_t u = 0; u < relation->unit_count && requires_moves; ++u) {
        const loom_low_move_location_t source =
            loom_low_allocation_assignment_unit_location(
                state->descriptor_set, &fixed_source->assignment,
                relation->source_unit_offset + u);
        const loom_low_move_location_t target =
            loom_low_allocation_assignment_unit_location(
                state->descriptor_set, &fixed_destination->assignment,
                relation->result_unit_offset + u);
        requires_moves =
            !loom_low_allocation_unit_locations_equal(&source, &target);
      }
      if (!requires_moves) {
        continue;
      }
      const loom_liveness_interval_t* source_interval =
          loom_low_placement_interval_for_ordinal(state,
                                                  relation->source_ordinal);
      // Selecting another provider must retire this original component's
      // transport demand, not extend two independently observable lifetimes.
      if (source_interval->end_point != relation->write_point) {
        continue;
      }
      const uint32_t saved_start =
          available[loom_low_placement_unit_origin(origins, source_start)];
      if (saved_start == UINT32_MAX) {
        continue;
      }
      const loom_value_ordinal_t saved_value =
          loom_low_placement_unit_value(starts, value_count, saved_start);
      const loom_liveness_interval_t* saved_interval =
          loom_low_placement_interval_for_ordinal(state, saved_value);
      const loom_op_t* saved_op = loom_value_def_op(loom_module_value(
          state->module, state->liveness->value_ids[saved_value]));
      if (saved_value == relation->source_ordinal ||
          saved_op->parent_block != relation->op->parent_block ||
          saved_interval->definition_point <=
              source_interval->definition_point ||
          saved_interval->value_class.register_class_id !=
              source_interval->value_class.register_class_id ||
          relation->unit_count > starts[saved_value + 1] - saved_start) {
        continue;
      }
      bool contiguous = true;
      for (uint32_t u = 1; u < relation->unit_count && contiguous; ++u) {
        contiguous = available[loom_low_placement_unit_origin(
                         origins, source_start + u)] == saved_start + u;
      }
      if (!contiguous) {
        continue;
      }
      relation->source_ordinal = saved_value;
      relation->source_unit_offset = saved_start - starts[saved_value];
      relation->source_operand_index = LOOM_LOW_PLACEMENT_SOURCE_OPERAND_NONE;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_placement_build(
    loom_low_placement_build_state_t* state,
    loom_low_placement_table_t* out_table,
    loom_low_placement_preference_index_t* out_preferences) {
  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  if (value_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count, sizeof(*state->ranges_by_result_ordinal),
        (void**)&state->ranges_by_result_ordinal));
    memset(state->ranges_by_result_ordinal, 0,
           value_count * sizeof(*state->ranges_by_result_ordinal));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count, sizeof(*state->ranges_by_source_ordinal),
        (void**)&state->ranges_by_source_ordinal));
    memset(state->ranges_by_source_ordinal, 0,
           value_count * sizeof(*state->ranges_by_source_ordinal));
  }

  IREE_RETURN_IF_ERROR(loom_low_placement_visit_ops(state));
  state->preferences.instruction_use_count =
      (uint32_t)state->preferences.use_count;
  for (iree_host_size_t i = 0; i < state->pair_uses.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_placement_collect_pair_relations(
        state, &state->pair_uses.values[i]));
  }

  const iree_host_size_t relation_count = state->relation_count;
  if (relation_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, relation_count,
                                                   sizeof(*state->relations),
                                                   (void**)&state->relations));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, relation_count,
        sizeof(*state->relation_indices_by_source_ordinal),
        (void**)&state->relation_indices_by_source_ordinal));
  }
  if (state->edge_relation_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, state->edge_relation_count,
                                  sizeof(*state->edge_relation_indices),
                                  (void**)&state->edge_relation_indices));
  }
  if (state->storage.write_relation_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->storage.write_relation_count,
        sizeof(*state->storage.write_relation_indices),
        (void**)&state->storage.write_relation_indices));
  }
  // Every exact tied component uses one base. Retain its strongest packet
  // requirement once, before fixed-input validation or allocation can place
  // any member. Optional copy and slice relations do not constrain the source.
  if (state->operand_constraints_by_interval != NULL &&
      state->tied_storage_origins_by_value_ordinal != NULL) {
    const uint32_t* interval_indices = state->liveness->value_interval_indices;
    for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
      const uint32_t interval_index = interval_indices[i];
      if (interval_index == UINT32_MAX) {
        continue;
      }
      const uint32_t origin_index =
          interval_indices[state->tied_storage_origins_by_value_ordinal[i]];
      loom_low_placement_operand_constraints_t* origin =
          &state->operand_constraints_by_interval[origin_index];
      const loom_low_placement_operand_constraints_t* member =
          &state->operand_constraints_by_interval[interval_index];
      origin->unit_alignment_log2 =
          iree_max(origin->unit_alignment_log2, member->unit_alignment_log2);
      if (member->addressable_unit_count != 0 &&
          (origin->addressable_unit_count == 0 ||
           member->addressable_unit_count < origin->addressable_unit_count)) {
        origin->addressable_unit_count = member->addressable_unit_count;
      }
      origin->has_target_address_state |= member->has_target_address_state;
    }
    for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
      const uint32_t interval_index = interval_indices[i];
      if (interval_index == UINT32_MAX) {
        continue;
      }
      const uint32_t origin_index =
          interval_indices[state->tied_storage_origins_by_value_ordinal[i]];
      state->operand_constraints_by_interval[interval_index] =
          state->operand_constraints_by_interval[origin_index];
    }
  }

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_target_constraints_resolve_fixed_locations(
          state->target_constraints, state->liveness, state->value_domain,
          state->operand_constraints_by_interval,
          state->tied_storage_origins_by_value_ordinal, state->fixed_values,
          state->fixed_value_count, state->arena));
  if (state->target_constraints->error_count != 0) {
    return iree_ok_status();
  }
  const iree_arena_checkpoint_t identity_checkpoint =
      iree_arena_checkpoint_save(state->scratch_arena);
  iree_status_t identity_status = loom_low_placement_select_copy_sources(state);
  iree_arena_checkpoint_restore(&identity_checkpoint);
  IREE_RETURN_IF_ERROR(identity_status);
  loom_low_placement_prefix_ranges(state);
  // The prefixed ranges partition both output arrays. Each collected row
  // initializes its complete record and source index at its reserved slot.
  for (const loom_low_placement_relation_chunk_t* chunk =
           state->collected_relations.head;
       chunk != NULL; chunk = chunk->next) {
    const uint32_t count = chunk->next != NULL
                               ? LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK
                               : 1 + (state->relation_count - 1) %
                                         LOOM_LOW_PLACEMENT_RELATIONS_PER_CHUNK;
    for (uint32_t i = 0; i < count; ++i) {
      loom_low_placement_append_relation(state, &chunk->values[i]);
    }
  }
  IREE_ASSERT_EQ(state->appended_relation_count, relation_count);
  IREE_ASSERT_EQ(state->appended_source_relation_count, relation_count);
  IREE_ASSERT_EQ(state->appended_edge_relation_count,
                 state->edge_relation_count);
  IREE_RETURN_IF_ERROR(loom_low_placement_build_storage_value_order(state));
  IREE_RETURN_IF_ERROR(
      loom_low_placement_index_preferences(state, out_preferences));

  *out_table = (loom_low_placement_table_t){
      .module = state->module,
      .region = state->region,
      .value_ids = state->liveness->value_ids,
      .value_count = (loom_value_ordinal_t)state->liveness->value_count,
      .relations = state->relations,
      .relation_count = relation_count,
      .edge_relation_indices = state->edge_relation_indices,
      .edge_relation_count = state->edge_relation_count,
      .storage =
          {
              .write_relation_indices = state->storage.write_relation_indices,
              .write_relation_count = state->storage.write_relation_count,
              .flags = state->storage.flags,
          },
      .location_relation_count = state->location_relation_count,
      .hard_location_relation_count = state->hard_location_relation_count,
      .packet_move_group_count = state->packet_move_group_count,
      .packet_move_unit_count = state->packet_move_unit_count,
      .edge_copy_group_count = state->edge_copy_group_count,
      .branch_unit_count = state->branch_unit_count,
      .max_move_group_unit_count = state->max_move_group_unit_count,
      .ranges_by_result_ordinal = state->ranges_by_result_ordinal,
      .relation_indices_by_source_ordinal =
          state->relation_indices_by_source_ordinal,
      .ranges_by_source_ordinal = state->ranges_by_source_ordinal,
      .storage_value_order = state->storage_value_order,
      .storage_value_order_count = state->storage_value_order_count,
      .tied_storage_origins_by_value_ordinal =
          state->tied_storage_origins_by_value_ordinal,
      .operand_constraints_by_interval = state->operand_constraints_by_interval,
  };
  return loom_low_placement_captures_build(state->liveness, out_table,
                                           state->arena, state->scratch_arena);
}

iree_status_t loom_low_allocation_placement_build(
    loom_low_allocation_target_constraints_t* target_constraints,
    const loom_region_t* region, const loom_local_value_domain_t* value_domain,
    const loom_value_ordinal_t* storage_origins,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_fixed_value_t* fixed_values,
    iree_host_size_t fixed_value_count,
    loom_low_placement_pair_use_list_t pair_uses,
    loom_low_placement_instruction_preferences_t instruction_preferences,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* preference_arena,
    loom_low_placement_table_t* out_table,
    loom_low_placement_preference_index_t* out_preferences) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  *out_table = (loom_low_placement_table_t){0};
  *out_preferences = (loom_low_placement_preference_index_t){0};
  IREE_ASSERT(value_domain->value_count == liveness->value_count &&
                  value_domain->value_ids == liveness->value_ids,
              "low placement requires liveness over the same local value "
              "domain");

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  loom_low_placement_build_state_t state = {
      .module = value_domain->module,
      .target_constraints = target_constraints,
      .fixed_values = fixed_values,
      .fixed_value_count = fixed_value_count,
      .region = region,
      .descriptor_set = target_constraints->target->descriptor_set,
      .value_domain = value_domain,
      .tied_storage_origins_by_value_ordinal = storage_origins,
      .liveness = liveness,
      .pair_uses = pair_uses,
      .instruction_preferences = instruction_preferences,
      .preference_arena = preference_arena,
      .arena = arena,
      .scratch_arena = &scratch_arena,
  };
  const iree_status_t status =
      loom_low_placement_build(&state, out_table, out_preferences);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
