// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/pressure_alias.h"

#include <string.h>

#include "loom/codegen/low/schedule/pressure.h"

#define LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_SOURCE_OWNS_BIT (UINT32_C(1) << 31)
#define LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_EPOCH_MASK \
  (~LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_SOURCE_OWNS_BIT)

// Mutable ownership for one compact storage relation.
struct loom_low_schedule_pressure_alias_record_t {
  // Number of units covered by this alias relation.
  uint32_t unit_count;
  // Next alias relation for the same source value.
  uint32_t next_source_relation;
  // Current traversal epoch and source-ownership bit.
  uint32_t epoch_and_flags;
};

static_assert(sizeof(loom_low_schedule_pressure_alias_record_t) == 12,
              "dense pressure alias state must remain compact");

static bool loom_low_schedule_storage_relation_can_alias_pressure(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_storage_relation_t* relation,
    loom_value_ordinal_t result_ordinal, loom_value_ordinal_t source_ordinal) {
  if (relation->kind != LOOM_LOW_STORAGE_RELATION_SAME_STORAGE &&
      relation->kind != LOOM_LOW_STORAGE_RELATION_SUBRANGE &&
      relation->kind != LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART) {
    return false;
  }
  const loom_low_schedule_value_record_t* result =
      &state->values[result_ordinal];
  const loom_low_schedule_value_record_t* source =
      &state->values[source_ordinal];
  if (result->register_class_id == LOOM_LOW_REG_CLASS_NONE ||
      result->register_class_id != source->register_class_id) {
    return false;
  }
  IREE_ASSERT(relation->destination_unit_offset <= result->unit_count &&
                  relation->unit_count <=
                      result->unit_count - relation->destination_unit_offset,
              "verified storage relation destination units must fit result");
  IREE_ASSERT(relation->source_unit_offset <= source->unit_count &&
                  relation->unit_count <=
                      source->unit_count - relation->source_unit_offset,
              "verified storage relation source units must fit source");
  return relation->unit_count != 0;
}

static bool loom_low_schedule_pressure_alias_is_active(
    const loom_low_schedule_pressure_alias_state_t* alias_state,
    const loom_low_schedule_pressure_alias_record_t* record) {
  return (record->epoch_and_flags &
          LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_EPOCH_MASK) == alias_state->epoch;
}

static bool loom_low_schedule_pressure_alias_source_owns(
    const loom_low_schedule_pressure_alias_record_t* record) {
  return iree_any_bit_set(record->epoch_and_flags,
                          LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_SOURCE_OWNS_BIT);
}

static bool loom_low_schedule_value_is_live_with_full_pressure(
    const loom_low_schedule_value_record_t* value, uint32_t unit_count) {
  return iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE) &&
         value->live_unit_count == value->unit_count &&
         value->unit_count >= unit_count;
}

iree_status_t loom_low_schedule_pressure_alias_initialize(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_alias_state_t* out_alias_state) {
  *out_alias_state = (loom_low_schedule_pressure_alias_state_t){0};
  const iree_host_size_t relation_count =
      state->storage_relations.relation_count;
  if (relation_count == 0) return iree_ok_status();

  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, value_count, sizeof(*out_alias_state->source_heads),
      (void**)&out_alias_state->source_heads));
  memset(out_alias_state->source_heads, 0xFF,
         value_count * sizeof(*out_alias_state->source_heads));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, value_count, sizeof(*out_alias_state->source_ordinals),
      (void**)&out_alias_state->source_ordinals));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, value_count, sizeof(*out_alias_state->source_unit_counts),
      (void**)&out_alias_state->source_unit_counts));
  memset(out_alias_state->source_unit_counts, 0,
         value_count * sizeof(*out_alias_state->source_unit_counts));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, relation_count, sizeof(*out_alias_state->records),
      (void**)&out_alias_state->records));
  memset(out_alias_state->records, 0,
         relation_count * sizeof(*out_alias_state->records));
  out_alias_state->epoch = 1;
  return iree_ok_status();
}

void loom_low_schedule_pressure_alias_reset(
    loom_low_schedule_pressure_alias_state_t* alias_state) {
  if (alias_state->source_heads == NULL) return;
  for (iree_host_size_t i = 0; i < alias_state->source_count; ++i) {
    const loom_value_ordinal_t source_ordinal = alias_state->source_ordinals[i];
    alias_state->source_heads[source_ordinal] = LOOM_LOW_SCHEDULE_NODE_NONE;
    alias_state->source_unit_counts[source_ordinal] = 0;
  }
  alias_state->source_count = 0;
  IREE_ASSERT_LT(alias_state->epoch,
                 LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_EPOCH_MASK);
  ++alias_state->epoch;
}

static void loom_low_schedule_pressure_alias_activate(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_alias_state_t* alias_state,
    uint32_t relation_index,
    const loom_low_schedule_storage_relation_t* relation, uint32_t unit_count,
    uint32_t ownership_flags) {
  IREE_ASSERT_NE(unit_count, 0u);
  IREE_ASSERT_EQ(
      ownership_flags & ~LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_SOURCE_OWNS_BIT, 0u);
  loom_low_schedule_pressure_alias_record_t* record =
      &alias_state->records[relation_index];
  IREE_ASSERT(!loom_low_schedule_pressure_alias_is_active(alias_state, record));
  const loom_value_ordinal_t source_ordinal = relation->source_ordinal;
  if (alias_state->source_heads[source_ordinal] ==
      LOOM_LOW_SCHEDULE_NODE_NONE) {
    alias_state->source_ordinals[alias_state->source_count++] = source_ordinal;
  }
  *record = (loom_low_schedule_pressure_alias_record_t){
      .unit_count = unit_count,
      .next_source_relation = alias_state->source_heads[source_ordinal],
      .epoch_and_flags = alias_state->epoch | ownership_flags,
  };
  alias_state->source_heads[source_ordinal] = relation_index;
  IREE_ASSERT_LE(unit_count,
                 state->values[source_ordinal].unit_count -
                     alias_state->source_unit_counts[source_ordinal]);
  alias_state->source_unit_counts[source_ordinal] += unit_count;
  state->values[relation->destination_ordinal].flags |=
      LOOM_LOW_SCHEDULE_VALUE_FLAG_ACTIVE_PRESSURE_ALIAS;
}

void loom_low_schedule_pressure_alias_deactivate_result(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t result_ordinal) {
  loom_low_schedule_value_record_t* result = &state->values[result_ordinal];
  if (!iree_any_bit_set(result->flags,
                        LOOM_LOW_SCHEDULE_VALUE_FLAG_ACTIVE_PRESSURE_ALIAS)) {
    return;
  }
  const uint32_t producer_node = result->producer_node;
  if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE) return;
  loom_low_schedule_pressure_alias_state_t* alias_state =
      &pressure_state->storage_aliases;
  const uint32_t relation_begin =
      loom_low_schedule_storage_relation_index_begin(&state->storage_relations,
                                                     producer_node);
  const uint32_t relation_end = loom_low_schedule_storage_relation_index_end(
      &state->storage_relations, producer_node);
  for (uint32_t relation_index = relation_begin; relation_index < relation_end;
       ++relation_index) {
    const loom_low_schedule_storage_relation_t* relation =
        loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                    relation_index);
    if (relation->destination_ordinal != result_ordinal) continue;
    loom_low_schedule_pressure_alias_record_t* record =
        &alias_state->records[relation_index];
    if (!loom_low_schedule_pressure_alias_is_active(alias_state, record)) {
      continue;
    }
    uint32_t* source_unit_count =
        &alias_state->source_unit_counts[relation->source_ordinal];
    IREE_ASSERT_LE(record->unit_count, *source_unit_count);
    *source_unit_count -= record->unit_count;
    record->epoch_and_flags = 0;
  }
  result->flags &= ~LOOM_LOW_SCHEDULE_VALUE_FLAG_ACTIVE_PRESSURE_ALIAS;
}

uint32_t loom_low_schedule_pressure_alias_transfer_from_source(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal) {
  loom_low_schedule_pressure_alias_state_t* alias_state =
      &pressure_state->storage_aliases;
  if (alias_state->source_heads == NULL) return 0;
  uint32_t transfer_units = 0;
  for (uint32_t relation_index = alias_state->source_heads[source_ordinal];
       relation_index != LOOM_LOW_SCHEDULE_NODE_NONE;) {
    loom_low_schedule_pressure_alias_record_t* record =
        &alias_state->records[relation_index];
    const uint32_t next_relation_index = record->next_source_relation;
    if (loom_low_schedule_pressure_alias_is_active(alias_state, record) &&
        loom_low_schedule_pressure_alias_source_owns(record)) {
      const loom_low_schedule_storage_relation_t* relation =
          loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                      relation_index);
      IREE_ASSERT_EQ(relation->source_ordinal, source_ordinal);
      loom_low_schedule_value_record_t* result =
          &state->values[relation->destination_ordinal];
      if (iree_any_bit_set(result->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
        IREE_ASSERT(record->unit_count <= result->unit_count &&
                        result->live_unit_count <=
                            result->unit_count - record->unit_count,
                    "alias units must fit result pressure units");
        result->live_unit_count += record->unit_count;
        record->epoch_and_flags &=
            ~LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_SOURCE_OWNS_BIT;
        transfer_units += record->unit_count;
      }
    }
    relation_index = next_relation_index;
  }
  return transfer_units;
}

uint32_t loom_low_schedule_pressure_alias_transfer_to_source(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal) {
  loom_low_schedule_pressure_alias_state_t* alias_state =
      &pressure_state->storage_aliases;
  if (alias_state->source_heads == NULL) return 0;
  uint32_t transfer_units = 0;
  for (uint32_t relation_index = alias_state->source_heads[source_ordinal];
       relation_index != LOOM_LOW_SCHEDULE_NODE_NONE;) {
    loom_low_schedule_pressure_alias_record_t* record =
        &alias_state->records[relation_index];
    const uint32_t next_relation_index = record->next_source_relation;
    if (loom_low_schedule_pressure_alias_is_active(alias_state, record) &&
        !loom_low_schedule_pressure_alias_source_owns(record)) {
      const loom_low_schedule_storage_relation_t* relation =
          loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                      relation_index);
      IREE_ASSERT_EQ(relation->source_ordinal, source_ordinal);
      loom_low_schedule_value_record_t* result =
          &state->values[relation->destination_ordinal];
      if (iree_any_bit_set(result->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
        IREE_ASSERT_LE(record->unit_count, result->live_unit_count);
        result->live_unit_count -= record->unit_count;
        record->epoch_and_flags |=
            LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_SOURCE_OWNS_BIT;
        transfer_units += record->unit_count;
      }
    }
    relation_index = next_relation_index;
  }
  return transfer_units;
}

uint32_t loom_low_schedule_pressure_alias_append_source_baseline_result(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_block_t* block, loom_value_ordinal_t result_ordinal) {
  loom_low_schedule_pressure_alias_state_t* alias_state =
      &pressure_state->storage_aliases;
  if (alias_state->records == NULL) return 0;
  const uint32_t producer_node = state->values[result_ordinal].producer_node;
  if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
      state->nodes[producer_node].block != block ||
      state->nodes[producer_node].storage_relation_count == 0) {
    return 0;
  }
  const loom_low_schedule_value_record_t* result =
      &state->values[result_ordinal];
  uint32_t result_alias_units = 0;
  uint32_t source_owned_units = 0;
  const uint32_t relation_begin =
      loom_low_schedule_storage_relation_index_begin(&state->storage_relations,
                                                     producer_node);
  const uint32_t relation_end = loom_low_schedule_storage_relation_index_end(
      &state->storage_relations, producer_node);
  for (uint32_t relation_index = relation_begin; relation_index < relation_end;
       ++relation_index) {
    const loom_low_schedule_storage_relation_t* relation =
        loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                    relation_index);
    if (relation->destination_ordinal != result_ordinal ||
        !loom_low_schedule_storage_relation_can_alias_pressure(
            state, relation, result_ordinal, relation->source_ordinal)) {
      continue;
    }
    const loom_value_ordinal_t source_ordinal = relation->source_ordinal;
    const loom_low_schedule_value_record_t* source =
        &state->values[source_ordinal];
    const uint32_t claimed_source_units =
        alias_state->source_unit_counts[source_ordinal];
    IREE_ASSERT_LE(claimed_source_units, source->unit_count);
    if (claimed_source_units == source->unit_count) continue;
    const uint32_t aliasable_units = iree_min(
        relation->unit_count, source->unit_count - claimed_source_units);
    const bool source_owns =
        loom_low_schedule_value_is_live_with_full_pressure(source,
                                                           aliasable_units) &&
        !iree_any_bit_set(source->flags,
                          LOOM_LOW_SCHEDULE_VALUE_FLAG_ACTIVE_PRESSURE_ALIAS);
    loom_low_schedule_pressure_alias_activate(
        state, alias_state, relation_index, relation, aliasable_units,
        source_owns ? LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_SOURCE_OWNS_BIT : 0);
    IREE_ASSERT_LE(aliasable_units, result->unit_count - result_alias_units);
    result_alias_units += aliasable_units;
    if (source_owns) source_owned_units += aliasable_units;
  }
  return source_owned_units;
}

static bool loom_low_schedule_value_lives_after_scored_candidate(
    const loom_low_schedule_value_record_t* value,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal) {
  if (!iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
    return false;
  }
  const uint32_t candidate_use_count =
      pressure_state->candidate_operand_use_counts[value_ordinal];
  return value->remaining_use_count > candidate_use_count;
}

static bool loom_low_schedule_relation_source_is_available_after_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal) {
  const loom_low_schedule_value_record_t* source =
      &state->values[source_ordinal];
  return source->live_unit_count == source->unit_count &&
         loom_low_schedule_value_lives_after_scored_candidate(
             source, pressure_state, source_ordinal);
}

void loom_low_schedule_pressure_alias_note_candidate_result_releases(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t result_ordinal) {
  const loom_low_schedule_value_record_t* result =
      &state->values[result_ordinal];
  if (!iree_any_bit_set(result->flags,
                        LOOM_LOW_SCHEDULE_VALUE_FLAG_ACTIVE_PRESSURE_ALIAS)) {
    return;
  }
  loom_low_schedule_pressure_alias_state_t* alias_state =
      &pressure_state->storage_aliases;
  const uint32_t producer_node = result->producer_node;
  const uint32_t relation_begin =
      loom_low_schedule_storage_relation_index_begin(&state->storage_relations,
                                                     producer_node);
  const uint32_t relation_end = loom_low_schedule_storage_relation_index_end(
      &state->storage_relations, producer_node);
  for (uint32_t relation_index = relation_begin; relation_index < relation_end;
       ++relation_index) {
    const loom_low_schedule_storage_relation_t* relation =
        loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                    relation_index);
    if (relation->destination_ordinal != result_ordinal) continue;
    const loom_low_schedule_pressure_alias_record_t* record =
        &alias_state->records[relation_index];
    if (!loom_low_schedule_pressure_alias_is_active(alias_state, record)) {
      continue;
    }
    const loom_value_ordinal_t source_ordinal = relation->source_ordinal;
    uint32_t* released_units =
        &pressure_state->candidate_scratch_counts[source_ordinal];
    if (pressure_state->candidate_operand_use_counts[source_ordinal] == 0 &&
        *released_units == 0) {
      pressure_state->candidate_operand_ordinals
          [pressure_state->candidate_operand_count++] = source_ordinal;
    }
    IREE_ASSERT_LE(
        record->unit_count,
        alias_state->source_unit_counts[source_ordinal] - *released_units);
    *released_units += record->unit_count;
  }
}

static uint32_t loom_low_schedule_candidate_claimed_source_units(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal) {
  loom_low_schedule_pressure_alias_state_t* alias_state =
      &pressure_state->storage_aliases;
  loom_low_schedule_value_record_t* source = &state->values[source_ordinal];
  uint32_t* claimed_units =
      &pressure_state->candidate_scratch_counts[source_ordinal];
  if (!iree_any_bit_set(source->flags,
                        LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_ALIAS_CLAIM)) {
    const uint32_t active_units =
        alias_state->source_unit_counts[source_ordinal];
    IREE_ASSERT_LE(*claimed_units, active_units);
    *claimed_units = active_units - *claimed_units;
    source->flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_ALIAS_CLAIM;
  }
  return iree_min(*claimed_units, source->unit_count);
}

uint32_t loom_low_schedule_pressure_alias_candidate_result_units(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    loom_value_ordinal_t result_ordinal) {
  if (state->nodes[node_index].storage_relation_count == 0) return 0;
  const loom_low_schedule_value_record_t* result =
      &state->values[result_ordinal];
  uint32_t alias_units = 0;
  const uint32_t relation_begin =
      loom_low_schedule_storage_relation_index_begin(&state->storage_relations,
                                                     node_index);
  const uint32_t relation_end = loom_low_schedule_storage_relation_index_end(
      &state->storage_relations, node_index);
  for (uint32_t relation_index = relation_begin; relation_index < relation_end;
       ++relation_index) {
    const loom_low_schedule_storage_relation_t* relation =
        loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                    relation_index);
    if (relation->destination_ordinal != result_ordinal) continue;
    const loom_value_ordinal_t source_ordinal = relation->source_ordinal;
    if (!loom_low_schedule_storage_relation_can_alias_pressure(
            state, relation, result_ordinal, source_ordinal) ||
        !loom_low_schedule_relation_source_is_available_after_candidate(
            state, pressure_state, source_ordinal)) {
      continue;
    }
    const loom_low_schedule_value_record_t* source =
        &state->values[source_ordinal];
    const uint32_t claimed_units =
        loom_low_schedule_candidate_claimed_source_units(state, pressure_state,
                                                         source_ordinal);
    if (claimed_units >= source->unit_count) continue;
    const uint32_t relation_alias_units =
        iree_min(relation->unit_count, source->unit_count - claimed_units);
    pressure_state->candidate_scratch_counts[source_ordinal] +=
        relation_alias_units;
    IREE_ASSERT_LE(relation_alias_units, result->unit_count - alias_units);
    alias_units += relation_alias_units;
  }
  return alias_units;
}

uint32_t loom_low_schedule_pressure_alias_candidate_transfer_from_source(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal) {
  const loom_low_schedule_pressure_alias_state_t* alias_state =
      &pressure_state->storage_aliases;
  if (alias_state->source_heads == NULL) return 0;
  uint32_t transfer_units = 0;
  for (uint32_t relation_index = alias_state->source_heads[source_ordinal];
       relation_index != LOOM_LOW_SCHEDULE_NODE_NONE;) {
    const loom_low_schedule_pressure_alias_record_t* record =
        &alias_state->records[relation_index];
    const uint32_t next_relation_index = record->next_source_relation;
    if (loom_low_schedule_pressure_alias_is_active(alias_state, record) &&
        loom_low_schedule_pressure_alias_source_owns(record)) {
      const loom_low_schedule_storage_relation_t* relation =
          loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                      relation_index);
      IREE_ASSERT_EQ(relation->source_ordinal, source_ordinal);
      const loom_low_schedule_value_record_t* result =
          &state->values[relation->destination_ordinal];
      if (loom_low_schedule_value_lives_after_scored_candidate(
              result, pressure_state, relation->destination_ordinal)) {
        IREE_ASSERT(record->unit_count <= result->unit_count &&
                        result->live_unit_count <=
                            result->unit_count - record->unit_count,
                    "alias units must fit result pressure units");
        transfer_units += record->unit_count;
      }
    }
    relation_index = next_relation_index;
  }
  return transfer_units;
}

uint32_t loom_low_schedule_pressure_alias_append_scheduled_result(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    loom_value_ordinal_t result_ordinal) {
  loom_low_schedule_pressure_alias_state_t* alias_state =
      &pressure_state->storage_aliases;
  if (alias_state->source_heads == NULL ||
      state->nodes[node_index].storage_relation_count == 0) {
    return 0;
  }
  const loom_low_schedule_value_record_t* result =
      &state->values[result_ordinal];
  uint32_t alias_units = 0;
  const uint32_t relation_begin =
      loom_low_schedule_storage_relation_index_begin(&state->storage_relations,
                                                     node_index);
  const uint32_t relation_end = loom_low_schedule_storage_relation_index_end(
      &state->storage_relations, node_index);
  for (uint32_t relation_index = relation_begin; relation_index < relation_end;
       ++relation_index) {
    const loom_low_schedule_storage_relation_t* relation =
        loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                    relation_index);
    if (relation->destination_ordinal != result_ordinal) continue;
    const loom_value_ordinal_t source_ordinal = relation->source_ordinal;
    if (!loom_low_schedule_storage_relation_can_alias_pressure(
            state, relation, result_ordinal, source_ordinal)) {
      continue;
    }
    const loom_low_schedule_value_record_t* source =
        &state->values[source_ordinal];
    if (!loom_low_schedule_value_is_live_with_full_pressure(
            source, relation->unit_count)) {
      continue;
    }
    const uint32_t claimed_source_units =
        alias_state->source_unit_counts[source_ordinal];
    IREE_ASSERT_LE(claimed_source_units, source->unit_count);
    if (claimed_source_units == source->unit_count) continue;
    const uint32_t aliasable_units = iree_min(
        relation->unit_count, source->unit_count - claimed_source_units);
    loom_low_schedule_pressure_alias_activate(
        state, alias_state, relation_index, relation, aliasable_units,
        LOOM_LOW_SCHEDULE_PRESSURE_ALIAS_SOURCE_OWNS_BIT);
    IREE_ASSERT_LE(aliasable_units, result->unit_count - alias_units);
    alias_units += aliasable_units;
  }
  return alias_units;
}
