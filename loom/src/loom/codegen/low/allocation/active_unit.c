// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/active_unit.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"

enum {
  // Tiny functions are cheaper to scan linearly than to index.
  LOOM_LOW_ALLOCATION_ACTIVE_UNIT_INDEX_MIN_CAPACITY = 32,
};

struct loom_low_allocation_active_unit_entry_t {
  // Assignment occupying this active unit.
  uint32_t assignment_index;
  // Next entry in the hashed unit bucket.
  uint32_t next_entry;
  // Previous entry in the hashed unit bucket.
  uint32_t previous_entry;
  // Hash bucket containing this entry.
  uint32_t bucket_index;
  // Target-storage identity key for the register class owning this unit.
  uint32_t storage_key;
  // Target-visible storage kind for this unit.
  loom_low_allocation_location_kind_t location_kind;
  // Physical register or target ID for this unit.
  uint32_t location;
};

static uint32_t loom_low_allocation_round_up_to_power_of_two_u32(
    uint32_t value) {
  if (value <= 1) {
    return 1;
  }
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value == UINT32_MAX ? 0 : value + 1u;
}

static uint32_t loom_low_allocation_active_unit_hash(
    loom_low_allocation_location_kind_t location_kind, uint32_t storage_key,
    uint32_t location) {
  uint32_t hash = location ^ ((uint32_t)location_kind * 0x9E3779B9u);
  hash ^= storage_key * 0x7F4A7C15u;
  hash ^= hash >> 16;
  hash *= 0x85EBCA6Bu;
  hash ^= hash >> 13;
  hash *= 0xC2B2AE35u;
  hash ^= hash >> 16;
  return hash;
}

static uint32_t loom_low_allocation_active_unit_bucket_index(
    const loom_low_allocation_active_unit_index_t* index,
    loom_low_allocation_location_kind_t location_kind, uint32_t storage_key,
    uint32_t location) {
  IREE_ASSERT(loom_low_allocation_active_unit_index_is_enabled(index));
  return loom_low_allocation_active_unit_hash(location_kind, storage_key,
                                              location) &
         (index->bucket_count - 1u);
}

static uint32_t loom_low_allocation_active_unit_next_seen_generation(
    loom_low_allocation_active_unit_index_t* index) {
  if (index->seen_generations_by_assignment_index == NULL) {
    return 0;
  }
  if (index->seen_generation == UINT32_MAX) {
    memset(index->seen_generations_by_assignment_index, 0,
           index->assignment_capacity *
               sizeof(*index->seen_generations_by_assignment_index));
    index->seen_generation = 0;
  }
  return ++index->seen_generation;
}

static bool loom_low_allocation_active_unit_mark_assignment_seen(
    loom_low_allocation_active_unit_index_t* index, uint32_t assignment_index,
    uint32_t generation) {
  if (generation == 0) {
    return false;
  }
  IREE_ASSERT_LT(assignment_index, index->assignment_capacity);
  uint32_t* seen_generations = index->seen_generations_by_assignment_index;
  uint32_t* assignment_generation = &seen_generations[assignment_index];
  if (*assignment_generation == generation) {
    return true;
  }
  *assignment_generation = generation;
  return false;
}

static bool loom_low_allocation_value_id_is_ignored(
    loom_value_id_t value_id, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count) {
  for (uint16_t i = 0; i < ignored_value_count; ++i) {
    if (ignored_value_ids[i] == value_id) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_active_assignment_conflicts(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* existing,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  if (loom_low_allocation_value_id_is_ignored(
          existing->value_id, ignored_value_ids, ignored_value_count)) {
    return false;
  }
  if (existing->location_kind != candidate->location_kind) {
    return false;
  }
  return loom_low_allocation_live_range_assignments_conflict(
      descriptor_set, liveness, unit_end_points, unit_end_point_count, existing,
      candidate);
}

static bool loom_low_allocation_active_unit_index_can_insert_assignment(
    const loom_low_allocation_active_unit_index_t* index,
    const loom_low_allocation_assignment_t* assignment) {
  if (!loom_low_allocation_active_unit_index_is_enabled(index) ||
      !loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind) ||
      assignment->location_count == 0 ||
      assignment->location_base > UINT32_MAX - assignment->location_count) {
    return false;
  }
  return assignment->location_count <=
         index->entry_capacity - index->entry_count;
}

iree_status_t loom_low_allocation_active_unit_index_initialize(
    iree_host_size_t assignment_capacity, iree_host_size_t unit_capacity,
    iree_arena_allocator_t* arena,
    loom_low_allocation_active_unit_index_t* out_index) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_index);
  *out_index = (loom_low_allocation_active_unit_index_t){0};
  if (unit_capacity < LOOM_LOW_ALLOCATION_ACTIVE_UNIT_INDEX_MIN_CAPACITY ||
      unit_capacity > UINT32_MAX / 2u) {
    return iree_ok_status();
  }

  const uint32_t bucket_count =
      loom_low_allocation_round_up_to_power_of_two_u32((uint32_t)unit_capacity *
                                                       2u);
  if (bucket_count == 0) {
    return iree_ok_status();
  }

  out_index->bucket_count = bucket_count;
  out_index->entry_capacity = unit_capacity;
  out_index->assignment_capacity = assignment_capacity;

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, bucket_count, sizeof(*out_index->bucket_heads),
      (void**)&out_index->bucket_heads));
  for (uint32_t i = 0; i < bucket_count; ++i) {
    out_index->bucket_heads[i] = UINT32_MAX;
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, unit_capacity,
                                                 sizeof(*out_index->entries),
                                                 (void**)&out_index->entries));
  memset(out_index->entries, 0, unit_capacity * sizeof(*out_index->entries));

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assignment_capacity,
      sizeof(*out_index->entry_starts_by_assignment_index),
      (void**)&out_index->entry_starts_by_assignment_index));
  for (iree_host_size_t i = 0; i < assignment_capacity; ++i) {
    out_index->entry_starts_by_assignment_index[i] = UINT32_MAX;
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assignment_capacity,
      sizeof(*out_index->seen_generations_by_assignment_index),
      (void**)&out_index->seen_generations_by_assignment_index));
  memset(out_index->seen_generations_by_assignment_index, 0,
         assignment_capacity *
             sizeof(*out_index->seen_generations_by_assignment_index));
  return iree_ok_status();
}

bool loom_low_allocation_active_unit_index_is_enabled(
    const loom_low_allocation_active_unit_index_t* index) {
  IREE_ASSERT_ARGUMENT(index);
  return index->bucket_heads != NULL && index->bucket_count != 0 &&
         index->entries != NULL &&
         index->entry_starts_by_assignment_index != NULL;
}

bool loom_low_allocation_active_unit_index_contains_assignment(
    const loom_low_allocation_active_unit_index_t* index,
    uint32_t assignment_index) {
  IREE_ASSERT_ARGUMENT(index);
  if (!loom_low_allocation_active_unit_index_is_enabled(index) ||
      assignment_index >= index->assignment_capacity) {
    return false;
  }
  return index->entry_starts_by_assignment_index[assignment_index] !=
         UINT32_MAX;
}

iree_host_size_t loom_low_allocation_active_unit_index_unindexed_count(
    const loom_low_allocation_active_unit_index_t* index) {
  IREE_ASSERT_ARGUMENT(index);
  return index->unindexed_count;
}

bool loom_low_allocation_active_unit_index_conflicts(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness, const uint32_t* unit_end_points,
    iree_host_size_t unit_end_point_count,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(liveness);
  IREE_ASSERT_ARGUMENT(assignments);
  IREE_ASSERT_ARGUMENT(candidate);
  if (!loom_low_allocation_location_kind_is_register_like(
          candidate->location_kind)) {
    return false;
  }
  const uint32_t storage_key = loom_low_reg_class_storage_key(
      descriptor_set, candidate->descriptor_reg_class_id);
  const uint32_t generation =
      loom_low_allocation_active_unit_next_seen_generation(index);
  for (uint32_t unit_offset = 0; unit_offset < candidate->location_count;
       ++unit_offset) {
    if (candidate->location_base > UINT32_MAX - unit_offset) {
      break;
    }
    const uint32_t location = candidate->location_base + unit_offset;
    const uint32_t bucket_index = loom_low_allocation_active_unit_bucket_index(
        index, candidate->location_kind, storage_key, location);
    uint32_t entry_index = index->bucket_heads[bucket_index];
    while (entry_index != UINT32_MAX) {
      IREE_ASSERT_LT(entry_index, index->entry_count);
      const loom_low_allocation_active_unit_entry_t* entry =
          &index->entries[entry_index];
      const uint32_t assignment_index = entry->assignment_index;
      IREE_ASSERT_LT(assignment_index, assignment_count);
      const loom_low_allocation_assignment_t* existing =
          &assignments[assignment_index];
      if (entry->location_kind == candidate->location_kind &&
          entry->storage_key == storage_key && entry->location == location &&
          existing->end_point > candidate->start_point &&
          !loom_low_allocation_active_unit_mark_assignment_seen(
              index, assignment_index, generation) &&
          loom_low_allocation_active_assignment_conflicts(
              descriptor_set, liveness, unit_end_points, unit_end_point_count,
              existing, candidate, ignored_value_ids, ignored_value_count)) {
        return true;
      }
      entry_index = entry->next_entry;
    }
  }
  return false;
}

void loom_low_allocation_active_unit_index_insert_assignment(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(assignments);
  if (!loom_low_allocation_active_unit_index_is_enabled(index)) {
    return;
  }
  IREE_ASSERT_LT(assignment_index, assignment_count);
  IREE_ASSERT_LT(assignment_index, index->assignment_capacity);
  const loom_low_allocation_assignment_t* assignment =
      &assignments[assignment_index];
  if (!loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind)) {
    return;
  }
  if (!loom_low_allocation_active_unit_index_can_insert_assignment(
          index, assignment)) {
    ++index->unindexed_count;
    return;
  }

  const iree_host_size_t entry_start = index->entry_count;
  IREE_ASSERT(entry_start <= UINT32_MAX);
  index->entry_starts_by_assignment_index[assignment_index] =
      (uint32_t)entry_start;
  const uint32_t storage_key = loom_low_reg_class_storage_key(
      descriptor_set, assignment->descriptor_reg_class_id);
  for (uint32_t unit_offset = 0; unit_offset < assignment->location_count;
       ++unit_offset) {
    const uint32_t location = assignment->location_base + unit_offset;
    const uint32_t bucket_index = loom_low_allocation_active_unit_bucket_index(
        index, assignment->location_kind, storage_key, location);
    const uint32_t entry_index = (uint32_t)index->entry_count++;
    loom_low_allocation_active_unit_entry_t* entry =
        &index->entries[entry_index];
    *entry = (loom_low_allocation_active_unit_entry_t){
        .assignment_index = assignment_index,
        .next_entry = index->bucket_heads[bucket_index],
        .previous_entry = UINT32_MAX,
        .bucket_index = bucket_index,
        .storage_key = storage_key,
        .location_kind = assignment->location_kind,
        .location = location,
    };
    if (entry->next_entry != UINT32_MAX) {
      IREE_ASSERT_LT(entry->next_entry, index->entry_count);
      index->entries[entry->next_entry].previous_entry = entry_index;
    }
    index->bucket_heads[bucket_index] = entry_index;
  }
}

void loom_low_allocation_active_unit_index_remove_assignment(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(assignments);
  if (!loom_low_allocation_active_unit_index_is_enabled(index)) {
    return;
  }
  IREE_ASSERT_LT(assignment_index, assignment_count);
  IREE_ASSERT_LT(assignment_index, index->assignment_capacity);
  const loom_low_allocation_assignment_t* assignment =
      &assignments[assignment_index];
  if (!loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind)) {
    return;
  }
  const uint32_t entry_start =
      index->entry_starts_by_assignment_index[assignment_index];
  if (entry_start == UINT32_MAX) {
    IREE_ASSERT_GT(index->unindexed_count, 0);
    --index->unindexed_count;
    return;
  }
  for (uint32_t unit_offset = 0; unit_offset < assignment->location_count;
       ++unit_offset) {
    const uint32_t entry_index = entry_start + unit_offset;
    IREE_ASSERT_LT(entry_index, index->entry_count);
    loom_low_allocation_active_unit_entry_t* entry =
        &index->entries[entry_index];
    if (entry->previous_entry != UINT32_MAX) {
      IREE_ASSERT_LT(entry->previous_entry, index->entry_count);
      index->entries[entry->previous_entry].next_entry = entry->next_entry;
    } else {
      IREE_ASSERT_EQ(index->bucket_heads[entry->bucket_index], entry_index);
      index->bucket_heads[entry->bucket_index] = entry->next_entry;
    }
    if (entry->next_entry != UINT32_MAX) {
      IREE_ASSERT_LT(entry->next_entry, index->entry_count);
      index->entries[entry->next_entry].previous_entry = entry->previous_entry;
    }
    entry->next_entry = UINT32_MAX;
    entry->previous_entry = UINT32_MAX;
  }
  index->entry_starts_by_assignment_index[assignment_index] = UINT32_MAX;
}
