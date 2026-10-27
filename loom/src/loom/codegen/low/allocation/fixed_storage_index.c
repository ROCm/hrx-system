// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/fixed_storage_index.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_location.h"

struct loom_low_allocation_fixed_storage_record_t {
  // Descriptor-defined namespace shared by aliasing register classes.
  uint32_t storage_key;
  // Atomic location within the storage namespace.
  uint32_t location;
  // Inclusive exact storage-claim start point.
  uint32_t start_point;
  // Exclusive exact storage-claim end point.
  uint32_t end_point;
  // Maximum exclusive storage-claim end in this implicit subtree.
  uint32_t subtree_end_point;
  // Index into the resolved fixed-value array.
  uint32_t fixed_value_index;
};

// One location's ordered immutable record span and attempt-local cursor.
struct loom_low_allocation_fixed_location_entry_t {
  // Atomic location within the owning storage group.
  uint32_t location;
  // One past the last fixed-storage record for this location.
  uint32_t record_end;
  // First record whose end point exceeds the attempt's current start point.
  uint32_t current_record;
  // Start point of |current_record|, or UINT32_MAX when none remains.
  uint32_t next_conflict_start;
  // Maximum |next_conflict_start| in this implicit balanced subtree.
  uint32_t subtree_max_conflict_start;
};

// Contiguous location entries sharing one storage identity and location kind.
struct loom_low_allocation_fixed_location_group_t {
  // Descriptor-defined namespace shared by aliasing register classes.
  uint32_t storage_key;
  // First entry belonging to this group.
  uint32_t entry_start;
  // Number of entries belonging to this group.
  uint32_t entry_count;
  // Register-like location-kind ordinal used by the fixed record index.
  uint32_t kind_ordinal;
};

static uint32_t loom_low_allocation_fixed_storage_sort_word(
    const loom_low_allocation_fixed_storage_record_t* record,
    uint32_t word_ordinal) {
  switch (word_ordinal) {
    case 0:
      return record->start_point;
    case 1:
      return record->location;
    default:
      return record->storage_key;
  }
}

static bool loom_low_allocation_fixed_storage_sort_key_less(
    const loom_low_allocation_fixed_storage_record_t* lhs,
    const loom_low_allocation_fixed_storage_record_t* rhs) {
  if (lhs->storage_key != rhs->storage_key) {
    return lhs->storage_key < rhs->storage_key;
  }
  if (lhs->location != rhs->location) {
    return lhs->location < rhs->location;
  }
  return lhs->start_point < rhs->start_point;
}

static bool loom_low_allocation_fixed_storage_records_are_ordered(
    const loom_low_allocation_fixed_storage_record_t* records,
    uint32_t record_count) {
  for (uint32_t i = 1; i < record_count; ++i) {
    if (loom_low_allocation_fixed_storage_sort_key_less(&records[i],
                                                        &records[i - 1])) {
      return false;
    }
  }
  return true;
}

// Performs stable least-significant-digit passes over start, location, then
// storage key. Constant byte lanes do not affect order and are skipped after
// one linear inspection per word.
static void loom_low_allocation_fixed_storage_radix_sort(
    loom_low_allocation_fixed_storage_record_t* records,
    loom_low_allocation_fixed_storage_record_t* temporary,
    uint32_t record_count) {
  if (record_count <= 1) {
    return;
  }
  loom_low_allocation_fixed_storage_record_t* source = records;
  loom_low_allocation_fixed_storage_record_t* destination = temporary;
  for (uint32_t word_ordinal = 0; word_ordinal < 3; ++word_ordinal) {
    const uint32_t reference =
        loom_low_allocation_fixed_storage_sort_word(&source[0], word_ordinal);
    uint32_t varying_bits = 0;
    for (uint32_t i = 1; i < record_count; ++i) {
      varying_bits |= reference ^ loom_low_allocation_fixed_storage_sort_word(
                                      &source[i], word_ordinal);
    }
    for (uint32_t shift = 0; shift < 32; shift += 8) {
      if (((varying_bits >> shift) & 0xFFu) == 0) {
        continue;
      }
      uint32_t offsets[256] = {0};
      for (uint32_t i = 0; i < record_count; ++i) {
        ++offsets[(loom_low_allocation_fixed_storage_sort_word(&source[i],
                                                               word_ordinal) >>
                   shift) &
                  0xFFu];
      }
      uint32_t next_offset = 0;
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(offsets); ++i) {
        const uint32_t count = offsets[i];
        offsets[i] = next_offset;
        next_offset += count;
      }
      for (uint32_t i = 0; i < record_count; ++i) {
        destination[offsets[(loom_low_allocation_fixed_storage_sort_word(
                                 &source[i], word_ordinal) >>
                             shift) &
                            0xFFu]++] = source[i];
      }
      loom_low_allocation_fixed_storage_record_t* swap = source;
      source = destination;
      destination = swap;
    }
  }
  if (source != records) {
    memcpy(records, source, record_count * sizeof(*records));
  }
}

static uint32_t loom_low_allocation_fixed_storage_build_subtree(
    const loom_low_allocation_resolved_fixed_value_t* fixed_values,
    loom_low_allocation_fixed_storage_record_t* records,
    loom_value_ordinal_t* subtree_tied_roots, uint32_t begin, uint32_t end) {
  if (begin == end) {
    return 0;
  }
  const uint32_t middle = begin + (end - begin) / 2;
  const uint32_t left_end = loom_low_allocation_fixed_storage_build_subtree(
      fixed_values, records, subtree_tied_roots, begin, middle);
  const uint32_t right_end = loom_low_allocation_fixed_storage_build_subtree(
      fixed_values, records, subtree_tied_roots, middle + 1, end);
  records[middle].subtree_end_point =
      iree_max(records[middle].end_point, iree_max(left_end, right_end));
  if (subtree_tied_roots == NULL) {
    return records[middle].subtree_end_point;
  }
  loom_value_ordinal_t tied_root =
      fixed_values[records[middle].fixed_value_index].tied_root_ordinal;
  if (begin < middle) {
    const uint32_t left_middle = begin + (middle - begin) / 2;
    if (subtree_tied_roots[left_middle] != tied_root) {
      tied_root = LOOM_VALUE_ORDINAL_INVALID;
    }
  }
  if (middle + 1 < end) {
    const uint32_t right_middle = middle + 1 + (end - middle - 1) / 2;
    if (subtree_tied_roots[right_middle] != tied_root) {
      tied_root = LOOM_VALUE_ORDINAL_INVALID;
    }
  }
  subtree_tied_roots[middle] = tied_root;
  return records[middle].subtree_end_point;
}

static uint32_t loom_low_allocation_fixed_storage_kind_ordinal(
    loom_low_allocation_location_kind_t location_kind) {
  IREE_ASSERT(
      loom_low_allocation_location_kind_is_register_like(location_kind));
  return location_kind == LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ? 0
                                                                         : 1;
}

static loom_low_allocation_assignment_t
loom_low_allocation_fixed_storage_unit_assignment(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index) {
  const loom_low_move_location_t location =
      loom_low_allocation_assignment_unit_location(descriptor_set, assignment,
                                                   unit_index);
  return (loom_low_allocation_assignment_t){
      .descriptor_reg_class_id = location.descriptor_reg_class_id,
      .location_kind = location.location_kind,
      .location_base = location.location,
      .location_count = 1,
  };
}

typedef struct loom_low_allocation_fixed_storage_time_iterator_t {
  // Sparse assignment segments, or NULL for one continuous unit lifetime.
  const loom_liveness_segment_t* segments;
  // Number of sparse segments, or one for a continuous unit lifetime.
  uint32_t segment_count;
  // Next sparse segment or continuous lifetime to inspect.
  uint32_t segment_index;
  // Inclusive refined unit-lifetime start point.
  uint32_t unit_start_point;
  // Exclusive refined unit-lifetime end point.
  uint32_t unit_end_point;
} loom_low_allocation_fixed_storage_time_iterator_t;

static loom_low_allocation_fixed_storage_time_iterator_t
loom_low_allocation_fixed_storage_time_iterator(
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index) {
  const loom_liveness_segment_range_t segment_range =
      assignment->liveness_segments;
  return (loom_low_allocation_fixed_storage_time_iterator_t){
      .segments =
          segment_range.count == 0
              ? NULL
              : unit_liveness->storage_segments.entries + segment_range.start,
      .segment_count = segment_range.count == 0 ? 1 : segment_range.count,
      .unit_start_point =
          loom_low_allocation_live_range_assignment_unit_start_point(
              unit_liveness->start_points, unit_liveness->point_count,
              assignment, unit_index),
      .unit_end_point =
          loom_low_allocation_live_range_assignment_unit_end_point(
              unit_liveness->end_points, unit_liveness->point_count, assignment,
              unit_index),
  };
}

static bool loom_low_allocation_fixed_storage_time_iterator_next(
    loom_low_allocation_fixed_storage_time_iterator_t* iterator,
    uint32_t* out_start_point, uint32_t* out_end_point) {
  while (iterator->segment_index < iterator->segment_count) {
    uint32_t start_point = iterator->unit_start_point;
    uint32_t end_point = iterator->unit_end_point;
    if (iterator->segments != NULL) {
      const loom_liveness_segment_t* segment =
          &iterator->segments[iterator->segment_index];
      start_point = iree_max(start_point, segment->start_point);
      end_point = iree_min(end_point, segment->end_point);
    }
    ++iterator->segment_index;
    if (start_point < end_point) {
      *out_start_point = start_point;
      *out_end_point = end_point;
      return true;
    }
  }
  return false;
}

iree_status_t loom_low_allocation_fixed_storage_index_initialize(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena) {
  loom_low_allocation_fixed_storage_index_t* index = &constraints->fixed_index;
  *index = (loom_low_allocation_fixed_storage_index_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  iree_host_size_t record_counts[2] = {0, 0};
  for (iree_host_size_t i = 0; i < constraints->fixed_value_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &constraints->fixed_values[i].assignment;
    if (!loom_low_allocation_assignment_is_register_like(assignment)) {
      continue;
    }
    const uint32_t kind_ordinal =
        loom_low_allocation_fixed_storage_kind_ordinal(
            assignment->location_kind);
    for (uint32_t unit_index = 0; unit_index < assignment->location_count;
         ++unit_index) {
      const loom_low_allocation_assignment_t unit_assignment =
          loom_low_allocation_fixed_storage_unit_assignment(
              descriptor_set, assignment, unit_index);
      const uint32_t atomic_unit_count =
          loom_low_allocation_storage_assignment_atomic_unit_count(
              descriptor_set, &unit_assignment);
      iree_host_size_t temporal_claim_count = 0;
      loom_low_allocation_fixed_storage_time_iterator_t time_iterator =
          loom_low_allocation_fixed_storage_time_iterator(
              unit_liveness, assignment, unit_index);
      uint32_t claim_start_point = 0;
      uint32_t claim_end_point = 0;
      while (loom_low_allocation_fixed_storage_time_iterator_next(
          &time_iterator, &claim_start_point, &claim_end_point)) {
        ++temporal_claim_count;
      }
      iree_host_size_t unit_record_count = 0;
      if (!iree_host_size_checked_mul(temporal_claim_count, atomic_unit_count,
                                      &unit_record_count) ||
          !iree_host_size_checked_add(record_counts[kind_ordinal],
                                      unit_record_count,
                                      &record_counts[kind_ordinal])) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "fixed storage index record count exceeds host size");
      }
    }
  }

  iree_host_size_t record_count = 0;
  if (!iree_host_size_checked_add(record_counts[0], record_counts[1],
                                  &record_count) ||
      record_count > UINT32_MAX ||
      constraints->fixed_value_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "fixed storage index exceeds uint32_t range");
  }
  index->record_starts[1] = (uint32_t)record_counts[0];
  index->record_counts[0] = (uint32_t)record_counts[0];
  index->record_counts[1] = (uint32_t)record_counts[1];
  if (record_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, record_count, sizeof(*index->records), (void**)&index->records));
  for (uint32_t i = 0; i < constraints->fixed_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &constraints->fixed_values[i];
    if (fixed_value->tied_root_ordinal != fixed_value->value_ordinal) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, record_count, sizeof(*index->subtree_tied_roots),
          (void**)&index->subtree_tied_roots));
      break;
    }
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, constraints->fixed_value_count,
                                sizeof(*index->excluded_generations),
                                (void**)&index->excluded_generations));
  memset(index->excluded_generations, 0,
         constraints->fixed_value_count * sizeof(*index->excluded_generations));

  uint32_t record_cursors[2] = {0, index->record_starts[1]};
  for (uint32_t i = 0; i < constraints->fixed_value_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &constraints->fixed_values[i].assignment;
    if (!loom_low_allocation_assignment_is_register_like(assignment)) {
      continue;
    }
    const uint32_t kind_ordinal =
        loom_low_allocation_fixed_storage_kind_ordinal(
            assignment->location_kind);
    for (uint32_t unit_index = 0; unit_index < assignment->location_count;
         ++unit_index) {
      const loom_low_allocation_assignment_t unit_assignment =
          loom_low_allocation_fixed_storage_unit_assignment(
              descriptor_set, assignment, unit_index);
      const uint32_t atomic_unit_count =
          loom_low_allocation_storage_assignment_atomic_unit_count(
              descriptor_set, &unit_assignment);
      loom_low_allocation_fixed_storage_time_iterator_t time_iterator =
          loom_low_allocation_fixed_storage_time_iterator(
              unit_liveness, assignment, unit_index);
      uint32_t claim_start_point = 0;
      uint32_t claim_end_point = 0;
      while (loom_low_allocation_fixed_storage_time_iterator_next(
          &time_iterator, &claim_start_point, &claim_end_point)) {
        for (uint32_t atomic_unit = 0; atomic_unit < atomic_unit_count;
             ++atomic_unit) {
          loom_low_allocation_fixed_storage_record_t* record =
              &index->records[record_cursors[kind_ordinal]++];
          loom_low_allocation_storage_assignment_atomic_unit(
              descriptor_set, &unit_assignment, atomic_unit,
              &record->storage_key, &record->location);
          record->start_point = claim_start_point;
          record->end_point = claim_end_point;
          record->fixed_value_index = i;
        }
      }
    }
  }
  IREE_ASSERT_EQ(record_cursors[0], index->record_counts[0]);
  IREE_ASSERT_EQ(record_cursors[1], record_count);

  bool sort_required[2];
  uint32_t temporary_count = 0;
  for (uint32_t kind_ordinal = 0; kind_ordinal < 2; ++kind_ordinal) {
    const uint32_t start = index->record_starts[kind_ordinal];
    const uint32_t count = index->record_counts[kind_ordinal];
    sort_required[kind_ordinal] =
        !loom_low_allocation_fixed_storage_records_are_ordered(
            index->records + start, count);
    if (sort_required[kind_ordinal]) {
      temporary_count = iree_max(temporary_count, count);
    }
  }
  iree_arena_allocator_t build_arena;
  iree_arena_initialize(arena->block_pool, &build_arena);
  loom_low_allocation_fixed_storage_record_t* temporary = NULL;
  iree_status_t status = iree_ok_status();
  if (temporary_count != 0) {
    status = iree_arena_allocate_array(&build_arena, temporary_count,
                                       sizeof(*temporary), (void**)&temporary);
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t kind_ordinal = 0; kind_ordinal < 2; ++kind_ordinal) {
      const uint32_t start = index->record_starts[kind_ordinal];
      const uint32_t count = index->record_counts[kind_ordinal];
      if (sort_required[kind_ordinal]) {
        loom_low_allocation_fixed_storage_radix_sort(index->records + start,
                                                     temporary, count);
      }
      uint32_t group_start = start;
      const uint32_t end = start + count;
      while (group_start < end) {
        uint32_t group_end = group_start + 1;
        while (group_end < end &&
               index->records[group_end].storage_key ==
                   index->records[group_start].storage_key &&
               index->records[group_end].location ==
                   index->records[group_start].location) {
          ++group_end;
        }
        loom_low_allocation_fixed_storage_build_subtree(
            constraints->fixed_values, index->records,
            index->subtree_tied_roots, group_start, group_end);
        group_start = group_end;
      }
    }
  }
  iree_arena_deinitialize(&build_arena);
  return status;
}

static uint32_t loom_low_allocation_fixed_availability_build_subtree(
    loom_low_allocation_fixed_location_entry_t* entries, uint32_t begin,
    uint32_t end) {
  if (begin == end) {
    return 0;
  }
  const uint32_t middle = begin + (end - begin) / 2u;
  const uint32_t left_max =
      loom_low_allocation_fixed_availability_build_subtree(entries, begin,
                                                           middle);
  const uint32_t right_max =
      loom_low_allocation_fixed_availability_build_subtree(entries, middle + 1,
                                                           end);
  entries[middle].subtree_max_conflict_start = iree_max(
      entries[middle].next_conflict_start, iree_max(left_max, right_max));
  return entries[middle].subtree_max_conflict_start;
}

static uint32_t loom_low_allocation_fixed_availability_current_end(
    const loom_low_allocation_fixed_availability_t* availability,
    uint32_t entry_index) {
  const loom_low_allocation_fixed_location_entry_t* entry =
      &availability->entries[entry_index];
  IREE_ASSERT_LT(entry->current_record, entry->record_end);
  return availability->constraints->fixed_index.records[entry->current_record]
      .end_point;
}

static bool loom_low_allocation_fixed_availability_heap_less(
    const loom_low_allocation_fixed_availability_t* availability,
    uint32_t lhs_entry_index, uint32_t rhs_entry_index) {
  const uint32_t lhs_end = loom_low_allocation_fixed_availability_current_end(
      availability, lhs_entry_index);
  const uint32_t rhs_end = loom_low_allocation_fixed_availability_current_end(
      availability, rhs_entry_index);
  return lhs_end < rhs_end ||
         (lhs_end == rhs_end && lhs_entry_index < rhs_entry_index);
}

static void loom_low_allocation_fixed_availability_heap_sift_down(
    loom_low_allocation_fixed_availability_t* availability,
    uint32_t heap_index) {
  while (true) {
    const uint32_t left = heap_index * 2u + 1u;
    if (left >= availability->expiration_heap_count) {
      return;
    }
    const uint32_t right = left + 1u;
    uint32_t selected = left;
    if (right < availability->expiration_heap_count &&
        loom_low_allocation_fixed_availability_heap_less(
            availability, availability->expiration_heap[right],
            availability->expiration_heap[left])) {
      selected = right;
    }
    if (!loom_low_allocation_fixed_availability_heap_less(
            availability, availability->expiration_heap[selected],
            availability->expiration_heap[heap_index])) {
      return;
    }
    const uint32_t swap = availability->expiration_heap[heap_index];
    availability->expiration_heap[heap_index] =
        availability->expiration_heap[selected];
    availability->expiration_heap[selected] = swap;
    heap_index = selected;
  }
}

static uint32_t loom_low_allocation_fixed_availability_subtree_max(
    const loom_low_allocation_fixed_location_entry_t* entries, uint32_t begin,
    uint32_t end) {
  if (begin == end) {
    return 0;
  }
  return entries[begin + (end - begin) / 2u].subtree_max_conflict_start;
}

static uint32_t loom_low_allocation_fixed_availability_update_subtree(
    loom_low_allocation_fixed_location_entry_t* entries, uint32_t begin,
    uint32_t end, uint32_t entry_index) {
  IREE_ASSERT_LT(begin, end);
  const uint32_t middle = begin + (end - begin) / 2u;
  if (entry_index < middle) {
    loom_low_allocation_fixed_availability_update_subtree(entries, begin,
                                                          middle, entry_index);
  } else if (entry_index > middle) {
    loom_low_allocation_fixed_availability_update_subtree(entries, middle + 1u,
                                                          end, entry_index);
  }
  entries[middle].subtree_max_conflict_start =
      iree_max(entries[middle].next_conflict_start,
               iree_max(loom_low_allocation_fixed_availability_subtree_max(
                            entries, begin, middle),
                        loom_low_allocation_fixed_availability_subtree_max(
                            entries, middle + 1u, end)));
  return entries[middle].subtree_max_conflict_start;
}

static const loom_low_allocation_fixed_location_group_t*
loom_low_allocation_fixed_availability_group_for_entry(
    const loom_low_allocation_fixed_availability_t* availability,
    uint32_t entry_index) {
  uint32_t begin = 0;
  uint32_t end = availability->group_count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2u;
    const loom_low_allocation_fixed_location_group_t* group =
        &availability->groups[middle];
    if (entry_index < group->entry_start) {
      end = middle;
    } else if (entry_index >= group->entry_start + group->entry_count) {
      begin = middle + 1u;
    } else {
      return group;
    }
  }
  IREE_ASSERT_UNREACHABLE("fixed availability entry has no storage group");
  return NULL;
}

static void loom_low_allocation_fixed_availability_update_entry(
    loom_low_allocation_fixed_availability_t* availability,
    uint32_t entry_index) {
  const loom_low_allocation_fixed_location_group_t* group =
      loom_low_allocation_fixed_availability_group_for_entry(availability,
                                                             entry_index);
  loom_low_allocation_fixed_availability_update_subtree(
      availability->entries, group->entry_start,
      group->entry_start + group->entry_count, entry_index);
}

static void loom_low_allocation_fixed_availability_advance(
    loom_low_allocation_fixed_availability_t* availability,
    uint32_t start_point) {
  IREE_ASSERT_GE(start_point, availability->start_point);
  const loom_low_allocation_fixed_storage_record_t* records =
      availability->constraints->fixed_index.records;
  while (availability->expiration_heap_count != 0) {
    const uint32_t entry_index = availability->expiration_heap[0];
    loom_low_allocation_fixed_location_entry_t* entry =
        &availability->entries[entry_index];
    if (records[entry->current_record].end_point > start_point) {
      break;
    }
    do {
      ++entry->current_record;
    } while (entry->current_record < entry->record_end &&
             records[entry->current_record].end_point <= start_point);
    if (entry->current_record == entry->record_end) {
      entry->next_conflict_start = UINT32_MAX;
      --availability->expiration_heap_count;
      if (availability->expiration_heap_count != 0) {
        availability->expiration_heap[0] =
            availability->expiration_heap[availability->expiration_heap_count];
        loom_low_allocation_fixed_availability_heap_sift_down(availability, 0);
      }
    } else {
      entry->next_conflict_start = records[entry->current_record].start_point;
      loom_low_allocation_fixed_availability_heap_sift_down(availability, 0);
    }
    loom_low_allocation_fixed_availability_update_entry(availability,
                                                        entry_index);
  }
  availability->start_point = start_point;
}

iree_status_t loom_low_allocation_fixed_availability_initialize(
    const loom_low_allocation_target_constraints_t* constraints,
    iree_arena_allocator_t* arena,
    loom_low_allocation_fixed_availability_t* out_availability) {
  IREE_ASSERT_ARGUMENT(constraints);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_availability);
  *out_availability = (loom_low_allocation_fixed_availability_t){
      .constraints = constraints,
  };
  const loom_low_allocation_fixed_storage_index_t* index =
      &constraints->fixed_index;
  if (index->records == NULL) {
    return iree_ok_status();
  }

  uint32_t entry_count = 0;
  uint32_t group_count = 0;
  for (uint32_t kind_ordinal = 0; kind_ordinal < 2; ++kind_ordinal) {
    const uint32_t begin = index->record_starts[kind_ordinal];
    const uint32_t end = begin + index->record_counts[kind_ordinal];
    uint32_t record_index = begin;
    while (record_index < end) {
      const loom_low_allocation_fixed_storage_record_t* record =
          &index->records[record_index];
      ++group_count;
      const uint32_t storage_key = record->storage_key;
      while (record_index < end &&
             index->records[record_index].storage_key == storage_key) {
        const uint32_t location = index->records[record_index].location;
        ++entry_count;
        do {
          ++record_index;
        } while (record_index < end &&
                 index->records[record_index].storage_key == storage_key &&
                 index->records[record_index].location == location);
      }
    }
  }
  IREE_ASSERT_NE(entry_count, 0u);
  IREE_ASSERT_NE(group_count, 0u);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, entry_count, sizeof(*out_availability->entries),
      (void**)&out_availability->entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count, sizeof(*out_availability->groups),
      (void**)&out_availability->groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, entry_count, sizeof(*out_availability->expiration_heap),
      (void**)&out_availability->expiration_heap));

  uint32_t next_entry = 0;
  uint32_t next_group = 0;
  for (uint32_t kind_ordinal = 0; kind_ordinal < 2; ++kind_ordinal) {
    const uint32_t begin = index->record_starts[kind_ordinal];
    const uint32_t end = begin + index->record_counts[kind_ordinal];
    uint32_t record_index = begin;
    while (record_index < end) {
      const uint32_t storage_key = index->records[record_index].storage_key;
      loom_low_allocation_fixed_location_group_t* group =
          &out_availability->groups[next_group++];
      *group = (loom_low_allocation_fixed_location_group_t){
          .storage_key = storage_key,
          .entry_start = next_entry,
          .kind_ordinal = kind_ordinal,
      };
      while (record_index < end &&
             index->records[record_index].storage_key == storage_key) {
        const uint32_t location = index->records[record_index].location;
        const uint32_t record_start = record_index;
        do {
          ++record_index;
        } while (record_index < end &&
                 index->records[record_index].storage_key == storage_key &&
                 index->records[record_index].location == location);
        out_availability->entries[next_entry] =
            (loom_low_allocation_fixed_location_entry_t){
                .location = location,
                .record_end = record_index,
                .current_record = record_start,
                .next_conflict_start = index->records[record_start].start_point,
            };
        out_availability->expiration_heap[next_entry] = next_entry;
        ++next_entry;
        ++group->entry_count;
      }
      loom_low_allocation_fixed_availability_build_subtree(
          out_availability->entries, group->entry_start,
          group->entry_start + group->entry_count);
    }
  }
  IREE_ASSERT_EQ(next_entry, entry_count);
  IREE_ASSERT_EQ(next_group, group_count);
  out_availability->group_count = group_count;
  out_availability->expiration_heap_count = entry_count;
  for (uint32_t i = entry_count / 2u; i > 0; --i) {
    loom_low_allocation_fixed_availability_heap_sift_down(out_availability,
                                                          i - 1u);
  }
  return iree_ok_status();
}

static const loom_low_allocation_fixed_location_group_t*
loom_low_allocation_fixed_availability_find_group(
    const loom_low_allocation_fixed_availability_t* availability,
    uint32_t kind_ordinal, uint32_t storage_key) {
  uint32_t begin = 0;
  uint32_t end = availability->group_count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2u;
    const loom_low_allocation_fixed_location_group_t* group =
        &availability->groups[middle];
    if (group->kind_ordinal < kind_ordinal ||
        (group->kind_ordinal == kind_ordinal &&
         group->storage_key < storage_key)) {
      begin = middle + 1u;
    } else if (group->kind_ordinal > kind_ordinal ||
               group->storage_key > storage_key) {
      end = middle;
    } else {
      return group;
    }
  }
  return NULL;
}

static bool loom_low_allocation_fixed_availability_location_is_available(
    const loom_low_allocation_fixed_availability_t* availability,
    const loom_low_allocation_fixed_location_group_t* group, uint32_t location,
    uint32_t candidate_end_point) {
  uint32_t begin = group->entry_start;
  uint32_t end = begin + group->entry_count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2u;
    const loom_low_allocation_fixed_location_entry_t* entry =
        &availability->entries[middle];
    if (entry->location < location) {
      begin = middle + 1u;
    } else if (entry->location > location) {
      end = middle;
    } else {
      return entry->next_conflict_start >= candidate_end_point;
    }
  }
  return true;
}

static bool loom_low_allocation_fixed_availability_find_first(
    const loom_low_allocation_fixed_availability_t* availability,
    uint32_t begin, uint32_t end, uint64_t maximum,
    uint32_t candidate_end_point, uint64_t* cursor, uint64_t* out_location) {
  if (begin == end || *cursor > maximum) {
    return false;
  }
  const loom_low_allocation_fixed_location_entry_t* entries =
      availability->entries;
  const uint64_t subtree_minimum = entries[begin].location;
  const uint64_t subtree_maximum = entries[end - 1u].location;
  if (subtree_maximum < *cursor || subtree_minimum > maximum) {
    return false;
  }
  if (subtree_minimum > *cursor) {
    *out_location = *cursor;
    return true;
  }
  const uint32_t middle = begin + (end - begin) / 2u;
  const bool subtree_is_dense =
      subtree_maximum - subtree_minimum + 1u == end - begin;
  if (subtree_is_dense && *cursor >= subtree_minimum &&
      *cursor <= subtree_maximum &&
      entries[middle].subtree_max_conflict_start < candidate_end_point) {
    *cursor = subtree_maximum + 1u;
    return false;
  }
  if (loom_low_allocation_fixed_availability_find_first(
          availability, begin, middle, maximum, candidate_end_point, cursor,
          out_location)) {
    return true;
  }
  if (*cursor > maximum) {
    return false;
  }
  const loom_low_allocation_fixed_location_entry_t* entry = &entries[middle];
  if (*cursor < entry->location) {
    *out_location = *cursor;
    return true;
  }
  if (*cursor == entry->location) {
    if (entry->next_conflict_start >= candidate_end_point) {
      *out_location = *cursor;
      return true;
    }
    ++*cursor;
  }
  return loom_low_allocation_fixed_availability_find_first(
      availability, middle + 1u, end, maximum, candidate_end_point, cursor,
      out_location);
}

static bool loom_low_allocation_fixed_availability_find_last(
    const loom_low_allocation_fixed_availability_t* availability,
    uint32_t begin, uint32_t end, uint64_t minimum,
    uint32_t candidate_end_point, uint64_t* cursor, uint64_t* out_location) {
  if (begin == end || *cursor < minimum) {
    return false;
  }
  const loom_low_allocation_fixed_location_entry_t* entries =
      availability->entries;
  const uint64_t subtree_minimum = entries[begin].location;
  const uint64_t subtree_maximum = entries[end - 1u].location;
  if (subtree_maximum < minimum || subtree_minimum > *cursor) {
    return false;
  }
  if (subtree_maximum < *cursor) {
    *out_location = *cursor;
    return true;
  }
  const uint32_t middle = begin + (end - begin) / 2u;
  const bool subtree_is_dense =
      subtree_maximum - subtree_minimum + 1u == end - begin;
  if (subtree_is_dense && *cursor >= subtree_minimum &&
      *cursor <= subtree_maximum &&
      entries[middle].subtree_max_conflict_start < candidate_end_point) {
    if (subtree_minimum == 0) {
      *cursor = 0;
      return false;
    }
    *cursor = subtree_minimum - 1u;
    return false;
  }
  if (loom_low_allocation_fixed_availability_find_last(
          availability, middle + 1u, end, minimum, candidate_end_point, cursor,
          out_location)) {
    return true;
  }
  if (*cursor < minimum) {
    return false;
  }
  const loom_low_allocation_fixed_location_entry_t* entry = &entries[middle];
  if (*cursor > entry->location) {
    *out_location = *cursor;
    return true;
  }
  if (*cursor == entry->location) {
    if (entry->next_conflict_start >= candidate_end_point) {
      *out_location = *cursor;
      return true;
    }
    if (*cursor == 0) {
      return false;
    }
    --*cursor;
  }
  return loom_low_allocation_fixed_availability_find_last(
      availability, begin, middle, minimum, candidate_end_point, cursor,
      out_location);
}

static const loom_low_allocation_fixed_location_group_t*
loom_low_allocation_fixed_availability_prepare_query(
    loom_low_allocation_fixed_availability_t* availability,
    const loom_low_allocation_assignment_t* candidate) {
  IREE_ASSERT_ARGUMENT(availability);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_TRUE(loom_low_allocation_fixed_availability_can_order_candidate(
      availability, candidate));
  loom_low_allocation_fixed_availability_advance(availability,
                                                 candidate->start_point);
  uint32_t storage_key = 0;
  uint32_t ignored_location = 0;
  IREE_ASSERT_EQ(
      loom_low_allocation_storage_assignment_atomic_unit_count(
          availability->constraints->target->descriptor_set, candidate),
      1u);
  loom_low_allocation_storage_assignment_atomic_unit(
      availability->constraints->target->descriptor_set, candidate,
      /*atomic_unit_ordinal=*/0, &storage_key, &ignored_location);
  return loom_low_allocation_fixed_availability_find_group(
      availability,
      loom_low_allocation_fixed_storage_kind_ordinal(candidate->location_kind),
      storage_key);
}

bool loom_low_allocation_fixed_availability_can_order_candidate(
    const loom_low_allocation_fixed_availability_t* availability,
    const loom_low_allocation_assignment_t* candidate) {
  IREE_ASSERT_ARGUMENT(availability);
  IREE_ASSERT_ARGUMENT(candidate);
  const loom_low_allocation_target_constraints_t* constraints =
      availability->constraints;
  if (availability->group_count == 0 || constraints == NULL ||
      candidate->descriptor_reg_class_id >=
          constraints->target->descriptor_set->reg_class_count ||
      !loom_low_allocation_assignment_is_register_like(candidate) ||
      candidate->unit_count != 1 || candidate->location_count != 1 ||
      candidate->liveness_segments.count != 0 ||
      iree_any_bit_set(
          candidate->flags,
          LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS) ||
      loom_low_allocation_storage_assignment_uses_explicit_physical_register(
          constraints->target->descriptor_set, candidate)) {
    return false;
  }
  return loom_low_allocation_target_constraints_fixed_value_for_value(
             constraints, candidate->value_id) == NULL;
}

bool loom_low_allocation_fixed_availability_find_next_location(
    loom_low_allocation_fixed_availability_t* availability,
    const loom_low_allocation_assignment_t* candidate, uint32_t minimum_base,
    uint32_t maximum_base, uint32_t* out_base) {
  IREE_ASSERT_ARGUMENT(out_base);
  const loom_low_allocation_fixed_location_group_t* group =
      loom_low_allocation_fixed_availability_prepare_query(availability,
                                                           candidate);
  if (minimum_base > maximum_base) {
    return false;
  }
  if (group == NULL) {
    *out_base = minimum_base;
    return true;
  }
  uint64_t cursor = minimum_base;
  uint64_t result = 0;
  bool found = loom_low_allocation_fixed_availability_find_first(
      availability, group->entry_start, group->entry_start + group->entry_count,
      maximum_base, candidate->end_point, &cursor, &result);
  if (!found && cursor <= maximum_base &&
      loom_low_allocation_fixed_availability_location_is_available(
          availability, group, (uint32_t)cursor, candidate->end_point)) {
    result = cursor;
    found = true;
  }
  if (found) {
    *out_base = (uint32_t)result;
  }
  return found;
}

bool loom_low_allocation_fixed_availability_find_previous_location(
    loom_low_allocation_fixed_availability_t* availability,
    const loom_low_allocation_assignment_t* candidate, uint32_t minimum_base,
    uint32_t maximum_base, uint32_t* out_base) {
  IREE_ASSERT_ARGUMENT(out_base);
  const loom_low_allocation_fixed_location_group_t* group =
      loom_low_allocation_fixed_availability_prepare_query(availability,
                                                           candidate);
  if (minimum_base > maximum_base) {
    return false;
  }
  if (group == NULL) {
    *out_base = maximum_base;
    return true;
  }
  uint64_t cursor = maximum_base;
  uint64_t result = 0;
  bool found = loom_low_allocation_fixed_availability_find_last(
      availability, group->entry_start, group->entry_start + group->entry_count,
      minimum_base, candidate->end_point, &cursor, &result);
  if (!found && cursor >= minimum_base &&
      loom_low_allocation_fixed_availability_location_is_available(
          availability, group, (uint32_t)cursor, candidate->end_point)) {
    result = cursor;
    found = true;
  }
  if (found) {
    *out_base = (uint32_t)result;
  }
  return found;
}

static int loom_low_allocation_fixed_storage_record_key_compare(
    const loom_low_allocation_fixed_storage_record_t* record,
    uint32_t storage_key, uint32_t location) {
  if (record->storage_key != storage_key) {
    return record->storage_key < storage_key ? -1 : 1;
  }
  if (record->location != location) {
    return record->location < location ? -1 : 1;
  }
  return 0;
}

static uint32_t loom_low_allocation_fixed_storage_record_bound(
    const loom_low_allocation_fixed_storage_record_t* records, uint32_t begin,
    uint32_t end, uint32_t storage_key, uint32_t location, bool upper) {
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2u;
    const int comparison = loom_low_allocation_fixed_storage_record_key_compare(
        &records[middle], storage_key, location);
    if (comparison < 0 || (upper && comparison == 0)) {
      begin = middle + 1u;
    } else {
      end = middle;
    }
  }
  return begin;
}

static bool loom_low_allocation_fixed_storage_range_conflicts(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_low_allocation_resolved_fixed_value_t* candidate_fixed_value,
    uint32_t candidate_start_point, uint32_t candidate_end_point,
    uint32_t generation, uint32_t begin, uint32_t end) {
  loom_low_allocation_fixed_storage_index_t* index = &constraints->fixed_index;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    const loom_low_allocation_fixed_storage_record_t* record =
        &index->records[middle];
    if (record->subtree_end_point <= candidate_start_point) {
      return false;
    }
    if (record->start_point >= candidate_end_point) {
      end = middle;
      continue;
    }
    if (candidate_fixed_value != NULL && index->subtree_tied_roots != NULL &&
        index->subtree_tied_roots[middle] ==
            candidate_fixed_value->tied_root_ordinal) {
      return false;
    }
    if (loom_low_allocation_fixed_storage_range_conflicts(
            constraints, candidate_fixed_value, candidate_start_point,
            candidate_end_point, generation, begin, middle)) {
      return true;
    }
    begin = middle + 1;
    if (record->end_point <= candidate_start_point) {
      continue;
    }
    const uint32_t fixed_value_index = record->fixed_value_index;
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &constraints->fixed_values[fixed_value_index];
    const uint32_t root_index =
        constraints
            ->fixed_value_indices_by_ordinal[fixed_value->tied_root_ordinal] -
        1;
    if (index->excluded_generations[root_index] == generation) {
      continue;
    }
    if (candidate_fixed_value != NULL &&
        fixed_value->tied_root_ordinal ==
            candidate_fixed_value->tied_root_ordinal) {
      continue;
    }
    return true;
  }
  return false;
}

static uint32_t loom_low_allocation_fixed_storage_next_generation(
    loom_low_allocation_target_constraints_t* constraints) {
  loom_low_allocation_fixed_storage_index_t* index = &constraints->fixed_index;
  if (index->generation == UINT32_MAX) {
    memset(
        index->excluded_generations, 0,
        constraints->fixed_value_count * sizeof(*index->excluded_generations));
    index->generation = 0;
  }
  return ++index->generation;
}

bool loom_low_allocation_target_constraints_fixed_storage_conflicts(
    loom_low_allocation_target_constraints_t* constraints,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  const loom_low_allocation_resolved_fixed_value_t* fixed_value =
      loom_low_allocation_target_constraints_fixed_value_for_value(
          constraints, candidate->value_id);
  if (fixed_value != NULL &&
      !loom_low_allocation_storage_assignment_locations_correspond(
          constraints->target->descriptor_set, &fixed_value->assignment,
          candidate)) {
    return true;
  }
  if (loom_low_allocation_unit_liveness_clobber_conflicts(
          unit_liveness, constraints->target->descriptor_set, candidate)) {
    return true;
  }
  loom_low_allocation_fixed_storage_index_t* index = &constraints->fixed_index;
  if (index->records == NULL ||
      !loom_low_allocation_assignment_is_register_like(candidate)) {
    return false;
  }

  const uint32_t generation =
      loom_low_allocation_fixed_storage_next_generation(constraints);
  if (fixed_value != NULL) {
    index->excluded_generations[constraints->fixed_value_indices_by_ordinal
                                    [fixed_value->tied_root_ordinal] -
                                1] = generation;
  }
  for (uint16_t i = 0; i < ignored_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* ignored_fixed_value =
        loom_low_allocation_target_constraints_fixed_value_for_value(
            constraints, ignored_value_ids[i]);
    if (ignored_fixed_value != NULL) {
      index->excluded_generations[constraints->fixed_value_indices_by_ordinal
                                      [ignored_fixed_value->tied_root_ordinal] -
                                  1] = generation;
    }
  }

  const loom_low_descriptor_set_t* descriptor_set =
      constraints->target->descriptor_set;
  const uint32_t kind_ordinal =
      loom_low_allocation_fixed_storage_kind_ordinal(candidate->location_kind);
  const uint32_t index_start = index->record_starts[kind_ordinal];
  const uint32_t index_end = index_start + index->record_counts[kind_ordinal];
  for (uint32_t unit_index = 0; unit_index < candidate->location_count;
       ++unit_index) {
    const loom_low_allocation_assignment_t unit_assignment =
        loom_low_allocation_fixed_storage_unit_assignment(
            descriptor_set, candidate, unit_index);
    const uint32_t atomic_unit_count =
        loom_low_allocation_storage_assignment_atomic_unit_count(
            descriptor_set, &unit_assignment);
    for (uint32_t atomic_unit = 0; atomic_unit < atomic_unit_count;
         ++atomic_unit) {
      uint32_t storage_key = 0;
      uint32_t location = 0;
      loom_low_allocation_storage_assignment_atomic_unit(
          descriptor_set, &unit_assignment, atomic_unit, &storage_key,
          &location);
      const uint32_t begin = loom_low_allocation_fixed_storage_record_bound(
          index->records, index_start, index_end, storage_key, location,
          /*upper=*/false);
      const uint32_t end = loom_low_allocation_fixed_storage_record_bound(
          index->records, begin, index_end, storage_key, location,
          /*upper=*/true);
      loom_low_allocation_fixed_storage_time_iterator_t time_iterator =
          loom_low_allocation_fixed_storage_time_iterator(
              unit_liveness, candidate, unit_index);
      uint32_t claim_start_point = 0;
      uint32_t claim_end_point = 0;
      while (loom_low_allocation_fixed_storage_time_iterator_next(
          &time_iterator, &claim_start_point, &claim_end_point)) {
        if (loom_low_allocation_fixed_storage_range_conflicts(
                constraints, fixed_value, claim_start_point, claim_end_point,
                generation, begin, end)) {
          return true;
        }
      }
    }
  }
  return false;
}
