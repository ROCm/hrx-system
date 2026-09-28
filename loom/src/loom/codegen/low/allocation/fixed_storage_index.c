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
    if (index->excluded_generations[fixed_value_index] == generation) {
      continue;
    }
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &constraints->fixed_values[fixed_value_index];
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
      !loom_low_allocation_storage_assignment_ranges_equal(
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
    index->excluded_generations[fixed_value - constraints->fixed_values] =
        generation;
  }
  for (uint16_t i = 0; i < ignored_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* ignored_fixed_value =
        loom_low_allocation_target_constraints_fixed_value_for_value(
            constraints, ignored_value_ids[i]);
    if (ignored_fixed_value != NULL) {
      index->excluded_generations[ignored_fixed_value -
                                  constraints->fixed_values] = generation;
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
