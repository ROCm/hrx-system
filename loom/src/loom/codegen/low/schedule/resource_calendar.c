// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/resource_calendar.h"

#include <inttypes.h>
#include <string.h>

static uint16_t loom_low_schedule_resource_calendar_find_prior_group_row(
    const loom_low_schedule_resource_calendar_t* calendar,
    uint16_t resource_id) {
  const loom_low_resource_t* resource =
      &calendar->descriptor_set->resources[resource_id];
  if (resource->contention_group_id == 0) {
    return LOOM_LOW_ID_NONE;
  }
  for (uint16_t prior_id = 0; prior_id < resource_id; ++prior_id) {
    const loom_low_resource_t* prior =
        &calendar->descriptor_set->resources[prior_id];
    if (prior->contention_group_id == resource->contention_group_id) {
      return calendar->resource_row_indices[prior_id];
    }
  }
  return LOOM_LOW_ID_NONE;
}

iree_status_t loom_low_schedule_resource_calendar_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_resource_calendar_t* out_calendar) {
  *out_calendar = (loom_low_schedule_resource_calendar_t){
      .descriptor_set = descriptor_set,
      .arena = arena,
  };
  if (descriptor_set->resource_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_LE(descriptor_set->resource_count, UINT16_MAX);
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, descriptor_set->resource_count,
                                sizeof(*out_calendar->resource_row_indices),
                                (void**)&out_calendar->resource_row_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptor_set->resource_count, sizeof(*out_calendar->rows),
      (void**)&out_calendar->rows));
  memset(out_calendar->rows, 0,
         descriptor_set->resource_count * sizeof(*out_calendar->rows));
  for (uint16_t resource_id = 0; resource_id < descriptor_set->resource_count;
       ++resource_id) {
    const loom_low_resource_t* resource =
        &descriptor_set->resources[resource_id];
    IREE_ASSERT_NE(resource->capacity_per_cycle, 0);
    uint16_t row_index =
        loom_low_schedule_resource_calendar_find_prior_group_row(out_calendar,
                                                                 resource_id);
    if (row_index == LOOM_LOW_ID_NONE) {
      row_index = out_calendar->row_count++;
      out_calendar->rows[row_index].capacity_per_cycle =
          resource->capacity_per_cycle;
    } else {
      IREE_ASSERT_EQ(out_calendar->rows[row_index].capacity_per_cycle,
                     resource->capacity_per_cycle);
    }
    out_calendar->resource_row_indices[resource_id] = row_index;
  }
  return iree_ok_status();
}

void loom_low_schedule_resource_calendar_reset(
    loom_low_schedule_resource_calendar_t* calendar) {
  for (uint16_t i = 0; i < calendar->row_count; ++i) {
    loom_low_schedule_resource_calendar_row_t* row = &calendar->rows[i];
    if (row->occupied_cycle_count != 0) {
      memset(row->occupied_units, 0,
             row->occupied_cycle_count * sizeof(*row->occupied_units));
      row->occupied_cycle_count = 0;
    }
    row->base_issue_cycle = 0;
  }
}

// Drops occupancy before |issue_cycle|. List scheduling advances issue cycles
// monotonically, so those entries can never constrain another candidate.
static void loom_low_schedule_resource_calendar_advance(
    loom_low_schedule_resource_calendar_t* calendar, uint32_t issue_cycle) {
  for (uint16_t i = 0; i < calendar->row_count; ++i) {
    loom_low_schedule_resource_calendar_row_t* row = &calendar->rows[i];
    IREE_ASSERT_LE(row->base_issue_cycle, issue_cycle);
    const uint32_t elapsed_cycles = issue_cycle - row->base_issue_cycle;
    if (elapsed_cycles >= row->occupied_cycle_count) {
      if (row->occupied_cycle_count != 0) {
        memset(row->occupied_units, 0,
               row->occupied_cycle_count * sizeof(*row->occupied_units));
      }
      row->occupied_cycle_count = 0;
      row->base_issue_cycle = issue_cycle;
      continue;
    }
    if (elapsed_cycles == 0) {
      continue;
    }
    const uint32_t retained_cycle_count =
        row->occupied_cycle_count - elapsed_cycles;
    memmove(row->occupied_units, &row->occupied_units[elapsed_cycles],
            retained_cycle_count * sizeof(*row->occupied_units));
    memset(&row->occupied_units[retained_cycle_count], 0,
           elapsed_cycles * sizeof(*row->occupied_units));
    row->occupied_cycle_count = retained_cycle_count;
    row->base_issue_cycle = issue_cycle;
  }
}

static uint16_t loom_low_schedule_resource_calendar_candidate_units(
    const loom_low_schedule_resource_calendar_t* calendar,
    const loom_low_schedule_class_t* schedule_class, uint16_t row_index,
    uint32_t relative_cycle, uint16_t* out_resource_id) {
  uint32_t units = 0;
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &calendar->descriptor_set
             ->issue_uses[schedule_class->issue_use_start + i];
    IREE_ASSERT_LT(issue_use->resource_id,
                   calendar->descriptor_set->resource_count);
    if (calendar->resource_row_indices[issue_use->resource_id] != row_index ||
        relative_cycle < issue_use->stage ||
        relative_cycle >= (uint32_t)issue_use->stage + issue_use->cycles) {
      continue;
    }
    units += issue_use->units;
    *out_resource_id = issue_use->resource_id;
  }
  IREE_ASSERT_LE(units, calendar->rows[row_index].capacity_per_cycle);
  return (uint16_t)units;
}

static bool loom_low_schedule_resource_calendar_issue_fits(
    const loom_low_schedule_resource_calendar_t* calendar,
    const loom_low_schedule_class_t* schedule_class, uint32_t issue_cycle,
    uint16_t* out_bottleneck_resource_id) {
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &calendar->descriptor_set
             ->issue_uses[schedule_class->issue_use_start + i];
    const uint16_t row_index =
        calendar->resource_row_indices[issue_use->resource_id];
    const loom_low_schedule_resource_calendar_row_t* row =
        &calendar->rows[row_index];
    const uint32_t relative_end =
        (uint32_t)issue_use->stage + issue_use->cycles;
    for (uint32_t relative_cycle = issue_use->stage;
         relative_cycle < relative_end; ++relative_cycle) {
      const uint64_t absolute_cycle = (uint64_t)issue_cycle + relative_cycle;
      if (absolute_cycle > UINT32_MAX) {
        *out_bottleneck_resource_id = issue_use->resource_id;
        return false;
      }
      uint16_t candidate_resource_id = issue_use->resource_id;
      const uint16_t candidate_units =
          loom_low_schedule_resource_calendar_candidate_units(
              calendar, schedule_class, row_index, relative_cycle,
              &candidate_resource_id);
      IREE_ASSERT_LE(row->base_issue_cycle, absolute_cycle);
      const uint64_t occupied_cycle_index =
          absolute_cycle - row->base_issue_cycle;
      const uint16_t occupied_units =
          occupied_cycle_index < row->occupied_cycle_count
              ? row->occupied_units[(uint32_t)occupied_cycle_index]
              : 0;
      if (occupied_units > row->capacity_per_cycle - candidate_units) {
        *out_bottleneck_resource_id = candidate_resource_id;
        return false;
      }
    }
  }
  return true;
}

uint32_t loom_low_schedule_resource_calendar_find_earliest_issue_cycle(
    const loom_low_schedule_resource_calendar_t* calendar,
    const loom_low_schedule_class_t* schedule_class,
    uint32_t proposed_issue_cycle, uint16_t* out_bottleneck_resource_id) {
  *out_bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE;
  if (schedule_class == NULL || schedule_class->issue_use_count == 0 ||
      calendar->row_count == 0) {
    return proposed_issue_cycle;
  }
  uint32_t issue_cycle = proposed_issue_cycle;
  while (true) {
    uint16_t conflict_resource_id = LOOM_LOW_RESOURCE_NONE;
    if (loom_low_schedule_resource_calendar_issue_fits(
            calendar, schedule_class, issue_cycle, &conflict_resource_id)) {
      return issue_cycle;
    }
    if (*out_bottleneck_resource_id == LOOM_LOW_RESOURCE_NONE) {
      *out_bottleneck_resource_id = conflict_resource_id;
    }
    if (issue_cycle == UINT32_MAX) {
      return UINT32_MAX;
    }
    ++issue_cycle;
  }
}

iree_status_t loom_low_schedule_resource_calendar_commit(
    loom_low_schedule_resource_calendar_t* calendar,
    const loom_low_schedule_class_t* schedule_class, uint32_t issue_cycle) {
  if (schedule_class == NULL) {
    return iree_ok_status();
  }
  uint16_t bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE;
  if (!loom_low_schedule_resource_calendar_issue_fits(
          calendar, schedule_class, issue_cycle, &bottleneck_resource_id)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low schedule resource use has no representable capacity at issue "
        "cycle %" PRIu32,
        issue_cycle);
  }
  loom_low_schedule_resource_calendar_advance(calendar, issue_cycle);
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &calendar->descriptor_set
             ->issue_uses[schedule_class->issue_use_start + i];
    const uint16_t row_index =
        calendar->resource_row_indices[issue_use->resource_id];
    loom_low_schedule_resource_calendar_row_t* row = &calendar->rows[row_index];
    IREE_ASSERT_EQ(row->base_issue_cycle, issue_cycle);
    const uint32_t use_start = issue_use->stage;
    const uint32_t use_end = use_start + issue_use->cycles;
    if (use_end > row->occupied_cycle_capacity) {
      const iree_host_size_t old_count = row->occupied_cycle_count;
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          calendar->arena, old_count, (iree_host_size_t)use_end,
          sizeof(*row->occupied_units), &row->occupied_cycle_capacity,
          (void**)&row->occupied_units));
      memset(&row->occupied_units[old_count], 0,
             (row->occupied_cycle_capacity - old_count) *
                 sizeof(*row->occupied_units));
    }
    if (use_end > row->occupied_cycle_count) {
      memset(
          &row->occupied_units[row->occupied_cycle_count], 0,
          (use_end - row->occupied_cycle_count) * sizeof(*row->occupied_units));
    }
    for (uint32_t cycle = use_start; cycle < use_end; ++cycle) {
      IREE_ASSERT_LE(row->occupied_units[cycle],
                     row->capacity_per_cycle - issue_use->units);
      row->occupied_units[cycle] += issue_use->units;
    }
    if (use_end > row->occupied_cycle_count) {
      row->occupied_cycle_count = (uint32_t)use_end;
    }
  }
  return iree_ok_status();
}
