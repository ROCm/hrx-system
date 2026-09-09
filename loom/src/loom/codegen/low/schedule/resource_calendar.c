// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/resource_calendar.h"

#include <inttypes.h>
#include <string.h>

iree_status_t loom_low_schedule_resource_calendar_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_resource_calendar_t* out_calendar) {
  *out_calendar = (loom_low_schedule_resource_calendar_t){
      .descriptor_set = descriptor_set,
  };
  const uint32_t slot_count = descriptor_set->resource_calendar_slot_count;
  if (slot_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, slot_count,
                                                 sizeof(*out_calendar->slots),
                                                 (void**)&out_calendar->slots));
  memset(out_calendar->slots, 0, slot_count * sizeof(*out_calendar->slots));
  return iree_ok_status();
}

void loom_low_schedule_resource_calendar_reset(
    loom_low_schedule_resource_calendar_t* calendar) {
  const uint32_t slot_count =
      calendar->descriptor_set->resource_calendar_slot_count;
  if (slot_count != 0) {
    memset(calendar->slots, 0, slot_count * sizeof(*calendar->slots));
  }
}

static loom_low_schedule_resource_occupancy_t
loom_low_schedule_resource_calendar_candidate_occupancy(
    const loom_low_schedule_resource_calendar_t* calendar,
    const loom_low_schedule_class_t* schedule_class,
    const loom_low_resource_t* resource, uint32_t relative_cycle,
    uint16_t* out_resource_id) {
  uint32_t required_units = 0;
  uint32_t reserved_units = 0;
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &calendar->descriptor_set
             ->issue_uses[schedule_class->issue_use_start + i];
    if (calendar->descriptor_set->resources[issue_use->resource_id]
                .calendar.slot_start != resource->calendar.slot_start ||
        relative_cycle < issue_use->stage ||
        relative_cycle >= (uint32_t)issue_use->stage + issue_use->cycles) {
      continue;
    }
    if (issue_use->kind == LOOM_LOW_ISSUE_USE_KIND_REQUIRED) {
      required_units += issue_use->units;
    } else {
      reserved_units = iree_max(reserved_units, issue_use->units);
    }
    *out_resource_id = issue_use->resource_id;
  }
  return (loom_low_schedule_resource_occupancy_t){
      .required_units = (uint16_t)required_units,
      .reserved_units = (uint16_t)reserved_units,
  };
}

static bool loom_low_schedule_resource_calendar_issue_fits(
    const loom_low_schedule_resource_calendar_t* calendar,
    const loom_low_schedule_class_t* schedule_class, uint32_t issue_cycle,
    uint16_t* out_bottleneck_resource_id) {
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &calendar->descriptor_set
             ->issue_uses[schedule_class->issue_use_start + i];
    const loom_low_resource_t* resource =
        &calendar->descriptor_set->resources[issue_use->resource_id];
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
      const loom_low_schedule_resource_occupancy_t candidate =
          loom_low_schedule_resource_calendar_candidate_occupancy(
              calendar, schedule_class, resource, relative_cycle,
              &candidate_resource_id);
      const uint32_t slot_index =
          resource->calendar.slot_start +
          ((uint32_t)absolute_cycle & resource->calendar.slot_mask);
      const loom_low_schedule_resource_calendar_slot_t* slot =
          &calendar->slots[slot_index];
      const loom_low_schedule_resource_occupancy_t occupied =
          slot->issue_cycle == absolute_cycle
              ? slot->occupancy
              : (loom_low_schedule_resource_occupancy_t){0};
      const uint32_t combined_required_units =
          (uint32_t)occupied.required_units + candidate.required_units;
      const uint32_t combined_reserved_units =
          iree_max(occupied.reserved_units, candidate.reserved_units);
      if (combined_required_units + combined_reserved_units >
          resource->capacity_per_cycle) {
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
  if (schedule_class == NULL || schedule_class->issue_use_count == 0) {
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
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &calendar->descriptor_set
             ->issue_uses[schedule_class->issue_use_start + i];
    const loom_low_resource_t* resource =
        &calendar->descriptor_set->resources[issue_use->resource_id];
    const uint32_t use_start = issue_use->stage;
    const uint32_t use_end = use_start + issue_use->cycles;
    if (issue_cycle > UINT32_MAX - (use_end - 1u)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low schedule resource use exceeds the cycle domain at issue "
          "cycle %" PRIu32,
          issue_cycle);
    }
    for (uint32_t cycle = use_start; cycle < use_end; ++cycle) {
      const uint32_t absolute_cycle = issue_cycle + cycle;
      const uint32_t slot_index =
          resource->calendar.slot_start +
          (absolute_cycle & resource->calendar.slot_mask);
      loom_low_schedule_resource_calendar_slot_t* slot =
          &calendar->slots[slot_index];
      if (slot->issue_cycle != absolute_cycle) {
        *slot = (loom_low_schedule_resource_calendar_slot_t){
            .issue_cycle = absolute_cycle,
        };
      }
      loom_low_schedule_resource_occupancy_t* occupancy = &slot->occupancy;
      if (issue_use->kind == LOOM_LOW_ISSUE_USE_KIND_REQUIRED) {
        occupancy->required_units += issue_use->units;
      } else {
        occupancy->reserved_units =
            iree_max(occupancy->reserved_units, issue_use->units);
      }
      IREE_ASSERT_LE(
          (uint32_t)occupancy->required_units + occupancy->reserved_units,
          resource->capacity_per_cycle);
    }
  }
  return iree_ok_status();
}
