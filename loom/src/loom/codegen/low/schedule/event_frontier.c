// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/event_frontier.h"

#include <string.h>

iree_status_t loom_low_schedule_event_frontier_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_event_frontier_t* out_frontier) {
  *out_frontier = (loom_low_schedule_event_frontier_t){
      .descriptor_set = descriptor_set,
  };
  const iree_host_size_t count =
      (iree_host_size_t)descriptor_set->physical_register_unit_count *
      descriptor_set->timing_event_count;
  if (count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*out_frontier->deadlines),
                                (void**)&out_frontier->deadlines));
  memset(out_frontier->deadlines, 0, count * sizeof(*out_frontier->deadlines));
  return iree_ok_status();
}

uint32_t loom_low_schedule_event_frontier_query(
    const loom_low_schedule_event_frontier_t* frontier,
    uint16_t physical_register_id, uint16_t event_id) {
  if (event_id == LOOM_LOW_TIMING_EVENT_NONE) return 0;
  const loom_low_descriptor_set_t* descriptor_set = frontier->descriptor_set;
  const loom_low_physical_register_t* physical_register =
      &descriptor_set->physical_registers[physical_register_id];
  uint32_t issue_cycle = 0;
  for (uint16_t i = 0; i < physical_register->atomic_unit_count; ++i) {
    const uint16_t unit = descriptor_set->physical_register_atomic_units
                              [physical_register->atomic_unit_start + i];
    const uint32_t deadline =
        frontier->deadlines[(iree_host_size_t)unit *
                                descriptor_set->timing_event_count +
                            event_id];
    issue_cycle = iree_max(issue_cycle, deadline);
  }
  return issue_cycle;
}

iree_status_t loom_low_schedule_event_frontier_commit(
    loom_low_schedule_event_frontier_t* frontier, uint16_t physical_register_id,
    uint16_t event_id, uint32_t issue_cycle) {
  if (event_id == LOOM_LOW_TIMING_EVENT_NONE) return iree_ok_status();
  const loom_low_descriptor_set_t* descriptor_set = frontier->descriptor_set;
  const loom_low_timing_event_t* event =
      &descriptor_set->timing_events[event_id];
  if (issue_cycle > UINT32_MAX - event->maximum_issue_separation_cycles) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "physical event exceeds the issue-cycle domain");
  }
  frontier->quiescent_cycle =
      iree_max(frontier->quiescent_cycle,
               issue_cycle + event->maximum_issue_separation_cycles);
  const loom_low_physical_register_t* physical_register =
      &descriptor_set->physical_registers[physical_register_id];
  for (uint16_t i = 0; i < physical_register->atomic_unit_count; ++i) {
    const uint16_t unit = descriptor_set->physical_register_atomic_units
                              [physical_register->atomic_unit_start + i];
    uint32_t* deadlines =
        frontier->deadlines +
        (iree_host_size_t)unit * descriptor_set->timing_event_count;
    for (uint16_t j = 0; j < event->separation_count; ++j) {
      const loom_low_event_separation_t* separation =
          &descriptor_set->event_separations[event->separation_start + j];
      if (separation->minimum_issue_separation_cycles <= 0) continue;
      const uint32_t deadline =
          issue_cycle + (uint32_t)separation->minimum_issue_separation_cycles;
      uint32_t* retained = &deadlines[separation->consumer_event_id];
      *retained = iree_max(*retained, deadline);
    }
  }
  return iree_ok_status();
}
