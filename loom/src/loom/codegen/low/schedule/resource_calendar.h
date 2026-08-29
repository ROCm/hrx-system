// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Capacity-aware target resource occupancy calendar.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_RESOURCE_CALENDAR_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_RESOURCE_CALENDAR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_schedule_resource_calendar_row_t {
  // Occupied units indexed relative to base_issue_cycle.
  uint16_t* occupied_units;
  // Absolute issue cycle represented by occupied_units[0].
  uint32_t base_issue_cycle;
  // Number of meaningful entries in occupied_units.
  uint32_t occupied_cycle_count;
  // Allocated entries in occupied_units.
  iree_host_size_t occupied_cycle_capacity;
  // Units available in each cycle.
  uint16_t capacity_per_cycle;
} loom_low_schedule_resource_calendar_row_t;

typedef struct loom_low_schedule_resource_calendar_t {
  // Descriptor set owning all resource and issue-use rows.
  const loom_low_descriptor_set_t* descriptor_set;
  // Arena owning the calendar and its lazily grown occupancy rows.
  iree_arena_allocator_t* arena;
  // Calendar row index keyed by descriptor resource identifier.
  uint16_t* resource_row_indices;
  // Independent resource calendars, including shared contention groups.
  loom_low_schedule_resource_calendar_row_t* rows;
  // Number of populated rows.
  uint16_t row_count;
} loom_low_schedule_resource_calendar_t;

// Initializes the compact resource-to-calendar mapping. Resources with the
// same nonzero contention group share one calendar row.
iree_status_t loom_low_schedule_resource_calendar_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_resource_calendar_t* out_calendar);

// Clears all occupancy while retaining arena-owned row capacity for reuse.
void loom_low_schedule_resource_calendar_reset(
    loom_low_schedule_resource_calendar_t* calendar);

// Returns the earliest issue cycle at or after |proposed_issue_cycle| where
// all uses fit. Proposed cycles must not precede the most recently committed
// cycle. |out_bottleneck_resource_id| identifies a resource involved in the
// first rejected cycle, or LOOM_LOW_RESOURCE_NONE when no stall is needed.
uint32_t loom_low_schedule_resource_calendar_find_earliest_issue_cycle(
    const loom_low_schedule_resource_calendar_t* calendar,
    const loom_low_schedule_class_t* schedule_class,
    uint32_t proposed_issue_cycle, uint16_t* out_bottleneck_resource_id);

// Commits all uses in |schedule_class| at |issue_cycle|. Commit cycles must be
// monotonically nondecreasing between resets.
iree_status_t loom_low_schedule_resource_calendar_commit(
    loom_low_schedule_resource_calendar_t* calendar,
    const loom_low_schedule_class_t* schedule_class, uint32_t issue_cycle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_RESOURCE_CALENDAR_H_
