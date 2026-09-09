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

typedef struct loom_low_schedule_resource_occupancy_t {
  // Capacity consumed by required uses in this cycle.
  uint16_t required_units;
  // Maximum capacity reserved by any overlapping reservation in this cycle.
  uint16_t reserved_units;
} loom_low_schedule_resource_occupancy_t;

typedef struct loom_low_schedule_resource_calendar_slot_t {
  // Absolute cycle owning this ring slot. Other cycles observe zero occupancy.
  uint32_t issue_cycle;
  // Required and reserved capacity at |issue_cycle|. Zero-initialized slots
  // represent an empty calendar, including at issue cycle zero.
  loom_low_schedule_resource_occupancy_t occupancy;
} loom_low_schedule_resource_calendar_slot_t;

typedef struct loom_low_schedule_resource_calendar_t {
  // Descriptor set owning all resource and issue-use rows.
  const loom_low_descriptor_set_t* descriptor_set;
  // Fixed occupancy storage indexed by generated resource calendar layouts.
  // Only slots touched by an issued class are updated; advancing time neither
  // moves retained occupancy nor visits unrelated resources.
  loom_low_schedule_resource_calendar_slot_t* slots;
  // First cycle after every committed resource stage has completed.
  uint64_t quiescent_cycle;
} loom_low_schedule_resource_calendar_t;

// Allocates the exact target-declared occupancy storage. Resources with the
// same nonzero contention group share a ring. Rings cover each group's maximum
// stage plus duration, so monotonically advancing issue cycles only overwrite
// expired occupancy. No allocations occur after initialization.
iree_status_t loom_low_schedule_resource_calendar_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_resource_calendar_t* out_calendar);

// Clears all occupancy while retaining the fixed storage for reuse.
void loom_low_schedule_resource_calendar_reset(
    loom_low_schedule_resource_calendar_t* calendar);

// Tests the intrinsic resource legality of simultaneous instruction classes.
// Single descriptor classes already satisfy this during generation. Native
// packet formation uses this query before admitting a multi-instruction group.
bool loom_low_schedule_resource_group_fits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_class_t* const* schedule_classes,
    uint16_t schedule_class_count);

// Returns the earliest issue cycle at or after |proposed_issue_cycle| where
// all uses of an intrinsically legal group fit. Proposed cycles must not
// precede the most recently committed cycle. |out_bottleneck_resource_id|
// identifies a resource involved in the first rejected cycle, or
// LOOM_LOW_RESOURCE_NONE when no stall is needed.
uint32_t loom_low_schedule_resource_calendar_find_earliest_issue_cycle(
    const loom_low_schedule_resource_calendar_t* calendar,
    const loom_low_schedule_class_t* const* schedule_classes,
    uint16_t schedule_class_count, uint32_t proposed_issue_cycle,
    uint16_t* out_bottleneck_resource_id);

// Commits all uses in the group at a cycle admitted by find_earliest.
// No intervening commit may change that admission. Commit cycles must be
// monotonically nondecreasing between resets. Reports cycle-domain overflow;
// resource capacity is established by admission, not checked a second time.
iree_status_t loom_low_schedule_resource_calendar_commit(
    loom_low_schedule_resource_calendar_t* calendar,
    const loom_low_schedule_class_t* const* schedule_classes,
    uint16_t schedule_class_count, uint32_t issue_cycle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_RESOURCE_CALENDAR_H_
