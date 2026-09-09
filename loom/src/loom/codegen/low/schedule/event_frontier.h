// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_EVENT_FRONTIER_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_EVENT_FRONTIER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Physical event admission for a realized instruction stream. Each atomic
// storage unit retains one deadline per consumer event, not a history of SSA
// values or instructions. Committing an access advances the deadlines selected
// by its outgoing RAW/WAR/WAW rules. In particular, a newer write never erases
// an outstanding reader or a slower older write.
//
// The stream owner resolves register aliases into the descriptor set's atomic
// units and queries every access before committing any access of one issue
// group. Accesses within a group must independently permit coissue. Partial
// accesses conservatively occupy their containing atomic unit; they cannot
// kill another part's pending events. No allocation occurs after initialize.
typedef struct loom_low_schedule_event_frontier_t {
  // Descriptor set owning the event rules and physical storage geometry.
  const loom_low_descriptor_set_t* descriptor_set;
  // Unit-major deadlines indexed by the consumer timing-event ID.
  uint32_t* deadlines;
  // First cycle at which no retained event constrains any future access.
  uint32_t quiescent_cycle;
} loom_low_schedule_event_frontier_t;

iree_status_t loom_low_schedule_event_frontier_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_event_frontier_t* out_frontier);

// Returns the earliest legal issue cycle for this event on one physical
// register, considering every overlapping atomic storage unit.
uint32_t loom_low_schedule_event_frontier_query(
    const loom_low_schedule_event_frontier_t* frontier,
    uint16_t physical_register_id, uint16_t event_id);

// Commits an admitted physical access. The only possible failure is overflow
// of the issue-cycle domain. Events with zero or negative separations impose
// no delay on an instruction stream that preserves issue order.
iree_status_t loom_low_schedule_event_frontier_commit(
    loom_low_schedule_event_frontier_t* frontier, uint16_t physical_register_id,
    uint16_t event_id, uint32_t issue_cycle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_EVENT_FRONTIER_H_
