// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_PHYSICAL_ISSUE_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_PHYSICAL_ISSUE_H_

#include "loom/codegen/low/schedule/event_frontier.h"
#include "loom/codegen/low/schedule/resource_calendar.h"

#ifdef __cplusplus
extern "C" {
#endif

// A concrete instruction, including helpers that have no source descriptor
// packet. Bindings describe the selected native instruction, not the semantic
// register class of a source SSA value. The view is borrowed for one query.
typedef struct loom_low_physical_instruction_t {
  // Selected descriptor in the issue model's descriptor set.
  uint32_t descriptor_ordinal;
  // One physical-register ID per descriptor operand, including implicit state.
  const uint16_t* physical_registers;
} loom_low_physical_instruction_t;

// Final issue admission for software-timed register machines. Logical ordering
// has already established semantic dependencies. This owner adds the hard
// dependencies exposed by concrete storage reuse and generated instructions,
// and admits the collective resource demand of each physical issue group.
// Targets without software-visible event rules retain their hardware hazard
// protocol instead of inventing fixed issue delays.
typedef struct loom_low_physical_issue_t {
  // Register-access deadlines, including every outstanding reader and writer.
  loom_low_schedule_event_frontier_t events;
  // Resource occupancy for the realized instruction stream.
  loom_low_schedule_resource_calendar_t resources;
  // Effect-event completion bound used at control-flow boundaries.
  uint32_t effect_quiescent_cycle;
} loom_low_physical_issue_t;

iree_status_t loom_low_physical_issue_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, loom_low_physical_issue_t* out_issue);

// Tests coissue without touching retained state. Instruction order is the
// semantic order when zero-distance dependencies share one issue cycle.
bool loom_low_physical_issue_group_fits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_physical_instruction_t* instructions, uint16_t count);

// Returns the register-event lower bound for a candidate group. Querying all
// accesses before committing any preserves legal same-cycle forwarding.
uint32_t loom_low_physical_issue_register_ready_cycle(
    const loom_low_physical_issue_t* issue,
    const loom_low_physical_instruction_t* instructions, uint16_t count);

// Places one intrinsically legal group at or after the proposed cycle and the
// previously placed group. The caller retains the returned cycle in its native
// plan; this function neither mutates IR nor grows storage or rescans the plan.
iree_status_t loom_low_physical_issue_place(
    loom_low_physical_issue_t* issue,
    const loom_low_physical_instruction_t* instructions, uint16_t count,
    uint32_t proposed_cycle, uint32_t* out_cycle);

// Earliest successor-entry cycle that is independent of predecessor history.
// A control-flow owner accounts for branch-delay cycles before enforcing this
// boundary. A scoreboard reset alone is not an entry/exit timing contract.
static inline uint64_t loom_low_physical_issue_quiescent_cycle(
    const loom_low_physical_issue_t* issue) {
  return iree_max(
      issue->resources.quiescent_cycle,
      iree_max(issue->events.quiescent_cycle, issue->effect_quiescent_cycle));
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_PHYSICAL_ISSUE_H_
