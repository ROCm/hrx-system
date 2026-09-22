// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/physical_issue.h"

#include <string.h>

#include "loom/codegen/low/schedule/dependency_index.h"
#include "loom/codegen/low/schedule/types.h"

iree_status_t loom_low_physical_issue_initialize(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_low_physical_issue_t* out_issue) {
  *out_issue = (loom_low_physical_issue_t){.schedule = schedule};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->node_count, sizeof(*out_issue->node_ready_cycles),
      (void**)&out_issue->node_ready_cycles));
  memset(out_issue->node_ready_cycles, 0,
         schedule->node_count * sizeof(*out_issue->node_ready_cycles));
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  IREE_RETURN_IF_ERROR(loom_low_schedule_event_frontier_initialize(
      descriptor_set, arena, &out_issue->events));
  return loom_low_schedule_resource_calendar_initialize(descriptor_set, arena,
                                                        &out_issue->resources);
}

uint32_t loom_low_physical_issue_source_ready_cycle(
    const loom_low_physical_issue_t* issue, uint32_t scheduled_packet_index) {
  const uint32_t node =
      issue->schedule->scheduled_node_indices[scheduled_packet_index];
  return iree_any_bit_set(issue->schedule->nodes[node].flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP)
             ? 0
             : issue->node_ready_cycles[node];
}

void loom_low_physical_issue_commit_source(loom_low_physical_issue_t* issue,
                                           uint32_t scheduled_packet_index,
                                           uint32_t issue_cycle) {
  const loom_low_schedule_table_t* schedule = issue->schedule;
  const uint32_t node =
      schedule->scheduled_node_indices[scheduled_packet_index];
  const uint32_t available_cycle =
      iree_max(issue_cycle, issue->node_ready_cycles[node]);
  const loom_low_schedule_dependency_index_t* index =
      schedule->dependency_index;
  const uint32_t begin =
      loom_low_schedule_dependency_index_group_begin(index, node);
  const uint32_t end =
      loom_low_schedule_dependency_index_group_end(index, node);
  for (uint32_t i = begin; i < end; ++i) {
    const loom_low_schedule_dependency_group_t* group =
        loom_low_schedule_dependency_index_group_at(index, i);
    const uint32_t consumer = group->consumer_node;
    if (schedule->nodes[consumer].block_index !=
        schedule->nodes[node].block_index) {
      continue;
    }
    issue->node_ready_cycles[consumer] =
        iree_max(issue->node_ready_cycles[consumer],
                 loom_low_schedule_add_signed_issue_separation(
                     available_cycle, group->minimum_issue_separation_cycles));
  }
}

static const loom_low_schedule_class_t* loom_low_physical_instruction_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_physical_instruction_t* instruction) {
  const loom_low_descriptor_view_t* view =
      loom_low_descriptor_set_descriptor_view_at(
          descriptor_set, instruction->descriptor_ordinal);
  return &descriptor_set->schedule_classes[view->schedule_class_id];
}

static bool loom_low_physical_registers_overlap(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t lhs_id,
    uint16_t rhs_id) {
  const loom_low_physical_register_t* lhs =
      &descriptor_set->physical_registers[lhs_id];
  const loom_low_physical_register_t* rhs =
      &descriptor_set->physical_registers[rhs_id];
  const uint16_t* units = descriptor_set->physical_register_atomic_units;
  uint16_t i = 0;
  uint16_t j = 0;
  while (i < lhs->atomic_unit_count && j < rhs->atomic_unit_count) {
    const uint16_t lhs_unit = units[lhs->atomic_unit_start + i];
    const uint16_t rhs_unit = units[rhs->atomic_unit_start + j];
    if (lhs_unit == rhs_unit) {
      return true;
    }
    if (lhs_unit < rhs_unit) {
      ++i;
    } else {
      ++j;
    }
  }
  return false;
}

static bool loom_low_physical_events_require_separation(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t producer,
    uint16_t consumer) {
  const loom_low_event_separation_t* separation =
      loom_low_descriptor_set_lookup_event_separation(descriptor_set, producer,
                                                      consumer);
  return separation != NULL && separation->minimum_issue_separation_cycles > 0;
}

bool loom_low_physical_issue_group_fits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_physical_instruction_t* instructions, uint16_t count) {
  const loom_low_schedule_class_t** classes =
      (const loom_low_schedule_class_t**)iree_alloca(count * sizeof(*classes));
  for (uint16_t i = 0; i < count; ++i) {
    classes[i] =
        loom_low_physical_instruction_class(descriptor_set, &instructions[i]);
    const loom_low_descriptor_t* consumer =
        &descriptor_set->descriptors[instructions[i].descriptor_ordinal];
    for (uint16_t j = 0; j < i; ++j) {
      const loom_low_descriptor_t* producer =
          &descriptor_set->descriptors[instructions[j].descriptor_ordinal];
      for (uint16_t p = 0; p < producer->operand_count; ++p) {
        const loom_low_operand_t* lhs =
            &descriptor_set->operands[producer->operand_start + p];
        for (uint16_t c = 0; c < consumer->operand_count; ++c) {
          const loom_low_operand_t* rhs =
              &descriptor_set->operands[consumer->operand_start + c];
          if (!loom_low_physical_events_require_separation(
                  descriptor_set, lhs->read_event_id, rhs->write_event_id) &&
              !loom_low_physical_events_require_separation(
                  descriptor_set, lhs->write_event_id, rhs->read_event_id) &&
              !loom_low_physical_events_require_separation(
                  descriptor_set, lhs->write_event_id, rhs->write_event_id)) {
            continue;
          }
          if (loom_low_physical_registers_overlap(
                  descriptor_set, instructions[j].physical_registers[p],
                  instructions[i].physical_registers[c])) {
            return false;
          }
        }
      }
    }
  }
  return loom_low_schedule_resource_group_fits(descriptor_set, classes, count);
}

uint32_t loom_low_physical_issue_register_ready_cycle(
    const loom_low_physical_issue_t* issue,
    const loom_low_physical_instruction_t* instructions, uint16_t count) {
  const loom_low_descriptor_set_t* descriptor_set =
      issue->events.descriptor_set;
  uint32_t cycle = 0;
  for (uint16_t i = 0; i < count; ++i) {
    const loom_low_descriptor_t* descriptor =
        &descriptor_set->descriptors[instructions[i].descriptor_ordinal];
    for (uint16_t j = 0; j < descriptor->operand_count; ++j) {
      const loom_low_operand_t* operand =
          &descriptor_set->operands[descriptor->operand_start + j];
      const uint16_t physical_register = instructions[i].physical_registers[j];
      cycle = iree_max(cycle, loom_low_schedule_event_frontier_query(
                                  &issue->events, physical_register,
                                  operand->read_event_id));
      cycle = iree_max(cycle, loom_low_schedule_event_frontier_query(
                                  &issue->events, physical_register,
                                  operand->write_event_id));
    }
  }
  return cycle;
}

iree_status_t loom_low_physical_issue_place(
    loom_low_physical_issue_t* issue,
    const loom_low_physical_instruction_t* instructions, uint16_t count,
    uint32_t proposed_cycle, uint32_t* out_cycle) {
  const loom_low_descriptor_set_t* descriptor_set =
      issue->events.descriptor_set;
  const loom_low_schedule_class_t** classes =
      (const loom_low_schedule_class_t**)iree_alloca(count * sizeof(*classes));
  for (uint16_t i = 0; i < count; ++i) {
    classes[i] =
        loom_low_physical_instruction_class(descriptor_set, &instructions[i]);
  }
  uint32_t cycle = iree_max(
      proposed_cycle,
      loom_low_physical_issue_register_ready_cycle(issue, instructions, count));
  uint16_t bottleneck = LOOM_LOW_RESOURCE_NONE;
  cycle = loom_low_schedule_resource_calendar_find_earliest_issue_cycle(
      &issue->resources, classes, count, cycle, &bottleneck);
  IREE_RETURN_IF_ERROR(loom_low_schedule_resource_calendar_commit(
      &issue->resources, classes, count, cycle));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const loom_low_descriptor_t* descriptor =
        &descriptor_set->descriptors[instructions[i].descriptor_ordinal];
    for (uint16_t j = 0;
         j < descriptor->operand_count && iree_status_is_ok(status); ++j) {
      const loom_low_operand_t* operand =
          &descriptor_set->operands[descriptor->operand_start + j];
      const uint16_t physical_register = instructions[i].physical_registers[j];
      status = loom_low_schedule_event_frontier_commit(
          &issue->events, physical_register, operand->read_event_id, cycle);
      if (iree_status_is_ok(status)) {
        status = loom_low_schedule_event_frontier_commit(
            &issue->events, physical_register, operand->write_event_id, cycle);
      }
    }
    for (uint16_t j = 0;
         j < descriptor->effect_count && iree_status_is_ok(status); ++j) {
      const loom_low_effect_t* effect =
          &descriptor_set->effects[descriptor->effect_start + j];
      if (effect->producer_event_id == LOOM_LOW_TIMING_EVENT_NONE) {
        continue;
      }
      const uint32_t separation =
          descriptor_set->timing_events[effect->producer_event_id]
              .maximum_issue_separation_cycles;
      if (cycle > UINT32_MAX - separation) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "physical effect exceeds the issue-cycle domain");
      } else {
        issue->effect_quiescent_cycle =
            iree_max(issue->effect_quiescent_cycle, cycle + separation);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_cycle = cycle;
  }
  return status;
}
