// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/block.h"

#include "loom/codegen/low/schedule/descriptor_rows.h"
#include "loom/codegen/low/schedule/ready_policy.h"

void loom_low_schedule_block_begin(loom_low_schedule_build_state_t* state,
                                   uint32_t block_index) {
  loom_low_schedule_block_t* block = &state->blocks[block_index];
  block->scheduled_node_start = (uint32_t)state->scheduled_node_count;
  block->scheduled_node_count = 0;
  block->issue_group_start = (uint32_t)state->issue_group_count;
  block->issue_group_count = 0;
  block->candidate_decisions.start = (uint32_t)state->candidate_decision_count;
  block->candidate_decisions.count = 0;
  state->liveness_block_orders[block_index] = (loom_liveness_block_order_t){
      .block = block->block,
      .ops = block->node_count != 0
                 ? &state->scheduled_ops[state->scheduled_node_count]
                 : NULL,
      .op_count = block->node_count,
  };
  state->current_block_index = block_index;
  state->current_issue_cycle = 0;
  state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  loom_low_schedule_resource_calendar_reset(&state->resource_calendar);
}

static void loom_low_schedule_block_note_issue_group(
    loom_low_schedule_build_state_t* state, uint32_t node_index) {
  loom_low_schedule_node_t* node = &state->nodes[node_index];
  if (state->issue_group_count != 0) {
    loom_low_schedule_issue_group_t* group =
        &state->issue_groups[state->issue_group_count - 1];
    if (group->block_index == node->block_index) {
      IREE_ASSERT_LE(group->issue_cycle, node->issue_cycle);
    }
    if (group->block_index == node->block_index &&
        group->issue_cycle == node->issue_cycle) {
      IREE_ASSERT_NE(group->scheduled_node_count, UINT32_MAX);
      ++group->scheduled_node_count;
      node->issue_group_ordinal = (uint32_t)state->issue_group_count - 1;
      return;
    }
  }
  IREE_ASSERT_LT(state->issue_group_count, UINT32_MAX);
  node->issue_group_ordinal = (uint32_t)state->issue_group_count;
  state->issue_groups[state->issue_group_count++] =
      (loom_low_schedule_issue_group_t){
          .block_index = node->block_index,
          .issue_cycle = node->issue_cycle,
          .scheduled_node_start = (uint32_t)state->scheduled_node_count,
          .scheduled_node_count = 1,
      };
}

void loom_low_schedule_block_append(loom_low_schedule_build_state_t* state,
                                    uint32_t node_index, uint32_t issue_cycle) {
  loom_low_schedule_block_t* block = &state->blocks[state->current_block_index];
  loom_low_schedule_node_t* node = &state->nodes[node_index];
  node->scheduled_ordinal = block->scheduled_node_count++;
  node->issue_cycle = issue_cycle;
  state->current_issue_cycle = issue_cycle;
  loom_low_schedule_block_note_issue_group(state, node_index);
  block->issue_group_count =
      (uint32_t)state->issue_group_count - block->issue_group_start;
  state->scheduled_node_indices[state->scheduled_node_count] = node_index;
  state->scheduled_ops[state->scheduled_node_count] = node->op;
  ++state->scheduled_node_count;
  loom_low_schedule_ready_policy_note_node_scheduled(state, node_index);
}

void loom_low_schedule_block_finish(loom_low_schedule_build_state_t* state) {
  loom_low_schedule_block_t* block = &state->blocks[state->current_block_index];
  block->candidate_decisions.count = (uint32_t)state->candidate_decision_count -
                                     block->candidate_decisions.start;
}

iree_status_t loom_low_schedule_block_retain(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_table_t* previous) {
  const uint32_t block_index = state->current_block_index;
  const loom_low_schedule_block_t* block = &state->blocks[block_index];
  const loom_low_schedule_block_t* previous_block =
      &previous->blocks[block_index];
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       iree_status_is_ok(status) && i < previous_block->scheduled_node_count;
       ++i) {
    const uint32_t previous_index =
        previous
            ->scheduled_node_indices[previous_block->scheduled_node_start + i];
    const uint32_t node_index =
        block->node_start + (previous_index - previous_block->node_start);
    const loom_low_schedule_node_t* previous_node =
        &previous->nodes[previous_index];
    loom_low_schedule_node_t* node = &state->nodes[node_index];
    // Source offsets are stable only inside the transform's unchanged blocks.
    IREE_ASSERT_EQ(node->op, previous_node->op);
    node->descriptor = previous_node->descriptor;
    node->schedule_class_id = previous_node->schedule_class_id;
    node->schedule_class = previous_node->schedule_class;
    loom_low_schedule_block_append(state, node_index,
                                   previous_node->issue_cycle);
    if (state->pressure_steps != NULL) {
      loom_low_schedule_pressure_step_t* step =
          &state->pressure_steps[state->pressure_step_count++];
      *step =
          previous->pressure_steps[previous_block->scheduled_node_start + i];
      step->node_index = node_index;
    }
    status = loom_low_schedule_note_descriptor_rows_for_node(state, node_index);
  }
  if (iree_status_is_ok(status)) {
    if (state->candidate_decisions != NULL) {
      for (uint32_t i = 0; i < previous_block->candidate_decisions.count; ++i) {
        loom_low_schedule_candidate_decision_t* decision =
            &state->candidate_decisions[state->candidate_decision_count++];
        *decision = previous->candidate_decisions
                        [previous_block->candidate_decisions.start + i];
        decision->chosen_node =
            block->node_start +
            (decision->chosen_node - previous_block->node_start);
        decision->rejected_node =
            block->node_start +
            (decision->rejected_node - previous_block->node_start);
      }
    }
    loom_low_schedule_pressure_retain_block(state, pressure_state, previous,
                                            block_index);
    loom_low_schedule_block_finish(state);
  }
  return status;
}
