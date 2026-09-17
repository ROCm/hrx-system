// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/scopes.h"

#include "loom/codegen/low/schedule/graph.h"

typedef struct loom_low_schedule_phase_frontier_t {
  // Block in which this frontier was initialized, or UINT32_MAX.
  uint32_t block_index;
  // Start control of the active phase in this block, or NODE_NONE.
  uint32_t start_node;
  // Head of the active phase's direct member list, or NODE_NONE.
  uint32_t first_member_node;
} loom_low_schedule_phase_frontier_t;

static loom_low_schedule_phase_frontier_t* loom_low_schedule_phase_frontier(
    loom_low_schedule_phase_frontier_t* frontiers, uint32_t scope,
    uint32_t block_index) {
  loom_low_schedule_phase_frontier_t* frontier = &frontiers[scope];
  if (frontier->block_index != block_index) {
    *frontier = (loom_low_schedule_phase_frontier_t){
        .block_index = block_index,
        .start_node = LOOM_LOW_SCHEDULE_NODE_NONE,
        .first_member_node = LOOM_LOW_SCHEDULE_NODE_NONE,
    };
  }
  return frontier;
}

static iree_status_t loom_low_schedule_add_order_dependency(
    loom_low_schedule_build_state_t* state, uint32_t producer,
    uint32_t consumer) {
  if (producer == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return iree_ok_status();
  }
  return loom_low_schedule_add_dependency(state, producer, consumer,
                                          LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER,
                                          UINT32_MAX);
}

static iree_status_t loom_low_schedule_phase_add_member(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_phase_frontier_t* frontiers, uint32_t* next_members,
    uint32_t scope, uint32_t block_index, uint32_t node_index) {
  if (scope == 0) {
    return iree_ok_status();
  }
  loom_low_schedule_phase_frontier_t* frontier =
      loom_low_schedule_phase_frontier(frontiers, scope, block_index);
  IREE_RETURN_IF_ERROR(loom_low_schedule_add_order_dependency(
      state, frontier->start_node, node_index));
  next_members[node_index] = frontier->first_member_node;
  frontier->first_member_node = node_index;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_phase_join(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_phase_frontier_t* frontier, const uint32_t* next_members,
    uint32_t node_index) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_add_order_dependency(
      state, frontier->start_node, node_index));
  iree_status_t status = iree_ok_status();
  for (uint32_t member = frontier->first_member_node;
       member != LOOM_LOW_SCHEDULE_NODE_NONE && iree_status_is_ok(status);
       member = next_members[member]) {
    status = loom_low_schedule_add_order_dependency(state, member, node_index);
  }
  frontier->start_node = node_index;
  frontier->first_member_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  return status;
}

iree_status_t loom_low_schedule_build_scope_dependencies(
    loom_low_schedule_build_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_scope_builder_finish(
      &state->scope_builder, state->cfg_graph, state->options->emitter,
      state->arena, &state->scopes));
  state->error_count += state->scopes.error_count;
  if (state->scopes.control_count == 0 || state->scopes.error_count != 0) {
    return iree_ok_status();
  }

  loom_low_schedule_phase_frontier_t* frontiers = NULL;
  const iree_host_size_t frontier_count = state->scopes.control_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, frontier_count, sizeof(*frontiers), (void**)&frontiers));
  for (iree_host_size_t i = 0; i < frontier_count; ++i) {
    frontiers[i].block_index = UINT32_MAX;
  }
  uint32_t* next_members = NULL;
  const loom_low_schedule_block_t* last_block =
      &state->blocks[state->body->block_count - 1];
  const uint32_t node_count = last_block->node_start + last_block->node_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*next_members), (void**)&next_members));

  uint32_t control_index = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < state->body->block_count && iree_status_is_ok(status);
       ++block_index) {
    const loom_low_schedule_block_t* block = &state->blocks[block_index];
    const uint32_t node_end = block->node_start + block->node_count;
    uint32_t active_scope = state->scopes.block_entry_scopes[block_index];
    for (uint32_t node_index = block->node_start;
         node_index < node_end && iree_status_is_ok(status); ++node_index) {
      const loom_low_schedule_control_t* control =
          control_index < state->scopes.control_count &&
                  state->scopes.controls[control_index].node_index == node_index
              ? &state->scopes.controls[control_index++]
              : NULL;
      if (active_scope == LOOM_LOW_SCHEDULE_SCOPE_UNREACHABLE) {
        continue;
      }
      if (control == NULL) {
        status = loom_low_schedule_phase_add_member(state, frontiers,
                                                    next_members, active_scope,
                                                    block_index, node_index);
        continue;
      }
      if (control->kind == LOOM_LOW_SCHEDULE_CONTROL_BEGIN) {
        if (control->scope_before != 0) {
          loom_low_schedule_phase_frontier_t* parent =
              loom_low_schedule_phase_frontier(frontiers, control->scope_before,
                                               block_index);
          status = loom_low_schedule_add_order_dependency(
              state, parent->start_node, node_index);
        }
        loom_low_schedule_phase_frontier_t* child =
            loom_low_schedule_phase_frontier(frontiers, control->scope_after,
                                             block_index);
        child->start_node = node_index;
      } else {
        loom_low_schedule_phase_frontier_t* frontier =
            loom_low_schedule_phase_frontier(frontiers, control->scope_before,
                                             block_index);
        status = loom_low_schedule_phase_join(state, frontier, next_members,
                                              node_index);
        if (iree_status_is_ok(status) &&
            control->kind == LOOM_LOW_SCHEDULE_CONTROL_END) {
          status = loom_low_schedule_phase_add_member(
              state, frontiers, next_members, control->scope_after, block_index,
              node_index);
        }
      }
      active_scope = control->scope_after;
    }
  }
  return status;
}
