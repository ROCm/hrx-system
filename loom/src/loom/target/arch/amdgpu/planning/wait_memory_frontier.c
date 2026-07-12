// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_memory_frontier.h"

#include <string.h>

#include "loom/codegen/low/memory_access.h"
#include "loom/util/cfg_graph.h"

enum {
  LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT = LOOM_LOW_MEMORY_SPACE_WASM_MEMORY + 1u,
  LOOM_AMDGPU_WAIT_MEMORY_READ_SPACE_SHIFT = 0,
  LOOM_AMDGPU_WAIT_MEMORY_WRITE_SPACE_SHIFT = 8,
  LOOM_AMDGPU_WAIT_MEMORY_BLOCK_FLAG_QUEUED = 1u << 0,
  LOOM_AMDGPU_WAIT_MEMORY_BLOCK_FLAG_RESOLVED = 1u << 1,
};

static_assert(LOOM_AMDGPU_WAIT_MEMORY_SPACE_COUNT <= 8,
              "memory-space frontier flags must fit in one byte");
static_assert(sizeof(loom_amdgpu_wait_memory_state_t) == 10,
              "memory frontier state must remain compact");

loom_amdgpu_wait_memory_space_flags_t loom_amdgpu_wait_memory_space_flag(
    loom_low_memory_space_t memory_space) {
  const loom_low_memory_space_t normalized_space =
      loom_low_memory_access_normalize_space(memory_space);
  return (loom_amdgpu_wait_memory_space_flags_t)(1u << normalized_space);
}

static bool loom_amdgpu_wait_memory_state_union_changed(
    loom_amdgpu_wait_memory_state_t* target,
    const loom_amdgpu_wait_memory_state_t* source) {
  bool changed = false;
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint16_t access_space_flags =
        target->counter_access_space_flags[slot] |
        source->counter_access_space_flags[slot];
    changed |= access_space_flags != target->counter_access_space_flags[slot];
    target->counter_access_space_flags[slot] = access_space_flags;
  }
  return changed;
}

static bool loom_amdgpu_wait_memory_state_is_empty(
    const loom_amdgpu_wait_memory_state_t* state) {
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    if (state->counter_access_space_flags[slot] != 0) return false;
  }
  return true;
}

static void loom_amdgpu_wait_memory_state_add_node(
    loom_amdgpu_wait_memory_state_t* state,
    const loom_amdgpu_wait_memory_node_t* node, uint32_t read_counter_mask,
    uint32_t write_counter_mask) {
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    uint16_t access_space_flags = 0;
    if (iree_any_bit_set(read_counter_mask, counter_mask)) {
      access_space_flags |= (uint16_t)node->read_space_flags
                            << LOOM_AMDGPU_WAIT_MEMORY_READ_SPACE_SHIFT;
    }
    if (iree_any_bit_set(write_counter_mask, counter_mask)) {
      access_space_flags |= (uint16_t)node->write_space_flags
                            << LOOM_AMDGPU_WAIT_MEMORY_WRITE_SPACE_SHIFT;
    }
    state->counter_access_space_flags[slot] |= access_space_flags;
  }
}

static void loom_amdgpu_wait_memory_frontier_build_local_states(
    loom_amdgpu_wait_memory_frontier_t* frontier) {
  const loom_low_schedule_table_t* schedule = frontier->schedule;
  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    loom_amdgpu_wait_memory_state_t* state =
        &frontier->static_outgoing_states[block_index];
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const uint32_t node_index =
          schedule->scheduled_node_indices[block->scheduled_node_start + i];
      const loom_amdgpu_wait_memory_node_t* node = &frontier->nodes[node_index];
      loom_amdgpu_wait_memory_state_add_node(
          state, node, node->read_counter_mask, node->write_counter_mask);
    }
  }
}

static void loom_amdgpu_wait_memory_worklist_push(
    uint16_t block_index, uint16_t* worklist, uint32_t block_count,
    uint32_t* tail, uint32_t* count, uint8_t* block_flags) {
  if (iree_any_bit_set(block_flags[block_index],
                       LOOM_AMDGPU_WAIT_MEMORY_BLOCK_FLAG_QUEUED)) {
    return;
  }
  worklist[*tail] = block_index;
  if (++*tail == block_count) {
    *tail = 0;
  }
  ++*count;
  block_flags[block_index] |= LOOM_AMDGPU_WAIT_MEMORY_BLOCK_FLAG_QUEUED;
}

static void loom_amdgpu_wait_memory_frontier_propagate_static_states(
    loom_amdgpu_wait_memory_frontier_t* frontier, uint16_t* worklist) {
  const loom_cfg_graph_t* graph = &frontier->schedule->cfg_graph;
  const uint32_t block_count = (uint32_t)frontier->schedule->block_count;
  if (block_count <= 1 || graph->blocks == NULL) {
    return;
  }

  uint32_t head = 0;
  uint32_t tail = 0;
  uint32_t count = 0;
  for (iree_host_size_t block_index = 0; block_index < block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index) ||
        loom_amdgpu_wait_memory_state_is_empty(
            &frontier->static_outgoing_states[block_index])) {
      continue;
    }
    loom_amdgpu_wait_memory_worklist_push((uint16_t)block_index, worklist,
                                          block_count, &tail, &count,
                                          frontier->block_flags);
  }

  while (count != 0) {
    const uint16_t block_index = worklist[head];
    if (++head == block_count) {
      head = 0;
    }
    --count;
    frontier->block_flags[block_index] &=
        (uint8_t)~LOOM_AMDGPU_WAIT_MEMORY_BLOCK_FLAG_QUEUED;
    const loom_amdgpu_wait_memory_state_t* outgoing_state =
        &frontier->static_outgoing_states[block_index];
    const loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, block_index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      const uint16_t successor_index = successors.values[i];
      if (!loom_cfg_graph_block_is_reachable(graph, successor_index) ||
          !loom_amdgpu_wait_memory_state_union_changed(
              &frontier->static_outgoing_states[successor_index],
              outgoing_state)) {
        continue;
      }
      loom_amdgpu_wait_memory_worklist_push(successor_index, worklist,
                                            block_count, &tail, &count,
                                            frontier->block_flags);
    }
  }
}

iree_status_t loom_amdgpu_wait_memory_frontier_initialize(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_wait_memory_node_t* nodes, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_memory_frontier_t* out_frontier) {
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_frontier);
  IREE_ASSERT(schedule->node_count == 0 || nodes != NULL);
  *out_frontier = (loom_amdgpu_wait_memory_frontier_t){
      .schedule = schedule,
      .nodes = nodes,
      .active_block_index = UINT16_MAX,
  };
  if (schedule->block_count <= 1 || schedule->cfg_graph.blocks == NULL) {
    return iree_ok_status();
  }

  bool has_memory_producer = false;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    if (nodes[i].read_counter_mask != 0 || nodes[i].write_counter_mask != 0) {
      has_memory_producer = true;
      break;
    }
  }
  if (!has_memory_producer) return iree_ok_status();

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, schedule->block_count,
                                sizeof(*out_frontier->static_outgoing_states),
                                (void**)&out_frontier->static_outgoing_states));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count,
      sizeof(*out_frontier->resolved_outgoing_states),
      (void**)&out_frontier->resolved_outgoing_states));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count, sizeof(*out_frontier->block_flags),
      (void**)&out_frontier->block_flags));
  uint16_t* worklist = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, schedule->block_count, sizeof(*worklist), (void**)&worklist));
  memset(out_frontier->static_outgoing_states, 0,
         schedule->block_count * sizeof(*out_frontier->static_outgoing_states));
  memset(
      out_frontier->resolved_outgoing_states, 0,
      schedule->block_count * sizeof(*out_frontier->resolved_outgoing_states));
  memset(out_frontier->block_flags, 0,
         schedule->block_count * sizeof(*out_frontier->block_flags));

  loom_amdgpu_wait_memory_frontier_build_local_states(out_frontier);
  loom_amdgpu_wait_memory_frontier_propagate_static_states(out_frontier,
                                                           worklist);
  return iree_ok_status();
}

void loom_amdgpu_wait_memory_frontier_begin_block(
    loom_amdgpu_wait_memory_frontier_t* frontier, uint16_t block_index) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(block_index < frontier->schedule->block_count);
  IREE_ASSERT(frontier->active_block_index == UINT16_MAX);
  frontier->active_state = (loom_amdgpu_wait_memory_state_t){0};
  frontier->active_block_index = block_index;

  const loom_cfg_graph_t* graph = &frontier->schedule->cfg_graph;
  if (frontier->static_outgoing_states == NULL || graph->blocks == NULL) {
    return;
  }
  const loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    const uint16_t predecessor_index = predecessors.values[i];
    if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index)) {
      continue;
    }
    const bool predecessor_resolved =
        iree_any_bit_set(frontier->block_flags[predecessor_index],
                         LOOM_AMDGPU_WAIT_MEMORY_BLOCK_FLAG_RESOLVED);
    const loom_amdgpu_wait_memory_state_t* predecessor_state =
        predecessor_resolved
            ? &frontier->resolved_outgoing_states[predecessor_index]
            : &frontier->static_outgoing_states[predecessor_index];
    loom_amdgpu_wait_memory_state_union_changed(&frontier->active_state,
                                                predecessor_state);
  }
}

static bool loom_amdgpu_wait_memory_spaces_may_alias(
    loom_amdgpu_wait_memory_space_flags_t left,
    loom_amdgpu_wait_memory_space_flags_t right) {
  if (left == 0 || right == 0) return false;
  const loom_amdgpu_wait_memory_space_flags_t generic_flag =
      loom_amdgpu_wait_memory_space_flag(LOOM_LOW_MEMORY_SPACE_GENERIC);
  return iree_any_bit_set(left | right, generic_flag) ||
         iree_any_bit_set(left, right);
}

uint32_t loom_amdgpu_wait_memory_frontier_query(
    const loom_amdgpu_wait_memory_frontier_t* frontier,
    loom_amdgpu_wait_memory_space_flags_t space_flags,
    loom_amdgpu_wait_memory_access_flags_t access_flags) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  if (space_flags == 0 || access_flags == 0) return 0;
  uint32_t counter_mask = 0;
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    const uint16_t packed_space_flags =
        frontier->active_state.counter_access_space_flags[slot];
    loom_amdgpu_wait_memory_space_flags_t producer_space_flags = 0;
    if (iree_any_bit_set(access_flags,
                         LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ)) {
      producer_space_flags |=
          (loom_amdgpu_wait_memory_space_flags_t)(packed_space_flags >>
                                                  LOOM_AMDGPU_WAIT_MEMORY_READ_SPACE_SHIFT);
    }
    if (iree_any_bit_set(access_flags,
                         LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE)) {
      producer_space_flags |=
          (loom_amdgpu_wait_memory_space_flags_t)(packed_space_flags >>
                                                  LOOM_AMDGPU_WAIT_MEMORY_WRITE_SPACE_SHIFT);
    }
    if (loom_amdgpu_wait_memory_spaces_may_alias(producer_space_flags,
                                                 space_flags)) {
      counter_mask |= loom_amdgpu_wait_counter_mask_from_slot(slot);
    }
  }
  return counter_mask;
}

uint32_t loom_amdgpu_wait_memory_frontier_dependency_mask(
    const loom_amdgpu_wait_memory_frontier_t* frontier,
    const loom_amdgpu_wait_memory_node_t* node) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT_ARGUMENT(node);
  if (frontier->static_outgoing_states == NULL ||
      (node->read_space_flags == 0 && node->write_space_flags == 0)) {
    return 0;
  }
  const uint32_t prior_writes =
      node->read_space_flags == 0
          ? 0
          : loom_amdgpu_wait_memory_frontier_query(
                frontier, node->read_space_flags,
                LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_WRITE);
  const uint32_t prior_reads =
      node->write_space_flags == 0
          ? 0
          : loom_amdgpu_wait_memory_frontier_query(
                frontier, node->write_space_flags,
                LOOM_AMDGPU_WAIT_MEMORY_ACCESS_FLAG_READ);
  return prior_writes | prior_reads;
}

void loom_amdgpu_wait_memory_frontier_drain(
    loom_amdgpu_wait_memory_frontier_t* frontier, uint32_t counter_mask) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_SLOT_COUNT; ++slot) {
    if (iree_any_bit_set(counter_mask,
                         loom_amdgpu_wait_counter_mask_from_slot(slot))) {
      frontier->active_state.counter_access_space_flags[slot] = 0;
    }
  }
}

void loom_amdgpu_wait_memory_frontier_end_block(
    loom_amdgpu_wait_memory_frontier_t* frontier) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(frontier->active_block_index < frontier->schedule->block_count);
  if (frontier->resolved_outgoing_states == NULL) {
    frontier->active_state = (loom_amdgpu_wait_memory_state_t){0};
    frontier->active_block_index = UINT16_MAX;
    return;
  }
  const uint16_t block_index = frontier->active_block_index;
  loom_amdgpu_wait_memory_state_t* outgoing_state =
      &frontier->resolved_outgoing_states[block_index];
  *outgoing_state = frontier->active_state;

  const loom_low_schedule_block_t* block =
      &frontier->schedule->blocks[block_index];
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    const uint32_t node_index =
        frontier->schedule
            ->scheduled_node_indices[block->scheduled_node_start + i];
    const loom_amdgpu_wait_memory_node_t* node = &frontier->nodes[node_index];
    const uint32_t read_counter_mask =
        node->read_counter_mask & ~node->drained_after_production_counter_mask;
    const uint32_t write_counter_mask =
        node->write_counter_mask & ~node->drained_after_production_counter_mask;
    loom_amdgpu_wait_memory_state_add_node(
        outgoing_state, node, read_counter_mask, write_counter_mask);
  }
  frontier->block_flags[block_index] |=
      LOOM_AMDGPU_WAIT_MEMORY_BLOCK_FLAG_RESOLVED;
  frontier->active_state = (loom_amdgpu_wait_memory_state_t){0};
  frontier->active_block_index = UINT16_MAX;
}
