// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_loop.h"

#include "loom/ops/low/ops.h"
#include "loom/util/cfg_graph.h"

// Number of binary-ancestor levels needed for uint16_t CFG block indices.
#define LOOM_AMDGPU_WAIT_LOOP_ANCESTOR_LEVEL_COUNT 16u

// AMDGPU eligibility layered over one generic canonical loop interval.
struct loom_amdgpu_wait_loop_t {
  // Nearest eligible enclosing loop, or LOOM_CFG_LOOP_NONE.
  uint32_t valid_parent_loop_index;
  // Whether both loop edges have the required target-low branch shape.
  bool is_valid;
};

static bool loom_amdgpu_wait_loop_interval_is_supported(
    const loom_low_schedule_table_t* schedule,
    const loom_cfg_loop_interval_t* interval) {
  if (!interval->is_canonical) return false;
  const loom_cfg_graph_t* graph = &schedule->cfg_graph;
  const loom_cfg_edge_info_t* entry_edge =
      loom_cfg_graph_edge(graph, interval->entry_edge_index);
  const loom_cfg_edge_info_t* backedge =
      loom_cfg_graph_edge(graph, interval->backedge_edge_index);
  const loom_block_t* header = graph->blocks[interval->header_index].block;
  if (entry_edge == NULL || backedge == NULL ||
      entry_edge->terminator == NULL || backedge->terminator == NULL ||
      !loom_low_br_isa(entry_edge->terminator) ||
      loom_low_br_dest(entry_edge->terminator) != header ||
      !loom_low_br_isa(backedge->terminator) ||
      loom_low_br_dest(backedge->terminator) != header ||
      backedge->successor_index != 0) {
    return false;
  }
  const loom_cfg_block_index_span_t entry_successors =
      loom_cfg_graph_successors(graph, entry_edge->source_block_index);
  if (entry_successors.count != 1 || entry_edge->successor_index != 0) {
    return false;
  }

  const loom_low_schedule_block_t* preheader =
      &schedule->blocks[interval->entry_predecessor_index];
  if (preheader->node_count == 0) return false;
  const uint32_t insertion_node =
      preheader->node_start + preheader->node_count - 1;
  IREE_ASSERT_LT(insertion_node, schedule->node_count);
  return schedule->nodes[insertion_node].op == entry_edge->terminator;
}

static iree_host_size_t loom_amdgpu_wait_loop_ancestor_slot_index(
    uint32_t level, iree_host_size_t loop_count, uint32_t loop_index) {
  IREE_ASSERT_LT(level, LOOM_AMDGPU_WAIT_LOOP_ANCESTOR_LEVEL_COUNT);
  IREE_ASSERT_LT(loop_index, loop_count);
  return (iree_host_size_t)level * loop_count + loop_index;
}

iree_status_t loom_amdgpu_wait_loop_analysis_initialize(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_loop_analysis_t* out_analysis) {
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_analysis);
  *out_analysis = (loom_amdgpu_wait_loop_analysis_t){
      .schedule = schedule,
  };
  const loom_cfg_loop_forest_t* forest = &schedule->loop_forest;
  if (forest->interval_count == 0) return iree_ok_status();

  loom_amdgpu_wait_loop_t* loops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, forest->interval_count, sizeof(*loops), (void**)&loops));
  for (iree_host_size_t i = 0; i < forest->interval_count; ++i) {
    const loom_cfg_loop_interval_t* interval = &forest->intervals[i];
    const uint32_t parent_loop = interval->parent_loop_index;
    uint32_t valid_parent_loop = LOOM_CFG_LOOP_NONE;
    if (parent_loop != LOOM_CFG_LOOP_NONE) {
      valid_parent_loop = loops[parent_loop].is_valid
                              ? parent_loop
                              : loops[parent_loop].valid_parent_loop_index;
    }
    loops[i] = (loom_amdgpu_wait_loop_t){
        .valid_parent_loop_index = valid_parent_loop,
        .is_valid =
            loom_amdgpu_wait_loop_interval_is_supported(schedule, interval),
    };
  }

  // Sixteen rows cover every ancestor distance representable by uint16_t block
  // indices and make each dependency query a fixed number of probes.
  const iree_host_size_t ancestor_slot_count =
      forest->interval_count * LOOM_AMDGPU_WAIT_LOOP_ANCESTOR_LEVEL_COUNT;
  uint32_t* valid_ancestor_loops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, ancestor_slot_count, sizeof(*valid_ancestor_loops),
      (void**)&valid_ancestor_loops));
  for (iree_host_size_t i = 0; i < forest->interval_count; ++i) {
    valid_ancestor_loops[loom_amdgpu_wait_loop_ancestor_slot_index(
        /*level=*/0, forest->interval_count, (uint32_t)i)] =
        loops[i].is_valid ? loops[i].valid_parent_loop_index
                          : LOOM_CFG_LOOP_NONE;
  }
  for (uint32_t level = 1; level < LOOM_AMDGPU_WAIT_LOOP_ANCESTOR_LEVEL_COUNT;
       ++level) {
    for (iree_host_size_t i = 0; i < forest->interval_count; ++i) {
      const uint32_t prior_ancestor =
          valid_ancestor_loops[loom_amdgpu_wait_loop_ancestor_slot_index(
              level - 1, forest->interval_count, (uint32_t)i)];
      valid_ancestor_loops[loom_amdgpu_wait_loop_ancestor_slot_index(
          level, forest->interval_count, (uint32_t)i)] =
          prior_ancestor == LOOM_CFG_LOOP_NONE
              ? LOOM_CFG_LOOP_NONE
              : valid_ancestor_loops[loom_amdgpu_wait_loop_ancestor_slot_index(
                    level - 1, forest->interval_count, prior_ancestor)];
    }
  }

  *out_analysis = (loom_amdgpu_wait_loop_analysis_t){
      .schedule = schedule,
      .loops = loops,
      .valid_ancestor_loops = valid_ancestor_loops,
      .loop_count = forest->interval_count,
  };
  return iree_ok_status();
}

uint16_t loom_amdgpu_wait_loop_analysis_preheader(
    const loom_amdgpu_wait_loop_analysis_t* analysis, uint32_t producer_node,
    uint32_t consumer_node) {
  if (analysis->loop_count == 0) return UINT16_MAX;
  const loom_cfg_loop_forest_t* forest = &analysis->schedule->loop_forest;
  const uint16_t producer_block =
      analysis->schedule->nodes[producer_node].block_index;
  const uint16_t consumer_block =
      analysis->schedule->nodes[consumer_node].block_index;
  const uint32_t innermost_loop =
      forest->innermost_loop_indices[consumer_block];
  if (innermost_loop == LOOM_CFG_LOOP_NONE) return UINT16_MAX;

  uint32_t loop_index =
      analysis->loops[innermost_loop].is_valid
          ? innermost_loop
          : analysis->loops[innermost_loop].valid_parent_loop_index;
  if (loop_index == LOOM_CFG_LOOP_NONE ||
      forest->intervals[loop_index].header_index <= producer_block) {
    return UINT16_MAX;
  }

  for (uint32_t i = LOOM_AMDGPU_WAIT_LOOP_ANCESTOR_LEVEL_COUNT; i > 0; --i) {
    const uint32_t level = i - 1;
    const uint32_t ancestor_loop =
        analysis
            ->valid_ancestor_loops[loom_amdgpu_wait_loop_ancestor_slot_index(
                level, analysis->loop_count, loop_index)];
    if (ancestor_loop != LOOM_CFG_LOOP_NONE &&
        forest->intervals[ancestor_loop].header_index > producer_block) {
      loop_index = ancestor_loop;
    }
  }
  return forest->intervals[loop_index].entry_predecessor_index;
}
