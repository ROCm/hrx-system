// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/cfg_loop.h"

#include <string.h>

// Transient construction state for one public loop interval.
typedef struct loom_cfg_loop_build_interval_t {
  // Public interval retained after construction.
  loom_cfg_loop_interval_t interval;
  // Minimum predecessor entering the header.
  uint16_t header_predecessor_min;
  // Maximum predecessor entering the header.
  uint16_t header_predecessor_max;
  // Minimum predecessor entering a non-header interval block.
  uint16_t body_predecessor_min;
  // Maximum predecessor entering a non-header interval block.
  uint16_t body_predecessor_max;
  // Whether the header predecessor range is populated.
  bool has_header_predecessor;
  // Whether the body predecessor range is populated.
  bool has_body_predecessor;
} loom_cfg_loop_build_interval_t;

static bool loom_cfg_loop_find_candidate(
    const loom_cfg_graph_t* graph, uint16_t header_index,
    iree_host_size_t* inout_reachable_backward_edge_count,
    loom_cfg_loop_interval_t* out_interval) {
  *out_interval = (loom_cfg_loop_interval_t){0};
  loom_cfg_edge_index_t entry_edge_index = LOOM_CFG_EDGE_INDEX_INVALID;
  loom_cfg_edge_index_t backedge_edge_index = LOOM_CFG_EDGE_INDEX_INVALID;
  bool has_unique_entry_edge = true;
  bool has_unique_backedge = true;
  const loom_cfg_edge_index_span_t predecessor_edges =
      loom_cfg_graph_predecessor_edges(graph, header_index);
  for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
    const loom_cfg_edge_index_t edge_index = predecessor_edges.values[i];
    const loom_cfg_edge_info_t* edge = loom_cfg_graph_edge(graph, edge_index);
    if (edge == NULL ||
        !loom_cfg_graph_block_is_reachable(graph, edge->source_block_index)) {
      continue;
    }
    if (edge->source_block_index < header_index) {
      if (entry_edge_index == LOOM_CFG_EDGE_INDEX_INVALID) {
        entry_edge_index = edge_index;
      } else {
        has_unique_entry_edge = false;
      }
    } else {
      ++*inout_reachable_backward_edge_count;
      if (backedge_edge_index == LOOM_CFG_EDGE_INDEX_INVALID) {
        backedge_edge_index = edge_index;
      } else {
        has_unique_backedge = false;
      }
    }
  }
  if (!has_unique_entry_edge || !has_unique_backedge ||
      entry_edge_index == LOOM_CFG_EDGE_INDEX_INVALID ||
      backedge_edge_index == LOOM_CFG_EDGE_INDEX_INVALID) {
    return false;
  }

  const loom_cfg_edge_info_t* entry_edge =
      loom_cfg_graph_edge(graph, entry_edge_index);
  const loom_cfg_edge_info_t* backedge =
      loom_cfg_graph_edge(graph, backedge_edge_index);

  *out_interval = (loom_cfg_loop_interval_t){
      .header_index = header_index,
      .latch_index = backedge->source_block_index,
      .entry_predecessor_index = entry_edge->source_block_index,
      .parent_loop_index = LOOM_CFG_LOOP_NONE,
      .entry_edge_index = entry_edge_index,
      .backedge_edge_index = backedge_edge_index,
      .is_canonical = true,
  };
  return true;
}

static void loom_cfg_loop_include_predecessor(uint16_t predecessor_index,
                                              bool* has_predecessor,
                                              uint16_t* predecessor_min,
                                              uint16_t* predecessor_max) {
  if (!*has_predecessor) {
    *has_predecessor = true;
    *predecessor_min = predecessor_index;
    *predecessor_max = predecessor_index;
    return;
  }
  *predecessor_min = iree_min(*predecessor_min, predecessor_index);
  *predecessor_max = iree_max(*predecessor_max, predecessor_index);
}

static void loom_cfg_loop_include_predecessor_range(
    bool source_has_predecessor, uint16_t source_predecessor_min,
    uint16_t source_predecessor_max, bool* target_has_predecessor,
    uint16_t* target_predecessor_min, uint16_t* target_predecessor_max) {
  if (!source_has_predecessor) return;
  loom_cfg_loop_include_predecessor(
      source_predecessor_min, target_has_predecessor, target_predecessor_min,
      target_predecessor_max);
  loom_cfg_loop_include_predecessor(
      source_predecessor_max, target_has_predecessor, target_predecessor_min,
      target_predecessor_max);
}

static iree_status_t loom_cfg_loop_forest_build_impl(
    const loom_cfg_graph_t* graph, iree_arena_allocator_t* transient_arena,
    iree_arena_allocator_t* arena, loom_cfg_loop_forest_t* out_forest) {
  const iree_host_size_t candidate_capacity =
      iree_min(graph->block_count, graph->backward_edge_count);
  loom_cfg_loop_build_interval_t* build_intervals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      transient_arena, candidate_capacity, sizeof(*build_intervals),
      (void**)&build_intervals));
  uint32_t* loop_by_header = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      transient_arena, graph->block_count, sizeof(*loop_by_header),
      (void**)&loop_by_header));
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    loop_by_header[i] = LOOM_CFG_LOOP_NONE;
  }

  iree_host_size_t interval_count = 0;
  iree_host_size_t reachable_backward_edge_count = 0;
  // Predecessor spans partition CFG edges, making discovery O(B+E).
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    loom_cfg_loop_interval_t interval = {0};
    if (!loom_cfg_loop_find_candidate(
            graph, (uint16_t)i, &reachable_backward_edge_count, &interval)) {
      continue;
    }
    IREE_ASSERT_LT(interval_count, candidate_capacity);
    const uint32_t loop_index = (uint32_t)interval_count++;
    loop_by_header[i] = loop_index;
    build_intervals[loop_index] = (loom_cfg_loop_build_interval_t){
        .interval = interval,
    };
  }
  out_forest->reachable_backward_edge_count = reachable_backward_edge_count;
  if (interval_count == 0) return iree_ok_status();

  uint32_t* innermost_loop_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      transient_arena, graph->block_count, sizeof(*innermost_loop_indices),
      (void**)&innermost_loop_indices));
  uint32_t* loop_stack = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(transient_arena, interval_count,
                                sizeof(*loop_stack), (void**)&loop_stack));

  iree_host_size_t loop_stack_count = 0;
  // Each interval is pushed and popped at most once, making this O(B+L).
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    while (
        loop_stack_count > 0 &&
        build_intervals[loop_stack[loop_stack_count - 1]].interval.latch_index <
            i) {
      --loop_stack_count;
    }
    const uint32_t loop_index = loop_by_header[i];
    if (loop_index != LOOM_CFG_LOOP_NONE) {
      loom_cfg_loop_interval_t* interval =
          &build_intervals[loop_index].interval;
      if (loop_stack_count > 0 &&
          build_intervals[loop_stack[loop_stack_count - 1]]
                  .interval.latch_index < interval->latch_index) {
        // Crossing intervals do not form a tree. Leave the forest empty so
        // consumers preserve exact point-local behavior.
        return iree_ok_status();
      }
      interval->parent_loop_index = loop_stack_count > 0
                                        ? loop_stack[loop_stack_count - 1]
                                        : LOOM_CFG_LOOP_NONE;
      loop_stack[loop_stack_count++] = loop_index;
    }
    innermost_loop_indices[i] = loop_stack_count > 0
                                    ? loop_stack[loop_stack_count - 1]
                                    : LOOM_CFG_LOOP_NONE;
  }

  // Predecessor spans partition CFG edges, so nested intervals do not cause
  // their contained blocks or edges to be revisited.
  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    const uint16_t block_index = (uint16_t)i;
    const uint32_t loop_index = innermost_loop_indices[block_index];
    if (loop_index == LOOM_CFG_LOOP_NONE ||
        !loom_cfg_graph_block_is_reachable(graph, block_index)) {
      continue;
    }
    loom_cfg_loop_build_interval_t* build_interval =
        &build_intervals[loop_index];
    const bool is_loop_header =
        block_index == build_interval->interval.header_index;
    const loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(graph, block_index);
    for (iree_host_size_t j = 0; j < predecessors.count; ++j) {
      const uint16_t predecessor_index = predecessors.values[j];
      if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index)) {
        continue;
      }
      if (is_loop_header) {
        loom_cfg_loop_include_predecessor(
            predecessor_index, &build_interval->has_header_predecessor,
            &build_interval->header_predecessor_min,
            &build_interval->header_predecessor_max);
      } else {
        loom_cfg_loop_include_predecessor(
            predecessor_index, &build_interval->has_body_predecessor,
            &build_interval->body_predecessor_min,
            &build_interval->body_predecessor_max);
      }
    }
  }

  // Children follow parents in header order. Folding each child once carries
  // every predecessor range through the forest in O(L).
  for (iree_host_size_t i = interval_count; i > 0; --i) {
    loom_cfg_loop_build_interval_t* build_interval = &build_intervals[i - 1];
    loom_cfg_loop_interval_t* interval = &build_interval->interval;
    if (build_interval->has_body_predecessor &&
        (build_interval->body_predecessor_min < interval->header_index ||
         build_interval->body_predecessor_max > interval->latch_index)) {
      interval->is_canonical = false;
    }
    if (interval->parent_loop_index != LOOM_CFG_LOOP_NONE) {
      loom_cfg_loop_build_interval_t* parent =
          &build_intervals[interval->parent_loop_index];
      loom_cfg_loop_include_predecessor_range(
          build_interval->has_header_predecessor,
          build_interval->header_predecessor_min,
          build_interval->header_predecessor_max, &parent->has_body_predecessor,
          &parent->body_predecessor_min, &parent->body_predecessor_max);
      loom_cfg_loop_include_predecessor_range(
          build_interval->has_body_predecessor,
          build_interval->body_predecessor_min,
          build_interval->body_predecessor_max, &parent->has_body_predecessor,
          &parent->body_predecessor_min, &parent->body_predecessor_max);
    }
  }

  loom_cfg_loop_interval_t* intervals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, interval_count, sizeof(*intervals), (void**)&intervals));
  uint32_t* retained_innermost_loop_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*retained_innermost_loop_indices),
      (void**)&retained_innermost_loop_indices));
  for (iree_host_size_t i = 0; i < interval_count; ++i) {
    intervals[i] = build_intervals[i].interval;
  }
  memcpy(retained_innermost_loop_indices, innermost_loop_indices,
         graph->block_count * sizeof(*retained_innermost_loop_indices));
  *out_forest = (loom_cfg_loop_forest_t){
      .intervals = intervals,
      .innermost_loop_indices = retained_innermost_loop_indices,
      .interval_count = interval_count,
      .reachable_backward_edge_count = reachable_backward_edge_count,
  };
  return iree_ok_status();
}

iree_status_t loom_cfg_loop_forest_build(const loom_cfg_graph_t* graph,
                                         iree_arena_allocator_t* arena,
                                         loom_cfg_loop_forest_t* out_forest) {
  if (!graph || !arena || !out_forest) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "CFG loop forest build requires a graph, arena, and output forest");
  }
  *out_forest = (loom_cfg_loop_forest_t){0};
  if (graph->malformed || graph->block_count <= 1 || graph->blocks == NULL ||
      graph->backward_edge_count == 0) {
    return iree_ok_status();
  }

  iree_arena_allocator_t transient_arena;
  iree_arena_initialize(arena->block_pool, &transient_arena);
  iree_status_t status = loom_cfg_loop_forest_build_impl(
      graph, &transient_arena, arena, out_forest);
  iree_arena_deinitialize(&transient_arena);
  return status;
}

bool loom_cfg_loop_forest_calculate_block_execution_counts(
    const loom_cfg_loop_forest_t* forest, const loom_cfg_graph_t* graph,
    const uint64_t* trip_counts, uint64_t* out_block_counts) {
  IREE_ASSERT(forest->interval_count == 0 || trip_counts != NULL);

  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    const bool is_reachable =
        graph->blocks == NULL ||
        loom_cfg_graph_block_is_reachable(graph, (uint16_t)i);
    out_block_counts[i] = is_reachable ? 1 : 0;
    if (!is_reachable || forest->interval_count == 0) continue;
    const uint32_t loop_index = forest->innermost_loop_indices[i];
    if (loop_index != LOOM_CFG_LOOP_NONE &&
        forest->intervals[loop_index].header_index != i &&
        graph->blocks[i].successor_count > 1) {
      return false;
    }
  }
  if (forest->interval_count != forest->reachable_backward_edge_count) {
    return false;
  }
  if (forest->interval_count == 0) {
    return true;
  }

  // Use each loop header's output slot as its body execution count until all
  // non-header blocks have consumed it below.
  for (iree_host_size_t i = 0; i < forest->interval_count; ++i) {
    const loom_cfg_loop_interval_t* interval = &forest->intervals[i];
    if (!interval->is_canonical) return false;
    const uint64_t parent_body_count =
        interval->parent_loop_index == LOOM_CFG_LOOP_NONE
            ? 1
            : out_block_counts[forest->intervals[interval->parent_loop_index]
                                   .header_index];
    if (!iree_checked_mul_u64(parent_body_count, trip_counts[i],
                              &out_block_counts[interval->header_index])) {
      return false;
    }
  }

  for (iree_host_size_t i = 0; i < graph->block_count; ++i) {
    if (!loom_cfg_graph_block_is_reachable(graph, (uint16_t)i)) continue;
    const uint32_t loop_index = forest->innermost_loop_indices[i];
    if (loop_index == LOOM_CFG_LOOP_NONE ||
        forest->intervals[loop_index].header_index == i) {
      continue;
    }
    out_block_counts[i] =
        out_block_counts[forest->intervals[loop_index].header_index];
  }

  // Children follow parents in interval order. Finalizing in reverse preserves
  // each parent's temporary body count until all child headers consume it.
  for (iree_host_size_t i = forest->interval_count; i > 0; --i) {
    const loom_cfg_loop_interval_t* interval = &forest->intervals[i - 1];
    const uint64_t parent_body_count =
        interval->parent_loop_index == LOOM_CFG_LOOP_NONE
            ? 1
            : out_block_counts[forest->intervals[interval->parent_loop_index]
                                   .header_index];
    uint64_t header_trip_count = 0;
    if (!iree_checked_add_u64(trip_counts[i - 1], 1, &header_trip_count) ||
        !iree_checked_mul_u64(parent_body_count, header_trip_count,
                              &out_block_counts[interval->header_index])) {
      return false;
    }
  }
  return true;
}
