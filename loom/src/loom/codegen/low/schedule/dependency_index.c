// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/dependency_index.h"

#include <string.h>

static iree_status_t loom_low_schedule_dependency_index_allocate_segments(
    loom_segmented_storage_t* storage, uint32_t row_count,
    uint32_t segment_capacity, iree_arena_allocator_t* arena) {
  const uint32_t segment_count =
      row_count == 0 ? 0 : 1 + (row_count - 1) / segment_capacity;
  for (uint32_t segment_index = 0; segment_index < segment_count;
       ++segment_index) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(storage, arena, &segment));
  }
  return iree_ok_status();
}

static uint32_t* loom_low_schedule_dependency_detail_index_mutable_at(
    loom_low_schedule_dependency_detail_index_t* detail_index,
    uint32_t detail_ordinal) {
  IREE_ASSERT_LT(detail_ordinal, detail_index->dependency_count);
  loom_low_schedule_dependency_detail_segment_t* segment =
      (loom_low_schedule_dependency_detail_segment_t*)
          loom_segmented_storage_segment(
              &detail_index->dependency_indices,
              detail_ordinal >>
                  LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_SHIFT);
  return &segment->rows[detail_ordinal &
                        LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_MASK];
}

static loom_low_schedule_dependency_group_t*
loom_low_schedule_dependency_index_mutable_group_at(
    loom_low_schedule_dependency_index_t* index, uint32_t group_index) {
  IREE_ASSERT_LT(group_index, index->group_count);
  loom_low_schedule_dependency_group_segment_t* segment =
      (loom_low_schedule_dependency_group_segment_t*)
          loom_segmented_storage_segment(
              &index->groups,
              group_index >> LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_SHIFT);
  return &segment->rows[group_index &
                        LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_MASK];
}

static void loom_low_schedule_dependency_index_count_raw_dependencies(
    const loom_low_schedule_dependency_graph_t* graph, uint32_t node_count,
    uint32_t* producer_dependency_starts, uint32_t* out_indegrees) {
  for (uint32_t dependency_index = 0; dependency_index < graph->count;
       ++dependency_index) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(graph, dependency_index);
    IREE_ASSERT_LT(dependency->producer_node, node_count);
    IREE_ASSERT_LT(dependency->consumer_node, node_count);
    IREE_ASSERT_NE(producer_dependency_starts[dependency->producer_node + 1],
                   UINT32_MAX);
    IREE_ASSERT_NE(out_indegrees[dependency->consumer_node], UINT32_MAX);
    ++producer_dependency_starts[dependency->producer_node + 1];
    ++out_indegrees[dependency->consumer_node];
  }
  for (uint32_t producer_node = 0; producer_node < node_count;
       ++producer_node) {
    IREE_ASSERT_LE(producer_dependency_starts[producer_node],
                   UINT32_MAX - producer_dependency_starts[producer_node + 1]);
    producer_dependency_starts[producer_node + 1] +=
        producer_dependency_starts[producer_node];
  }
}

static void loom_low_schedule_dependency_index_scatter_raw_dependencies(
    const loom_low_schedule_dependency_graph_t* graph, uint32_t node_count,
    const uint32_t* producer_dependency_starts, uint32_t* producer_cursors,
    loom_low_schedule_dependency_detail_index_t* detail_index) {
  memcpy(producer_cursors, producer_dependency_starts,
         node_count * sizeof(*producer_cursors));
  for (uint32_t dependency_index = 0; dependency_index < graph->count;
       ++dependency_index) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(graph, dependency_index);
    *loom_low_schedule_dependency_detail_index_mutable_at(
        detail_index, producer_cursors[dependency->producer_node]++) =
        dependency_index;
  }
}

static uint32_t loom_low_schedule_dependency_index_count_groups(
    const loom_low_schedule_dependency_graph_t* graph, uint32_t node_count,
    const uint32_t* producer_dependency_starts,
    const loom_low_schedule_dependency_detail_index_t* detail_index,
    uint32_t* last_producer_nodes, uint32_t* producer_group_starts) {
  if (node_count != 0) {
    memset(last_producer_nodes, 0xFF,
           node_count * sizeof(*last_producer_nodes));
  }
  uint32_t group_count = 0;
  producer_group_starts[0] = 0;
  for (uint32_t producer_node = 0; producer_node < node_count;
       ++producer_node) {
    const uint32_t dependency_begin = producer_dependency_starts[producer_node];
    const uint32_t dependency_end =
        producer_dependency_starts[producer_node + 1];
    for (uint32_t i = dependency_begin; i < dependency_end; ++i) {
      const loom_low_schedule_dependency_t* dependency =
          loom_low_schedule_dependency_graph_at(
              graph,
              loom_low_schedule_dependency_detail_index_at(detail_index, i));
      const uint32_t consumer_node = dependency->consumer_node;
      if (last_producer_nodes[consumer_node] == producer_node) continue;
      last_producer_nodes[consumer_node] = producer_node;
      IREE_ASSERT_NE(group_count, UINT32_MAX);
      ++group_count;
    }
    producer_group_starts[producer_node + 1] = group_count;
  }
  return group_count;
}

static void loom_low_schedule_dependency_index_fill_groups(
    const loom_low_schedule_dependency_graph_t* graph, uint32_t node_count,
    const uint32_t* producer_dependency_starts,
    const loom_low_schedule_dependency_detail_index_t* detail_index,
    const uint32_t* producer_group_starts, uint32_t* last_producer_nodes,
    uint32_t* consumer_group_indices,
    loom_low_schedule_dependency_index_t* index) {
  memset(last_producer_nodes, 0xFF, node_count * sizeof(*last_producer_nodes));
  for (uint32_t producer_node = 0; producer_node < node_count;
       ++producer_node) {
    uint32_t next_group_index = producer_group_starts[producer_node];
    const uint32_t dependency_begin = producer_dependency_starts[producer_node];
    const uint32_t dependency_end =
        producer_dependency_starts[producer_node + 1];
    for (uint32_t i = dependency_begin; i < dependency_end; ++i) {
      const loom_low_schedule_dependency_t* dependency =
          loom_low_schedule_dependency_graph_at(
              graph,
              loom_low_schedule_dependency_detail_index_at(detail_index, i));
      const uint32_t consumer_node = dependency->consumer_node;
      if (last_producer_nodes[consumer_node] != producer_node) {
        last_producer_nodes[consumer_node] = producer_node;
        consumer_group_indices[consumer_node] = next_group_index;
        *loom_low_schedule_dependency_index_mutable_group_at(
            index, next_group_index++) = (loom_low_schedule_dependency_group_t){
            .consumer_node = consumer_node,
            .minimum_issue_separation_cycles = INT32_MIN,
        };
      }
      const uint32_t group_index = consumer_group_indices[consumer_node];
      loom_low_schedule_dependency_group_t* group =
          loom_low_schedule_dependency_index_mutable_group_at(index,
                                                              group_index);
      IREE_ASSERT_NE(group->dependency_count, UINT32_MAX);
      ++group->dependency_count;
      group->minimum_issue_separation_cycles =
          iree_max(group->minimum_issue_separation_cycles,
                   dependency->minimum_issue_separation_cycles);
      if (dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
        index->ssa_group_bits[group_index >> 3] |=
            (uint8_t)(1u << (group_index & 7u));
      } else if (dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT) {
        index->effect_group_bits[group_index >> 3] |=
            (uint8_t)(1u << (group_index & 7u));
      }
    }
    IREE_ASSERT_EQ(next_group_index, producer_group_starts[producer_node + 1]);
  }
}

iree_status_t loom_low_schedule_dependency_index_initialize(
    const loom_low_schedule_dependency_graph_t* graph, uint32_t node_count,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* arena,
    uint32_t* out_indegrees, loom_low_schedule_dependency_index_t* out_index,
    loom_low_schedule_dependency_detail_index_t* out_detail_index) {
  IREE_ASSERT_ARGUMENT(graph);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_index);
  IREE_ASSERT_ARGUMENT(out_detail_index);
  IREE_ASSERT_LE(graph->count, UINT32_MAX);
  IREE_ASSERT(node_count == 0 || out_indegrees != NULL);
  *out_index = (loom_low_schedule_dependency_index_t){
      .node_count = node_count,
  };
  *out_detail_index = (loom_low_schedule_dependency_detail_index_t){
      .dependency_count = (uint32_t)graph->count,
  };
  loom_segmented_storage_initialize(
      sizeof(loom_low_schedule_dependency_group_segment_t),
      iree_alignof(loom_low_schedule_dependency_group_segment_t),
      &out_index->groups);
  loom_segmented_storage_initialize(
      sizeof(loom_low_schedule_dependency_detail_segment_t),
      iree_alignof(loom_low_schedule_dependency_detail_segment_t),
      &out_detail_index->dependency_indices);

  const iree_host_size_t node_sentinel_count = (iree_host_size_t)node_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_sentinel_count, sizeof(*out_index->producer_group_starts),
      (void**)&out_index->producer_group_starts));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, node_sentinel_count,
      sizeof(*out_detail_index->producer_dependency_starts),
      (void**)&out_detail_index->producer_dependency_starts));
  memset(out_index->producer_group_starts, 0,
         node_sentinel_count * sizeof(*out_index->producer_group_starts));
  memset(out_detail_index->producer_dependency_starts, 0,
         node_sentinel_count *
             sizeof(*out_detail_index->producer_dependency_starts));
  if (node_count != 0) {
    memset(out_indegrees, 0, node_count * sizeof(*out_indegrees));
  }

  loom_low_schedule_dependency_index_count_raw_dependencies(
      graph, node_count, out_detail_index->producer_dependency_starts,
      out_indegrees);
  const uint32_t dependency_count = (uint32_t)graph->count;
  uint32_t* producer_cursors = NULL;
  uint32_t* last_producer_nodes = NULL;
  if (node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, node_count,
                                                   sizeof(*producer_cursors),
                                                   (void**)&producer_cursors));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, node_count, sizeof(*last_producer_nodes),
        (void**)&last_producer_nodes));
  }
  if (dependency_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_index_allocate_segments(
        &out_detail_index->dependency_indices, dependency_count,
        LOOM_LOW_SCHEDULE_DEPENDENCY_DETAIL_SEGMENT_CAPACITY, scratch_arena));
    loom_low_schedule_dependency_index_scatter_raw_dependencies(
        graph, node_count, out_detail_index->producer_dependency_starts,
        producer_cursors, out_detail_index);
  }

  out_index->group_count = loom_low_schedule_dependency_index_count_groups(
      graph, node_count, out_detail_index->producer_dependency_starts,
      out_detail_index, last_producer_nodes, out_index->producer_group_starts);
  if (out_index->group_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_index_allocate_segments(
      &out_index->groups, out_index->group_count,
      LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_SEGMENT_CAPACITY, arena));
  const iree_host_size_t group_bitset_size =
      ((iree_host_size_t)out_index->group_count + 7) / 8;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_bitset_size, sizeof(*out_index->ssa_group_bits),
      (void**)&out_index->ssa_group_bits));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_bitset_size, sizeof(*out_index->effect_group_bits),
      (void**)&out_index->effect_group_bits));
  memset(out_index->ssa_group_bits, 0, group_bitset_size);
  memset(out_index->effect_group_bits, 0, group_bitset_size);
  loom_low_schedule_dependency_index_fill_groups(
      graph, node_count, out_detail_index->producer_dependency_starts,
      out_detail_index, out_index->producer_group_starts, last_producer_nodes,
      producer_cursors, out_index);
  return iree_ok_status();
}

iree_status_t loom_low_schedule_dependency_frontier_initialize(
    const loom_low_schedule_dependency_index_t* index,
    iree_arena_allocator_t* arena,
    loom_low_schedule_dependency_frontier_t* out_frontier) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_frontier);
  *out_frontier = (loom_low_schedule_dependency_frontier_t){
      .node_count = index->node_count,
  };
  if (index->node_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, index->node_count,
      sizeof(*out_frontier->remaining_producer_counts),
      (void**)&out_frontier->remaining_producer_counts));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, index->node_count, sizeof(*out_frontier->remaining_producer_xors),
      (void**)&out_frontier->remaining_producer_xors));
  memset(out_frontier->remaining_producer_counts, 0,
         index->node_count * sizeof(*out_frontier->remaining_producer_counts));
  memset(out_frontier->remaining_producer_xors, 0,
         index->node_count * sizeof(*out_frontier->remaining_producer_xors));
  for (uint32_t producer_node = 0; producer_node < index->node_count;
       ++producer_node) {
    const uint32_t group_begin =
        loom_low_schedule_dependency_index_group_begin(index, producer_node);
    const uint32_t group_end =
        loom_low_schedule_dependency_index_group_end(index, producer_node);
    for (uint32_t group_index = group_begin; group_index < group_end;
         ++group_index) {
      const uint32_t consumer_node =
          loom_low_schedule_dependency_index_group_at(index, group_index)
              ->consumer_node;
      IREE_ASSERT_NE(out_frontier->remaining_producer_counts[consumer_node],
                     UINT32_MAX);
      ++out_frontier->remaining_producer_counts[consumer_node];
      out_frontier->remaining_producer_xors[consumer_node] ^= producer_node;
    }
  }
  return iree_ok_status();
}

uint32_t loom_low_schedule_dependency_frontier_consume_group(
    loom_low_schedule_dependency_frontier_t* frontier, uint32_t producer_node,
    const loom_low_schedule_dependency_group_t* group) {
  const uint32_t consumer_node = group->consumer_node;
  IREE_ASSERT_LT(consumer_node, frontier->node_count);
  uint32_t* remaining_count =
      &frontier->remaining_producer_counts[consumer_node];
  IREE_ASSERT_NE(*remaining_count, 0u);
  --*remaining_count;
  frontier->remaining_producer_xors[consumer_node] ^= producer_node;
  ++frontier->consumed_group_count;
  return *remaining_count == 1
             ? frontier->remaining_producer_xors[consumer_node]
             : LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE;
}
