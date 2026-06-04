// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// DAG scheduling algorithm: topological sort, partitioning, workstream
// detection. Pure algorithm with no binding-layer dependencies.

#include "hrx_internal.h"
#include "iree/base/internal/math.h"

#ifndef IREE_PREFETCH_RO
#if defined(__GNUC__) || defined(__clang__)
#define IREE_PREFETCH_RO(ptr, locality) \
  __builtin_prefetch((ptr), /*rw=*/0, locality)
#else
#define IREE_PREFETCH_RO(ptr, locality) ((void)0)
#endif
#endif

//===----------------------------------------------------------------------===//
// Tuning parameters
//===----------------------------------------------------------------------===//

#define HRX_GRAPH_MIN_PARTITION_SIZE_FOR_STREAMS 16
#define HRX_GRAPH_MAX_PARTITION_SIZE 2048
#define HRX_GRAPH_MAX_WORKSTREAMS 8
#define HRX_GRAPH_MIN_NODES_PER_STREAM 32
#define HRX_GRAPH_MAX_REACHABILITY_BITS 64
#define HRX_GRAPH_MAX_FAN_OUT 8

//===----------------------------------------------------------------------===//
// Phase 1: Preparation
//===----------------------------------------------------------------------===//

static bool hrx_graph_prepare_nodes(hrx_graph_node_block_t* node_blocks,
                                    iree_host_size_t node_count,
                                    hrx_graph_sort_node_t* sort_nodes,
                                    uint32_t* node_index_map) {
  uint32_t index = 0;
  for (hrx_graph_node_block_t* block = node_blocks; block;
       block = block->next) {
    if (block->next) {
      IREE_PREFETCH_RO(block->next, 1);
    }
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      hrx_graph_node_s* node = block->nodes[i];
      sort_nodes[index] = (hrx_graph_sort_node_t){
          .original_index = index,
          .sorted_index = index,
          .max_dependency_index = 0,
          .partition_id = 0,
          .in_degree = 0,
          .type = (uint8_t)node->type,
          .stream_id = 0,
          .node = node,
      };
      node_index_map[node->node_index] = index;
      ++index;
    }
  }

  bool is_sorted = true;
  for (uint32_t i = 0; is_sorted && i < node_count; ++i) {
    hrx_graph_node_s* node = sort_nodes[i].node;
    if (node->dependency_count > 0) {
      for (uint32_t j = 0; j < node->dependency_count; ++j) {
        uint32_t dep_index = node_index_map[node->dependencies[j]->node_index];
        if (dep_index >= i) {
          is_sorted = false;
          break;
        }
      }
    }
  }

  return is_sorted;
}

//===----------------------------------------------------------------------===//
// Phase 2: Topological sort
//===----------------------------------------------------------------------===//

static iree_status_t hrx_graph_topological_sort(
    hrx_graph_sort_node_t* nodes, uint32_t node_count, uint32_t* node_index_map,
    hrx_graph_edge_t* additional_edges, iree_arena_allocator_t* arena,
    bool is_already_sorted) {
  if (is_already_sorted && !additional_edges) {
    for (uint32_t i = 0; i < node_count; ++i) {
      uint32_t max_dep = 0;
      for (uint32_t j = 0; j < nodes[i].node->dependency_count; ++j) {
        const uint32_t dep_index =
            node_index_map[nodes[i].node->dependencies[j]->node_index];
        if (dep_index < i && dep_index != UINT32_MAX) {
          max_dep = iree_max(max_dep, dep_index);
        }
      }
      nodes[i].max_dependency_index = max_dep;
    }
    return iree_ok_status();
  }

  uint32_t* queue = NULL;
  const iree_host_size_t queue_size = node_count * sizeof(*queue);
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, queue_size, (void**)&queue));

  for (uint32_t i = 0; i < node_count; ++i) {
    nodes[i].in_degree = (uint16_t)nodes[i].node->dependency_count;
  }

  hrx_graph_edge_t* edge = additional_edges;
  while (edge) {
    uint32_t to_index = node_index_map[edge->to->node_index];
    if (to_index < node_count) {
      ++nodes[to_index].in_degree;
    }
    edge = edge->next;
  }

  uint32_t queue_head = 0;
  uint32_t queue_tail = 0;
  for (uint32_t i = 0; i < node_count; ++i) {
    if (nodes[i].in_degree == 0) {
      queue[queue_tail++] = i;
    }
  }

  uint32_t sorted_count = 0;
  while (queue_head < queue_tail) {
    uint32_t current = queue[queue_head++];
    nodes[current].sorted_index = sorted_count++;

    uint32_t max_dep = 0;
    for (uint32_t j = 0; j < nodes[current].node->dependency_count; ++j) {
      uint32_t dep_index =
          node_index_map[nodes[current].node->dependencies[j]->node_index];
      if (dep_index != UINT32_MAX) {
        uint32_t dep_sorted_index = nodes[dep_index].sorted_index;
        max_dep = iree_max(max_dep, dep_sorted_index);
      }
    }

    edge = additional_edges;
    while (edge) {
      if (edge->to == nodes[current].node) {
        uint32_t from_index = node_index_map[edge->from->node_index];
        if (from_index != UINT32_MAX) {
          uint32_t from_sorted_index = nodes[from_index].sorted_index;
          max_dep = iree_max(max_dep, from_sorted_index);
        }
      }
      edge = edge->next;
    }
    nodes[current].max_dependency_index = max_dep;

    for (uint32_t i = 0; i < node_count; ++i) {
      if (i == current) continue;
      hrx_graph_node_s* node = nodes[i].node;
      for (uint32_t j = 0; j < node->dependency_count; ++j) {
        if (node->dependencies[j] == nodes[current].node) {
          if (--nodes[i].in_degree == 0) {
            queue[queue_tail++] = i;
          }
          break;
        }
      }
    }

    edge = additional_edges;
    while (edge) {
      if (edge->from == nodes[current].node) {
        uint32_t to_index = node_index_map[edge->to->node_index];
        if (to_index < node_count && to_index != current) {
          if (--nodes[to_index].in_degree == 0) {
            queue[queue_tail++] = to_index;
          }
        }
      }
      edge = edge->next;
    }
  }

  if (sorted_count != node_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "graph contains cycles (%u nodes processed of %u)",
                            sorted_count, node_count);
  }

  hrx_graph_sort_node_t temp;
  for (uint32_t i = 0; i < node_count; ++i) {
    while (nodes[i].sorted_index != i) {
      uint32_t target = nodes[i].sorted_index;
      temp = nodes[target];
      nodes[target] = nodes[i];
      nodes[i] = temp;
    }
  }

  for (uint32_t i = 0; i < node_count; ++i) {
    node_index_map[nodes[i].node->node_index] = i;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Phase 3: Partitioning with workstream detection
//===----------------------------------------------------------------------===//

typedef struct hrx_graph_stream_state_t {
  uint64_t reachability_mask;
  uint32_t last_node_index;
} hrx_graph_stream_state_t;

typedef struct hrx_uint32x2_t {
  uint32_t values[2];
} hrx_uint32x2_t;

static hrx_uint32x2_t hrx_graph_partition_with_streams(
    hrx_graph_sort_node_t* nodes, uint32_t node_count, uint32_t* node_index_map,
    hrx_graph_partition_t* partitions) {
  uint32_t partition_count = 0;
  uint32_t block_count = 0;

  hrx_graph_stream_state_t streams[HRX_GRAPH_MAX_WORKSTREAMS];
  uint8_t active_streams = 0;

  for (uint32_t i = 0; i < node_count;) {
    bool is_recordable = hrx_graph_node_is_recordable(nodes[i].type);
    if (!is_recordable) {
      hrx_graph_partition_type_t partition_type;
      switch (nodes[i].type) {
        case HRX_GRAPH_NODE_TYPE_INTERNAL_EMPTY:
          partition_type = HRX_GRAPH_PARTITION_TYPE_EMPTY;
          break;
        case HRX_GRAPH_NODE_TYPE_INTERNAL_HOST_CALL:
          partition_type = HRX_GRAPH_PARTITION_TYPE_HOST_CALL;
          break;
        default:
          partition_type = HRX_GRAPH_PARTITION_TYPE_EMPTY;
          break;
      }
      partitions[partition_count] = (hrx_graph_partition_t){
          .start_index = i,
          .count = 1,
          .type = partition_type,
          .stream_count = 1,
      };
      nodes[i].partition_id = partition_count;
      nodes[i].stream_id = 0;
      ++partition_count;
      ++block_count;
      ++i;
      active_streams = 0;
    } else {
      uint32_t recordable_start = i;
      memset(streams, 0, sizeof(streams));
      active_streams = 0;

      uint32_t partition_size = 0;
      while (i < node_count && partition_size < HRX_GRAPH_MAX_PARTITION_SIZE &&
             hrx_graph_node_is_recordable(nodes[i].type)) {
        bool deps_satisfied = true;
        for (uint32_t j = 0; j < nodes[i].node->dependency_count; ++j) {
          uint32_t dep_index =
              node_index_map[nodes[i].node->dependencies[j]->node_index];
          if (dep_index >= recordable_start && dep_index < i) {
            // OK
          } else if (dep_index >= i) {
            deps_satisfied = false;
            break;
          }
        }
        if (!deps_satisfied) break;

        const bool use_workstreams =
            (i - recordable_start) >= HRX_GRAPH_MIN_PARTITION_SIZE_FOR_STREAMS;

        uint8_t assigned_stream = 0;
        uint8_t connected_streams = 0;

        for (uint32_t j = 0; j < nodes[i].node->dependency_count; ++j) {
          uint32_t dep_index =
              node_index_map[nodes[i].node->dependencies[j]->node_index];
          if (dep_index >= recordable_start && dep_index < i) {
            uint8_t dep_stream = nodes[dep_index].stream_id;
            connected_streams |= (1 << dep_stream);
          }
        }

        if (use_workstreams && connected_streams == 0 &&
            active_streams < HRX_GRAPH_MAX_WORKSTREAMS) {
          uint32_t remaining_in_partition =
              HRX_GRAPH_MAX_PARTITION_SIZE - partition_size;
          if (remaining_in_partition >= HRX_GRAPH_MIN_NODES_PER_STREAM) {
            assigned_stream = active_streams++;
            streams[assigned_stream].last_node_index = i;
            if ((i - recordable_start) < HRX_GRAPH_MAX_REACHABILITY_BITS) {
              streams[assigned_stream].reachability_mask =
                  1ULL << (i - recordable_start);
            }
          } else {
            assigned_stream = 0;
            if (active_streams == 0) active_streams = 1;
          }
        } else if (use_workstreams &&
                   iree_math_count_ones_u32(connected_streams) == 1) {
          assigned_stream =
              iree_math_count_trailing_zeros_u32(connected_streams);
          streams[assigned_stream].last_node_index = i;
          if ((i - recordable_start) < HRX_GRAPH_MAX_REACHABILITY_BITS) {
            streams[assigned_stream].reachability_mask |=
                1ULL << (i - recordable_start);
          }
        } else {
          const uint32_t dep_count =
              iree_math_count_ones_u32(connected_streams);
          const bool is_sync_point =
              (nodes[i].node->dependency_count > HRX_GRAPH_MAX_FAN_OUT);
          if (!use_workstreams || dep_count > 1 || is_sync_point) {
            assigned_stream = 0;
            for (uint32_t k = recordable_start; k < i; ++k) {
              nodes[k].stream_id = 0;
            }
            active_streams = 1;
          } else {
            assigned_stream = 0;
            if (active_streams == 0) active_streams = 1;
          }
        }

        nodes[i].partition_id = partition_count;
        nodes[i].stream_id = assigned_stream;
        ++i;
        ++partition_size;
      }

      const uint32_t stream_count = active_streams > 0 ? active_streams : 1;
      partitions[partition_count] = (hrx_graph_partition_t){
          .start_index = recordable_start,
          .count = i - recordable_start,
          .type = HRX_GRAPH_PARTITION_TYPE_RECORDABLE,
          .stream_count = stream_count,
      };
      ++partition_count;
      block_count += stream_count;
    }
  }

  return (hrx_uint32x2_t){{partition_count, block_count}};
}

//===----------------------------------------------------------------------===//
// Main entry point
//===----------------------------------------------------------------------===//

iree_status_t hrx_graph_schedule_nodes(hrx_graph_node_block_t* node_blocks,
                                       iree_host_size_t node_count,
                                       hrx_graph_edge_t* additional_edges,
                                       iree_arena_allocator_t* arena,
                                       hrx_graph_schedule_t* out_schedule) {
  IREE_ASSERT_ARGUMENT(out_schedule);

  if (node_count == 0) {
    memset(out_schedule, 0, sizeof(*out_schedule));
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, node_count);

  hrx_graph_sort_node_t* sorted_nodes = NULL;
  const iree_host_size_t sorted_nodes_size = node_count * sizeof(*sorted_nodes);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, sorted_nodes_size, (void**)&sorted_nodes));

  uint32_t* node_index_map = NULL;
  const iree_host_size_t map_size = node_count * sizeof(*node_index_map);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, map_size, (void**)&node_index_map));

  const bool is_sorted = hrx_graph_prepare_nodes(node_blocks, node_count,
                                                 sorted_nodes, node_index_map);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, is_sorted);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, hrx_graph_topological_sort(sorted_nodes, node_count, node_index_map,
                                     additional_edges, arena, is_sorted));

  hrx_graph_partition_t* partitions = NULL;
  const iree_host_size_t partitions_size = node_count * sizeof(*partitions);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, partitions_size, (void**)&partitions));

  const hrx_uint32x2_t partition_block_counts =
      hrx_graph_partition_with_streams(sorted_nodes, node_count, node_index_map,
                                       partitions);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, partition_block_counts.values[0]);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, partition_block_counts.values[1]);

  out_schedule->sorted_nodes = sorted_nodes;
  out_schedule->node_index_map = node_index_map;
  out_schedule->partitions = partitions;
  out_schedule->partition_count = partition_block_counts.values[0];
  out_schedule->block_count = partition_block_counts.values[1];

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
