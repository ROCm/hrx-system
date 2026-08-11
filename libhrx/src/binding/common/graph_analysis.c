// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"
#include "common/internal.h"
#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// Tuning Parameters and Heuristics
//===----------------------------------------------------------------------===//

// Minimum number of nodes in a partition to consider workstream detection.
// Smaller partitions don't benefit from the overhead of multiple streams.
#define IREE_HAL_STREAMING_GRAPH_MIN_PARTITION_SIZE_FOR_STREAMS 16

// Maximum number of nodes in a single partition before forcing a split.
// Prevents command buffers from growing too large.
#define IREE_HAL_STREAMING_GRAPH_MAX_PARTITION_SIZE 2048

// Maximum number of concurrent workstreams to detect within a partition.
// Hardware typically has limited concurrent execution contexts.
#define IREE_HAL_STREAMING_GRAPH_MAX_WORKSTREAMS 8

// Minimum nodes per workstream to justify creating separate streams.
// Each stream has overhead, so ensure sufficient work.
#define IREE_HAL_STREAMING_GRAPH_MIN_NODES_PER_STREAM 32

// Maximum reachability mask bits for efficient workstream detection.
// Beyond this, fall back to single stream for simplicity.
#define IREE_HAL_STREAMING_GRAPH_MAX_REACHABILITY_BITS 64

// Threshold for "small" graphs that use simplified algorithms.
#define IREE_HAL_STREAMING_GRAPH_SMALL_GRAPH_THRESHOLD 32

// Maximum dependency fan-in before considering a node a synchronization point.
#define IREE_HAL_STREAMING_GRAPH_MAX_FAN_IN 8

//===----------------------------------------------------------------------===//
// DAG Scheduling Algorithm Overview
//===----------------------------------------------------------------------===//
//
// This file implements an efficient three-phase algorithm for transforming a
// directed acyclic graph (DAG) of heterogeneous nodes into an optimized
// execution schedule with automatic workstream detection.
//
// Algorithm phases:
// 1. PREPARE: Linearize nodes and detect if already topologically sorted
// 2. SORT: Perform topological sorting if needed (with fast path)
// 3. PARTITION: Group nodes into executable blocks with workstream detection
//
// Example 1: Linear Stream with Small Concurrency
// ================================================
// Input DAG:
//   K1 ---> K2 ---> K3 ---> H1 ---> K4 ---> K5
//            \                 /
//             M1 -------------+
//
// Node types: K=Kernel(recordable), M=Memcpy(recordable),
// H=Host(non-recordable)
//
// After scheduling:
// +-------------------------------------+
// | Partition 0: RECORDABLE (2 streams) |
// |  Stream 0: K1 -> K2 -> K3           |
// |  Stream 1: M1                       |
// +-------------------------------------+
// +-------------------------------------+
// | Partition 1: HOST (1 stream)        |
// |  Stream 0: H1                       |
// +-------------------------------------+
// +-------------------------------------+
// | Partition 2: RECORDABLE (1 stream)  |
// |  Stream 0: K4 -> K5                 |
// +-------------------------------------+
//
// Efficiency: O(N + E) where N=6 nodes, E=6 edges
// Memory: 6 * 24 bytes (sort nodes) + 3 * 16 bytes (partitions) = 192 bytes
//
// Example 2: Multiple Concurrent Streams
// =======================================
// Input DAG:
//   K1 ---> K2 ---> K5 ---> H1 ---> K9
//    |               \              ^
//    +--> M1 --> M2 --> K6 ---------+
//    |                              |
//    +--> K3 --> K4 --> K7 --> K8 --+
//
// After scheduling:
// +-------------------------------------+
// | Partition 0: RECORDABLE (3 streams) |
// |  Stream 0: K1 -> K2 -> K5 -> K6     |
// |  Stream 1: M1 -> M2                 |
// |  Stream 2: K3 -> K4 -> K7 -> K8     |
// +-------------------------------------+
// +-------------------------------------+
// | Partition 1: HOST (1 stream)        |
// |  Stream 0: H1                       |
// +-------------------------------------+
// +-------------------------------------+
// | Partition 2: RECORDABLE (1 stream)  |
// |  Stream 0: K9                       |
// +-------------------------------------+
//
// Efficiency: O(N + E) where N=10 nodes, E=11 edges
// Memory: 10 * 24 bytes (sort nodes) + 3 * 16 bytes (partitions) = 288 bytes
//
// Performance characteristics:
// - Time Complexity: O(N + E) for all phases
// - Space Complexity: O(N) with 24 bytes per node + O(P) partitions
// - Cache Behavior: Sequential access patterns, prefetch-friendly
// - Scalability: Handles 1-100,000 nodes efficiently within the ordering and
//                mutability constraints of the graph API.
//
// Worst case scenarios:
// 1. All non-recordable nodes: P = N partitions, O(N) space
// 2. Fully connected DAG: E = N*(N-1)/2 edges, O(N²) edge processing
// 3. Deep linear chain: No parallelism opportunities
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Phase 1: Preparation - Linearize and Detect Sorting
//===----------------------------------------------------------------------===//

typedef struct iree_hal_streaming_graph_prepare_result_t {
  // Number of active nodes copied into the sort array.
  uint32_t active_node_count;
  // True if active nodes are already in topological order.
  bool is_sorted;
} iree_hal_streaming_graph_prepare_result_t;

static uint32_t iree_hal_streaming_graph_node_map_lookup(
    const uint32_t* node_index_map, iree_host_size_t node_index_map_count,
    const iree_hal_streaming_graph_node_t* node) {
  if (!node || node->node_index >= node_index_map_count) return UINT32_MAX;
  return node_index_map[node->node_index];
}

// Linearizes nodes from chained blocks and detects if already sorted.
// Returns true if nodes are already in topological order.
//
// Algorithm:
// 1. Walk chained blocks copying to sort_nodes array
// 2. Build index mapping from node_index to sort_nodes position
// 3. For each node, check if all dependencies have lower indices
// 4. Cache node type to avoid indirection later
//
// Complexity: O(N * avg_deps) ~= O(N) for sparse graphs
static iree_hal_streaming_graph_prepare_result_t
iree_hal_streaming_graph_prepare_nodes(
    iree_hal_streaming_node_block_t* node_blocks, const uint8_t* disabled_nodes,
    iree_host_size_t disabled_node_count,
    iree_hal_streaming_graph_sort_node_t* sort_nodes, uint32_t* node_index_map,
    iree_host_size_t node_index_map_count) {
  for (iree_host_size_t i = 0; i < node_index_map_count; ++i) {
    node_index_map[i] = UINT32_MAX;
  }

  // Linearize from chained blocks.
  uint32_t index = 0;
  for (iree_hal_streaming_node_block_t* block = node_blocks; block;
       block = block->next) {
    if (block->next) {
      IREE_BUILTIN_PREFETCH_RO(block->next, IREE_BUILTIN_PREFETCH_LOCALITY_L3);
    }

    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      const bool is_disabled =
          (node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_DISABLED) != 0 ||
          (disabled_nodes && node->node_index < disabled_node_count &&
           disabled_nodes[node->node_index]);
      // Initialize all fields explicitly because arena allocation does not
      // guarantee zeroed storage.
      sort_nodes[index] = (iree_hal_streaming_graph_sort_node_t){
          .original_index = index,
          .sorted_index = index,  // Initially assume sorted.
          .max_dependency_index = 0,
          .partition_id = 0,
          .in_degree = 0,
          .type = is_disabled ? IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY
                              : (uint8_t)node->type,
          .stream_id = 0,
          .node = node,
      };
      // Map the node's original index to its position in sort_nodes.
      if (node->node_index < node_index_map_count) {
        node_index_map[node->node_index] = index;
      }
      ++index;
    }
  }

  // Now check if dependencies maintain topological order.
  // This requires a second pass since we need all nodes linearized first.
  // Note that we can early exit on the first unsorted node we find. We could
  // possibly store this bit back on the graph to avoid needing to do this walk
  // on subsequent instantiations.
  bool is_sorted = true;
  for (uint32_t i = 0; is_sorted && i < index; ++i) {
    iree_hal_streaming_graph_node_t* node = sort_nodes[i].node;
    if (node->dependency_count > 0) {
      for (uint32_t j = 0; j < node->dependency_count; ++j) {
        // Look up dependency's position in sort_nodes using the mapping.
        uint32_t dep_index = iree_hal_streaming_graph_node_map_lookup(
            node_index_map, node_index_map_count, node->dependencies[j]);
        if (dep_index != UINT32_MAX && dep_index >= i) {
          is_sorted = false;
          break;
        }
      }
    }
  }

  return (iree_hal_streaming_graph_prepare_result_t){
      .active_node_count = index,
      .is_sorted = is_sorted,
  };
}

//===----------------------------------------------------------------------===//
// Phase 2: Topological Sort with Optimization
//===----------------------------------------------------------------------===//

// Performs topological sort using modified Kahn's algorithm.
// Fast path for already-sorted graphs (common case).
//
// Algorithm when sorting needed:
// 1. Calculate in-degrees for all nodes (including additional edges)
// 2. Queue nodes with zero in-degree
// 3. Process queue, updating in-degrees
// 4. Reorder array in-place using cycle detection
//
// Complexity: O(N + E) where E = total edges (embedded + additional)
static iree_status_t iree_hal_streaming_graph_topological_sort(
    iree_hal_streaming_graph_sort_node_t* nodes, uint32_t node_count,
    uint32_t* node_index_map, iree_host_size_t node_index_map_count,
    iree_hal_streaming_graph_edge_t* additional_edges,
    iree_arena_allocator_t* arena, bool is_already_sorted) {
  if (is_already_sorted && !additional_edges) {
    // Fast path: just compute max dependencies.
    // This is needed for partition boundary detection.
    // Note: Cannot use fast path if there are additional edges since they
    // may change the order.
    for (uint32_t i = 0; i < node_count; ++i) {
      uint32_t max_dep = 0;
      for (uint32_t j = 0; j < nodes[i].node->dependency_count; ++j) {
        // Use the mapping for O(1) lookup.
        const uint32_t dep_index = iree_hal_streaming_graph_node_map_lookup(
            node_index_map, node_index_map_count,
            nodes[i].node->dependencies[j]);
        if (dep_index < i && dep_index != UINT32_MAX) {
          max_dep = iree_max(max_dep, dep_index);
        }
      }
      nodes[i].max_dependency_index = max_dep;
    }
    return iree_ok_status();
  }

  // Full topological sort using Kahn's algorithm.
  uint32_t* queue = NULL;
  const iree_host_size_t queue_size = node_count * sizeof(*queue);
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, queue_size, (void**)&queue));

  // Step 1: Calculate in-degrees from embedded dependencies.
  for (uint32_t i = 0; i < node_count; ++i) {
    uint16_t in_degree = 0;
    for (uint32_t j = 0; j < nodes[i].node->dependency_count; ++j) {
      const uint32_t dep_index = iree_hal_streaming_graph_node_map_lookup(
          node_index_map, node_index_map_count, nodes[i].node->dependencies[j]);
      if (dep_index != UINT32_MAX) {
        ++in_degree;
      }
    }
    nodes[i].in_degree = in_degree;
  }

  // Step 1b: Add in-degrees from additional edges.
  iree_hal_streaming_graph_edge_t* edge = additional_edges;
  while (edge) {
    // Find the 'to' node in our nodes array and increment its in-degree.
    const uint32_t from_index = iree_hal_streaming_graph_node_map_lookup(
        node_index_map, node_index_map_count, edge->from);
    const uint32_t to_index = iree_hal_streaming_graph_node_map_lookup(
        node_index_map, node_index_map_count, edge->to);
    if (from_index != UINT32_MAX && to_index < node_count) {
      ++nodes[to_index].in_degree;
    }
    edge = edge->next;
  }

  // Step 2: Find zero in-degree nodes.
  uint32_t queue_head = 0;
  uint32_t queue_tail = 0;
  for (uint32_t i = 0; i < node_count; ++i) {
    if (nodes[i].in_degree == 0) {
      queue[queue_tail++] = i;
    }
  }

  // Step 3: Process queue.
  uint32_t sorted_count = 0;
  while (queue_head < queue_tail) {
    uint32_t current = queue[queue_head++];
    nodes[current].sorted_index = sorted_count++;

    // Update max dependency for this node from embedded dependencies.
    uint32_t max_dep = 0;
    for (uint32_t j = 0; j < nodes[current].node->dependency_count; ++j) {
      // Use the mapping for O(1) lookup.
      uint32_t dep_index = iree_hal_streaming_graph_node_map_lookup(
          node_index_map, node_index_map_count,
          nodes[current].node->dependencies[j]);
      if (dep_index != UINT32_MAX) {
        uint32_t dep_sorted_index = nodes[dep_index].sorted_index;
        max_dep = iree_max(max_dep, dep_sorted_index);
      }
    }

    // Also check additional edges for max dependency.
    edge = additional_edges;
    while (edge) {
      if (edge->to == nodes[current].node) {
        uint32_t from_index = iree_hal_streaming_graph_node_map_lookup(
            node_index_map, node_index_map_count, edge->from);
        if (from_index != UINT32_MAX) {
          uint32_t from_sorted_index = nodes[from_index].sorted_index;
          max_dep = iree_max(max_dep, from_sorted_index);
        }
      }
      edge = edge->next;
    }
    nodes[current].max_dependency_index = max_dep;

    // Decrement in-degrees of nodes that depend on current.
    // This requires finding reverse edges (who depends on current).
    // First check embedded dependencies.
    for (uint32_t i = 0; i < node_count; ++i) {
      if (i == current) continue;
      iree_hal_streaming_graph_node_t* node = nodes[i].node;
      for (uint32_t j = 0; j < node->dependency_count; ++j) {
        if (node->dependencies[j] == nodes[current].node) {
          if (--nodes[i].in_degree == 0) {
            queue[queue_tail++] = i;
          }
          break;  // Each node appears at most once in dependency list.
        }
      }
    }

    // Also check additional edges for reverse dependencies.
    edge = additional_edges;
    while (edge) {
      if (edge->from == nodes[current].node) {
        uint32_t to_index = iree_hal_streaming_graph_node_map_lookup(
            node_index_map, node_index_map_count, edge->to);
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

  // Step 4: Reorder array in-place based on sorted_index. The cycle-following
  // algorithm minimizes copies and keeps memory usage bounded.
  iree_hal_streaming_graph_sort_node_t temp;
  for (uint32_t i = 0; i < node_count; ++i) {
    while (nodes[i].sorted_index != i) {
      uint32_t target = nodes[i].sorted_index;
      temp = nodes[target];
      nodes[target] = nodes[i];
      nodes[i] = temp;
    }
  }

  // Update the mapping to reflect the new sorted order.
  for (uint32_t i = 0; i < node_count; ++i) {
    if (nodes[i].node->node_index < node_index_map_count) {
      node_index_map[nodes[i].node->node_index] = i;
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Phase 3: Partitioning with Workstream Detection
//===----------------------------------------------------------------------===//

// State for tracking independent workstreams during partitioning.
typedef struct iree_hal_streaming_graph_stream_state_t {
  // Nodes reachable in this stream.
  uint64_t reachability_mask;
  // Last node added to stream.
  uint32_t last_node_index;
} iree_hal_streaming_graph_stream_state_t;

// One incoming dependency added after node creation, indexed by its dependent
// node. Source indices refer to the final topological order.
typedef struct iree_hal_streaming_graph_additional_dependency_t {
  // Source node position in the final topological order.
  uint32_t source_index;
  // Next incoming additional dependency, or UINT32_MAX at the end.
  uint32_t next_index;
} iree_hal_streaming_graph_additional_dependency_t;

typedef struct iree_hal_streaming_graph_additional_dependency_index_t {
  // First incoming additional dependency for each sorted node.
  uint32_t* incoming_heads;
  // Compact storage for active additional dependencies.
  iree_hal_streaming_graph_additional_dependency_t* dependencies;
} iree_hal_streaming_graph_additional_dependency_index_t;

static iree_status_t iree_hal_streaming_graph_build_additional_dependency_index(
    iree_hal_streaming_graph_edge_t* additional_edges,
    const uint32_t* node_index_map, iree_host_size_t node_index_map_count,
    uint32_t node_count, iree_arena_allocator_t* arena,
    iree_hal_streaming_graph_additional_dependency_index_t* out_index) {
  memset(out_index, 0, sizeof(*out_index));
  if (!additional_edges) return iree_ok_status();

  iree_host_size_t active_edge_count = 0;
  for (iree_hal_streaming_graph_edge_t* edge = additional_edges; edge;
       edge = edge->next) {
    const uint32_t source_index = iree_hal_streaming_graph_node_map_lookup(
        node_index_map, node_index_map_count, edge->from);
    const uint32_t target_index = iree_hal_streaming_graph_node_map_lookup(
        node_index_map, node_index_map_count, edge->to);
    if (source_index == UINT32_MAX || target_index == UINT32_MAX) continue;
    if (IREE_UNLIKELY(active_edge_count == UINT32_MAX)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph has too many additional dependencies");
    }
    ++active_edge_count;
  }

  iree_host_size_t heads_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          node_count, sizeof(*out_index->incoming_heads), &heads_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph dependency index size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, heads_size,
                                           (void**)&out_index->incoming_heads));
  memset(out_index->incoming_heads, 0xFF, heads_size);
  if (active_edge_count == 0) return iree_ok_status();

  iree_host_size_t dependencies_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          active_edge_count, sizeof(*out_index->dependencies),
          &dependencies_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph dependency storage size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, dependencies_size,
                                           (void**)&out_index->dependencies));

  uint32_t dependency_index = 0;
  for (iree_hal_streaming_graph_edge_t* edge = additional_edges; edge;
       edge = edge->next) {
    const uint32_t source_index = iree_hal_streaming_graph_node_map_lookup(
        node_index_map, node_index_map_count, edge->from);
    const uint32_t target_index = iree_hal_streaming_graph_node_map_lookup(
        node_index_map, node_index_map_count, edge->to);
    if (source_index == UINT32_MAX || target_index == UINT32_MAX) continue;
    out_index->dependencies[dependency_index] =
        (iree_hal_streaming_graph_additional_dependency_t){
            .source_index = source_index,
            .next_index = out_index->incoming_heads[target_index],
        };
    out_index->incoming_heads[target_index] = dependency_index++;
  }
  return iree_ok_status();
}

typedef struct iree_uint32x2_t {
  uint32_t values[2];
} iree_uint32x2_t;

// Partitions sorted nodes into executable blocks and detects independent
// workstreams within recordable partitions.
//
// Algorithm:
// 1. Scan for recordable vs non-recordable boundaries
// 2. Within recordable sections, detect independent workstreams
// 3. Track reachability masks to determine stream assignment
// 4. Merge streams when convergence detected
//
// Tuning heuristics:
// - Partitions < MIN_PARTITION_SIZE_FOR_STREAMS (~32) use single stream
// - Partitions limited to MAX_PARTITION_SIZE (~1024) nodes
// - Maximum MAX_WORKSTREAMS (~4) concurrent streams per partition
// - New streams require MIN_NODES_PER_STREAM (~4) nodes
// - Reachability analysis limited to MAX_REACHABILITY_BITS (64) nodes
// - High fan-out (> MAX_FAN_OUT=~8) forces stream convergence
//
// Complexity: O(N) with up to MAX_WORKSTREAMS workstreams per partition
//
// Returns [partition_count, block_count].
static iree_uint32x2_t iree_hal_streaming_graph_partition_with_streams(
    iree_hal_streaming_graph_sort_node_t* nodes, uint32_t node_count,
    uint32_t* node_index_map, iree_host_size_t node_index_map_count,
    const iree_hal_streaming_graph_additional_dependency_index_t*
        additional_dependency_index,
    iree_hal_streaming_graph_partition_t* partitions) {
  uint32_t partition_count = 0;
  uint32_t block_count = 0;

  // State for tracking independent workstreams.
  iree_hal_streaming_graph_stream_state_t
      streams[IREE_HAL_STREAMING_GRAPH_MAX_WORKSTREAMS];
  uint8_t active_streams = 0;

  for (uint32_t i = 0; i < node_count;) {
    bool is_recordable =
        iree_hal_streaming_graph_node_is_recordable(nodes[i].type);
    if (!is_recordable) {
      // Non-recordable node gets its own partition.
      iree_hal_streaming_graph_partition_type_t partition_type;
      switch (nodes[i].type) {
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EMPTY:
          partition_type = IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_EMPTY;
          break;
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_HOST_CALL:
          partition_type = IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_HOST_CALL;
          break;
        case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH:
          partition_type = IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_GRAPH;
          break;
        default:
          partition_type = IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_EMPTY;
          break;
      }
      partitions[partition_count] = (iree_hal_streaming_graph_partition_t){
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
      // Start recordable partition.
      uint32_t recordable_start = i;
      memset(streams, 0, sizeof(streams));
      active_streams = 0;

      // Extend partition with compatible nodes up to size limit.
      // A node can be added if all its dependencies are before the partition
      // start (i.e., max_dependency_index < recordable_start).
      uint32_t partition_size = 0;
      while (i < node_count &&
             partition_size < IREE_HAL_STREAMING_GRAPH_MAX_PARTITION_SIZE &&
             iree_hal_streaming_graph_node_is_recordable(nodes[i].type)) {
        // Check if dependencies are satisfied.
        bool deps_satisfied = true;
        for (uint32_t j = 0; j < nodes[i].node->dependency_count; ++j) {
          // Use the mapping for O(1) lookup.
          uint32_t dep_index = iree_hal_streaming_graph_node_map_lookup(
              node_index_map, node_index_map_count,
              nodes[i].node->dependencies[j]);
          if (dep_index == UINT32_MAX) {
            // Disabled dependencies are absent from the executable graph.
          } else if (dep_index >= recordable_start && dep_index < i) {
            // Dependency is within this partition - OK.
          } else if (dep_index >= i) {
            // Dependency is ahead - can't include this node.
            deps_satisfied = false;
            break;
          }
        }
        if (!deps_satisfied) {
          break;  // End this partition.
        }

        // Only perform workstream detection for sufficiently large partitions.
        const bool use_workstreams =
            (i - recordable_start) >=
            IREE_HAL_STREAMING_GRAPH_MIN_PARTITION_SIZE_FOR_STREAMS;

        // Determine which stream this node belongs to.
        uint8_t assigned_stream = 0;
        uint8_t connected_streams = 0;
        uint32_t effective_dependency_count = nodes[i].node->dependency_count;
        bool is_sync_point = nodes[i].node->dependency_count >
                             IREE_HAL_STREAMING_GRAPH_MAX_FAN_IN;

        // Check dependencies within this partition.
        for (uint32_t j = 0; j < nodes[i].node->dependency_count; ++j) {
          // Use the mapping for O(1) lookup.
          uint32_t dep_index = iree_hal_streaming_graph_node_map_lookup(
              node_index_map, node_index_map_count,
              nodes[i].node->dependencies[j]);
          if (dep_index >= recordable_start && dep_index < i) {
            // Dependency is within this partition.
            uint8_t dep_stream = nodes[dep_index].stream_id;
            connected_streams |= (1 << dep_stream);
          }
        }

        if (additional_dependency_index->incoming_heads) {
          for (uint32_t dependency_index =
                   additional_dependency_index->incoming_heads[i];
               dependency_index != UINT32_MAX;
               dependency_index =
                   additional_dependency_index->dependencies[dependency_index]
                       .next_index) {
            const uint32_t dep_index =
                additional_dependency_index->dependencies[dependency_index]
                    .source_index;
            if (!is_sync_point) {
              is_sync_point = ++effective_dependency_count >
                              IREE_HAL_STREAMING_GRAPH_MAX_FAN_IN;
            }
            if (dep_index >= recordable_start && dep_index < i) {
              const uint8_t dep_stream = nodes[dep_index].stream_id;
              connected_streams |= (1 << dep_stream);
            }
          }
        }

        if (use_workstreams && connected_streams == 0 &&
            active_streams < IREE_HAL_STREAMING_GRAPH_MAX_WORKSTREAMS) {
          // Check if enough nodes remain to justify a new stream.
          uint32_t remaining_in_partition =
              IREE_HAL_STREAMING_GRAPH_MAX_PARTITION_SIZE - partition_size;
          if (remaining_in_partition >=
              IREE_HAL_STREAMING_GRAPH_MIN_NODES_PER_STREAM) {
            // No dependencies - new independent stream!
            assigned_stream = active_streams++;
            streams[assigned_stream].last_node_index = i;
            if ((i - recordable_start) <
                IREE_HAL_STREAMING_GRAPH_MAX_REACHABILITY_BITS) {
              streams[assigned_stream].reachability_mask =
                  1ULL << (i - recordable_start);
            }
          } else {
            // Not enough nodes - use stream 0.
            assigned_stream = 0;
            if (active_streams == 0) active_streams = 1;
          }
        } else if (use_workstreams &&
                   iree_math_count_ones_u32(connected_streams) == 1) {
          // Depends on single stream.
          assigned_stream =
              iree_math_count_trailing_zeros_u32(connected_streams);
          streams[assigned_stream].last_node_index = i;
          if ((i - recordable_start) <
              IREE_HAL_STREAMING_GRAPH_MAX_REACHABILITY_BITS) {
            streams[assigned_stream].reachability_mask |=
                1ULL << (i - recordable_start);
          }
        } else {
          // Either: no workstreams, multiple dependencies, or merge point.
          const uint32_t dep_count =
              iree_math_count_ones_u32(connected_streams);
          // Check for a high fan-in synchronization point.
          if (!use_workstreams || dep_count > 1 || is_sync_point) {
            // Collapse to single stream.
            assigned_stream = 0;
            for (uint32_t k = recordable_start; k < i; ++k) {
              nodes[k].stream_id = 0;  // Reset all to stream 0.
            }
            active_streams = 1;
          } else {
            // Single dependency or first node - use stream 0.
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
      partitions[partition_count] = (iree_hal_streaming_graph_partition_t){
          .start_index = recordable_start,
          .count = i - recordable_start,
          .type = IREE_HAL_STREAMING_GRAPH_PARTITION_TYPE_RECORDABLE,
          .stream_count = stream_count,
      };
      ++partition_count;
      block_count += stream_count;
    }
  }

  return (iree_uint32x2_t){{partition_count, block_count}};
}

//===----------------------------------------------------------------------===//
// Main Scheduling Entry Point
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_graph_schedule_nodes(
    iree_hal_streaming_node_block_t* node_blocks, iree_host_size_t node_count,
    const uint8_t* disabled_nodes, iree_host_size_t disabled_node_count,
    iree_hal_streaming_graph_edge_t* additional_edges,
    iree_arena_allocator_t* arena,
    iree_hal_streaming_graph_schedule_t* out_schedule) {
  IREE_ASSERT_ARGUMENT(out_schedule);

  iree_host_size_t actual_node_count = 0;
  iree_host_size_t node_index_map_count = 0;
  uint32_t max_node_index = 0;
  bool has_node = false;
  for (iree_hal_streaming_node_block_t* block = node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* node = block->nodes[i];
      ++actual_node_count;
      if (!has_node || node->node_index > max_node_index) {
        max_node_index = node->node_index;
      }
      has_node = true;
    }
  }
  for (iree_hal_streaming_graph_edge_t* edge = additional_edges; edge;
       edge = edge->next) {
    if (edge->from && (!has_node || edge->from->node_index > max_node_index)) {
      max_node_index = edge->from->node_index;
      has_node = true;
    }
    if (edge->to && edge->to->node_index > max_node_index) {
      max_node_index = edge->to->node_index;
      has_node = true;
    }
  }

  if (actual_node_count == 0) {
    memset(out_schedule, 0, sizeof(*out_schedule));
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(
          actual_node_count > UINT32_MAX ||
          !iree_host_size_checked_add((iree_host_size_t)max_node_index, 1,
                                      &node_index_map_count))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph node index map size overflow");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, node_count);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, actual_node_count);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, node_index_map_count);

  // Allocate all working memory from arena.
  iree_hal_streaming_graph_sort_node_t* sorted_nodes = NULL;
  iree_host_size_t sorted_nodes_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          actual_node_count, sizeof(*sorted_nodes), &sorted_nodes_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph sort node allocation size overflow");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, sorted_nodes_size, (void**)&sorted_nodes));

  // Allocate mapping from original node_index to sorted position.
  uint32_t* node_index_map = NULL;
  iree_host_size_t map_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          node_index_map_count, sizeof(*node_index_map), &map_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph node index map allocation size overflow");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, map_size, (void**)&node_index_map));

  // Phase 1: Prepare - linearize and detect if sorted.
  const iree_hal_streaming_graph_prepare_result_t prepare_result =
      iree_hal_streaming_graph_prepare_nodes(
          node_blocks, disabled_nodes, disabled_node_count, sorted_nodes,
          node_index_map, node_index_map_count);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, prepare_result.is_sorted);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, prepare_result.active_node_count);

  if (prepare_result.active_node_count == 0) {
    out_schedule->sorted_nodes = sorted_nodes;
    out_schedule->node_index_map = node_index_map;
    out_schedule->partitions = NULL;
    out_schedule->partition_count = 0;
    out_schedule->block_count = 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Phase 2: Sort - topological ordering.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_topological_sort(
              sorted_nodes, prepare_result.active_node_count, node_index_map,
              node_index_map_count, additional_edges, arena,
              prepare_result.is_sorted));

  iree_hal_streaming_graph_additional_dependency_index_t
      additional_dependency_index;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_graph_build_additional_dependency_index(
              additional_edges, node_index_map, node_index_map_count,
              prepare_result.active_node_count, arena,
              &additional_dependency_index));

  // Allocate the worst-case partition table up front. A graph where every
  // active node must be isolated needs one partition per node.
  iree_hal_streaming_graph_partition_t* partitions = NULL;
  iree_host_size_t partitions_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          actual_node_count, sizeof(*partitions), &partitions_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph partition allocation size overflow");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(arena, partitions_size, (void**)&partitions));

  // Phase 3: Partition - group into executable blocks.
  const iree_uint32x2_t partition_block_counts =
      iree_hal_streaming_graph_partition_with_streams(
          sorted_nodes, prepare_result.active_node_count, node_index_map,
          node_index_map_count, &additional_dependency_index, partitions);
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
