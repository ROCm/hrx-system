// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "graph_analysis.h"

#include "iree/base/internal/math.h"

// Bound recording size and amortize the extra command buffers/semaphores needed
// by independent workstreams. These bounds affect recording, never graph size.
#define HRX_GRAPH_MAX_PARTITION_SIZE 2048
#define HRX_GRAPH_MIN_PARTITION_SIZE_FOR_STREAMS 16
#define HRX_GRAPH_MIN_NODES_PER_STREAM 32
#define HRX_GRAPH_MAX_FAN_IN 8

iree_status_t hrx_graph_analysis_initialize(
    iree_host_size_t node_count, iree_arena_allocator_t* arena,
    hrx_graph_analysis_t* out_analysis) {
  memset(out_analysis, 0, sizeof(*out_analysis));
  if (!node_count) {
    return iree_ok_status();
  }
  iree_host_size_t allocation_size = 0;
  if (node_count > UINT32_MAX ||
      !iree_host_size_checked_mul(
          node_count, sizeof(void*) + 2 * sizeof(uint32_t) + sizeof(uint8_t),
          &allocation_size) ||
      !iree_host_size_checked_add(allocation_size, sizeof(uint32_t),
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph node tables exceed addressable storage");
  }
  void** sources = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, allocation_size, (void**)&sources));
  uint32_t* offsets = (uint32_t*)(sources + node_count);
  uint32_t* order = offsets + node_count + 1;
  uint8_t* flags = (uint8_t*)(order + node_count);
  memset(offsets, 0, (node_count + 1) * sizeof(*offsets));
  *out_analysis = (hrx_graph_analysis_t){
      .node_count = (uint32_t)node_count,
      .sources = sources,
      .offsets = offsets,
      .order = order,
      .flags = flags,
  };
  return iree_ok_status();
}

iree_status_t hrx_graph_analysis_allocate_dependencies(
    hrx_graph_analysis_t* analysis, iree_arena_allocator_t* arena) {
  if (!analysis->node_count) {
    return iree_ok_status();
  }
  uint64_t edge_count = 0;
  for (uint32_t i = 0; i < analysis->node_count; ++i) {
    uint32_t count = analysis->offsets[i + 1];
    analysis->offsets[i] = (uint32_t)edge_count;
    analysis->order[i] = (uint32_t)edge_count;
    edge_count += count;
  }
  iree_host_size_t allocation_size = 0;
  if (edge_count > UINT32_MAX ||
      !iree_host_size_checked_mul((iree_host_size_t)edge_count,
                                  sizeof(uint32_t), &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph dependencies exceed addressable storage");
  }
  analysis->offsets[analysis->node_count] = (uint32_t)edge_count;
  if (allocation_size) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, allocation_size,
                                             (void**)&analysis->dependencies));
  }
  return iree_ok_status();
}

// Builds reverse CSR adjacency only when creation order is not topological.
// order doubles as adjacency fill cursors and then the FIFO ready queue. After
// the sort, pending becomes the source-to-sorted inverse map.
static iree_status_t hrx_graph_topological_sort(hrx_graph_analysis_t* analysis,
                                                iree_arena_allocator_t* arena,
                                                const uint32_t** out_ranks) {
  const uint32_t count = analysis->node_count;
  if (analysis->is_topologically_ordered) {
    for (uint32_t node = 0; node < count; ++node) {
      analysis->order[node] = node;
    }
    *out_ranks = analysis->order;
    return iree_ok_status();
  }

  const uint32_t edge_count = analysis->offsets[count];
  iree_host_size_t word_count = (iree_host_size_t)count * 2 + 1 + edge_count;
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(word_count, sizeof(uint32_t),
                                  &allocation_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "graph reverse adjacency exceeds addressable storage");
  }
  uint32_t* outgoing_offsets = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, allocation_size, (void**)&outgoing_offsets));
  uint32_t* pending = outgoing_offsets + count + 1;
  uint32_t* outgoing = pending + count;
  memset(outgoing_offsets, 0, ((iree_host_size_t)count + 1) * sizeof(uint32_t));
  for (uint32_t e = 0; e < edge_count; ++e) {
    ++outgoing_offsets[analysis->dependencies[e] + 1];
  }
  for (uint32_t node = 0; node < count; ++node) {
    outgoing_offsets[node + 1] += outgoing_offsets[node];
    analysis->order[node] = outgoing_offsets[node];
    pending[node] = analysis->offsets[node + 1] - analysis->offsets[node];
  }
  for (uint32_t node = 0; node < count; ++node) {
    for (uint32_t e = analysis->offsets[node]; e < analysis->offsets[node + 1];
         ++e) {
      const uint32_t predecessor = analysis->dependencies[e];
      outgoing[analysis->order[predecessor]++] = node;
    }
  }
  uint32_t head = 0;
  uint32_t tail = 0;
  for (uint32_t node = 0; node < count; ++node) {
    if (!pending[node]) {
      analysis->order[tail++] = node;
    }
  }
  while (head < tail) {
    const uint32_t node = analysis->order[head++];
    for (uint32_t e = outgoing_offsets[node]; e < outgoing_offsets[node + 1];
         ++e) {
      const uint32_t successor = outgoing[e];
      if (!--pending[successor]) {
        analysis->order[tail++] = successor;
      }
    }
  }
  if (tail != count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "graph contains cycles (%u of %u nodes ordered)",
                            tail, count);
  }
  for (uint32_t position = 0; position < count; ++position) {
    pending[analysis->order[position]] = position;
  }
  *out_ranks = pending;
  return iree_ok_status();
}

static uint32_t hrx_graph_partition_end(const hrx_graph_analysis_t* analysis,
                                        uint32_t start) {
  uint32_t end = start + 1;
  if ((analysis->flags[analysis->order[start]] &
       HRX_GRAPH_ANALYSIS_NODE_KIND_MASK) ==
      HRX_GRAPH_ANALYSIS_NODE_RECORDABLE) {
    const uint32_t limit = start + iree_min(analysis->node_count - start,
                                            HRX_GRAPH_MAX_PARTITION_SIZE);
    while (end < limit && (analysis->flags[analysis->order[end]] &
                           HRX_GRAPH_ANALYSIS_NODE_KIND_MASK) ==
                              HRX_GRAPH_ANALYSIS_NODE_RECORDABLE) {
      ++end;
    }
  }
  return end;
}

// A multi-stream join collapses the preceding prefix. Record its endpoint and
// read earlier stream IDs as zero instead of rewriting the prefix at each join.
// One final pass publishes the assignments and exact barrier decisions.
static uint32_t hrx_graph_partition_streams(hrx_graph_analysis_t* analysis,
                                            const uint32_t* ranks,
                                            uint32_t start, uint32_t end,
                                            uint32_t* frontiers) {
  uint32_t collapsed_end = start;
  uint32_t stream_count = 1;
  for (uint32_t position = start; position < end; ++position) {
    const uint32_t node = analysis->order[position];
    uint32_t connected = 0;
    uint32_t frontier = 0;
    for (uint32_t e = analysis->offsets[node]; e < analysis->offsets[node + 1];
         ++e) {
      const uint32_t predecessor = analysis->dependencies[e];
      const uint32_t rank = ranks[predecessor];
      frontier = iree_max(frontier, rank + 1);
      if (rank >= start) {
        uint32_t stream = rank < collapsed_end
                              ? 0
                              : analysis->flags[predecessor] >>
                                    HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT;
        connected |= 1u << stream;
      }
    }
    frontiers[position - start] = frontier;
    uint32_t stream = 0;
    bool parallel =
        position - start >= HRX_GRAPH_MIN_PARTITION_SIZE_FOR_STREAMS;
    if (parallel && !connected && stream_count < HRX_GRAPH_MAX_WORKSTREAMS) {
      if (HRX_GRAPH_MAX_PARTITION_SIZE - (position - start) >=
          HRX_GRAPH_MIN_NODES_PER_STREAM) {
        stream = stream_count++;
      }
    } else if (parallel && iree_math_count_ones_u32(connected) == 1) {
      stream = iree_math_count_trailing_zeros_u32(connected);
    } else if (!parallel || iree_math_count_ones_u32(connected) > 1 ||
               analysis->offsets[node + 1] - analysis->offsets[node] >
                   HRX_GRAPH_MAX_FAN_IN) {
      collapsed_end = position;
      stream_count = 1;
    }
    analysis->flags[node] =
        (uint8_t)(stream << HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT);
  }

  // Every in-partition dependency has been assigned to the same final stream.
  // A barrier covers all earlier positions on that stream, so the largest
  // predecessor rank is sufficient: no pending-node set or edge walk is needed.
  // Earlier partitions are already ordered by their completion semaphores.
  uint32_t boundaries[HRX_GRAPH_MAX_WORKSTREAMS];
  for (uint32_t stream = 0; stream < stream_count; ++stream) {
    boundaries[stream] = start;
  }
  for (uint32_t position = start; position < end; ++position) {
    const uint32_t node = analysis->order[position];
    uint32_t stream =
        position < collapsed_end
            ? 0
            : analysis->flags[node] >> HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT;
    uint8_t flags = (uint8_t)(stream << HRX_GRAPH_ANALYSIS_NODE_STREAM_SHIFT);
    if (frontiers[position - start] > boundaries[stream]) {
      flags |= HRX_GRAPH_ANALYSIS_NODE_BARRIER_BEFORE;
      boundaries[stream] = position;
    }
    analysis->flags[node] = flags;
  }
  return stream_count;
}

iree_status_t hrx_graph_analyze(hrx_graph_analysis_t* analysis,
                                iree_arena_allocator_t* arena,
                                hrx_graph_schedule_t* out_schedule) {
  memset(out_schedule, 0, sizeof(*out_schedule));
  if (!analysis->node_count) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  const uint32_t* ranks = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, hrx_graph_topological_sort(analysis, arena, &ranks));

  // Count actual boundaries before allocation. A long recording needs a few
  // descriptors, not a worst-case node-sized table held for its whole lifetime.
  uint32_t partition_count = 0;
  uint32_t maximum_partition_size = 0;
  for (uint32_t start = 0; start < analysis->node_count;) {
    const uint32_t end = hrx_graph_partition_end(analysis, start);
    maximum_partition_size = iree_max(maximum_partition_size, end - start);
    ++partition_count;
    start = end;
  }
  hrx_graph_partition_t* partitions = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(
              arena, (iree_host_size_t)partition_count * sizeof(*partitions),
              (void**)&partitions));
  // Reused by every partition; these predecessor frontiers avoid an edge walk
  // after workstream joins have resolved their final assignments.
  uint32_t* frontiers = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_arena_allocate(arena, maximum_partition_size * sizeof(*frontiers),
                          (void**)&frontiers));
  uint32_t block_count = 0;
  uint32_t start = 0;
  for (uint32_t p = 0; p < partition_count; ++p) {
    const uint32_t end = hrx_graph_partition_end(analysis, start);
    uint32_t stream_count = 1;
    if ((analysis->flags[analysis->order[start]] &
         HRX_GRAPH_ANALYSIS_NODE_KIND_MASK) ==
        HRX_GRAPH_ANALYSIS_NODE_RECORDABLE) {
      stream_count =
          hrx_graph_partition_streams(analysis, ranks, start, end, frontiers);
    }
    partitions[p] = (hrx_graph_partition_t){
        .start_index = start,
        .count = end - start,
        .stream_count = stream_count,
    };
    block_count += stream_count;
    start = end;
  }
  *out_schedule = (hrx_graph_schedule_t){
      .node_count = analysis->node_count,
      .partition_count = partition_count,
      .block_count = block_count,
      .sources = analysis->sources,
      .order = analysis->order,
      .flags = analysis->flags,
      .partitions = partitions,
  };
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
