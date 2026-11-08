// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/progress.h"

typedef struct loom_pipeline_progress_edge_t {
  // Next source depending on the same destination's progress.
  uint32_t next;
  // Upstream channel whose record ordinal this edge preserves.
  uint32_t source;
} loom_pipeline_progress_edge_t;

typedef struct loom_pipeline_progress_node_t {
  // First upstream edge, or UINT32_MAX.
  uint32_t upstream;
  // Downstream channels still awaiting observer selection.
  uint32_t pending;
} loom_pipeline_progress_node_t;

iree_status_t loom_pipeline_progress_select(
    const loom_pipeline_invocations_t* invocations,
    const loom_pipeline_worker_t* workers,
    const loom_pipeline_progress_endpoint_t* endpoints,
    loom_pipeline_progress_callback_t callback, iree_arena_allocator_t* arena,
    uint32_t** out_readers) {
  *out_readers = NULL;
  const iree_host_size_t count = invocations->channel_count;
  if (!count) {
    return iree_ok_status();
  }
  loom_pipeline_progress_node_t* nodes;
  uint32_t* ready;
  uint32_t* readers;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*nodes), (void**)&nodes));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*ready), (void**)&ready));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*readers),
                                                 (void**)&readers));
  iree_host_size_t edge_capacity = 0;
  for (iree_host_size_t i = 0; i < invocations->worker_count; ++i) {
    const loom_pipeline_transport_t* transport = &workers[i].transport;
    for (iree_host_size_t k = 0; k < transport->cohort_count; ++k) {
      edge_capacity += transport->cohorts[k].destination_count;
    }
  }
  if (edge_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline progress exceeds edge index capacity");
  }
  loom_pipeline_progress_edge_t* edges;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, edge_capacity, sizeof(*edges), (void**)&edges));
  for (iree_host_size_t i = 0; i < count; ++i) {
    nodes[i] = (loom_pipeline_progress_node_t){.upstream = UINT32_MAX};
    readers[i] =
        endpoints[i].observer && endpoints[i].reads ? (uint32_t)i : UINT32_MAX;
  }
  uint32_t edge_count = 0;
  for (iree_host_size_t i = 0; i < invocations->worker_count; ++i) {
    const loom_pipeline_transport_t* transport = &workers[i].transport;
    for (iree_host_size_t k = 0; k < transport->batch_count; ++k) {
      const loom_pipeline_transport_batch_t* batch = &transport->batches[k];
      const loom_pipeline_transport_cohort_t* cohort = batch->records;
      if (batch->record_count != 1 || cohort->release ||
          cohort->admissions != cohort->transfers) {
        continue;
      }
      const uint32_t source = loom_pipeline_invocations_channel_index(
          invocations, cohort->admission->channel);
      if (!endpoints[source].reads ||
          endpoints[source].reads != cohort->admissions) {
        continue;
      }
      for (iree_host_size_t d = 0; d < cohort->destination_count; ++d) {
        const uint32_t destination = loom_pipeline_invocations_channel_index(
            invocations, cohort->destinations[d].channel);
        if (endpoints[destination].writes != cohort->transfers) {
          continue;
        }
        edges[edge_count] = (loom_pipeline_progress_edge_t){
            .next = nodes[destination].upstream, .source = source};
        nodes[destination].upstream = edge_count++;
        ++nodes[source].pending;
      }
    }
  }
  iree_host_size_t head = 0, tail = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!nodes[i].pending) {
      ready[tail++] = (uint32_t)i;
    }
  }
  while (head < tail) {
    const uint32_t destination = ready[head++];
    const uint32_t candidate = readers[destination];
    for (uint32_t edge = nodes[destination].upstream; edge != UINT32_MAX;
         edge = edges[edge].next) {
      const uint32_t source = edges[edge].source;
      const uint32_t selected = readers[source];
      if (candidate != UINT32_MAX &&
          endpoints[candidate].reads == endpoints[source].reads) {
        if (selected == UINT32_MAX) {
          readers[source] = candidate;
        } else {
          const uint64_t candidate_cost =
              callback.cost(callback.user_data, source, candidate);
          const uint64_t selected_cost =
              callback.cost(callback.user_data, source, selected);
          if (candidate_cost < selected_cost ||
              (candidate_cost == selected_cost && candidate < selected)) {
            readers[source] = candidate;
          }
        }
      }
      if (!--nodes[source].pending) {
        ready[tail++] = source;
      }
    }
  }
  // Pending nodes participate in or depend on a cycle. An eligible reader at
  // such a node cannot establish equal-ordinal feedback through that cycle.
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (nodes[i].pending) {
      readers[i] = UINT32_MAX;
    }
  }
  *out_readers = readers;
  return iree_ok_status();
}
