// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/write_interference_flow.h"

#include <string.h>

#include "loom/ops/low/ops.h"

typedef struct loom_low_write_flow_edge_t {
  // Source write point, after its events execute.
  uint32_t source;
  // Target write point, before its events execute.
  uint32_t target;
} loom_low_write_flow_edge_t;

typedef struct loom_low_write_flow_regions_t {
  // First child instruction write point in the first region.
  uint32_t first;
  // First child instruction write point in the second region, or zero.
  uint32_t second;
} loom_low_write_flow_regions_t;

iree_status_t loom_low_write_flow_build(
    const loom_liveness_analysis_t* liveness, const loom_cfg_graph_t* cfg_graph,
    uint32_t point_count, iree_arena_allocator_t* arena,
    loom_low_write_flow_t* out_flow) {
  *out_flow = (loom_low_write_flow_t){0};
  // Every structured operation contributes at most two explicit outgoing
  // edges. Ordinary instructions remain inside linear spans.
  const uint64_t edge_capacity =
      (uint64_t)cfg_graph->edge_count + 2ull * liveness->operation_count;
  if (edge_capacity > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "retained-write flow edges exceed u32 index capacity");
  }
  loom_low_write_flow_edge_t* edges = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, edge_capacity, sizeof(*edges), (void**)&edges));
  uint32_t edge_count = 0;
  uint8_t* leaders = NULL;
  uint8_t* exits = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, point_count, sizeof(*leaders), (void**)&leaders));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, point_count, sizeof(*exits), (void**)&exits));
  memset(leaders, 0, point_count);
  memset(exits, 0, point_count);
  for (iree_host_size_t b = 0; b < liveness->block_count; ++b) {
    exits[liveness->blocks[b].end_point] = 1;
  }
  for (iree_host_size_t e = 0; e < cfg_graph->edge_count; ++e) {
    const loom_cfg_edge_info_t* edge = &cfg_graph->edges[e];
    edges[edge_count++] = (loom_low_write_flow_edge_t){
        .source = liveness->blocks[edge->source_block_index].end_point,
        .target = liveness->blocks[edge->target_block_index].start_point + 1,
    };
  }

  loom_low_write_flow_regions_t* regions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, liveness->operation_count, sizeof(*regions), (void**)&regions));
  memset(regions, 0, liveness->operation_count * sizeof(*regions));
  for (uint32_t i = 0; i < liveness->operation_count; ++i) {
    const loom_liveness_operation_point_t* point =
        loom_liveness_operation_at(liveness, i);
    if (point->parent_operation_index == UINT32_MAX) {
      continue;
    }
    const uint32_t parent = point->parent_operation_index;
    const loom_op_t* parent_op =
        loom_liveness_operation_at(liveness, parent)->op;
    const loom_region_t* region = point->op->parent_block->parent_region;
    uint32_t* first = region == loom_op_regions(parent_op)[0]
                          ? &regions[parent].first
                          : &regions[parent].second;
    if (*first == 0) {
      *first = point->start_point + 1;
    }
  }

  for (uint32_t i = 0; i < liveness->operation_count; ++i) {
    const loom_liveness_operation_point_t* point =
        loom_liveness_operation_at(liveness, i);
    const loom_op_t* op = point->op;
    const uint32_t entry = point->start_point + 1;
    if (loom_low_scf_if_isa(op) || loom_low_scf_for_isa(op) ||
        loom_low_scf_while_isa(op)) {
      exits[entry] = 1;
      edges[edge_count++] =
          (loom_low_write_flow_edge_t){entry, regions[i].first};
      if (!loom_low_scf_while_isa(op)) {
        edges[edge_count++] = (loom_low_write_flow_edge_t){
            entry,
            regions[i].second != 0 ? regions[i].second : point->end_point};
      }
    } else if (loom_low_scf_yield_isa(op) || loom_low_scf_condition_isa(op)) {
      const uint32_t parent = point->parent_operation_index;
      const loom_liveness_operation_point_t* parent_point =
          loom_liveness_operation_at(liveness, parent);
      exits[point->end_point] = 1;
      if (loom_low_scf_condition_isa(op)) {
        edges[edge_count++] = (loom_low_write_flow_edge_t){
            point->end_point, regions[parent].second};
        edges[edge_count++] = (loom_low_write_flow_edge_t){
            point->end_point, parent_point->end_point};
      } else if (loom_low_scf_while_isa(parent_point->op)) {
        edges[edge_count++] = (loom_low_write_flow_edge_t){
            point->end_point, regions[parent].first};
      } else {
        edges[edge_count++] = (loom_low_write_flow_edge_t){
            point->end_point, parent_point->end_point};
        if (loom_low_scf_for_isa(parent_point->op)) {
          edges[edge_count++] = (loom_low_write_flow_edge_t){
              point->end_point, regions[parent].first};
        }
      }
    }
  }
  leaders[1] = 1;
  for (uint32_t point = 1; point + 1 < point_count; ++point) {
    if (exits[point]) {
      leaders[point + 1] = 1;
    }
  }
  for (uint32_t e = 0; e < edge_count; ++e) {
    leaders[edges[e].target] = 1;
  }

  uint32_t block_count = 0;
  for (uint32_t point = 1; point < point_count; ++point) {
    block_count += leaders[point] != 0;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, block_count,
                                                 sizeof(*out_flow->blocks),
                                                 (void**)&out_flow->blocks));
  memset(out_flow->blocks, 0, block_count * sizeof(*out_flow->blocks));
  out_flow->block_count = block_count;
  uint32_t* blocks_by_point = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, point_count, sizeof(*blocks_by_point), (void**)&blocks_by_point));
  uint32_t block = UINT32_MAX;
  for (uint32_t point = 1; point < point_count; ++point) {
    if (leaders[point]) {
      ++block;
      out_flow->blocks[block].begin = point;
      if (block != 0) {
        out_flow->blocks[block - 1].end = point;
      }
    }
    blocks_by_point[point] = block;
  }
  out_flow->blocks[block].end = point_count;
  uint64_t successor_count = edge_count;
  for (uint32_t b = 0; b < block_count; ++b) {
    const uint32_t last = out_flow->blocks[b].end - 1;
    out_flow->blocks[b].successor_count = !exits[last] && b + 1 < block_count;
    successor_count += out_flow->blocks[b].successor_count;
  }
  if (successor_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "retained-write flow successors exceed u32 index capacity");
  }
  for (uint32_t e = 0; e < edge_count; ++e) {
    ++out_flow->blocks[blocks_by_point[edges[e].source]].successor_count;
  }
  uint32_t successor_start = 0;
  for (uint32_t b = 0; b < block_count; ++b) {
    loom_low_write_flow_block_t* span = &out_flow->blocks[b];
    span->successor_start = successor_start;
    successor_start += span->successor_count;
    span->successor_count = 0;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, successor_count, sizeof(*out_flow->successors),
      (void**)&out_flow->successors));
  for (uint32_t b = 0; b < block_count; ++b) {
    loom_low_write_flow_block_t* span = &out_flow->blocks[b];
    if (!exits[span->end - 1] && b + 1 < block_count) {
      out_flow->successors[span->successor_start + span->successor_count++] =
          b + 1;
    }
  }
  for (uint32_t e = 0; e < edge_count; ++e) {
    loom_low_write_flow_block_t* span =
        &out_flow->blocks[blocks_by_point[edges[e].source]];
    out_flow->successors[span->successor_start + span->successor_count++] =
        blocks_by_point[edges[e].target];
  }
  // Point-to-span translation ends here. Its point_count entries cover the
  // solver's block_count links without extending the construction lifetime.
  out_flow->worklist = blocks_by_point;
  return iree_ok_status();
}
