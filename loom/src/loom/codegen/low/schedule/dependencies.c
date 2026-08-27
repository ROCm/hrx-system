// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/dependencies.h"

static iree_status_t loom_low_schedule_dependency_graph_prepare_append(
    loom_low_schedule_dependency_graph_t* graph,
    iree_arena_allocator_t* arena) {
  if ((graph->count & LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_MASK) != 0) {
    return iree_ok_status();
  }
  void* segment = NULL;
  return loom_segmented_storage_append(&graph->segments, arena, &segment);
}

void loom_low_schedule_dependency_graph_initialize(
    loom_low_schedule_dependency_graph_t* out_graph) {
  IREE_ASSERT_ARGUMENT(out_graph);
  *out_graph = (loom_low_schedule_dependency_graph_t){0};
  loom_segmented_storage_initialize(
      sizeof(loom_low_schedule_dependency_segment_t),
      iree_alignof(loom_low_schedule_dependency_segment_t),
      &out_graph->segments);
}

void loom_low_schedule_dependency_graph_move(
    loom_low_schedule_dependency_graph_t* source,
    loom_low_schedule_dependency_graph_t* out_graph) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(out_graph);
  out_graph->count = source->count;
  loom_segmented_storage_move(&source->segments, &out_graph->segments);
  source->count = 0;
}

iree_status_t loom_low_schedule_dependency_graph_append(
    loom_low_schedule_dependency_graph_t* graph,
    loom_low_schedule_dependency_t dependency, iree_arena_allocator_t* arena) {
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_dependency_graph_prepare_append(graph, arena));
  const iree_host_size_t dependency_index = graph->count;
  loom_low_schedule_dependency_segment_t* segment =
      (loom_low_schedule_dependency_segment_t*)loom_segmented_storage_segment(
          &graph->segments,
          (uint32_t)(dependency_index >>
                     LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_SHIFT));
  segment->rows[dependency_index & LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_MASK] =
      dependency;
  ++graph->count;
  return iree_ok_status();
}
