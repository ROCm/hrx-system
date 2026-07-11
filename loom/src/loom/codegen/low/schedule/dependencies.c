// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/dependencies.h"

static iree_status_t loom_low_schedule_dependency_storage_prepare_append(
    loom_low_schedule_dependency_storage_t* storage,
    iree_arena_allocator_t* arena) {
  if ((storage->count & LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_MASK) != 0) {
    return iree_ok_status();
  }
  void* segment = NULL;
  return loom_segmented_storage_append(&storage->segments, arena, &segment);
}

void loom_low_schedule_dependency_graph_initialize(
    loom_low_schedule_dependency_graph_t* out_graph) {
  IREE_ASSERT_ARGUMENT(out_graph);
  *out_graph = (loom_low_schedule_dependency_graph_t){0};
  loom_segmented_storage_initialize(
      sizeof(loom_low_schedule_dependency_segment_t),
      iree_alignof(loom_low_schedule_dependency_segment_t),
      &out_graph->ordering.segments);
  loom_segmented_storage_initialize(
      sizeof(loom_low_schedule_visibility_dependency_segment_t),
      iree_alignof(loom_low_schedule_visibility_dependency_segment_t),
      &out_graph->visibility.segments);
}

void loom_low_schedule_dependency_graph_move(
    loom_low_schedule_dependency_graph_t* source,
    loom_low_schedule_dependency_graph_t* out_graph) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(out_graph);
  out_graph->ordering.count = source->ordering.count;
  loom_segmented_storage_move(&source->ordering.segments,
                              &out_graph->ordering.segments);
  out_graph->visibility.count = source->visibility.count;
  loom_segmented_storage_move(&source->visibility.segments,
                              &out_graph->visibility.segments);
  source->ordering.count = 0;
  source->visibility.count = 0;
}

iree_status_t loom_low_schedule_dependency_graph_append_ordering(
    loom_low_schedule_dependency_graph_t* graph,
    loom_low_schedule_dependency_t dependency, iree_arena_allocator_t* arena) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_storage_prepare_append(
      &graph->ordering, arena));
  const iree_host_size_t dependency_index = graph->ordering.count;
  loom_low_schedule_dependency_segment_t* segment =
      (loom_low_schedule_dependency_segment_t*)loom_segmented_storage_segment(
          &graph->ordering.segments,
          (uint32_t)(dependency_index >>
                     LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_SHIFT));
  segment->rows[dependency_index & LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_MASK] =
      dependency;
  ++graph->ordering.count;
  return iree_ok_status();
}

iree_status_t loom_low_schedule_dependency_graph_append_visibility(
    loom_low_schedule_dependency_graph_t* graph,
    loom_low_schedule_visibility_dependency_t dependency,
    iree_arena_allocator_t* arena) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_storage_prepare_append(
      &graph->visibility, arena));
  const iree_host_size_t dependency_index = graph->visibility.count;
  loom_low_schedule_visibility_dependency_segment_t* segment =
      (loom_low_schedule_visibility_dependency_segment_t*)
          loom_segmented_storage_segment(
              &graph->visibility.segments,
              (uint32_t)(dependency_index >>
                         LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_SHIFT));
  segment->rows[dependency_index & LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_MASK] =
      dependency;
  ++graph->visibility.count;
  return iree_ok_status();
}
