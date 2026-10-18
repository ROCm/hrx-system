// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/dependencies.h"

#include <string.h>

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

iree_status_t loom_low_schedule_dependency_graph_retain_range(
    const loom_low_schedule_dependency_graph_t* graph,
    loom_low_schedule_dependency_range_t range, iree_arena_allocator_t* arena,
    loom_low_schedule_dependency_slice_t* out_slice) {
  *out_slice = (loom_low_schedule_dependency_slice_t){0};
  if (range.count == 0) {
    return iree_ok_status();
  }
  const uint32_t offset =
      range.start & LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_MASK;
  const uint32_t first_segment =
      range.start >> LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_SHIFT;
  const uint32_t segment_count = ((offset + range.count - 1) >>
                                  LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_SHIFT) +
                                 1;
  const loom_low_schedule_dependency_segment_t** segments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, segment_count, sizeof(*segments), (void**)&segments));
  for (uint32_t i = 0; i < segment_count; ++i) {
    segments[i] = (const loom_low_schedule_dependency_segment_t*)
        loom_segmented_storage_const_segment(&graph->segments,
                                             first_segment + i);
  }
  if (offset != 0) {
    loom_low_schedule_dependency_segment_t* first = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_aligned(
        arena, sizeof(*first),
        iree_alignof(loom_low_schedule_dependency_segment_t), (void**)&first));
    const uint32_t count = iree_min(
        range.count, LOOM_LOW_SCHEDULE_DEPENDENCY_SEGMENT_CAPACITY - offset);
    memcpy(first->rows + offset, segments[0]->rows + offset,
           count * sizeof(*first->rows));
    segments[0] = first;
  }
  *out_slice = (loom_low_schedule_dependency_slice_t){
      .segments = segments,
      .offset = offset,
      .count = range.count,
  };
  return iree_ok_status();
}
