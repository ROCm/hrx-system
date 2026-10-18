// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/dependency_incoming.h"

#include <string.h>

iree_status_t loom_low_schedule_dependency_incoming_index_initialize(
    const loom_low_schedule_dependency_index_t* index,
    const loom_low_schedule_node_t* nodes, iree_arena_allocator_t* arena,
    loom_low_schedule_dependency_incoming_index_t* out_incoming_index) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_incoming_index);
  IREE_ASSERT(index->node_count == 0 || nodes != NULL);
  *out_incoming_index = (loom_low_schedule_dependency_incoming_index_t){
      .node_count = index->node_count,
  };
  if (index->node_count == 0) {
    return iree_ok_status();
  }

  const iree_host_size_t start_count = (iree_host_size_t)index->node_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, start_count, sizeof(*out_incoming_index->consumer_group_starts),
      (void**)&out_incoming_index->consumer_group_starts));
  uint32_t* starts = out_incoming_index->consumer_group_starts;
  memset(starts, 0, start_count * sizeof(*starts));
  for (uint32_t producer_node = 0; producer_node < index->node_count;
       ++producer_node) {
    const uint32_t group_end =
        loom_low_schedule_dependency_index_group_end(index, producer_node);
    for (uint32_t group_index = loom_low_schedule_dependency_index_group_begin(
             index, producer_node);
         group_index < group_end; ++group_index) {
      const uint32_t consumer_node =
          loom_low_schedule_dependency_index_group_at(index, group_index)
              ->consumer_node;
      if (nodes[producer_node].block_index ==
          nodes[consumer_node].block_index) {
        IREE_ASSERT_NE(starts[consumer_node + 1], UINT32_MAX);
        ++starts[consumer_node + 1];
      }
    }
  }
  for (uint32_t node_index = 0; node_index < index->node_count; ++node_index) {
    IREE_ASSERT_LE(starts[node_index], UINT32_MAX - starts[node_index + 1]);
    starts[node_index + 1] += starts[node_index];
  }
  out_incoming_index->group_count = starts[index->node_count];
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, out_incoming_index->group_count,
                                sizeof(*out_incoming_index->group_indices),
                                (void**)&out_incoming_index->group_indices));
  // Fill backwards from each row's end, reusing the offsets as cursors. The
  // decremented ends become the starts, avoiding another node-sized array.
  for (uint32_t producer_node = 0; producer_node < index->node_count;
       ++producer_node) {
    const uint32_t group_end =
        loom_low_schedule_dependency_index_group_end(index, producer_node);
    for (uint32_t group_index = loom_low_schedule_dependency_index_group_begin(
             index, producer_node);
         group_index < group_end; ++group_index) {
      const uint32_t consumer_node =
          loom_low_schedule_dependency_index_group_at(index, group_index)
              ->consumer_node;
      if (nodes[producer_node].block_index ==
          nodes[consumer_node].block_index) {
        out_incoming_index->group_indices[--starts[consumer_node + 1]] =
            group_index;
      }
    }
  }
  memmove(starts, starts + 1, index->node_count * sizeof(*starts));
  starts[index->node_count] = out_incoming_index->group_count;
  return iree_ok_status();
}
