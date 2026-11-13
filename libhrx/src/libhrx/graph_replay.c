// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "graph_replay.h"

iree_status_t hrx_graph_block_allocate(iree_arena_allocator_t* arena,
                                       uint8_t type,
                                       hrx_graph_dependencies_t dependencies,
                                       iree_host_size_t payload_size,
                                       hrx_graph_block_t** out_block) {
  hrx_graph_block_t* block = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*block) + payload_size,
                                           (void**)&block));
  block->type = type;
  block->dependencies = dependencies;
  *out_block = block;
  return iree_ok_status();
}
