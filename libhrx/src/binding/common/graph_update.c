// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph_update.h"

// A cursor visits each physical node once, including empty blocks left by copy
// rollback. Only visible nodes participate in update correspondence.
typedef struct iree_hal_streaming_graph_update_nodes_t {
  // Borrowed current block; NULL at the end of the description.
  const iree_hal_streaming_node_block_t* block;
  // Next physical entry within |block|.
  iree_host_size_t position;
} iree_hal_streaming_graph_update_nodes_t;

static iree_hal_streaming_graph_node_t*
iree_hal_streaming_graph_update_next_node(
    iree_hal_streaming_graph_update_nodes_t* cursor) {
  while (cursor->block) {
    if (cursor->position == cursor->block->count) {
      cursor->block = cursor->block->next;
      cursor->position = 0;
    } else {
      iree_hal_streaming_graph_node_t* node =
          cursor->block->nodes[cursor->position++];
      if (!(node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN)) {
        return node;
      }
    }
  }
  return NULL;
}

typedef struct iree_hal_streaming_graph_update_dependencies_t {
  // Borrowed node owning the embedded predecessor array.
  const iree_hal_streaming_graph_node_t* node;
  // Next embedded predecessor.
  uint32_t position;
  // Next separately attached predecessor, after embedded predecessors.
  const iree_hal_streaming_graph_dependency_t* edge;
} iree_hal_streaming_graph_update_dependencies_t;

static const iree_hal_streaming_graph_node_t*
iree_hal_streaming_graph_update_next_dependency(
    iree_hal_streaming_graph_update_dependencies_t* cursor) {
  while (cursor->position < cursor->node->dependency_count || cursor->edge) {
    const iree_hal_streaming_graph_node_t* predecessor;
    if (cursor->position < cursor->node->dependency_count) {
      predecessor = iree_hal_streaming_graph_node_dependencies(
          cursor->node)[cursor->position++];
    } else {
      predecessor = cursor->edge->from;
      cursor->edge = cursor->edge->next;
    }
    if (!(predecessor->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_HIDDEN)) {
      return predecessor;
    }
  }
  return NULL;
}

// Scratch is shared by all predecessor comparisons at one nesting level. It is
// untouched for helper-free descriptions whose predecessors have matching
// order.
typedef struct iree_hal_streaming_graph_update_scratch_t {
  // Existing executable compilation arena, borrowed under its mutation lock.
  iree_arena_allocator_t* arena;
  // Physical old-node count, bounding all predecessor indices.
  iree_host_size_t node_count;
  // New physical indices translated to old physical indices; NULL is identity.
  uint32_t* correspondence;
  // One bit per old physical node, allocated only for unordered comparisons.
  uint64_t* remaining;
} iree_hal_streaming_graph_update_scratch_t;

static uint32_t iree_hal_streaming_graph_update_dependency_index(
    const iree_hal_streaming_graph_update_scratch_t* scratch,
    const iree_hal_streaming_graph_node_t* node) {
  return scratch->correspondence ? scratch->correspondence[node->node_index]
                                 : node->node_index;
}

static iree_status_t iree_hal_streaming_graph_update_dependencies_match(
    const iree_hal_streaming_graph_node_t* old_node,
    const iree_hal_streaming_graph_node_t* new_node,
    iree_hal_streaming_graph_update_scratch_t* scratch, bool* out_match) {
  *out_match = false;
  iree_hal_streaming_graph_update_dependencies_t old_cursor = {
      .node = old_node,
      .edge = iree_hal_streaming_graph_additional_dependencies(old_node)};
  iree_hal_streaming_graph_update_dependencies_t new_cursor = {
      .node = new_node,
      .edge = iree_hal_streaming_graph_additional_dependencies(new_node)};
  const iree_hal_streaming_graph_node_t* old_dependency =
      iree_hal_streaming_graph_update_next_dependency(&old_cursor);
  const iree_hal_streaming_graph_node_t* new_dependency =
      iree_hal_streaming_graph_update_next_dependency(&new_cursor);
  while (old_dependency && new_dependency &&
         old_dependency->node_index ==
             iree_hal_streaming_graph_update_dependency_index(scratch,
                                                              new_dependency)) {
    old_dependency =
        iree_hal_streaming_graph_update_next_dependency(&old_cursor);
    new_dependency =
        iree_hal_streaming_graph_update_next_dependency(&new_cursor);
  }
  if (!old_dependency || !new_dependency) {
    *out_match = old_dependency == new_dependency;
    return iree_ok_status();
  }

  // Graph mutation already guarantees unique predecessors. Mark only the
  // unmatched suffix, then consume each replacement predecessor exactly once.
  // Success leaves the bitset empty for reuse by the next node. A mismatch ends
  // the comparison, so its residual bits are discarded with the checkpoint.
  if (!scratch->remaining) {
    iree_host_size_t byte_length =
        iree_host_size_ceil_div(scratch->node_count, 64) * sizeof(uint64_t);
    IREE_RETURN_IF_ERROR(iree_arena_allocate(scratch->arena, byte_length,
                                             (void**)&scratch->remaining));
    memset(scratch->remaining, 0, byte_length);
  }
  iree_host_size_t remaining_count = 0;
  for (; old_dependency;
       old_dependency =
           iree_hal_streaming_graph_update_next_dependency(&old_cursor)) {
    uint32_t index = old_dependency->node_index;
    scratch->remaining[index / 64] |= UINT64_C(1) << (index % 64);
    ++remaining_count;
  }
  bool match = true;
  for (; match && new_dependency;
       new_dependency =
           iree_hal_streaming_graph_update_next_dependency(&new_cursor)) {
    uint32_t index = iree_hal_streaming_graph_update_dependency_index(
        scratch, new_dependency);
    uint64_t bit = UINT64_C(1) << (index % 64);
    match = (scratch->remaining[index / 64] & bit) != 0;
    if (match) {
      scratch->remaining[index / 64] &= ~bit;
      --remaining_count;
    }
  }
  *out_match = match && remaining_count == 0;
  return iree_ok_status();
}

static bool iree_hal_streaming_graph_memcpy_update_is_compatible(
    const iree_hal_streaming_graph_memcpy_node_attrs_t* old_attrs,
    const iree_hal_streaming_graph_memcpy_node_attrs_t* new_attrs) {
  return old_attrs->hip_kind == new_attrs->hip_kind &&
         old_attrs->size == new_attrs->size &&
         old_attrs->execution_extent_width ==
             new_attrs->execution_extent_width &&
         old_attrs->execution_extent_height ==
             new_attrs->execution_extent_height &&
         old_attrs->execution_extent_depth ==
             new_attrs->execution_extent_depth &&
         old_attrs->execution_dst_pitch == new_attrs->execution_dst_pitch &&
         old_attrs->execution_src_pitch == new_attrs->execution_src_pitch &&
         old_attrs->execution_dst_ysize == new_attrs->execution_dst_ysize &&
         old_attrs->execution_src_ysize == new_attrs->execution_src_ysize &&
         old_attrs->hip_extent_width == new_attrs->hip_extent_width &&
         old_attrs->hip_extent_height == new_attrs->hip_extent_height &&
         old_attrs->hip_extent_depth == new_attrs->hip_extent_depth &&
         old_attrs->hip_dst_pitch == new_attrs->hip_dst_pitch &&
         old_attrs->hip_src_pitch == new_attrs->hip_src_pitch &&
         old_attrs->hip_dst_xsize == new_attrs->hip_dst_xsize &&
         old_attrs->hip_src_xsize == new_attrs->hip_src_xsize &&
         old_attrs->hip_dst_ysize == new_attrs->hip_dst_ysize &&
         old_attrs->hip_src_ysize == new_attrs->hip_src_ysize;
}

static bool iree_hal_streaming_graph_memset_update_is_compatible(
    const iree_hal_streaming_graph_memset_node_attrs_t* old_attrs,
    const iree_hal_streaming_graph_memset_node_attrs_t* new_attrs) {
  return old_attrs->pattern_size == new_attrs->pattern_size &&
         old_attrs->count == new_attrs->count &&
         old_attrs->hip_width == new_attrs->hip_width &&
         old_attrs->hip_height == new_attrs->hip_height &&
         old_attrs->hip_pitch == new_attrs->hip_pitch;
}

static iree_status_t iree_hal_streaming_graph_check_update_pair(
    const iree_hal_streaming_graph_t* old_graph,
    const iree_hal_streaming_graph_t* new_graph, iree_arena_allocator_t* arena,
    iree_hal_streaming_graph_node_t** out_error_node,
    iree_hal_streaming_graph_exec_update_result_t* out_result) {
  iree_hal_streaming_graph_update_nodes_t old_cursor = {
      .block = old_graph->node_blocks};
  iree_hal_streaming_graph_update_nodes_t new_cursor = {
      .block = new_graph->node_blocks};
  if (old_graph->context != new_graph->context) {
    iree_hal_streaming_graph_node_t* first =
        iree_hal_streaming_graph_update_next_node(&new_cursor);
    iree_hal_streaming_graph_node_t* node = first;
    while (node && node->type != IREE_HAL_STREAMING_GRAPH_NODE_TYPE_KERNEL) {
      node = iree_hal_streaming_graph_update_next_node(&new_cursor);
    }
    *out_error_node = node ? node : first;
    *out_result =
        IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_UNSUPPORTED_FUNCTION_CHANGE;
    return iree_ok_status();
  }
  iree_host_size_t visible_count =
      old_graph->node_count - old_graph->copy_helper_count;
  if (visible_count != new_graph->node_count - new_graph->copy_helper_count) {
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED;
    return iree_ok_status();
  }
  iree_hal_streaming_graph_update_scratch_t scratch = {
      .arena = arena, .node_count = old_graph->node_count};
  if (visible_count &&
      (old_graph->copy_helper_count || new_graph->copy_helper_count)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        arena, new_graph->node_count * sizeof(*scratch.correspondence),
        (void**)&scratch.correspondence));
    for (iree_host_size_t i = 0; i < visible_count; ++i) {
      iree_hal_streaming_graph_node_t* old_node =
          iree_hal_streaming_graph_update_next_node(&old_cursor);
      iree_hal_streaming_graph_node_t* new_node =
          iree_hal_streaming_graph_update_next_node(&new_cursor);
      scratch.correspondence[new_node->node_index] = old_node->node_index;
    }
    old_cursor = (iree_hal_streaming_graph_update_nodes_t){
        .block = old_graph->node_blocks};
    new_cursor = (iree_hal_streaming_graph_update_nodes_t){
        .block = new_graph->node_blocks};
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < visible_count && iree_status_is_ok(status) &&
       *out_result == IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS;
       ++i) {
    iree_hal_streaming_graph_node_t* old_node =
        iree_hal_streaming_graph_update_next_node(&old_cursor);
    iree_hal_streaming_graph_node_t* new_node =
        iree_hal_streaming_graph_update_next_node(&new_cursor);
    if (old_node->type != new_node->type) {
      *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_NODE_TYPE_CHANGED;
    } else {
      bool dependencies_match = false;
      status = iree_hal_streaming_graph_update_dependencies_match(
          old_node, new_node, &scratch, &dependencies_match);
      if (!iree_status_is_ok(status)) {
        *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;
      } else if (!dependencies_match) {
        *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_TOPOLOGY_CHANGED;
      } else {
        switch (old_node->type) {
          case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMCPY:
            if (!iree_hal_streaming_graph_memcpy_update_is_compatible(
                    iree_hal_streaming_graph_node_memcpy(old_node),
                    iree_hal_streaming_graph_node_memcpy(new_node))) {
              *out_result =
                  IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_PARAMETERS_CHANGED;
            }
            break;
          case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_MEMSET:
            if (!iree_hal_streaming_graph_memset_update_is_compatible(
                    iree_hal_streaming_graph_node_memset(old_node),
                    iree_hal_streaming_graph_node_memset(new_node))) {
              *out_result =
                  IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_PARAMETERS_CHANGED;
            }
            break;
          case IREE_HAL_STREAMING_GRAPH_NODE_TYPE_GRAPH:
            status = iree_hal_streaming_graph_check_update(
                iree_hal_streaming_graph_node_child_graph(old_node)->graph,
                iree_hal_streaming_graph_node_child_graph(new_node)->graph,
                arena, out_error_node, out_result);
            break;
          default:
            break;
        }
      }
    }
    if (*out_result != IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS) {
      *out_error_node = new_node;
    }
  }
  return status;
}

iree_status_t iree_hal_streaming_graph_check_update(
    const iree_hal_streaming_graph_t* old_graph,
    const iree_hal_streaming_graph_t* new_graph, iree_arena_allocator_t* arena,
    iree_hal_streaming_graph_node_t** out_error_node,
    iree_hal_streaming_graph_exec_update_result_t* out_result) {
  *out_error_node = NULL;
  *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS;
  iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(arena);
  iree_status_t status = iree_hal_streaming_graph_check_update_pair(
      old_graph, new_graph, arena, out_error_node, out_result);
  iree_arena_checkpoint_restore(&checkpoint);
  if (!iree_status_is_ok(status)) {
    *out_result = IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;
  }
  return status;
}
