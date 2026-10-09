// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/branch_layout.h"

#include <inttypes.h>
#include <limits.h>
#include <string.h>

typedef struct loom_amdgpu_branch_layout_path_node_t {
  // Original edge whose trampoline path owns this island.
  uint32_t edge_index;
  // Input anchor holding this island.
  uint32_t anchor_index;
  // Next path node, or LOOM_AMDGPU_BRANCH_ISLAND_NONE for the final block.
  uint32_t next_node_index;
  // Creation-order position among islands sharing |anchor_index|.
  uint16_t anchor_ordinal;
} loom_amdgpu_branch_layout_path_node_t;

// Per-anchor scratch storage with phase-specific contents. Relaxation records
// the number of nodes at each anchor. Export replaces each nonzero count with
// the first final island-table index for that anchor.
typedef union loom_amdgpu_branch_layout_anchor_state_t {
  // Number of nodes at this anchor during relaxation.
  uint32_t node_count;
  // First final island-table index at this anchor during export.
  uint32_t island_start;
} loom_amdgpu_branch_layout_anchor_state_t;

typedef struct loom_amdgpu_branch_layout_build_state_t {
  // Immutable measured native layout.
  const loom_amdgpu_branch_layout_input_t* input;
  // Arena owning transient relaxation tables.
  iree_arena_allocator_t* arena;
  // First path node for each original edge.
  uint32_t* edge_head_node_indices;
  // Mutable island nodes in creation order.
  loom_amdgpu_branch_layout_path_node_t* nodes;
  // Number of initialized entries in |nodes|.
  uint32_t node_count;
  // Allocated entry capacity of |nodes|.
  iree_host_size_t node_capacity;
  // Per-anchor node counts, replaced with final starts during export.
  loom_amdgpu_branch_layout_anchor_state_t* anchors;
  // Fenwick tree of inserted byte lengths keyed by input anchor index.
  uint64_t* inserted_byte_tree;
  // Number of anchors containing at least one island.
  uint32_t group_count;
  // Total byte length inserted across all anchor groups.
  uint64_t inserted_byte_count;
} loom_amdgpu_branch_layout_build_state_t;

static bool loom_amdgpu_branch_layout_offset_fits(uint64_t source_byte_offset,
                                                  uint64_t target_byte_offset,
                                                  int16_t* out_displacement) {
  IREE_ASSERT_LE(source_byte_offset, (uint64_t)INT64_MAX - 4u);
  IREE_ASSERT_LE(target_byte_offset, (uint64_t)INT64_MAX);
  const int64_t relative_byte_offset =
      (int64_t)target_byte_offset - ((int64_t)source_byte_offset + 4);
  IREE_ASSERT_EQ(relative_byte_offset % 4, 0);
  const int64_t relative_dword_offset = relative_byte_offset / 4;
  if (relative_dword_offset < INT16_MIN || relative_dword_offset > INT16_MAX) {
    return false;
  }
  if (out_displacement != NULL) {
    *out_displacement = (int16_t)relative_dword_offset;
  }
  return true;
}

// Returns the first anchor after |byte_offset| when |include_equal| is true,
// or the first anchor at |byte_offset| otherwise. The returned index is also
// the number of anchor groups preceding the translated native position.
static iree_host_size_t loom_amdgpu_branch_layout_anchor_bound(
    const loom_amdgpu_branch_layout_input_t* input, uint64_t byte_offset,
    bool include_equal) {
  iree_host_size_t first = 0;
  iree_host_size_t last = input->anchor_count;
  while (first < last) {
    const iree_host_size_t mid = first + (last - first) / 2u;
    const uint64_t anchor_byte_offset = input->anchors[mid].byte_offset;
    if (anchor_byte_offset < byte_offset ||
        (include_equal && anchor_byte_offset == byte_offset)) {
      first = mid + 1u;
    } else {
      last = mid;
    }
  }
  return first;
}

static uint64_t loom_amdgpu_branch_layout_query_inserted_bytes(
    const loom_amdgpu_branch_layout_build_state_t* state,
    iree_host_size_t anchor_count) {
  uint64_t inserted_byte_count = 0;
  while (anchor_count != 0) {
    inserted_byte_count += state->inserted_byte_tree[anchor_count - 1u];
    anchor_count &= anchor_count - 1u;
  }
  return inserted_byte_count;
}

static void loom_amdgpu_branch_layout_add_inserted_bytes(
    loom_amdgpu_branch_layout_build_state_t* state, uint32_t anchor_index,
    uint64_t byte_count) {
  for (iree_host_size_t tree_index = anchor_index;
       tree_index < state->input->anchor_count; tree_index |= tree_index + 1u) {
    state->inserted_byte_tree[tree_index] += byte_count;
  }
}

static uint64_t loom_amdgpu_branch_layout_translate_offset(
    const loom_amdgpu_branch_layout_build_state_t* state, uint64_t byte_offset,
    bool include_equal_groups) {
  if (state->inserted_byte_tree == NULL) {
    return byte_offset;
  }
  const iree_host_size_t anchor_count = loom_amdgpu_branch_layout_anchor_bound(
      state->input, byte_offset, include_equal_groups);
  return byte_offset +
         loom_amdgpu_branch_layout_query_inserted_bytes(state, anchor_count);
}

static uint64_t loom_amdgpu_branch_layout_node_final_offset(
    const loom_amdgpu_branch_layout_build_state_t* state,
    const loom_amdgpu_branch_layout_path_node_t* node) {
  const uint64_t group_byte_offset =
      state->input->anchors[node->anchor_index].byte_offset +
      loom_amdgpu_branch_layout_query_inserted_bytes(state, node->anchor_index);
  return group_byte_offset + ((uint64_t)node->anchor_ordinal + 1u) * 4u;
}

// Selects the lowest-index anchor nearest the midpoint of the open segment.
// Path nodes are monotonic in measured byte offset: every node was itself
// selected strictly between its adjacent endpoints. The open segment therefore
// contains no anchor already used by this path and needs no membership scan.
static uint32_t loom_amdgpu_branch_layout_select_midpoint_anchor(
    const loom_amdgpu_branch_layout_build_state_t* state,
    uint64_t source_base_byte_offset, uint64_t target_base_byte_offset) {
  const uint64_t lower = source_base_byte_offset < target_base_byte_offset
                             ? source_base_byte_offset
                             : target_base_byte_offset;
  const uint64_t upper = source_base_byte_offset < target_base_byte_offset
                             ? target_base_byte_offset
                             : source_base_byte_offset;
  const uint64_t midpoint = lower + (upper - lower) / 2u;
  const iree_host_size_t begin = loom_amdgpu_branch_layout_anchor_bound(
      state->input, lower, /*include_equal=*/true);
  const iree_host_size_t end = loom_amdgpu_branch_layout_anchor_bound(
      state->input, upper, /*include_equal=*/false);
  if (begin == end) {
    return LOOM_AMDGPU_BRANCH_ISLAND_NONE;
  }

  iree_host_size_t first = begin;
  iree_host_size_t last = end;
  while (first < last) {
    const iree_host_size_t mid = first + (last - first) / 2u;
    if (state->input->anchors[mid].byte_offset < midpoint) {
      first = mid + 1u;
    } else {
      last = mid;
    }
  }
  const iree_host_size_t right_index = first;
  if (right_index == begin) {
    return (uint32_t)right_index;
  }
  const uint64_t left_offset =
      state->input->anchors[right_index - 1u].byte_offset;
  if (right_index != end) {
    const uint64_t right_offset =
        state->input->anchors[right_index].byte_offset;
    if (right_offset - midpoint < midpoint - left_offset) {
      return (uint32_t)right_index;
    }
  }

  // |right_index - 1| may be the last of several anchors at the selected
  // offset. Recover the first so ties retain input order.
  first = begin;
  last = right_index;
  while (first < last) {
    const iree_host_size_t mid = first + (last - first) / 2u;
    if (state->input->anchors[mid].byte_offset < left_offset) {
      first = mid + 1u;
    } else {
      last = mid;
    }
  }
  return (uint32_t)first;
}

static iree_status_t loom_amdgpu_branch_layout_initialize_anchor_index(
    loom_amdgpu_branch_layout_build_state_t* state) {
  if (state->anchors != NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->input->anchor_count, sizeof(*state->anchors),
      (void**)&state->anchors));
  memset(state->anchors, 0,
         state->input->anchor_count * sizeof(*state->anchors));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->input->anchor_count,
      sizeof(*state->inserted_byte_tree), (void**)&state->inserted_byte_tree));
  memset(state->inserted_byte_tree, 0,
         state->input->anchor_count * sizeof(*state->inserted_byte_tree));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_branch_layout_grow_nodes(
    loom_amdgpu_branch_layout_build_state_t* state) {
  if (state->node_count < state->node_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(state->arena, state->node_count,
                               state->node_count + 1u, sizeof(*state->nodes),
                               &state->node_capacity, (void**)&state->nodes);
}

static iree_status_t loom_amdgpu_branch_layout_insert_island(
    loom_amdgpu_branch_layout_build_state_t* state, uint32_t edge_index,
    uint32_t source_node_index, uint32_t target_node_index,
    uint64_t source_base_byte_offset, uint64_t target_base_byte_offset,
    uint32_t* out_node_index) {
  const uint32_t anchor_index =
      loom_amdgpu_branch_layout_select_midpoint_anchor(
          state, source_base_byte_offset, target_base_byte_offset);
  if (anchor_index == LOOM_AMDGPU_BRANCH_ISLAND_NONE) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU branch relaxation has no packet boundary between native "
        "offsets %" PRIu64 " and %" PRIu64,
        source_base_byte_offset, target_base_byte_offset);
  }
  if (state->node_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU branch-island count overflowed");
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_branch_layout_initialize_anchor_index(state));
  const uint32_t anchor_node_count = state->anchors[anchor_index].node_count;
  if (anchor_node_count == INT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU branch relaxation requires more than %d co-located islands, "
        "exceeding the SOPP skip range",
        INT16_MAX);
  }
  const uint64_t inserted_byte_count = anchor_node_count == 0 ? 8u : 4u;
  if (inserted_byte_count > (uint64_t)INT64_MAX - state->input->byte_length -
                                state->inserted_byte_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU relaxed branch layout overflowed");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_branch_layout_grow_nodes(state));

  const uint32_t new_node_index = state->node_count++;
  state->nodes[new_node_index] = (loom_amdgpu_branch_layout_path_node_t){
      .edge_index = edge_index,
      .anchor_index = anchor_index,
      .next_node_index = target_node_index,
      .anchor_ordinal = (uint16_t)anchor_node_count,
  };
  state->anchors[anchor_index].node_count = anchor_node_count + 1u;
  if (anchor_node_count == 0) {
    ++state->group_count;
  }
  state->inserted_byte_count += inserted_byte_count;
  loom_amdgpu_branch_layout_add_inserted_bytes(state, anchor_index,
                                               inserted_byte_count);
  if (source_node_index == LOOM_AMDGPU_BRANCH_ISLAND_NONE) {
    state->edge_head_node_indices[edge_index] = new_node_index;
  } else {
    state->nodes[source_node_index].next_node_index = new_node_index;
  }
  *out_node_index = new_node_index;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_branch_layout_relax_long_segments(
    loom_amdgpu_branch_layout_build_state_t* state, bool* out_changed) {
  *out_changed = false;
  for (uint32_t edge_index = 0; edge_index < state->input->edge_count;
       ++edge_index) {
    const loom_amdgpu_branch_layout_input_edge_t* edge =
        &state->input->edges[edge_index];
    uint32_t source_node_index = LOOM_AMDGPU_BRANCH_ISLAND_NONE;
    uint64_t source_base_byte_offset = edge->source_byte_offset;
    uint64_t source_final_byte_offset =
        loom_amdgpu_branch_layout_translate_offset(
            state, edge->source_byte_offset, /*include_equal_groups=*/true);
    uint32_t target_node_index = state->edge_head_node_indices[edge_index];
    while (true) {
      uint64_t target_base_byte_offset = 0;
      uint64_t target_final_byte_offset = 0;
      if (target_node_index == LOOM_AMDGPU_BRANCH_ISLAND_NONE) {
        const uint32_t target_block_index = edge->target_block_index;
        IREE_ASSERT_LT(target_block_index, state->input->block_count);
        target_base_byte_offset =
            state->input->blocks[target_block_index].byte_offset;
        target_final_byte_offset = loom_amdgpu_branch_layout_translate_offset(
            state, target_base_byte_offset, /*include_equal_groups=*/false);
      } else {
        const loom_amdgpu_branch_layout_path_node_t* target_node =
            &state->nodes[target_node_index];
        target_base_byte_offset =
            state->input->anchors[target_node->anchor_index].byte_offset;
        target_final_byte_offset =
            loom_amdgpu_branch_layout_node_final_offset(state, target_node);
      }
      if (!loom_amdgpu_branch_layout_offset_fits(
              source_final_byte_offset, target_final_byte_offset, NULL)) {
        uint32_t new_node_index = LOOM_AMDGPU_BRANCH_ISLAND_NONE;
        IREE_RETURN_IF_ERROR(loom_amdgpu_branch_layout_insert_island(
            state, edge_index, source_node_index, target_node_index,
            source_base_byte_offset, target_base_byte_offset, &new_node_index));
        *out_changed = true;
        source_final_byte_offset =
            source_node_index == LOOM_AMDGPU_BRANCH_ISLAND_NONE
                ? loom_amdgpu_branch_layout_translate_offset(
                      state, edge->source_byte_offset,
                      /*include_equal_groups=*/true)
                : loom_amdgpu_branch_layout_node_final_offset(
                      state, &state->nodes[source_node_index]);
        target_node_index = new_node_index;
        continue;
      }
      if (target_node_index == LOOM_AMDGPU_BRANCH_ISLAND_NONE) {
        break;
      }
      source_node_index = target_node_index;
      const loom_amdgpu_branch_layout_path_node_t* source_node =
          &state->nodes[source_node_index];
      source_base_byte_offset =
          state->input->anchors[source_node->anchor_index].byte_offset;
      source_final_byte_offset =
          loom_amdgpu_branch_layout_node_final_offset(state, source_node);
      target_node_index = source_node->next_node_index;
    }
  }
  return iree_ok_status();
}

static uint32_t loom_amdgpu_branch_layout_node_final_index(
    const loom_amdgpu_branch_layout_build_state_t* state,
    const loom_amdgpu_branch_layout_path_node_t* node) {
  return state->anchors[node->anchor_index].island_start + node->anchor_ordinal;
}

static loom_amdgpu_branch_target_t loom_amdgpu_branch_layout_node_target(
    const loom_amdgpu_branch_layout_build_state_t* state,
    const loom_amdgpu_branch_layout_input_edge_t* edge, uint32_t node_index) {
  if (node_index == LOOM_AMDGPU_BRANCH_ISLAND_NONE) {
    return (loom_amdgpu_branch_target_t){
        .kind = LOOM_AMDGPU_BRANCH_TARGET_BLOCK,
        .index = edge->target_block_index,
    };
  }
  return (loom_amdgpu_branch_target_t){
      .kind = LOOM_AMDGPU_BRANCH_TARGET_ISLAND,
      .index = loom_amdgpu_branch_layout_node_final_index(
          state, &state->nodes[node_index]),
  };
}

static uint64_t loom_amdgpu_branch_layout_target_final_offset(
    const loom_amdgpu_branch_layout_build_state_t* state,
    const loom_amdgpu_branch_layout_input_edge_t* edge, uint32_t node_index) {
  if (node_index != LOOM_AMDGPU_BRANCH_ISLAND_NONE) {
    return loom_amdgpu_branch_layout_node_final_offset(
        state, &state->nodes[node_index]);
  }
  const uint64_t block_byte_offset =
      state->input->blocks[edge->target_block_index].byte_offset;
  return loom_amdgpu_branch_layout_translate_offset(
      state, block_byte_offset, /*include_equal_groups=*/false);
}

static iree_status_t loom_amdgpu_branch_layout_export(
    loom_amdgpu_branch_layout_build_state_t* state,
    iree_arena_allocator_t* output_arena,
    loom_amdgpu_branch_layout_t* out_layout) {
  if (state->node_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_branch_layout_edge_t* edges = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      output_arena, state->input->edge_count, sizeof(*edges), (void**)&edges));
  loom_amdgpu_branch_layout_island_t* islands = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      output_arena, state->node_count, sizeof(*islands), (void**)&islands));
  loom_amdgpu_branch_layout_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      output_arena, state->group_count, sizeof(*groups), (void**)&groups));

  uint32_t group_index = 0;
  uint32_t island_start = 0;
  for (uint32_t anchor_index = 0; anchor_index < state->input->anchor_count;
       ++anchor_index) {
    const uint32_t node_count = state->anchors[anchor_index].node_count;
    if (node_count == 0) {
      continue;
    }
    groups[group_index++] = (loom_amdgpu_branch_layout_group_t){
        .packet_index = state->input->anchors[anchor_index].packet_index,
        .island_start = island_start,
        .island_count = node_count,
    };
    state->anchors[anchor_index].island_start = island_start;
    island_start += node_count;
  }
  IREE_ASSERT_EQ(group_index, state->group_count);
  IREE_ASSERT_EQ(island_start, state->node_count);

  for (uint32_t edge_index = 0; edge_index < state->input->edge_count;
       ++edge_index) {
    const loom_amdgpu_branch_layout_input_edge_t* input_edge =
        &state->input->edges[edge_index];
    const uint32_t target_node_index =
        state->edge_head_node_indices[edge_index];
    const uint64_t source_final_byte_offset =
        loom_amdgpu_branch_layout_translate_offset(
            state, input_edge->source_byte_offset,
            /*include_equal_groups=*/true);
    const uint64_t target_final_byte_offset =
        loom_amdgpu_branch_layout_target_final_offset(state, input_edge,
                                                      target_node_index);
    int16_t displacement = 0;
    const bool displacement_fits = loom_amdgpu_branch_layout_offset_fits(
        source_final_byte_offset, target_final_byte_offset, &displacement);
    IREE_ASSERT(displacement_fits);
    (void)displacement_fits;
    edges[edge_index] = (loom_amdgpu_branch_layout_edge_t){
        .target = loom_amdgpu_branch_layout_node_target(state, input_edge,
                                                        target_node_index),
        .relative_dword_offset = displacement,
    };
  }

  for (uint32_t node_index = 0; node_index < state->node_count; ++node_index) {
    const loom_amdgpu_branch_layout_path_node_t* node =
        &state->nodes[node_index];
    const uint32_t owner_edge_index = node->edge_index;
    IREE_ASSERT_LT(owner_edge_index, state->input->edge_count);
    const loom_amdgpu_branch_layout_input_edge_t* owner_edge =
        &state->input->edges[owner_edge_index];
    const uint64_t source_final_byte_offset =
        loom_amdgpu_branch_layout_node_final_offset(state, node);
    const uint64_t target_final_byte_offset =
        loom_amdgpu_branch_layout_target_final_offset(state, owner_edge,
                                                      node->next_node_index);
    int16_t displacement = 0;
    const bool displacement_fits = loom_amdgpu_branch_layout_offset_fits(
        source_final_byte_offset, target_final_byte_offset, &displacement);
    IREE_ASSERT(displacement_fits);
    (void)displacement_fits;
    const uint32_t final_island_index =
        loom_amdgpu_branch_layout_node_final_index(state, node);
    islands[final_island_index] = (loom_amdgpu_branch_layout_island_t){
        .target = loom_amdgpu_branch_layout_node_target(state, owner_edge,
                                                        node->next_node_index),
        .relative_dword_offset = displacement,
    };
  }

  *out_layout = (loom_amdgpu_branch_layout_t){
      .byte_length = state->input->byte_length + state->inserted_byte_count,
      .edges = edges,
      .edge_count = state->input->edge_count,
      .islands = islands,
      .island_count = state->node_count,
      .groups = groups,
      .group_count = state->group_count,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_branch_layout_build(
    const loom_amdgpu_branch_layout_input_t* input,
    iree_arena_allocator_t* arena, loom_amdgpu_branch_layout_t* out_layout) {
  *out_layout = (loom_amdgpu_branch_layout_t){0};
  if (input->edge_count == 0) {
    return iree_ok_status();
  }
  if (input->byte_length > (uint64_t)INT64_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native instruction stream exceeds signed layout range");
  }
  if (input->block_count > UINT32_MAX || input->edge_count > UINT32_MAX ||
      input->anchor_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU branch layout exceeds 32-bit table capacity");
  }
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  loom_amdgpu_branch_layout_build_state_t state = {
      .input = input,
      .arena = &scratch_arena,
  };
  iree_status_t status = iree_arena_allocate_array(
      &scratch_arena, input->edge_count, sizeof(*state.edge_head_node_indices),
      (void**)&state.edge_head_node_indices);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < input->edge_count; ++i) {
      state.edge_head_node_indices[i] = LOOM_AMDGPU_BRANCH_ISLAND_NONE;
    }
  }
  bool changed = true;
  while (iree_status_is_ok(status) && changed) {
    changed = false;
    status = loom_amdgpu_branch_layout_relax_long_segments(&state, &changed);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_branch_layout_export(&state, arena, out_layout);
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
