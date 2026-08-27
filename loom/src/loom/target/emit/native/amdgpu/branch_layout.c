// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/branch_layout.h"

#include <inttypes.h>
#include <limits.h>

typedef struct loom_amdgpu_branch_layout_path_node_t {
  // Original edge whose trampoline path owns this island.
  uint32_t edge_index;
  // Input anchor holding this island.
  uint32_t anchor_index;
  // Next path node, or LOOM_AMDGPU_BRANCH_ISLAND_NONE for the final block.
  uint32_t next_node_index;
  // Island-table index assigned by the most recent physical layout.
  uint32_t final_island_index;
  // Final byte offset of the island branch instruction.
  uint64_t final_byte_offset;
} loom_amdgpu_branch_layout_path_node_t;

typedef struct loom_amdgpu_branch_layout_scratch_group_t {
  // Input anchor shared by the co-located islands.
  uint32_t anchor_index;
  // First node in the physically sorted scratch node-index table.
  uint32_t sorted_node_start;
  // Number of co-located island nodes.
  uint32_t node_count;
  // Final byte offset of the normal-flow skip branch.
  uint64_t final_byte_offset;
  // Total bytes occupied by the skip branch and its islands.
  uint64_t byte_length;
} loom_amdgpu_branch_layout_scratch_group_t;

typedef struct loom_amdgpu_branch_layout_build_state_t {
  // Immutable measured native layout.
  const loom_amdgpu_branch_layout_input_t* input;
  // Arena owning scratch and final tables.
  iree_arena_allocator_t* arena;
  // First path node for each original edge.
  uint32_t* edge_head_node_indices;
  // Mutable island nodes in creation order.
  loom_amdgpu_branch_layout_path_node_t* nodes;
  // Number of initialized entries in |nodes|.
  uint32_t node_count;
  // Allocated entry capacity of |nodes| and parallel scratch tables.
  iree_host_size_t node_capacity;
  // Node indices sorted into physical anchor order.
  uint32_t* sorted_node_indices;
  // Allocated entry capacity of |sorted_node_indices|.
  iree_host_size_t sorted_node_capacity;
  // Co-located groups rebuilt after every island insertion.
  loom_amdgpu_branch_layout_scratch_group_t* groups;
  // Allocated entry capacity of |groups|.
  iree_host_size_t group_capacity;
  // Number of initialized entries in |groups|.
  uint32_t group_count;
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

static bool loom_amdgpu_branch_layout_node_precedes(
    const loom_amdgpu_branch_layout_build_state_t* state,
    uint32_t lhs_node_index, uint32_t rhs_node_index) {
  const loom_amdgpu_branch_layout_path_node_t* lhs =
      &state->nodes[lhs_node_index];
  const loom_amdgpu_branch_layout_path_node_t* rhs =
      &state->nodes[rhs_node_index];
  if (lhs->anchor_index != rhs->anchor_index) {
    return lhs->anchor_index < rhs->anchor_index;
  }
  return lhs_node_index < rhs_node_index;
}

static void loom_amdgpu_branch_layout_sort_nodes(
    loom_amdgpu_branch_layout_build_state_t* state) {
  for (uint32_t i = 0; i < state->node_count; ++i) {
    state->sorted_node_indices[i] = i;
  }
  for (uint32_t i = 1; i < state->node_count; ++i) {
    const uint32_t key = state->sorted_node_indices[i];
    uint32_t position = i;
    while (position != 0 &&
           loom_amdgpu_branch_layout_node_precedes(
               state, key, state->sorted_node_indices[position - 1])) {
      state->sorted_node_indices[position] =
          state->sorted_node_indices[position - 1];
      --position;
    }
    state->sorted_node_indices[position] = key;
  }
}

static uint64_t loom_amdgpu_branch_layout_translate_offset(
    const loom_amdgpu_branch_layout_build_state_t* state, uint64_t byte_offset,
    bool include_equal_groups) {
  uint64_t translated_offset = byte_offset;
  for (uint32_t i = 0; i < state->group_count; ++i) {
    const loom_amdgpu_branch_layout_scratch_group_t* group = &state->groups[i];
    const uint64_t group_base_offset =
        state->input->anchors[group->anchor_index].byte_offset;
    if (group_base_offset > byte_offset ||
        (!include_equal_groups && group_base_offset == byte_offset)) {
      break;
    }
    translated_offset += group->byte_length;
  }
  return translated_offset;
}

static iree_status_t loom_amdgpu_branch_layout_rebuild_physical_layout(
    loom_amdgpu_branch_layout_build_state_t* state) {
  state->group_count = 0;
  if (state->node_count == 0) return iree_ok_status();
  loom_amdgpu_branch_layout_sort_nodes(state);

  uint64_t inserted_byte_count = 0;
  uint32_t sorted_node_index = 0;
  while (sorted_node_index < state->node_count) {
    const uint32_t first_node_index =
        state->sorted_node_indices[sorted_node_index];
    const uint32_t anchor_index = state->nodes[first_node_index].anchor_index;
    uint32_t group_node_count = 1;
    while (sorted_node_index + group_node_count < state->node_count) {
      const uint32_t next_node_index =
          state->sorted_node_indices[sorted_node_index + group_node_count];
      if (state->nodes[next_node_index].anchor_index != anchor_index) break;
      ++group_node_count;
    }
    if (group_node_count > INT16_MAX) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "AMDGPU branch relaxation requires %" PRIu32
          " co-located islands, exceeding the SOPP skip range",
          group_node_count);
    }
    const uint64_t group_byte_length = ((uint64_t)group_node_count + 1u) * 4u;
    const uint64_t base_byte_offset =
        state->input->anchors[anchor_index].byte_offset;
    const uint64_t remaining_layout_bytes =
        (uint64_t)INT64_MAX - state->input->byte_length - inserted_byte_count;
    if (group_byte_length > remaining_layout_bytes) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU relaxed branch layout overflowed");
    }
    const uint64_t final_group_byte_offset =
        base_byte_offset + inserted_byte_count;
    state->groups[state->group_count++] =
        (loom_amdgpu_branch_layout_scratch_group_t){
            .anchor_index = anchor_index,
            .sorted_node_start = sorted_node_index,
            .node_count = group_node_count,
            .final_byte_offset = final_group_byte_offset,
            .byte_length = group_byte_length,
        };
    for (uint32_t i = 0; i < group_node_count; ++i) {
      const uint32_t node_index =
          state->sorted_node_indices[sorted_node_index + i];
      state->nodes[node_index].final_island_index = sorted_node_index + i;
      state->nodes[node_index].final_byte_offset =
          final_group_byte_offset + ((uint64_t)i + 1u) * 4u;
    }
    inserted_byte_count += group_byte_length;
    sorted_node_index += group_node_count;
  }
  return iree_ok_status();
}

static bool loom_amdgpu_branch_layout_path_uses_anchor(
    const loom_amdgpu_branch_layout_build_state_t* state, uint32_t edge_index,
    uint32_t anchor_index) {
  uint32_t node_index = state->edge_head_node_indices[edge_index];
  while (node_index != LOOM_AMDGPU_BRANCH_ISLAND_NONE) {
    const loom_amdgpu_branch_layout_path_node_t* node =
        &state->nodes[node_index];
    if (node->anchor_index == anchor_index) return true;
    node_index = node->next_node_index;
  }
  return false;
}

static uint32_t loom_amdgpu_branch_layout_select_midpoint_anchor(
    const loom_amdgpu_branch_layout_build_state_t* state, uint32_t edge_index,
    uint64_t source_base_byte_offset, uint64_t target_base_byte_offset) {
  const uint64_t lower = source_base_byte_offset < target_base_byte_offset
                             ? source_base_byte_offset
                             : target_base_byte_offset;
  const uint64_t upper = source_base_byte_offset < target_base_byte_offset
                             ? target_base_byte_offset
                             : source_base_byte_offset;
  const uint64_t midpoint = lower + (upper - lower) / 2u;
  uint32_t selected_anchor_index = LOOM_AMDGPU_BRANCH_ISLAND_NONE;
  uint64_t selected_distance = UINT64_MAX;
  for (iree_host_size_t i = 0; i < state->input->anchor_count; ++i) {
    const uint64_t anchor_byte_offset = state->input->anchors[i].byte_offset;
    if (anchor_byte_offset <= lower || anchor_byte_offset >= upper) continue;
    if (loom_amdgpu_branch_layout_path_uses_anchor(state, edge_index,
                                                   (uint32_t)i)) {
      continue;
    }
    const uint64_t distance = anchor_byte_offset < midpoint
                                  ? midpoint - anchor_byte_offset
                                  : anchor_byte_offset - midpoint;
    if (distance < selected_distance) {
      selected_anchor_index = (uint32_t)i;
      selected_distance = distance;
    }
  }
  return selected_anchor_index;
}

static iree_status_t loom_amdgpu_branch_layout_grow_nodes(
    loom_amdgpu_branch_layout_build_state_t* state) {
  if (state->node_count < state->node_capacity) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      state->arena, state->node_count, state->node_count + 1,
      sizeof(*state->nodes), &state->node_capacity, (void**)&state->nodes));
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      state->arena, state->node_count, state->node_capacity,
      sizeof(*state->sorted_node_indices), &state->sorted_node_capacity,
      (void**)&state->sorted_node_indices));
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      state->arena, state->node_count, state->node_capacity,
      sizeof(*state->groups), &state->group_capacity, (void**)&state->groups));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_branch_layout_insert_island(
    loom_amdgpu_branch_layout_build_state_t* state, uint32_t edge_index,
    uint32_t source_node_index, uint32_t target_node_index,
    uint64_t source_base_byte_offset, uint64_t target_base_byte_offset) {
  const uint32_t anchor_index =
      loom_amdgpu_branch_layout_select_midpoint_anchor(
          state, edge_index, source_base_byte_offset, target_base_byte_offset);
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_branch_layout_grow_nodes(state));
  const uint32_t new_node_index = state->node_count++;
  state->nodes[new_node_index] = (loom_amdgpu_branch_layout_path_node_t){
      .edge_index = edge_index,
      .anchor_index = anchor_index,
      .next_node_index = target_node_index,
      .final_island_index = LOOM_AMDGPU_BRANCH_ISLAND_NONE,
  };
  if (source_node_index == LOOM_AMDGPU_BRANCH_ISLAND_NONE) {
    state->edge_head_node_indices[edge_index] = new_node_index;
  } else {
    state->nodes[source_node_index].next_node_index = new_node_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_branch_layout_relax_first_long_segment(
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
        target_final_byte_offset = target_node->final_byte_offset;
      }
      if (!loom_amdgpu_branch_layout_offset_fits(
              source_final_byte_offset, target_final_byte_offset, NULL)) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_branch_layout_insert_island(
            state, edge_index, source_node_index, target_node_index,
            source_base_byte_offset, target_base_byte_offset));
        *out_changed = true;
        return iree_ok_status();
      }
      if (target_node_index == LOOM_AMDGPU_BRANCH_ISLAND_NONE) break;
      source_node_index = target_node_index;
      const loom_amdgpu_branch_layout_path_node_t* source_node =
          &state->nodes[source_node_index];
      source_base_byte_offset =
          state->input->anchors[source_node->anchor_index].byte_offset;
      source_final_byte_offset = source_node->final_byte_offset;
      target_node_index = source_node->next_node_index;
    }
  }
  return iree_ok_status();
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
      .index = state->nodes[node_index].final_island_index,
  };
}

static uint64_t loom_amdgpu_branch_layout_target_final_offset(
    const loom_amdgpu_branch_layout_build_state_t* state,
    const loom_amdgpu_branch_layout_input_edge_t* edge, uint32_t node_index) {
  if (node_index != LOOM_AMDGPU_BRANCH_ISLAND_NONE) {
    return state->nodes[node_index].final_byte_offset;
  }
  const uint64_t block_byte_offset =
      state->input->blocks[edge->target_block_index].byte_offset;
  return loom_amdgpu_branch_layout_translate_offset(
      state, block_byte_offset, /*include_equal_groups=*/false);
}

static iree_status_t loom_amdgpu_branch_layout_export(
    const loom_amdgpu_branch_layout_build_state_t* state,
    loom_amdgpu_branch_layout_t* out_layout) {
  if (state->node_count == 0) return iree_ok_status();

  loom_amdgpu_branch_layout_edge_t* edges = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->input->edge_count, sizeof(*edges), (void**)&edges));
  loom_amdgpu_branch_layout_island_t* islands = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->node_count, sizeof(*islands), (void**)&islands));
  loom_amdgpu_branch_layout_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->group_count, sizeof(*groups), (void**)&groups));

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

  for (uint32_t sorted_index = 0; sorted_index < state->node_count;
       ++sorted_index) {
    const uint32_t node_index = state->sorted_node_indices[sorted_index];
    const loom_amdgpu_branch_layout_path_node_t* node =
        &state->nodes[node_index];
    const uint32_t owner_edge_index = node->edge_index;
    IREE_ASSERT_LT(owner_edge_index, state->input->edge_count);
    const loom_amdgpu_branch_layout_input_edge_t* owner_edge =
        &state->input->edges[owner_edge_index];
    const uint64_t target_final_byte_offset =
        loom_amdgpu_branch_layout_target_final_offset(state, owner_edge,
                                                      node->next_node_index);
    int16_t displacement = 0;
    const bool displacement_fits = loom_amdgpu_branch_layout_offset_fits(
        node->final_byte_offset, target_final_byte_offset, &displacement);
    IREE_ASSERT(displacement_fits);
    (void)displacement_fits;
    islands[sorted_index] = (loom_amdgpu_branch_layout_island_t){
        .target = loom_amdgpu_branch_layout_node_target(state, owner_edge,
                                                        node->next_node_index),
        .relative_dword_offset = displacement,
    };
  }

  for (uint32_t group_index = 0; group_index < state->group_count;
       ++group_index) {
    const loom_amdgpu_branch_layout_scratch_group_t* scratch_group =
        &state->groups[group_index];
    groups[group_index] = (loom_amdgpu_branch_layout_group_t){
        .packet_index =
            state->input->anchors[scratch_group->anchor_index].packet_index,
        .island_start = scratch_group->sorted_node_start,
        .island_count = scratch_group->node_count,
    };
  }

  uint64_t inserted_byte_count = 0;
  for (uint32_t i = 0; i < state->group_count; ++i) {
    inserted_byte_count += state->groups[i].byte_length;
  }
  if (state->input->byte_length > UINT64_MAX - inserted_byte_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU relaxed branch layout overflowed");
  }
  *out_layout = (loom_amdgpu_branch_layout_t){
      .byte_length = state->input->byte_length + inserted_byte_count,
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
  if (input->edge_count == 0) return iree_ok_status();
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
  uint32_t* edge_head_node_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, input->edge_count, sizeof(*edge_head_node_indices),
      (void**)&edge_head_node_indices));
  for (iree_host_size_t i = 0; i < input->edge_count; ++i) {
    edge_head_node_indices[i] = LOOM_AMDGPU_BRANCH_ISLAND_NONE;
  }
  loom_amdgpu_branch_layout_build_state_t state = {
      .input = input,
      .arena = arena,
      .edge_head_node_indices = edge_head_node_indices,
  };
  while (true) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_branch_layout_rebuild_physical_layout(&state));
    bool changed = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_branch_layout_relax_first_long_segment(&state, &changed));
    if (!changed) break;
  }
  return loom_amdgpu_branch_layout_export(&state, out_layout);
}
