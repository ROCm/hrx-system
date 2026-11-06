// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/join.h"

typedef struct loom_aie2p_join_edge_t {
  // Physical tiles sharing one lock aperture, independent of tree orientation.
  iree_host_size_t nodes[2];
  // Memory module accessible to both endpoints.
  iree_host_size_t owner;
  // Endpoint-relative bases of the owner's lock selectors.
  uint16_t selectors[2];
} loom_aie2p_join_edge_t;

static loom_xdna_tile_coordinate_t loom_aie2p_join_coordinate(
    const loom_xdna_array_family_t* family, iree_host_size_t tile) {
  return (loom_xdna_tile_coordinate_t){tile / family->row_count,
                                       tile % family->row_count};
}

static iree_host_size_t loom_aie2p_join_root(iree_host_size_t* parents,
                                             iree_host_size_t node) {
  while (parents[node] != node) {
    parents[node] = parents[parents[node]];
    node = parents[node];
  }
  return node;
}

static loom_aie2p_join_action_t loom_aie2p_join_action(
    const loom_aie2p_join_edge_t* edge, uint16_t first_lock,
    iree_host_size_t node, bool departure, bool acquire) {
  return (loom_aie2p_join_action_t){
      .selector =
          edge->selectors[edge->nodes[1] == node] + first_lock + departure,
      .acquire = acquire};
}

iree_status_t loom_aie2p_join_plan_build(const loom_xdna_array_family_t* family,
                                         const bool* participants,
                                         uint16_t* allocated_locks,
                                         iree_arena_allocator_t* arena,
                                         loom_aie2p_join_plan_t* out_plan,
                                         bool* out_valid) {
  *out_plan = (loom_aie2p_join_plan_t){0};
  *out_valid = false;
  const iree_host_size_t tile_count =
      (iree_host_size_t)family->column_count * family->row_count;
  iree_host_size_t node_count = 0, root = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    if (participants[i]) {
      if (!node_count) {
        root = i;
      }
      ++node_count;
    }
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, tile_count,
                                                 sizeof(*out_plan->programs),
                                                 (void**)&out_plan->programs));
  memset(out_plan->programs, 0, tile_count * sizeof(*out_plan->programs));
  if (node_count < 2) {
    *out_valid = true;
    return iree_ok_status();
  }

  // Enumerate shared apertures once. At most one accessor per generated
  // window reaches an owner, so the candidate bound follows hardware topology.
  const iree_host_size_t windows = family->address_window_count;
  loom_aie2p_join_edge_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, tile_count * windows * (windows - 1) / 2,
                                sizeof(*candidates), (void**)&candidates));
  iree_host_size_t candidate_count = 0;
  uint16_t* available = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*available), (void**)&available));
  for (iree_host_size_t owner = 0; owner < tile_count; ++owner) {
    const loom_xdna_tile_coordinate_t coordinate =
        loom_aie2p_join_coordinate(family, owner);
    available[owner] =
        loom_xdna_array_tile_facts(family, coordinate)->lock_count -
        allocated_locks[owner];
    if (available[owner] < 2) {
      continue;
    }
    for (iree_host_size_t first = 0; first < tile_count; ++first) {
      if (!participants[first]) {
        continue;
      }
      const loom_xdna_address_window_t* first_window =
          loom_xdna_array_load_window(
              family, loom_aie2p_join_coordinate(family, first), coordinate);
      if (!first_window) {
        continue;
      }
      for (iree_host_size_t second = first + 1; second < tile_count; ++second) {
        if (!participants[second]) {
          continue;
        }
        const loom_xdna_address_window_t* second_window =
            loom_xdna_array_load_window(
                family, loom_aie2p_join_coordinate(family, second), coordinate);
        if (!second_window) {
          continue;
        }
        candidates[candidate_count++] = (loom_aie2p_join_edge_t){
            .nodes = {first, second},
            .owner = owner,
            .selectors = {first_window->lock_selector_base,
                          second_window->lock_selector_base}};
      }
    }
  }
  iree_host_size_t* parents = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*parents), (void**)&parents));
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    parents[i] = i;
  }
  loom_aie2p_join_edge_t* edges = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count - 1, sizeof(*edges), (void**)&edges));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, node_count - 1,
                                                 sizeof(*out_plan->locks),
                                                 (void**)&out_plan->locks));
  for (iree_host_size_t i = 0; i < node_count - 1; ++i) {
    iree_host_size_t best = IREE_HOST_SIZE_MAX;
    uint16_t most_available = 1;
    for (iree_host_size_t j = 0; j < candidate_count; ++j) {
      const loom_aie2p_join_edge_t* candidate = &candidates[j];
      if (available[candidate->owner] > most_available &&
          loom_aie2p_join_root(parents, candidate->nodes[0]) !=
              loom_aie2p_join_root(parents, candidate->nodes[1])) {
        best = j;
        most_available = available[candidate->owner];
      }
    }
    if (best == IREE_HOST_SIZE_MAX) {
      return iree_ok_status();
    }
    edges[i] = candidates[best];
    const iree_host_size_t owner = edges[i].owner;
    const uint16_t capacity =
        loom_xdna_array_tile_facts(family,
                                   loom_aie2p_join_coordinate(family, owner))
            ->lock_count;
    out_plan->locks[i] = (loom_aie2p_join_locks_t){
        .owner = owner, .first = capacity - available[owner]};
    available[owner] -= 2;
    parents[loom_aie2p_join_root(parents, edges[i].nodes[1])] =
        loom_aie2p_join_root(parents, edges[i].nodes[0]);
  }
  out_plan->edge_count = node_count - 1;

  // Orient the selected tree from one root. Every non-root has exactly one
  // parent rendezvous; child arrivals precede it and departures follow it.
  iree_host_size_t *parent_edges = NULL, *order = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*parent_edges), (void**)&parent_edges));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(*order), (void**)&order));
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    parents[i] = parent_edges[i] = IREE_HOST_SIZE_MAX;
  }
  parents[root] = root;
  order[0] = root;
  iree_host_size_t order_count = 1;
  for (iree_host_size_t i = 0; i < order_count; ++i) {
    const iree_host_size_t node = order[i];
    for (iree_host_size_t j = 0; j < out_plan->edge_count; ++j) {
      const loom_aie2p_join_edge_t* edge = &edges[j];
      if (edge->nodes[0] != node && edge->nodes[1] != node) {
        continue;
      }
      const iree_host_size_t peer = edge->nodes[edge->nodes[0] == node];
      if (parents[peer] != IREE_HOST_SIZE_MAX) {
        continue;
      }
      parents[peer] = node;
      parent_edges[peer] = j;
      order[order_count++] = peer;
    }
  }
  loom_aie2p_join_action_t* actions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, 4 * out_plan->edge_count, sizeof(*actions), (void**)&actions));
  for (iree_host_size_t i = 0; i < order_count; ++i) {
    const iree_host_size_t node = order[i];
    loom_aie2p_join_program_t* program = &out_plan->programs[node];
    program->actions = actions;
    for (iree_host_size_t j = 1; j < order_count; ++j) {
      const iree_host_size_t child = order[j], edge = parent_edges[child];
      if (parents[child] == node) {
        actions[program->count++] = loom_aie2p_join_action(
            &edges[edge], out_plan->locks[edge].first, node, false, true);
      }
    }
    if (node != root) {
      const iree_host_size_t edge = parent_edges[node];
      actions[program->count++] = loom_aie2p_join_action(
          &edges[edge], out_plan->locks[edge].first, node, false, false);
      actions[program->count++] = loom_aie2p_join_action(
          &edges[edge], out_plan->locks[edge].first, node, true, true);
    }
    for (iree_host_size_t j = 1; j < order_count; ++j) {
      const iree_host_size_t child = order[j], edge = parent_edges[child];
      if (parents[child] == node) {
        actions[program->count++] = loom_aie2p_join_action(
            &edges[edge], out_plan->locks[edge].first, node, true, false);
      }
    }
    actions += program->count;
  }
  for (iree_host_size_t i = 0; i < out_plan->edge_count; ++i) {
    allocated_locks[out_plan->locks[i].owner] += 2;
  }
  *out_valid = true;
  return iree_ok_status();
}
