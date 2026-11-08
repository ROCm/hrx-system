// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/ordinal_component.h"

#include <string.h>

typedef struct loom_low_allocation_ordinal_component_node_t {
  // Union-find parent, or LOOM_VALUE_ORDINAL_INVALID when unconstrained.
  loom_value_ordinal_t parent;
  // Union rank used only while constructing the retained component table.
  uint8_t rank;
} loom_low_allocation_ordinal_component_node_t;

static loom_value_ordinal_t loom_low_allocation_ordinal_component_find(
    loom_low_allocation_ordinal_component_node_t* nodes,
    loom_value_ordinal_t value_ordinal) {
  loom_value_ordinal_t root = value_ordinal;
  while (nodes[root].parent != root) {
    root = nodes[root].parent;
  }
  while (nodes[value_ordinal].parent != value_ordinal) {
    const loom_value_ordinal_t next = nodes[value_ordinal].parent;
    nodes[value_ordinal].parent = root;
    value_ordinal = next;
  }
  return root;
}

static void loom_low_allocation_ordinal_components_union(
    loom_low_allocation_ordinal_component_node_t* nodes,
    loom_value_ordinal_t lhs, loom_value_ordinal_t rhs) {
  lhs = loom_low_allocation_ordinal_component_find(nodes, lhs);
  rhs = loom_low_allocation_ordinal_component_find(nodes, rhs);
  if (lhs == rhs) {
    return;
  }
  if (nodes[lhs].rank < nodes[rhs].rank) {
    const loom_value_ordinal_t temporary = lhs;
    lhs = rhs;
    rhs = temporary;
  }
  nodes[rhs].parent = lhs;
  nodes[lhs].rank += nodes[lhs].rank == nodes[rhs].rank;
}

static loom_value_ordinal_t loom_low_allocation_ordinal_component_origin(
    const loom_value_ordinal_t* origins, loom_value_ordinal_t value_ordinal) {
  return origins != NULL ? origins[value_ordinal] : value_ordinal;
}

static iree_status_t loom_low_allocation_ordinal_components_build_impl(
    loom_value_ordinal_t value_count,
    const loom_value_ordinal_t* tied_storage_origins_by_value_ordinal,
    const loom_low_placement_relation_t* relations,
    iree_host_size_t relation_count, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_ordinal_components_t* out_components) {
  loom_low_allocation_ordinal_component_node_t* nodes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, value_count, sizeof(*nodes), (void**)&nodes));
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    nodes[i].parent = LOOM_VALUE_ORDINAL_INVALID;
    nodes[i].rank = 0;
  }

  for (iree_host_size_t i = 0; i < relation_count; ++i) {
    const loom_low_placement_relation_t* relation = &relations[i];
    if (relation->kind != LOOM_LOW_PLACEMENT_RELATION_SAME_REGISTER_ORDINAL ||
        !iree_any_bit_set(relation->flags,
                          LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD)) {
      continue;
    }
    const loom_value_ordinal_t result =
        loom_low_allocation_ordinal_component_origin(
            tied_storage_origins_by_value_ordinal, relation->result_ordinal);
    const loom_value_ordinal_t source =
        loom_low_allocation_ordinal_component_origin(
            tied_storage_origins_by_value_ordinal, relation->source_ordinal);
    if (result == source) {
      continue;
    }
    if (nodes[result].parent == LOOM_VALUE_ORDINAL_INVALID) {
      nodes[result].parent = result;
    }
    if (nodes[source].parent == LOOM_VALUE_ORDINAL_INVALID) {
      nodes[source].parent = source;
    }
    loom_low_allocation_ordinal_components_union(nodes, result, source);
  }

  uint32_t component_count = 0;
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    if (nodes[i].parent != LOOM_VALUE_ORDINAL_INVALID &&
        loom_low_allocation_ordinal_component_find(nodes, i) == i) {
      ++component_count;
    }
  }
  if (component_count == 0) {
    return iree_ok_status();
  }

  uint32_t* indices_by_root = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, value_count,
                                                 sizeof(*indices_by_root),
                                                 (void**)&indices_by_root));
  memset(indices_by_root, 0xFF, value_count * sizeof(*indices_by_root));
  uint32_t next_component = 0;
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    if (nodes[i].parent != LOOM_VALUE_ORDINAL_INVALID &&
        loom_low_allocation_ordinal_component_find(nodes, i) == i) {
      indices_by_root[i] = next_component++;
    }
  }

  uint32_t* indices_by_value_ordinal = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*indices_by_value_ordinal),
      (void**)&indices_by_value_ordinal));
  memset(indices_by_value_ordinal, 0xFF,
         value_count * sizeof(*indices_by_value_ordinal));
  loom_low_placement_ordinal_component_range_t* ranges = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, component_count, sizeof(*ranges), (void**)&ranges));
  memset(ranges, 0, component_count * sizeof(*ranges));

  uint32_t member_count = 0;
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    const loom_value_ordinal_t origin =
        loom_low_allocation_ordinal_component_origin(
            tied_storage_origins_by_value_ordinal, i);
    if (nodes[origin].parent == LOOM_VALUE_ORDINAL_INVALID) {
      continue;
    }
    const loom_value_ordinal_t root =
        loom_low_allocation_ordinal_component_find(nodes, origin);
    const uint32_t component_index = indices_by_root[root];
    indices_by_value_ordinal[i] = component_index;
    ++ranges[component_index].count;
    ranges[component_index].origin_count += origin == i;
    ++member_count;
  }

  uint32_t offset = 0;
  for (uint32_t i = 0; i < component_count; ++i) {
    ranges[i].start = offset;
    offset += ranges[i].count;
  }
  loom_value_ordinal_t* members = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, member_count, sizeof(*members), (void**)&members));
  uint32_t* cursors = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, component_count, 2 * sizeof(*cursors), (void**)&cursors));
  for (uint32_t i = 0; i < component_count; ++i) {
    cursors[i] = ranges[i].start;
    cursors[component_count + i] = ranges[i].start + ranges[i].origin_count;
  }
  for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
    const uint32_t component_index = indices_by_value_ordinal[i];
    if (component_index != UINT32_MAX) {
      const loom_value_ordinal_t origin =
          loom_low_allocation_ordinal_component_origin(
              tied_storage_origins_by_value_ordinal, i);
      const uint32_t cursor =
          component_index + (origin == i ? 0 : component_count);
      members[cursors[cursor]++] = i;
    }
  }
  *out_components = (loom_low_allocation_ordinal_components_t){
      .indices_by_value_ordinal = indices_by_value_ordinal,
      .ranges = ranges,
      .members = members,
      .count = component_count,
  };
  return iree_ok_status();
}

iree_status_t loom_low_allocation_ordinal_components_build(
    loom_value_ordinal_t value_count,
    const loom_value_ordinal_t* tied_storage_origins_by_value_ordinal,
    const loom_low_placement_relation_t* relations,
    iree_host_size_t relation_count, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_ordinal_components_t* out_components) {
  *out_components = (loom_low_allocation_ordinal_components_t){0};
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  const iree_status_t status =
      loom_low_allocation_ordinal_components_build_impl(
          value_count, tied_storage_origins_by_value_ordinal, relations,
          relation_count, arena, scratch_arena, out_components);
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}
