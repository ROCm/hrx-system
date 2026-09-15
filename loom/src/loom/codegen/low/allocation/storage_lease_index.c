// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage_lease_index.h"

#include <string.h>

struct loom_low_allocation_storage_lease_index_node_t {
  // Exact leaf key or the minimum key in a branch's shared radix prefix.
  uint64_t key;
  // Maximum end point of any temporal lease in this subtree.
  uint32_t maximum_end_point;
  // Parent in this radix tree, or UINT32_MAX at a root.
  uint32_t parent;
  // Kind-specific payload; directory and temporal trees have separate roots.
  union {
    // Children of a radix branch, selected by the branch bit.
    uint32_t children[2];
    // One exact physical-unit directory leaf.
    struct {
      // Root of the unit's temporal lease tree, or UINT32_MAX.
      uint32_t temporal_root;
    } unit;
    // One materialized temporal lease unit.
    struct {
      // Ordinal of the borrowed assignment-backed lease instance.
      uint32_t index;
      // Next temporal leaf owned by this lease, or UINT32_MAX.
      uint32_t next_node;
    } lease;
  } data;
  // One-based split bit, decreasing toward children; zero denotes a leaf.
  uint8_t level;
};

static uint32_t loom_low_allocation_storage_lease_unit_root_ordinal(
    loom_low_allocation_location_kind_t location_kind) {
  return location_kind == LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ? 0
                                                                         : 1;
}

static uint8_t loom_low_allocation_storage_lease_key_level(uint64_t key) {
  return key == 0 ? 0 : (uint8_t)(64 - iree_math_count_leading_zeros_u64(key));
}

static uint32_t loom_low_allocation_storage_lease_key_child(uint64_t key,
                                                            uint8_t level) {
  return (uint32_t)((key >> (level - 1u)) & 1u);
}

static uint32_t loom_low_allocation_storage_lease_index_allocate_node(
    loom_low_allocation_storage_lease_unit_index_t* index, uint64_t key) {
  const uint32_t node_index = index->node_count++;
  // Each indexed unit needs at most one directory leaf, one directory branch,
  // one temporal leaf and one temporal branch; shared directories reduce the
  // exact total to 2 * units + distinct_units - nonempty_directories.
  IREE_ASSERT_LT(node_index, index->node_capacity);
  index->nodes[node_index] = (loom_low_allocation_storage_lease_index_node_t){
      .key = key,
      .parent = UINT32_MAX,
  };
  return node_index;
}

// Finds an exact leaf or the link where a new prefix branches from this tree.
// The returned link stays valid because node storage never grows.
static uint32_t* loom_low_allocation_storage_lease_index_find_link(
    loom_low_allocation_storage_lease_unit_index_t* index, uint32_t* link,
    uint64_t key, uint32_t* out_parent) {
  *out_parent = UINT32_MAX;
  while (*link != UINT32_MAX) {
    loom_low_allocation_storage_lease_index_node_t* node = &index->nodes[*link];
    if (node->level == 0 || loom_low_allocation_storage_lease_key_level(
                                node->key ^ key) > node->level) {
      break;
    }
    *out_parent = *link;
    link = &node->data.children[loom_low_allocation_storage_lease_key_child(
        key, node->level)];
  }
  return link;
}

static void loom_low_allocation_storage_lease_index_refresh_ancestors(
    loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t node_index) {
  while (node_index != UINT32_MAX) {
    loom_low_allocation_storage_lease_index_node_t* node =
        &index->nodes[node_index];
    const loom_low_allocation_storage_lease_index_node_t* left =
        &index->nodes[node->data.children[0]];
    const loom_low_allocation_storage_lease_index_node_t* right =
        &index->nodes[node->data.children[1]];
    const uint32_t maximum_end_point =
        iree_max(left->maximum_end_point, right->maximum_end_point);
    if (node->maximum_end_point == maximum_end_point) {
      break;
    }
    node->maximum_end_point = maximum_end_point;
    node_index = node->parent;
  }
}

static void loom_low_allocation_storage_lease_index_insert_at(
    loom_low_allocation_storage_lease_unit_index_t* index, uint32_t* link,
    uint32_t parent, uint32_t leaf_index) {
  loom_low_allocation_storage_lease_index_node_t* leaf =
      &index->nodes[leaf_index];
  if (*link == UINT32_MAX) {
    *link = leaf_index;
    leaf->parent = parent;
    loom_low_allocation_storage_lease_index_refresh_ancestors(index, parent);
    return;
  }
  const uint32_t previous_index = *link;
  const uint8_t level = loom_low_allocation_storage_lease_key_level(
      leaf->key ^ index->nodes[previous_index].key);
  const uint64_t prefix = level == 64 ? 0 : (leaf->key >> level) << level;
  const uint32_t branch_index =
      loom_low_allocation_storage_lease_index_allocate_node(index, prefix);
  loom_low_allocation_storage_lease_index_node_t* branch =
      &index->nodes[branch_index];
  branch->level = level;
  branch->parent = parent;
  const uint32_t child =
      loom_low_allocation_storage_lease_key_child(leaf->key, level);
  branch->data.children[child] = leaf_index;
  branch->data.children[child ^ 1u] = previous_index;
  leaf->parent = branch_index;
  index->nodes[previous_index].parent = branch_index;
  *link = branch_index;
  loom_low_allocation_storage_lease_index_refresh_ancestors(index,
                                                            branch_index);
}

static uint32_t loom_low_allocation_storage_lease_index_unit(
    loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t root_ordinal, uint64_t key) {
  uint32_t parent = UINT32_MAX;
  uint32_t* link = loom_low_allocation_storage_lease_index_find_link(
      index, &index->unit_roots[root_ordinal], key, &parent);
  if (*link != UINT32_MAX && index->nodes[*link].level == 0 &&
      index->nodes[*link].key == key) {
    return *link;
  }
  const uint32_t unit_index =
      loom_low_allocation_storage_lease_index_allocate_node(index, key);
  index->nodes[unit_index].data.unit.temporal_root = UINT32_MAX;
  loom_low_allocation_storage_lease_index_insert_at(index, link, parent,
                                                    unit_index);
  return unit_index;
}

iree_status_t loom_low_allocation_storage_lease_unit_index_initialize(
    loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_allocation_storage_lease_t* instances,
    iree_host_size_t lease_count, iree_host_size_t lease_unit_capacity,
    iree_host_size_t distinct_unit_capacity, iree_arena_allocator_t* arena) {
  *index = (loom_low_allocation_storage_lease_unit_index_t){
      .instances = instances,
      .unit_roots = {UINT32_MAX, UINT32_MAX},
  };
  if (lease_unit_capacity == 0) return iree_ok_status();
  if (lease_count > UINT32_MAX || lease_unit_capacity > UINT32_MAX / 3u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation storage lease index exceeds u32 range");
  }
  index->node_capacity =
      (uint32_t)lease_unit_capacity * 2u +
      (uint32_t)iree_min(lease_unit_capacity, distinct_unit_capacity);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, index->node_capacity,
                                                 sizeof(*index->nodes),
                                                 (void**)&index->nodes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, lease_count, sizeof(*index->first_nodes_by_lease),
      (void**)&index->first_nodes_by_lease));
  for (iree_host_size_t i = 0; i < lease_count; ++i) {
    index->first_nodes_by_lease[i] = UINT32_MAX;
  }
  return iree_ok_status();
}

void loom_low_allocation_storage_lease_unit_index_insert(
    loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t storage_lease_index) {
  const loom_low_allocation_storage_lease_t* lease =
      &index->instances[storage_lease_index];
  const uint32_t root_ordinal =
      loom_low_allocation_storage_lease_unit_root_ordinal(lease->location_kind);
  const uint32_t storage_key = loom_low_reg_class_storage_key(
      descriptor_set, lease->descriptor_reg_class_id);
  for (uint32_t unit_offset = 0; unit_offset < lease->location_count;
       ++unit_offset) {
    const uint64_t unit_key =
        ((uint64_t)storage_key << 32) | (lease->location_base + unit_offset);
    const uint32_t unit_index = loom_low_allocation_storage_lease_index_unit(
        index, root_ordinal, unit_key);
    const uint64_t key =
        ((uint64_t)lease->start_point << 32) | index->node_count;
    const uint32_t leaf_index =
        loom_low_allocation_storage_lease_index_allocate_node(index, key);
    loom_low_allocation_storage_lease_index_node_t* leaf =
        &index->nodes[leaf_index];
    leaf->maximum_end_point = lease->end_point;
    leaf->data.lease.index = storage_lease_index;
    leaf->data.lease.next_node =
        index->first_nodes_by_lease[storage_lease_index];
    index->first_nodes_by_lease[storage_lease_index] = leaf_index;
    uint32_t parent = UINT32_MAX;
    uint32_t* link = loom_low_allocation_storage_lease_index_find_link(
        index, &index->nodes[unit_index].data.unit.temporal_root, key, &parent);
    loom_low_allocation_storage_lease_index_insert_at(index, link, parent,
                                                      leaf_index);
  }
}

void loom_low_allocation_storage_lease_unit_index_update(
    loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t storage_lease_index) {
  const loom_low_allocation_storage_lease_t* lease =
      &index->instances[storage_lease_index];
  uint32_t node_index = index->first_nodes_by_lease[storage_lease_index];
  while (node_index != UINT32_MAX) {
    loom_low_allocation_storage_lease_index_node_t* node =
        &index->nodes[node_index];
    node->maximum_end_point = lease->end_point;
    loom_low_allocation_storage_lease_index_refresh_ancestors(index,
                                                              node->parent);
    node_index = node->data.lease.next_node;
  }
}

bool loom_low_allocation_storage_lease_unit_index_is_enabled(
    const loom_low_allocation_storage_lease_unit_index_t* index) {
  return index != NULL && index->node_capacity != 0;
}

iree_status_t loom_low_allocation_storage_lease_selection_initialize(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    iree_arena_allocator_t* arena,
    loom_low_allocation_storage_lease_selection_t* out_selection) {
  *out_selection = (loom_low_allocation_storage_lease_selection_t){
      .index = index,
  };
  if (index == NULL || index->node_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, index->node_count, sizeof(*out_selection->subtree_counts),
      (void**)&out_selection->subtree_counts));
  memset(out_selection->subtree_counts, 0,
         index->node_count * sizeof(*out_selection->subtree_counts));
  return iree_ok_status();
}

void loom_low_allocation_storage_lease_selection_set_active(
    loom_low_allocation_storage_lease_selection_t* selection,
    uint32_t storage_lease_index, bool active) {
  const loom_low_allocation_storage_lease_unit_index_t* index =
      selection->index;
  uint32_t leaf_index = index->first_nodes_by_lease[storage_lease_index];
  while (leaf_index != UINT32_MAX) {
    IREE_ASSERT_EQ(selection->subtree_counts[leaf_index], active ? 0u : 1u);
    uint32_t node_index = leaf_index;
    while (node_index != UINT32_MAX) {
      if (active) {
        ++selection->subtree_counts[node_index];
      } else {
        --selection->subtree_counts[node_index];
      }
      node_index = index->nodes[node_index].parent;
    }
    leaf_index = index->nodes[leaf_index].data.lease.next_node;
  }
}

void loom_low_allocation_storage_lease_unit_query_initialize(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, uint64_t minimum_end_point,
    uint64_t start_point_limit,
    const loom_low_allocation_storage_lease_selection_t* selection,
    loom_low_allocation_storage_lease_unit_query_t* out_query) {
  out_query->index = index;
  out_query->selection = selection;
  out_query->storage_key =
      loom_low_reg_class_storage_key(descriptor_set, descriptor_reg_class_id);
  out_query->unit_root_ordinal =
      loom_low_allocation_storage_lease_unit_root_ordinal(location_kind);
  out_query->location_base = location_base;
  out_query->location_count = location_count;
  out_query->next_unit_offset = 0;
  out_query->active_location = 0;
  out_query->minimum_end_point = minimum_end_point;
  out_query->start_point_limit = start_point_limit;
  out_query->stack_count = 0;
}

bool loom_low_allocation_storage_lease_unit_query_next(
    loom_low_allocation_storage_lease_unit_query_t* query,
    uint32_t* out_storage_lease_index) {
  const loom_low_allocation_storage_lease_unit_index_t* index = query->index;
  if (!loom_low_allocation_storage_lease_unit_index_is_enabled(index)) {
    return false;
  }
  while (true) {
    while (query->stack_count != 0) {
      const uint32_t node_index = query->stack[--query->stack_count];
      const loom_low_allocation_storage_lease_index_node_t* node =
          &index->nodes[node_index];
      const bool temporal_match =
          (node->key >> 32) < query->start_point_limit &&
          node->maximum_end_point >= query->minimum_end_point;
      const bool selected_match =
          query->selection != NULL &&
          query->selection->subtree_counts[node_index] != 0;
      if (!temporal_match && !selected_match) continue;
      if (node->level == 0) {
        *out_storage_lease_index = node->data.lease.index;
        return true;
      }
      query->stack[query->stack_count++] = node->data.children[0];
      query->stack[query->stack_count++] = node->data.children[1];
    }
    if (query->next_unit_offset == query->location_count) return false;
    query->active_location = query->location_base + query->next_unit_offset++;
    const uint64_t key =
        ((uint64_t)query->storage_key << 32) | query->active_location;
    uint32_t unit_index = index->unit_roots[query->unit_root_ordinal];
    while (unit_index != UINT32_MAX && index->nodes[unit_index].level != 0) {
      const loom_low_allocation_storage_lease_index_node_t* node =
          &index->nodes[unit_index];
      unit_index =
          node->data.children[loom_low_allocation_storage_lease_key_child(
              key, node->level)];
    }
    if (unit_index != UINT32_MAX && index->nodes[unit_index].key == key) {
      query->stack[query->stack_count++] =
          index->nodes[unit_index].data.unit.temporal_root;
    }
  }
}
