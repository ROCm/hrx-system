// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/storage_lease_index.h"

#include <string.h>

struct loom_low_allocation_storage_lease_index_node_t {
  // Exact leaf key or normalized minimum of a branch's radix prefix.
  uint64_t key;
  // Metadata interpreted by the tree owning this node. The union keeps each
  // shared node at 32 bytes while retaining both pressure policy summaries.
  union {
    struct {
      // Maximum end point of any temporal lease in this subtree.
      uint32_t maximum_end_point;
      // Parent in this temporal tree, or UINT32_MAX at a root.
      uint32_t parent;
      // Maximum end point excluding pressure-releasable leases.
      uint32_t maximum_non_pressure_end_point;
    } temporal;
    struct {
      // Maximum earliest start over a dense run, or UINT32_MAX.
      uint32_t maximum_first_start;
      union {
        // Equivalent summary excluding pressure-releasable leases.
        uint32_t maximum_non_pressure_first_start;
        // Minimum active pressure-lease start in a homogeneous pressure table.
        uint32_t minimum_pressure_first_start;
      } policy;
      // Parent in this directory tree, or UINT32_MAX at a root.
      uint32_t parent;
    } directory;
  } metadata;
  // Kind-specific payload; directory and temporal trees have separate roots.
  union {
    // Children of a radix branch, selected by the branch bit.
    uint32_t children[2];
    // One exact physical-unit directory leaf.
    struct {
      // Root of the unit's temporal lease tree, or UINT32_MAX.
      uint32_t temporal_root;
      // Next materialized unit in this directory, or UINT32_MAX.
      uint32_t next_unit;
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

static_assert(sizeof(loom_low_allocation_storage_lease_index_node_t) == 32,
              "storage lease index nodes must remain cache compact");

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
      .metadata.temporal.parent = UINT32_MAX,
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

static void loom_low_allocation_storage_lease_index_refresh_temporal_ancestors(
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
        iree_max(left->metadata.temporal.maximum_end_point,
                 right->metadata.temporal.maximum_end_point);
    const uint32_t maximum_non_pressure_end_point =
        iree_max(left->metadata.temporal.maximum_non_pressure_end_point,
                 right->metadata.temporal.maximum_non_pressure_end_point);
    if (node->metadata.temporal.maximum_end_point == maximum_end_point &&
        node->metadata.temporal.maximum_non_pressure_end_point ==
            maximum_non_pressure_end_point) {
      break;
    }
    node->metadata.temporal.maximum_end_point = maximum_end_point;
    node->metadata.temporal.maximum_non_pressure_end_point =
        maximum_non_pressure_end_point;
    node_index = node->metadata.temporal.parent;
  }
}

typedef enum loom_low_allocation_storage_lease_tree_kind_e {
  LOOM_LOW_ALLOCATION_STORAGE_LEASE_TREE_DIRECTORY = 0,
  LOOM_LOW_ALLOCATION_STORAGE_LEASE_TREE_TEMPORAL = 1,
} loom_low_allocation_storage_lease_tree_kind_t;

static void loom_low_allocation_storage_lease_index_insert_at(
    loom_low_allocation_storage_lease_unit_index_t* index, uint32_t* link,
    uint32_t parent, uint32_t leaf_index,
    loom_low_allocation_storage_lease_tree_kind_t tree_kind) {
  loom_low_allocation_storage_lease_index_node_t* leaf =
      &index->nodes[leaf_index];
  if (*link == UINT32_MAX) {
    *link = leaf_index;
    if (tree_kind == LOOM_LOW_ALLOCATION_STORAGE_LEASE_TREE_TEMPORAL) {
      leaf->metadata.temporal.parent = parent;
      loom_low_allocation_storage_lease_index_refresh_temporal_ancestors(
          index, parent);
    } else {
      leaf->metadata.directory.parent = parent;
    }
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
  const uint32_t child =
      loom_low_allocation_storage_lease_key_child(leaf->key, level);
  branch->data.children[child] = leaf_index;
  branch->data.children[child ^ 1u] = previous_index;
  *link = branch_index;
  if (tree_kind == LOOM_LOW_ALLOCATION_STORAGE_LEASE_TREE_TEMPORAL) {
    branch->metadata.temporal.parent = parent;
    leaf->metadata.temporal.parent = branch_index;
    index->nodes[previous_index].metadata.temporal.parent = branch_index;
    loom_low_allocation_storage_lease_index_refresh_temporal_ancestors(
        index, branch_index);
  } else {
    branch->metadata.directory.maximum_first_start = UINT32_MAX;
    branch->metadata.directory.policy.maximum_non_pressure_first_start =
        UINT32_MAX;
    branch->metadata.directory.parent = parent;
    leaf->metadata.directory.parent = branch_index;
    index->nodes[previous_index].metadata.directory.parent = branch_index;
  }
}

static uint64_t loom_low_allocation_storage_lease_index_maximum_prefix_key(
    const loom_low_allocation_storage_lease_index_node_t* node) {
  return node->level == 64 ? UINT64_MAX
                           : node->key | ((UINT64_C(1) << node->level) - 1u);
}

static uint32_t loom_low_allocation_storage_lease_directory_summary(
    const loom_low_allocation_storage_lease_index_node_t* node,
    loom_low_allocation_storage_lease_conflict_class_t conflict_class) {
  return conflict_class == LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL
             ? node->metadata.directory.maximum_first_start
             : node->metadata.directory.policy.maximum_non_pressure_first_start;
}

static void loom_low_allocation_storage_lease_index_recompute_directory_node(
    loom_low_allocation_storage_lease_unit_index_t* index, uint32_t node_index,
    bool all_leases_pressure_releasable) {
  loom_low_allocation_storage_lease_index_node_t* node =
      &index->nodes[node_index];
  if (node->level == 0) {
    return;
  }
  const uint32_t left_index = node->data.children[0];
  const uint32_t right_index = node->data.children[1];
  const loom_low_allocation_storage_lease_index_node_t* left =
      &index->nodes[left_index];
  const loom_low_allocation_storage_lease_index_node_t* right =
      &index->nodes[right_index];
  // A compressed radix branch spans a dense prefix only when both children
  // cover its complete half-prefix. Lower-level children expose structural
  // holes and remain conservative even when their own leaves are dense.
  const bool structurally_dense =
      left->level + 1u == node->level && right->level + 1u == node->level;
  const uint32_t left_all = left->metadata.directory.maximum_first_start;
  const uint32_t right_all = right->metadata.directory.maximum_first_start;
  node->metadata.directory.maximum_first_start =
      structurally_dense && left_all != UINT32_MAX && right_all != UINT32_MAX
          ? iree_max(left_all, right_all)
          : UINT32_MAX;
  if (all_leases_pressure_releasable) {
    node->metadata.directory.policy.minimum_pressure_first_start =
        iree_min(left->metadata.directory.policy.minimum_pressure_first_start,
                 right->metadata.directory.policy.minimum_pressure_first_start);
  } else {
    const uint32_t left_non_pressure =
        left->metadata.directory.policy.maximum_non_pressure_first_start;
    const uint32_t right_non_pressure =
        right->metadata.directory.policy.maximum_non_pressure_first_start;
    node->metadata.directory.policy.maximum_non_pressure_first_start =
        structurally_dense && left_non_pressure != UINT32_MAX &&
                right_non_pressure != UINT32_MAX
            ? iree_max(left_non_pressure, right_non_pressure)
            : UINT32_MAX;
  }
}

static void loom_low_allocation_storage_lease_index_refresh_directory_ancestors(
    loom_low_allocation_storage_lease_unit_index_t* index, uint32_t node_index,
    bool all_leases_pressure_releasable) {
  while (node_index != UINT32_MAX) {
    loom_low_allocation_storage_lease_index_node_t* node =
        &index->nodes[node_index];
    loom_low_allocation_storage_lease_index_recompute_directory_node(
        index, node_index, all_leases_pressure_releasable);
    node_index = node->metadata.directory.parent;
  }
}

static uint32_t loom_low_allocation_storage_lease_index_minimum_unit(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t node_index) {
  while (index->nodes[node_index].level != 0) {
    node_index = index->nodes[node_index].data.children[0];
  }
  return node_index;
}

static uint32_t loom_low_allocation_storage_lease_index_maximum_unit(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t node_index) {
  while (index->nodes[node_index].level != 0) {
    node_index = index->nodes[node_index].data.children[1];
  }
  return node_index;
}

static uint32_t loom_low_allocation_storage_lease_index_previous_unit(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t unit_index) {
  uint32_t child_index = unit_index;
  uint32_t parent_index = index->nodes[child_index].metadata.directory.parent;
  while (parent_index != UINT32_MAX) {
    const loom_low_allocation_storage_lease_index_node_t* parent =
        &index->nodes[parent_index];
    if (parent->data.children[1] == child_index) {
      return loom_low_allocation_storage_lease_index_maximum_unit(
          index, parent->data.children[0]);
    }
    child_index = parent_index;
    parent_index = parent->metadata.directory.parent;
  }
  return UINT32_MAX;
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
  const uint32_t displaced_index = *link;
  const uint32_t unit_index =
      loom_low_allocation_storage_lease_index_allocate_node(index, key);
  loom_low_allocation_storage_lease_index_node_t* unit =
      &index->nodes[unit_index];
  unit->data.unit.temporal_root = UINT32_MAX;
  unit->data.unit.next_unit = UINT32_MAX;
  unit->metadata.directory.maximum_first_start = UINT32_MAX;
  unit->metadata.directory.policy.maximum_non_pressure_first_start = UINT32_MAX;
  loom_low_allocation_storage_lease_index_insert_at(
      index, link, parent, unit_index,
      LOOM_LOW_ALLOCATION_STORAGE_LEASE_TREE_DIRECTORY);
  if (displaced_index != UINT32_MAX) {
    const loom_low_allocation_storage_lease_index_node_t* branch =
        &index->nodes[unit->metadata.directory.parent];
    if (branch->data.children[1] == unit_index) {
      const uint32_t previous_index =
          loom_low_allocation_storage_lease_index_maximum_unit(index,
                                                               displaced_index);
      unit->data.unit.next_unit =
          index->nodes[previous_index].data.unit.next_unit;
      index->nodes[previous_index].data.unit.next_unit = unit_index;
    } else {
      unit->data.unit.next_unit =
          loom_low_allocation_storage_lease_index_minimum_unit(index,
                                                               displaced_index);
      const uint32_t previous_index =
          loom_low_allocation_storage_lease_index_previous_unit(index,
                                                                unit_index);
      if (previous_index != UINT32_MAX) {
        index->nodes[previous_index].data.unit.next_unit = unit_index;
      }
    }
  }
  loom_low_allocation_storage_lease_index_refresh_directory_ancestors(
      index, unit->metadata.directory.parent,
      /*all_leases_pressure_releasable=*/false);
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
  if (lease_unit_capacity == 0) {
    return iree_ok_status();
  }
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
    uint32_t storage_lease_index, loom_low_storage_lease_flags_t lease_flags) {
  const loom_low_allocation_storage_lease_t* lease =
      &index->instances[storage_lease_index];
  if (loom_low_reg_class_uses_explicit_physical_registers(
          &descriptor_set->reg_classes[lease->descriptor_reg_class_id])) {
    return;
  }
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
    leaf->metadata.temporal.maximum_end_point = lease->end_point;
    leaf->metadata.temporal.maximum_non_pressure_end_point =
        iree_any_bit_set(lease_flags,
                         LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE)
            ? 0
            : lease->end_point;
    leaf->data.lease.index = storage_lease_index;
    leaf->data.lease.next_node =
        index->first_nodes_by_lease[storage_lease_index];
    index->first_nodes_by_lease[storage_lease_index] = leaf_index;
    uint32_t parent = UINT32_MAX;
    uint32_t* link = loom_low_allocation_storage_lease_index_find_link(
        index, &index->nodes[unit_index].data.unit.temporal_root, key, &parent);
    loom_low_allocation_storage_lease_index_insert_at(
        index, link, parent, leaf_index,
        LOOM_LOW_ALLOCATION_STORAGE_LEASE_TREE_TEMPORAL);
  }
}

void loom_low_allocation_storage_lease_unit_index_rebuild(
    loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t lease_count) {
  index->unit_roots[0] = index->unit_roots[1] = UINT32_MAX;
  index->node_count = 0;
  for (iree_host_size_t i = 0; i < lease_count; ++i) {
    index->first_nodes_by_lease[i] = UINT32_MAX;
  }
  for (iree_host_size_t i = 0; i < lease_count; ++i) {
    loom_low_allocation_storage_lease_unit_index_insert(index, descriptor_set,
                                                        (uint32_t)i,
                                                        /*lease_flags=*/0);
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
    node->metadata.temporal.maximum_end_point = lease->end_point;
    if (node->metadata.temporal.maximum_non_pressure_end_point != 0) {
      node->metadata.temporal.maximum_non_pressure_end_point = lease->end_point;
    }
    loom_low_allocation_storage_lease_index_refresh_temporal_ancestors(
        index, node->metadata.temporal.parent);
    node_index = node->data.lease.next_node;
  }
}

static uint32_t loom_low_allocation_storage_lease_index_first_unexpired_start(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t node_index, uint32_t start_point,
    loom_low_allocation_storage_lease_conflict_class_t conflict_class) {
  if (node_index == UINT32_MAX) {
    return UINT32_MAX;
  }
  const loom_low_allocation_storage_lease_index_node_t* node =
      &index->nodes[node_index];
  const uint32_t maximum_end_point =
      conflict_class == LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL
          ? node->metadata.temporal.maximum_end_point
          : node->metadata.temporal.maximum_non_pressure_end_point;
  if (maximum_end_point <= start_point) {
    return UINT32_MAX;
  }
  if (node->level == 0) {
    return (uint32_t)(node->key >> 32);
  }
  const uint32_t left_start =
      loom_low_allocation_storage_lease_index_first_unexpired_start(
          index, node->data.children[0], start_point, conflict_class);
  return left_start != UINT32_MAX
             ? left_start
             : loom_low_allocation_storage_lease_index_first_unexpired_start(
                   index, node->data.children[1], start_point, conflict_class);
}

static uint32_t loom_low_allocation_storage_lease_index_find_unit(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t root_ordinal, uint64_t key) {
  uint32_t node_index = index->unit_roots[root_ordinal];
  while (node_index != UINT32_MAX && index->nodes[node_index].level != 0) {
    const loom_low_allocation_storage_lease_index_node_t* node =
        &index->nodes[node_index];
    node_index =
        node->data.children[loom_low_allocation_storage_lease_key_child(
            key, node->level)];
  }
  return node_index != UINT32_MAX && index->nodes[node_index].key == key
             ? node_index
             : UINT32_MAX;
}

void loom_low_allocation_storage_lease_unit_index_refresh_availability(
    loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t storage_lease_index, uint32_t start_point,
    bool all_leases_pressure_releasable) {
  const loom_low_allocation_storage_lease_t* lease =
      &index->instances[storage_lease_index];
  if (loom_low_reg_class_uses_explicit_physical_registers(
          &descriptor_set->reg_classes[lease->descriptor_reg_class_id])) {
    return;
  }
  const uint32_t root_ordinal =
      loom_low_allocation_storage_lease_unit_root_ordinal(lease->location_kind);
  const uint32_t storage_key = loom_low_reg_class_storage_key(
      descriptor_set, lease->descriptor_reg_class_id);
  for (uint32_t unit_offset = 0; unit_offset < lease->location_count;
       ++unit_offset) {
    const uint64_t key =
        ((uint64_t)storage_key << 32) | (lease->location_base + unit_offset);
    const uint32_t unit_index =
        loom_low_allocation_storage_lease_index_find_unit(index, root_ordinal,
                                                          key);
    IREE_ASSERT_NE(unit_index, UINT32_MAX);
    loom_low_allocation_storage_lease_index_node_t* unit =
        &index->nodes[unit_index];
    unit->metadata.directory.maximum_first_start =
        loom_low_allocation_storage_lease_index_first_unexpired_start(
            index, unit->data.unit.temporal_root, start_point,
            LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_ALL);
    if (all_leases_pressure_releasable) {
      unit->metadata.directory.policy.minimum_pressure_first_start =
          unit->metadata.directory.maximum_first_start;
    } else {
      unit->metadata.directory.policy.maximum_non_pressure_first_start =
          loom_low_allocation_storage_lease_index_first_unexpired_start(
              index, unit->data.unit.temporal_root, start_point,
              LOOM_LOW_ALLOCATION_STORAGE_LEASE_CONFLICT_NON_PRESSURE);
    }
    loom_low_allocation_storage_lease_index_refresh_directory_ancestors(
        index, unit->metadata.directory.parent, all_leases_pressure_releasable);
  }
}

uint32_t
loom_low_allocation_storage_lease_unit_index_minimum_pressure_start_point(
    const loom_low_allocation_storage_lease_unit_index_t* index) {
  uint32_t minimum_start_point = UINT32_MAX;
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(index->unit_roots); ++i) {
    if (index->unit_roots[i] != UINT32_MAX) {
      minimum_start_point =
          iree_min(minimum_start_point,
                   index->nodes[index->unit_roots[i]]
                       .metadata.directory.policy.minimum_pressure_first_start);
    }
  }
  return minimum_start_point;
}

static bool loom_low_allocation_storage_lease_index_find_first_available(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t node_index, uint64_t maximum_key, uint32_t candidate_end_point,
    loom_low_allocation_storage_lease_conflict_class_t conflict_class,
    uint64_t* cursor, uint64_t* out_key) {
  if (node_index == UINT32_MAX || *cursor > maximum_key) {
    return false;
  }
  const loom_low_allocation_storage_lease_index_node_t* node =
      &index->nodes[node_index];
  const uint64_t subtree_minimum = node->key;
  const uint64_t subtree_maximum =
      loom_low_allocation_storage_lease_index_maximum_prefix_key(node);
  if (subtree_maximum < *cursor || subtree_minimum > maximum_key) {
    return false;
  }
  if (subtree_minimum > *cursor) {
    *out_key = *cursor;
    return true;
  }

  const uint32_t summary =
      loom_low_allocation_storage_lease_directory_summary(node, conflict_class);
  if (summary != UINT32_MAX && summary < candidate_end_point &&
      *cursor >= subtree_minimum && *cursor <= subtree_maximum) {
    *cursor = subtree_maximum + 1u;
    return false;
  }
  if (node->level == 0) {
    if (*cursor < node->key || summary == UINT32_MAX ||
        summary >= candidate_end_point) {
      *out_key = *cursor;
      return true;
    }
    ++*cursor;
    return false;
  }
  if (loom_low_allocation_storage_lease_index_find_first_available(
          index, node->data.children[0], maximum_key, candidate_end_point,
          conflict_class, cursor, out_key)) {
    return true;
  }
  return loom_low_allocation_storage_lease_index_find_first_available(
      index, node->data.children[1], maximum_key, candidate_end_point,
      conflict_class, cursor, out_key);
}

static bool loom_low_allocation_storage_lease_index_find_last_available(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t node_index, uint64_t minimum_key, uint32_t candidate_end_point,
    loom_low_allocation_storage_lease_conflict_class_t conflict_class,
    uint64_t* cursor, bool* exhausted, uint64_t* out_key) {
  if (node_index == UINT32_MAX || *exhausted || *cursor < minimum_key) {
    return false;
  }
  const loom_low_allocation_storage_lease_index_node_t* node =
      &index->nodes[node_index];
  const uint64_t subtree_minimum = node->key;
  const uint64_t subtree_maximum =
      loom_low_allocation_storage_lease_index_maximum_prefix_key(node);
  if (subtree_maximum < minimum_key || subtree_minimum > *cursor) {
    return false;
  }
  if (subtree_maximum < *cursor) {
    *out_key = *cursor;
    return true;
  }

  const uint32_t summary =
      loom_low_allocation_storage_lease_directory_summary(node, conflict_class);
  if (summary != UINT32_MAX && summary < candidate_end_point &&
      *cursor >= subtree_minimum && *cursor <= subtree_maximum) {
    if (subtree_minimum <= minimum_key) {
      *exhausted = true;
    } else {
      *cursor = subtree_minimum - 1u;
    }
    return false;
  }
  if (node->level == 0) {
    if (*cursor > node->key || summary == UINT32_MAX ||
        summary >= candidate_end_point) {
      *out_key = *cursor;
      return true;
    }
    if (node->key <= minimum_key) {
      *exhausted = true;
    } else {
      *cursor = node->key - 1u;
    }
    return false;
  }
  if (loom_low_allocation_storage_lease_index_find_last_available(
          index, node->data.children[1], minimum_key, candidate_end_point,
          conflict_class, cursor, exhausted, out_key)) {
    return true;
  }
  return loom_low_allocation_storage_lease_index_find_last_available(
      index, node->data.children[0], minimum_key, candidate_end_point,
      conflict_class, cursor, exhausted, out_key);
}

bool loom_low_allocation_storage_lease_unit_index_find_next_available_location(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id,
    loom_low_allocation_location_kind_t location_kind,
    uint32_t candidate_end_point,
    loom_low_allocation_storage_lease_conflict_class_t conflict_class,
    uint32_t minimum_base, uint32_t maximum_base, uint32_t* out_base) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(out_base);
  if (minimum_base > maximum_base) {
    return false;
  }
  const uint32_t root_ordinal =
      loom_low_allocation_storage_lease_unit_root_ordinal(location_kind);
  const uint32_t storage_key =
      loom_low_reg_class_storage_key(descriptor_set, descriptor_reg_class_id);
  uint64_t cursor = ((uint64_t)storage_key << 32) | minimum_base;
  const uint64_t maximum_key = ((uint64_t)storage_key << 32) | maximum_base;
  uint64_t result = 0;
  const bool found =
      loom_low_allocation_storage_lease_index_find_first_available(
          index, index->unit_roots[root_ordinal], maximum_key,
          candidate_end_point, conflict_class, &cursor, &result);
  if (found) {
    *out_base = (uint32_t)result;
    return true;
  }
  if (cursor <= maximum_key) {
    *out_base = (uint32_t)cursor;
    return true;
  }
  return false;
}

bool loom_low_allocation_storage_lease_unit_index_find_previous_available_location(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id,
    loom_low_allocation_location_kind_t location_kind,
    uint32_t candidate_end_point,
    loom_low_allocation_storage_lease_conflict_class_t conflict_class,
    uint32_t minimum_base, uint32_t maximum_base, uint32_t* out_base) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(out_base);
  if (minimum_base > maximum_base) {
    return false;
  }
  const uint32_t root_ordinal =
      loom_low_allocation_storage_lease_unit_root_ordinal(location_kind);
  const uint32_t storage_key =
      loom_low_reg_class_storage_key(descriptor_set, descriptor_reg_class_id);
  const uint64_t minimum_key = ((uint64_t)storage_key << 32) | minimum_base;
  uint64_t cursor = ((uint64_t)storage_key << 32) | maximum_base;
  bool exhausted = false;
  uint64_t result = 0;
  const bool found =
      loom_low_allocation_storage_lease_index_find_last_available(
          index, index->unit_roots[root_ordinal], minimum_key,
          candidate_end_point, conflict_class, &cursor, &exhausted, &result);
  if (found) {
    *out_base = (uint32_t)result;
    return true;
  }
  if (!exhausted && cursor >= minimum_key) {
    *out_base = (uint32_t)cursor;
    return true;
  }
  return false;
}

bool loom_low_allocation_storage_lease_unit_index_is_enabled(
    const loom_low_allocation_storage_lease_unit_index_t* index) {
  return index != NULL && index->node_capacity != 0;
}

// Finds the first materialized unit leaf in the inclusive key range. Directory
// children are ordered by key, and compressed prefixes let the walk reject an
// entire subtree without visiting its leaves.
static uint32_t
loom_low_allocation_storage_lease_index_find_first_unit_in_range(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    uint32_t node_index, uint64_t minimum_key, uint64_t maximum_key) {
  if (node_index == UINT32_MAX) {
    return UINT32_MAX;
  }
  const loom_low_allocation_storage_lease_index_node_t* node =
      &index->nodes[node_index];
  if (loom_low_allocation_storage_lease_index_maximum_prefix_key(node) <
          minimum_key ||
      node->key > maximum_key) {
    return UINT32_MAX;
  }
  if (node->level == 0) {
    return node_index;
  }
  const uint32_t left_index =
      loom_low_allocation_storage_lease_index_find_first_unit_in_range(
          index, node->data.children[0], minimum_key, maximum_key);
  return left_index != UINT32_MAX
             ? left_index
             : loom_low_allocation_storage_lease_index_find_first_unit_in_range(
                   index, node->data.children[1], minimum_key, maximum_key);
}

iree_status_t loom_low_allocation_storage_lease_selection_initialize(
    const loom_low_allocation_storage_lease_unit_index_t* index,
    iree_arena_allocator_t* arena,
    loom_low_allocation_storage_lease_selection_t* out_selection) {
  *out_selection = (loom_low_allocation_storage_lease_selection_t){
      .index = index,
  };
  if (index == NULL || index->node_count == 0) {
    return iree_ok_status();
  }
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
      node_index = index->nodes[node_index].metadata.temporal.parent;
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
  out_query->maximum_unit_key = 0;
  out_query->next_unit_index = UINT32_MAX;
  out_query->active_location = 0;
  out_query->minimum_end_point = minimum_end_point;
  out_query->start_point_limit = start_point_limit;
  out_query->stack_count = 0;
  if (location_count == 0 ||
      !loom_low_allocation_storage_lease_unit_index_is_enabled(index)) {
    return;
  }
  const uint64_t maximum_location =
      (uint64_t)location_base + location_count - 1u;
  IREE_ASSERT_LE(maximum_location, UINT32_MAX);
  const uint64_t storage_key =
      loom_low_reg_class_storage_key(descriptor_set, descriptor_reg_class_id);
  const uint64_t minimum_unit_key = (storage_key << 32) | location_base;
  out_query->maximum_unit_key = (storage_key << 32) | maximum_location;
  const uint32_t unit_root_ordinal =
      loom_low_allocation_storage_lease_unit_root_ordinal(location_kind);
  if (location_count == 1) {
    const uint32_t unit_index =
        loom_low_allocation_storage_lease_index_find_unit(
            index, unit_root_ordinal, minimum_unit_key);
    if (unit_index != UINT32_MAX) {
      out_query->active_location = location_base;
      out_query->stack[out_query->stack_count++] =
          index->nodes[unit_index].data.unit.temporal_root;
    }
  } else {
    out_query->next_unit_index =
        loom_low_allocation_storage_lease_index_find_first_unit_in_range(
            index, index->unit_roots[unit_root_ordinal], minimum_unit_key,
            out_query->maximum_unit_key);
  }
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
          node->metadata.temporal.maximum_end_point >= query->minimum_end_point;
      const bool selected_match =
          query->selection != NULL &&
          query->selection->subtree_counts[node_index] != 0;
      if (!temporal_match && !selected_match) {
        continue;
      }
      if (node->level == 0) {
        *out_storage_lease_index = node->data.lease.index;
        return true;
      }
      query->stack[query->stack_count++] = node->data.children[0];
      query->stack[query->stack_count++] = node->data.children[1];
    }
    if (query->next_unit_index == UINT32_MAX) {
      return false;
    }
    const uint32_t unit_index = query->next_unit_index;
    query->active_location = (uint32_t)index->nodes[unit_index].key;
    const uint32_t next_unit_index =
        index->nodes[unit_index].data.unit.next_unit;
    query->next_unit_index =
        next_unit_index != UINT32_MAX &&
                index->nodes[next_unit_index].key <= query->maximum_unit_key
            ? next_unit_index
            : UINT32_MAX;
    query->stack[query->stack_count++] =
        index->nodes[unit_index].data.unit.temporal_root;
  }
}
