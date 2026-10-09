// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/active_unit.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"

enum {
  // Tiny functions are cheaper to scan linearly than to index.
  LOOM_LOW_ALLOCATION_ACTIVE_UNIT_INDEX_MIN_CAPACITY = 32,
};

struct loom_low_allocation_active_unit_entry_t {
  // Assignment occupying this active unit.
  uint32_t assignment_index;
  // Next entry in the hashed unit bucket, or free-list link when inactive.
  uint32_t next_entry;
  // Previous entry in the hashed unit bucket.
  uint32_t previous_entry;
  // Next unit owned by the same assignment, or UINT32_MAX.
  uint32_t next_assignment_entry;
  // Target-storage identity key for the register class owning this unit.
  uint32_t storage_key;
  // Target-visible storage kind for this unit.
  loom_low_allocation_location_kind_t location_kind;
  // Physical register or target ID for this unit.
  uint32_t location;
};

// One distinct continuously occupied scalar location in an AVL tree. Subtree
// bounds and density let ordered searches skip contiguous runs in one step
// while owner_count preserves overlapping tied assignments.
struct loom_low_allocation_active_location_node_t {
  // Linear physical-register or target-ID location.
  uint32_t location;
  // Smallest location in this subtree.
  uint32_t subtree_minimum;
  // Largest location in this subtree.
  uint32_t subtree_maximum;
  // Left child node, or UINT32_MAX.
  uint32_t left;
  // Right child node, or UINT32_MAX.
  uint32_t right;
  // AVL height in the low bits and subtree density in the high bit.
  uint32_t height_and_dense;
  // Number of active assignments occupying |location|.
  uint32_t owner_count;
};

enum {
  LOOM_LOW_ALLOCATION_ACTIVE_LOCATION_DENSE_BIT = UINT32_C(1) << 31,
  LOOM_LOW_ALLOCATION_ACTIVE_LOCATION_HEIGHT_MASK = UINT32_C(0xFF),
};

static uint32_t loom_low_allocation_round_up_to_power_of_two_u32(
    uint32_t value) {
  if (value <= 1) {
    return 1;
  }
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value == UINT32_MAX ? 0 : value + 1u;
}

static uint32_t loom_low_allocation_active_unit_hash(
    loom_low_allocation_location_kind_t location_kind, uint32_t storage_key,
    uint32_t location) {
  uint32_t hash = location ^ ((uint32_t)location_kind * 0x9E3779B9u);
  hash ^= storage_key * 0x7F4A7C15u;
  hash ^= hash >> 16;
  hash *= 0x85EBCA6Bu;
  hash ^= hash >> 13;
  hash *= 0xC2B2AE35u;
  hash ^= hash >> 16;
  return hash;
}

static uint32_t loom_low_allocation_active_location_kind_ordinal(
    loom_low_allocation_location_kind_t location_kind) {
  IREE_ASSERT(
      loom_low_allocation_location_kind_is_register_like(location_kind));
  return location_kind == LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ? 0
                                                                         : 1;
}

static bool loom_low_allocation_active_location_is_dense(
    const loom_low_allocation_active_location_node_t* node) {
  return (node->height_and_dense &
          LOOM_LOW_ALLOCATION_ACTIVE_LOCATION_DENSE_BIT) != 0;
}

static uint32_t loom_low_allocation_active_location_height(
    const loom_low_allocation_active_unit_index_t* index, uint32_t node_index) {
  return node_index == UINT32_MAX
             ? 0
             : index->active_location_nodes[node_index].height_and_dense &
                   LOOM_LOW_ALLOCATION_ACTIVE_LOCATION_HEIGHT_MASK;
}

static void loom_low_allocation_active_location_recompute(
    loom_low_allocation_active_unit_index_t* index, uint32_t node_index) {
  loom_low_allocation_active_location_node_t* node =
      &index->active_location_nodes[node_index];
  node->subtree_minimum =
      node->left == UINT32_MAX
          ? node->location
          : index->active_location_nodes[node->left].subtree_minimum;
  node->subtree_maximum =
      node->right == UINT32_MAX
          ? node->location
          : index->active_location_nodes[node->right].subtree_maximum;
  const bool left_dense =
      node->left == UINT32_MAX ||
      (loom_low_allocation_active_location_is_dense(
           &index->active_location_nodes[node->left]) &&
       index->active_location_nodes[node->left].subtree_maximum != UINT32_MAX &&
       index->active_location_nodes[node->left].subtree_maximum + 1u ==
           node->location);
  const bool right_dense =
      node->right == UINT32_MAX ||
      (loom_low_allocation_active_location_is_dense(
           &index->active_location_nodes[node->right]) &&
       node->location != UINT32_MAX &&
       node->location + 1u ==
           index->active_location_nodes[node->right].subtree_minimum);
  const uint32_t height =
      1u +
      iree_max(loom_low_allocation_active_location_height(index, node->left),
               loom_low_allocation_active_location_height(index, node->right));
  IREE_ASSERT_LE(height, LOOM_LOW_ALLOCATION_ACTIVE_LOCATION_HEIGHT_MASK);
  node->height_and_dense =
      height |
      (left_dense && right_dense ? LOOM_LOW_ALLOCATION_ACTIVE_LOCATION_DENSE_BIT
                                 : 0);
}

static uint32_t loom_low_allocation_active_location_rotate_left(
    loom_low_allocation_active_unit_index_t* index, uint32_t root_index) {
  loom_low_allocation_active_location_node_t* root =
      &index->active_location_nodes[root_index];
  const uint32_t new_root_index = root->right;
  loom_low_allocation_active_location_node_t* new_root =
      &index->active_location_nodes[new_root_index];
  root->right = new_root->left;
  new_root->left = root_index;
  loom_low_allocation_active_location_recompute(index, root_index);
  loom_low_allocation_active_location_recompute(index, new_root_index);
  return new_root_index;
}

static uint32_t loom_low_allocation_active_location_rotate_right(
    loom_low_allocation_active_unit_index_t* index, uint32_t root_index) {
  loom_low_allocation_active_location_node_t* root =
      &index->active_location_nodes[root_index];
  const uint32_t new_root_index = root->left;
  loom_low_allocation_active_location_node_t* new_root =
      &index->active_location_nodes[new_root_index];
  root->left = new_root->right;
  new_root->right = root_index;
  loom_low_allocation_active_location_recompute(index, root_index);
  loom_low_allocation_active_location_recompute(index, new_root_index);
  return new_root_index;
}

static uint32_t loom_low_allocation_active_location_rebalance(
    loom_low_allocation_active_unit_index_t* index, uint32_t root_index) {
  loom_low_allocation_active_location_recompute(index, root_index);
  loom_low_allocation_active_location_node_t* root =
      &index->active_location_nodes[root_index];
  const int32_t balance =
      (int32_t)loom_low_allocation_active_location_height(index, root->left) -
      (int32_t)loom_low_allocation_active_location_height(index, root->right);
  if (balance > 1) {
    loom_low_allocation_active_location_node_t* left =
        &index->active_location_nodes[root->left];
    if (loom_low_allocation_active_location_height(index, left->left) <
        loom_low_allocation_active_location_height(index, left->right)) {
      root->left =
          loom_low_allocation_active_location_rotate_left(index, root->left);
    }
    return loom_low_allocation_active_location_rotate_right(index, root_index);
  }
  if (balance < -1) {
    loom_low_allocation_active_location_node_t* right =
        &index->active_location_nodes[root->right];
    if (loom_low_allocation_active_location_height(index, right->right) <
        loom_low_allocation_active_location_height(index, right->left)) {
      root->right =
          loom_low_allocation_active_location_rotate_right(index, root->right);
    }
    return loom_low_allocation_active_location_rotate_left(index, root_index);
  }
  return root_index;
}

static uint32_t loom_low_allocation_active_location_insert_node(
    loom_low_allocation_active_unit_index_t* index, uint32_t root_index,
    uint32_t inserted_index) {
  if (root_index == UINT32_MAX) {
    return inserted_index;
  }
  loom_low_allocation_active_location_node_t* root =
      &index->active_location_nodes[root_index];
  const loom_low_allocation_active_location_node_t* inserted =
      &index->active_location_nodes[inserted_index];
  if (inserted->location < root->location) {
    root->left = loom_low_allocation_active_location_insert_node(
        index, root->left, inserted_index);
  } else {
    root->right = loom_low_allocation_active_location_insert_node(
        index, root->right, inserted_index);
  }
  return loom_low_allocation_active_location_rebalance(index, root_index);
}

static uint32_t loom_low_allocation_active_location_leftmost(
    const loom_low_allocation_active_unit_index_t* index, uint32_t root_index) {
  while (index->active_location_nodes[root_index].left != UINT32_MAX) {
    root_index = index->active_location_nodes[root_index].left;
  }
  return root_index;
}

static uint32_t loom_low_allocation_active_location_remove_node(
    loom_low_allocation_active_unit_index_t* index, uint32_t root_index,
    uint32_t location, uint32_t* out_removed_index) {
  loom_low_allocation_active_location_node_t* root =
      &index->active_location_nodes[root_index];
  if (location < root->location) {
    root->left = loom_low_allocation_active_location_remove_node(
        index, root->left, location, out_removed_index);
  } else if (location > root->location) {
    root->right = loom_low_allocation_active_location_remove_node(
        index, root->right, location, out_removed_index);
  } else {
    if (root->left == UINT32_MAX || root->right == UINT32_MAX) {
      *out_removed_index = root_index;
      return root->left == UINT32_MAX ? root->right : root->left;
    }
    const uint32_t successor_index =
        loom_low_allocation_active_location_leftmost(index, root->right);
    const loom_low_allocation_active_location_node_t* successor =
        &index->active_location_nodes[successor_index];
    root->location = successor->location;
    root->owner_count = successor->owner_count;
    root->right = loom_low_allocation_active_location_remove_node(
        index, root->right, successor->location, out_removed_index);
  }
  return loom_low_allocation_active_location_rebalance(index, root_index);
}

static uint32_t loom_low_allocation_active_location_find_node(
    const loom_low_allocation_active_unit_index_t* index, uint32_t root_index,
    uint32_t location) {
  while (root_index != UINT32_MAX) {
    const loom_low_allocation_active_location_node_t* node =
        &index->active_location_nodes[root_index];
    if (location == node->location) {
      return root_index;
    }
    root_index = location < node->location ? node->left : node->right;
  }
  return UINT32_MAX;
}

static uint32_t loom_low_allocation_active_location_space_index(
    const loom_low_allocation_active_unit_index_t* index,
    const loom_low_allocation_assignment_t* assignment) {
  const loom_low_reg_class_t* reg_class =
      &index->descriptor_set->reg_classes[assignment->descriptor_reg_class_id];
  const uint32_t storage_space = reg_class->alias_set_id != 0
                                     ? reg_class->alias_set_id - 1u
                                     : index->active_location_alias_set_count +
                                           assignment->descriptor_reg_class_id;
  const uint32_t kind_ordinal =
      loom_low_allocation_active_location_kind_ordinal(
          assignment->location_kind);
  return storage_space * 2u + kind_ordinal;
}

static void loom_low_allocation_active_location_insert(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_allocation_assignment_t* assignment) {
  const uint32_t location = assignment->location_base;
  const uint32_t space_index =
      loom_low_allocation_active_location_space_index(index, assignment);
  IREE_ASSERT_NE(space_index, UINT32_MAX);
  uint32_t* root = &index->active_location_roots[space_index];
  const uint32_t existing_index =
      loom_low_allocation_active_location_find_node(index, *root, location);
  if (existing_index != UINT32_MAX) {
    ++index->active_location_nodes[existing_index].owner_count;
    return;
  }
  uint32_t node_index = index->free_active_location_head;
  if (node_index != UINT32_MAX) {
    index->free_active_location_head =
        index->active_location_nodes[node_index].left;
  } else {
    IREE_ASSERT_LT(index->active_location_count,
                   index->active_location_capacity);
    node_index = index->active_location_count++;
  }
  index->active_location_nodes[node_index] =
      (loom_low_allocation_active_location_node_t){
          .location = location,
          .subtree_minimum = location,
          .subtree_maximum = location,
          .left = UINT32_MAX,
          .right = UINT32_MAX,
          .height_and_dense =
              1u | LOOM_LOW_ALLOCATION_ACTIVE_LOCATION_DENSE_BIT,
          .owner_count = 1,
      };
  *root =
      loom_low_allocation_active_location_insert_node(index, *root, node_index);
}

static void loom_low_allocation_active_location_remove(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_allocation_assignment_t* assignment) {
  const uint32_t location = assignment->location_base;
  const uint32_t space_index =
      loom_low_allocation_active_location_space_index(index, assignment);
  IREE_ASSERT_NE(space_index, UINT32_MAX);
  uint32_t* root = &index->active_location_roots[space_index];
  const uint32_t existing_index =
      loom_low_allocation_active_location_find_node(index, *root, location);
  IREE_ASSERT_NE(existing_index, UINT32_MAX);
  loom_low_allocation_active_location_node_t* existing =
      &index->active_location_nodes[existing_index];
  IREE_ASSERT_NE(existing->owner_count, 0u);
  if (--existing->owner_count != 0) {
    return;
  }
  uint32_t removed_index = UINT32_MAX;
  *root = loom_low_allocation_active_location_remove_node(
      index, *root, location, &removed_index);
  IREE_ASSERT_NE(removed_index, UINT32_MAX);
  index->active_location_nodes[removed_index].left =
      index->free_active_location_head;
  index->free_active_location_head = removed_index;
}

static bool loom_low_allocation_active_location_subtree_is_dense(
    const loom_low_allocation_active_location_node_t* node) {
  return loom_low_allocation_active_location_is_dense(node);
}

static bool loom_low_allocation_active_location_find_first_gap(
    const loom_low_allocation_active_unit_index_t* index, uint32_t node_index,
    uint64_t maximum, uint64_t* cursor, uint64_t* out_key) {
  if (node_index == UINT32_MAX || *cursor > maximum) {
    return false;
  }
  const loom_low_allocation_active_location_node_t* node =
      &index->active_location_nodes[node_index];
  if (node->subtree_maximum < *cursor || node->subtree_minimum > maximum) {
    return false;
  }
  if (node->subtree_minimum > *cursor) {
    *out_key = *cursor;
    return true;
  }
  if (loom_low_allocation_active_location_subtree_is_dense(node) &&
      *cursor >= node->subtree_minimum && *cursor <= node->subtree_maximum) {
    *cursor = node->subtree_maximum + 1u;
    return false;
  }
  if (loom_low_allocation_active_location_find_first_gap(
          index, node->left, maximum, cursor, out_key)) {
    return true;
  }
  if (*cursor > maximum) {
    return false;
  }
  if (*cursor < node->location) {
    *out_key = *cursor;
    return true;
  }
  if (*cursor == node->location) {
    ++*cursor;
  }
  return loom_low_allocation_active_location_find_first_gap(
      index, node->right, maximum, cursor, out_key);
}

static bool loom_low_allocation_active_location_find_last_gap(
    const loom_low_allocation_active_unit_index_t* index, uint32_t node_index,
    uint64_t minimum, uint64_t* cursor, uint64_t* out_key) {
  if (node_index == UINT32_MAX || *cursor < minimum) {
    return false;
  }
  const loom_low_allocation_active_location_node_t* node =
      &index->active_location_nodes[node_index];
  if (node->subtree_maximum < minimum || node->subtree_minimum > *cursor) {
    return false;
  }
  if (node->subtree_maximum < *cursor) {
    *out_key = *cursor;
    return true;
  }
  if (loom_low_allocation_active_location_subtree_is_dense(node) &&
      *cursor >= node->subtree_minimum && *cursor <= node->subtree_maximum) {
    if (node->subtree_minimum == 0) {
      *cursor = 0;
      return false;
    }
    *cursor = node->subtree_minimum - 1u;
    return false;
  }
  if (loom_low_allocation_active_location_find_last_gap(
          index, node->right, minimum, cursor, out_key)) {
    return true;
  }
  if (*cursor < minimum) {
    return false;
  }
  if (*cursor > node->location) {
    *out_key = *cursor;
    return true;
  }
  if (*cursor == node->location) {
    if (*cursor == 0) {
      return false;
    }
    --*cursor;
  }
  return loom_low_allocation_active_location_find_last_gap(
      index, node->left, minimum, cursor, out_key);
}

static uint32_t loom_low_allocation_active_unit_bucket_index(
    const loom_low_allocation_active_unit_index_t* index,
    loom_low_allocation_location_kind_t location_kind, uint32_t storage_key,
    uint32_t location) {
  return loom_low_allocation_active_unit_hash(location_kind, storage_key,
                                              location) &
         (index->bucket_count - 1u);
}

static uint32_t loom_low_allocation_active_unit_next_seen_generation(
    loom_low_allocation_active_unit_index_t* index) {
  if (index->seen_generations_by_assignment_index == NULL) {
    return 0;
  }
  if (index->seen_generation == UINT32_MAX) {
    memset(index->seen_generations_by_assignment_index, 0,
           index->assignment_capacity *
               sizeof(*index->seen_generations_by_assignment_index));
    index->seen_generation = 0;
  }
  return ++index->seen_generation;
}

static bool loom_low_allocation_active_unit_mark_assignment_seen(
    loom_low_allocation_active_unit_index_t* index, uint32_t assignment_index,
    uint32_t generation) {
  if (generation == 0) {
    return false;
  }
  uint32_t* seen_generations = index->seen_generations_by_assignment_index;
  uint32_t* assignment_generation = &seen_generations[assignment_index];
  if (*assignment_generation == generation) {
    return true;
  }
  *assignment_generation = generation;
  return false;
}

static bool loom_low_allocation_active_assignment_conflicts(
    loom_low_allocation_live_range_sweep_t* live_range_sweep,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t existing_assignment_index,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  IREE_ASSERT_ARGUMENT(live_range_sweep);
  IREE_ASSERT_ARGUMENT(unit_liveness);
  IREE_ASSERT_LT(existing_assignment_index, assignment_count);
  const loom_low_allocation_assignment_t* existing =
      &assignments[existing_assignment_index];
  if (loom_low_allocation_unit_liveness_storage_is_ignored(
          unit_liveness, existing->value_id, ignored_value_ids,
          ignored_value_count)) {
    return false;
  }
  if (existing->location_kind != candidate->location_kind) {
    return false;
  }
  if (existing->liveness_segments.count <
      LOOM_LOW_ALLOCATION_LIVE_RANGE_CURSOR_MIN_SEGMENT_COUNT) {
    return loom_low_allocation_live_range_assignments_conflict(
        descriptor_set, unit_liveness->storage_segments.entries,
        unit_liveness->start_points, unit_liveness->end_points,
        unit_liveness->point_count, existing, candidate);
  }
  return loom_low_allocation_live_range_sweep_assignments_conflict(
      live_range_sweep->point,
      &live_range_sweep
           ->segment_starts_by_assignment_index[existing_assignment_index],
      descriptor_set, unit_liveness->storage_segments.entries,
      unit_liveness->start_points, unit_liveness->end_points,
      unit_liveness->point_count, existing, candidate);
}

static iree_status_t loom_low_allocation_active_location_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t unit_capacity, iree_arena_allocator_t* arena,
    loom_low_allocation_active_unit_index_t* index) {
  uint32_t alias_set_count = 0;
  bool has_ordered_class = false;
  for (uint16_t reg_class_id = 0;
       reg_class_id < descriptor_set->reg_class_count; ++reg_class_id) {
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[reg_class_id];
    alias_set_count = iree_max(alias_set_count, reg_class->alias_set_id);
    has_ordered_class |=
        reg_class->allocatable_count == 0 &&
        !loom_low_reg_class_uses_explicit_physical_registers(reg_class);
  }
  if (!has_ordered_class) {
    return iree_ok_status();
  }

  iree_host_size_t storage_space_count = 0;
  iree_host_size_t location_space_count = 0;
  if (!iree_host_size_checked_add(alias_set_count,
                                  descriptor_set->reg_class_count,
                                  &storage_space_count) ||
      !iree_host_size_checked_mul(storage_space_count, 2,
                                  &location_space_count) ||
      location_space_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "active allocation location-space table exceeds uint32_t");
  }
  index->active_location_alias_set_count = alias_set_count;
  index->active_location_space_count = (uint32_t)location_space_count;

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, index->active_location_space_count,
                                sizeof(*index->active_location_roots),
                                (void**)&index->active_location_roots));
  for (uint32_t i = 0; i < index->active_location_space_count; ++i) {
    index->active_location_roots[i] = UINT32_MAX;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, unit_capacity, sizeof(*index->active_location_nodes),
      (void**)&index->active_location_nodes));
  index->active_location_capacity = (uint32_t)unit_capacity;
  index->free_active_location_head = UINT32_MAX;
  return iree_ok_status();
}

iree_status_t loom_low_allocation_active_unit_index_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t assignment_capacity, iree_host_size_t unit_capacity,
    iree_arena_allocator_t* arena,
    loom_low_allocation_active_unit_index_t* out_index) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_index);
  *out_index = (loom_low_allocation_active_unit_index_t){0};
  out_index->descriptor_set = descriptor_set;
  out_index->free_active_location_head = UINT32_MAX;
  if (unit_capacity < LOOM_LOW_ALLOCATION_ACTIVE_UNIT_INDEX_MIN_CAPACITY) {
    return iree_ok_status();
  }
  if (unit_capacity > UINT32_MAX / 2u) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "active allocation unit index exceeds uint32_t");
  }

  const uint32_t bucket_count =
      loom_low_allocation_round_up_to_power_of_two_u32((uint32_t)unit_capacity *
                                                       2u);
  if (bucket_count == 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "active allocation unit buckets exceed uint32_t");
  }

  out_index->bucket_count = bucket_count;
  out_index->entry_capacity = (uint32_t)unit_capacity;
  out_index->free_entry_head = UINT32_MAX;
  out_index->assignment_capacity = assignment_capacity;

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, bucket_count, sizeof(*out_index->bucket_heads),
      (void**)&out_index->bucket_heads));
  for (uint32_t i = 0; i < bucket_count; ++i) {
    out_index->bucket_heads[i] = UINT32_MAX;
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, unit_capacity,
                                                 sizeof(*out_index->entries),
                                                 (void**)&out_index->entries));

  IREE_RETURN_IF_ERROR(loom_low_allocation_active_location_initialize(
      descriptor_set, unit_capacity, arena, out_index));

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assignment_capacity,
      sizeof(*out_index->entry_starts_by_assignment_index),
      (void**)&out_index->entry_starts_by_assignment_index));
  for (iree_host_size_t i = 0; i < assignment_capacity; ++i) {
    out_index->entry_starts_by_assignment_index[i] = UINT32_MAX;
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, assignment_capacity,
      sizeof(*out_index->seen_generations_by_assignment_index),
      (void**)&out_index->seen_generations_by_assignment_index));
  memset(out_index->seen_generations_by_assignment_index, 0,
         assignment_capacity *
             sizeof(*out_index->seen_generations_by_assignment_index));
  return iree_ok_status();
}

bool loom_low_allocation_active_unit_index_is_enabled(
    const loom_low_allocation_active_unit_index_t* index) {
  IREE_ASSERT_ARGUMENT(index);
  return index->bucket_count != 0;
}

bool loom_low_allocation_active_unit_index_can_order_candidate(
    const loom_low_allocation_active_unit_index_t* index,
    const loom_low_allocation_assignment_t* candidate) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(candidate);
  if (index->active_location_nodes == NULL ||
      candidate->descriptor_reg_class_id >=
          index->descriptor_set->reg_class_count ||
      !loom_low_allocation_location_kind_is_register_like(
          candidate->location_kind)) {
    return false;
  }
  const loom_low_reg_class_t* reg_class =
      &index->descriptor_set->reg_classes[candidate->descriptor_reg_class_id];
  return reg_class->allocatable_count == 0 &&
         !loom_low_reg_class_uses_explicit_physical_registers(reg_class) &&
         candidate->unit_count == 1 && candidate->location_count == 1 &&
         candidate->liveness_segments.count == 0 &&
         !iree_any_bit_set(
             candidate->flags,
             LOOM_LOW_ALLOCATION_ASSIGNMENT_FLAG_REFINED_UNIT_STARTS) &&
         !loom_low_allocation_storage_assignment_uses_explicit_physical_register(
             index->descriptor_set, candidate);
}

bool loom_low_allocation_active_unit_index_find_unoccupied_location(
    const loom_low_allocation_active_unit_index_t* index,
    const loom_low_allocation_assignment_t* candidate, uint32_t minimum_base,
    uint32_t maximum_base,
    loom_low_allocation_location_search_direction_t direction,
    uint32_t* out_base) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(candidate);
  IREE_ASSERT_ARGUMENT(out_base);
  IREE_ASSERT_TRUE(loom_low_allocation_active_unit_index_can_order_candidate(
      index, candidate));
  if (minimum_base > maximum_base) {
    return false;
  }
  const uint64_t minimum = minimum_base;
  const uint64_t maximum = maximum_base;
  const uint32_t space_index =
      loom_low_allocation_active_location_space_index(index, candidate);
  const uint32_t root = index->active_location_roots[space_index];
  uint64_t result = 0;
  bool found = false;
  if (direction == LOOM_LOW_ALLOCATION_LOCATION_SEARCH_DESCENDING) {
    uint64_t cursor = maximum;
    found = loom_low_allocation_active_location_find_last_gap(
        index, root, minimum, &cursor, &result);
    if (!found && cursor >= minimum &&
        loom_low_allocation_active_location_find_node(
            index, root, (uint32_t)cursor) == UINT32_MAX) {
      result = cursor;
      found = true;
    }
  } else {
    uint64_t cursor = minimum;
    found = loom_low_allocation_active_location_find_first_gap(
        index, root, maximum, &cursor, &result);
    if (!found && cursor <= maximum &&
        loom_low_allocation_active_location_find_node(
            index, root, (uint32_t)cursor) == UINT32_MAX) {
      result = cursor;
      found = true;
    }
  }
  if (!found) {
    return false;
  }
  *out_base = (uint32_t)result;
  return true;
}

bool loom_low_allocation_active_unit_index_conflicts(
    loom_low_allocation_active_unit_index_t* index,
    loom_low_allocation_live_range_sweep_t* live_range_sweep,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(live_range_sweep);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(assignments);
  IREE_ASSERT_ARGUMENT(candidate);
  if (!loom_low_allocation_location_kind_is_register_like(
          candidate->location_kind)) {
    return false;
  }
  const uint32_t generation =
      loom_low_allocation_active_unit_next_seen_generation(index);
  const uint32_t atomic_unit_count =
      loom_low_allocation_storage_assignment_atomic_unit_count(descriptor_set,
                                                               candidate);
  for (uint32_t unit_offset = 0; unit_offset < atomic_unit_count;
       ++unit_offset) {
    uint32_t storage_key = 0;
    uint32_t location = 0;
    loom_low_allocation_storage_assignment_atomic_unit(
        descriptor_set, candidate, unit_offset, &storage_key, &location);
    const uint32_t bucket_index = loom_low_allocation_active_unit_bucket_index(
        index, candidate->location_kind, storage_key, location);
    uint32_t entry_index = index->bucket_heads[bucket_index];
    while (entry_index != UINT32_MAX) {
      IREE_ASSERT_LT(entry_index, index->entry_count);
      const loom_low_allocation_active_unit_entry_t* entry =
          &index->entries[entry_index];
      const uint32_t assignment_index = entry->assignment_index;
      IREE_ASSERT_LT(assignment_index, assignment_count);
      const loom_low_allocation_assignment_t* existing =
          &assignments[assignment_index];
      if (entry->location_kind == candidate->location_kind &&
          entry->storage_key == storage_key && entry->location == location &&
          existing->end_point > candidate->start_point &&
          !loom_low_allocation_active_unit_mark_assignment_seen(
              index, assignment_index, generation) &&
          loom_low_allocation_active_assignment_conflicts(
              live_range_sweep, descriptor_set, unit_liveness, assignments,
              assignment_count, assignment_index, candidate, ignored_value_ids,
              ignored_value_count)) {
        return true;
      }
      entry_index = entry->next_entry;
    }
  }
  return false;
}

iree_status_t loom_low_allocation_active_unit_index_collect_conflicts(
    loom_low_allocation_active_unit_index_t* index,
    loom_low_allocation_live_range_sweep_t* live_range_sweep,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    uint32_t* assignment_indices, uint16_t assignment_capacity,
    uint16_t* inout_assignment_count) {
  if (!loom_low_allocation_location_kind_is_register_like(
          candidate->location_kind)) {
    return iree_ok_status();
  }
  const uint32_t generation =
      loom_low_allocation_active_unit_next_seen_generation(index);
  const uint32_t atomic_unit_count =
      loom_low_allocation_storage_assignment_atomic_unit_count(descriptor_set,
                                                               candidate);
  for (uint32_t unit_offset = 0; unit_offset < atomic_unit_count;
       ++unit_offset) {
    uint32_t storage_key = 0;
    uint32_t location = 0;
    loom_low_allocation_storage_assignment_atomic_unit(
        descriptor_set, candidate, unit_offset, &storage_key, &location);
    const uint32_t bucket_index = loom_low_allocation_active_unit_bucket_index(
        index, candidate->location_kind, storage_key, location);
    uint32_t entry_index = index->bucket_heads[bucket_index];
    while (entry_index != UINT32_MAX) {
      IREE_ASSERT_LT(entry_index, index->entry_count);
      const loom_low_allocation_active_unit_entry_t* entry =
          &index->entries[entry_index];
      const uint32_t assignment_index = entry->assignment_index;
      IREE_ASSERT_LT(assignment_index, assignment_count);
      const loom_low_allocation_assignment_t* existing =
          &assignments[assignment_index];
      if (entry->location_kind == candidate->location_kind &&
          entry->storage_key == storage_key && entry->location == location &&
          existing->end_point > candidate->start_point &&
          !loom_low_allocation_active_unit_mark_assignment_seen(
              index, assignment_index, generation) &&
          loom_low_allocation_active_assignment_conflicts(
              live_range_sweep, descriptor_set, unit_liveness, assignments,
              assignment_count, assignment_index, candidate, ignored_value_ids,
              ignored_value_count)) {
        if (*inout_assignment_count == assignment_capacity) {
          return iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "active allocation conflict set exceeds capacity");
        }
        assignment_indices[(*inout_assignment_count)++] = assignment_index;
      }
      entry_index = entry->next_entry;
    }
  }
  return iree_ok_status();
}

void loom_low_allocation_active_unit_index_insert_assignment(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(assignments);
  if (!loom_low_allocation_active_unit_index_is_enabled(index)) {
    return;
  }
  IREE_ASSERT_LT(assignment_index, assignment_count);
  IREE_ASSERT_LT(assignment_index, index->assignment_capacity);
  const loom_low_allocation_assignment_t* assignment =
      &assignments[assignment_index];
  if (!loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind)) {
    return;
  }
  if (loom_low_allocation_active_unit_index_can_order_candidate(index,
                                                                assignment)) {
    loom_low_allocation_active_location_insert(index, assignment);
  }
  const uint32_t atomic_unit_count =
      loom_low_allocation_storage_assignment_atomic_unit_count(descriptor_set,
                                                               assignment);
  IREE_ASSERT(
      atomic_unit_count <= index->entry_capacity - index->active_entry_count,
      "planned active unit capacity must cover every assignment");

  uint32_t assignment_entry_head = UINT32_MAX;
  for (uint32_t unit_offset = 0; unit_offset < atomic_unit_count;
       ++unit_offset) {
    uint32_t storage_key = 0;
    uint32_t location = 0;
    loom_low_allocation_storage_assignment_atomic_unit(
        descriptor_set, assignment, unit_offset, &storage_key, &location);
    const uint32_t bucket_index = loom_low_allocation_active_unit_bucket_index(
        index, assignment->location_kind, storage_key, location);
    uint32_t entry_index = index->free_entry_head;
    if (entry_index != UINT32_MAX) {
      index->free_entry_head = index->entries[entry_index].next_entry;
    } else {
      entry_index = index->entry_count++;
    }
    loom_low_allocation_active_unit_entry_t* entry =
        &index->entries[entry_index];
    *entry = (loom_low_allocation_active_unit_entry_t){
        .assignment_index = assignment_index,
        .next_entry = index->bucket_heads[bucket_index],
        .previous_entry = UINT32_MAX,
        .next_assignment_entry = assignment_entry_head,
        .storage_key = storage_key,
        .location_kind = assignment->location_kind,
        .location = location,
    };
    if (entry->next_entry != UINT32_MAX) {
      IREE_ASSERT_LT(entry->next_entry, index->entry_count);
      index->entries[entry->next_entry].previous_entry = entry_index;
    }
    index->bucket_heads[bucket_index] = entry_index;
    assignment_entry_head = entry_index;
  }
  index->entry_starts_by_assignment_index[assignment_index] =
      assignment_entry_head;
  index->active_entry_count += atomic_unit_count;
}

void loom_low_allocation_active_unit_index_remove_assignment(
    loom_low_allocation_active_unit_index_t* index,
    const loom_low_allocation_assignment_t* assignments,
    iree_host_size_t assignment_count, uint32_t assignment_index) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(assignments);
  if (!loom_low_allocation_active_unit_index_is_enabled(index)) {
    return;
  }
  IREE_ASSERT_LT(assignment_index, assignment_count);
  IREE_ASSERT_LT(assignment_index, index->assignment_capacity);
  const loom_low_allocation_assignment_t* assignment =
      &assignments[assignment_index];
  if (!loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind)) {
    return;
  }
  if (loom_low_allocation_active_unit_index_can_order_candidate(index,
                                                                assignment)) {
    loom_low_allocation_active_location_remove(index, assignment);
  }
  const uint32_t entry_start =
      index->entry_starts_by_assignment_index[assignment_index];
  IREE_ASSERT_NE(entry_start, UINT32_MAX);
  uint32_t entry_index = entry_start;
  while (entry_index != UINT32_MAX) {
    IREE_ASSERT_LT(entry_index, index->entry_count);
    loom_low_allocation_active_unit_entry_t* entry =
        &index->entries[entry_index];
    if (entry->previous_entry != UINT32_MAX) {
      IREE_ASSERT_LT(entry->previous_entry, index->entry_count);
      index->entries[entry->previous_entry].next_entry = entry->next_entry;
    } else {
      const uint32_t bucket_index =
          loom_low_allocation_active_unit_bucket_index(
              index, entry->location_kind, entry->storage_key, entry->location);
      IREE_ASSERT_EQ(index->bucket_heads[bucket_index], entry_index);
      index->bucket_heads[bucket_index] = entry->next_entry;
    }
    if (entry->next_entry != UINT32_MAX) {
      IREE_ASSERT_LT(entry->next_entry, index->entry_count);
      index->entries[entry->next_entry].previous_entry = entry->previous_entry;
    }
    entry->next_entry = index->free_entry_head;
    index->free_entry_head = entry_index;
    --index->active_entry_count;
    entry_index = entry->next_assignment_entry;
  }
  index->entry_starts_by_assignment_index[assignment_index] = UINT32_MAX;
}
