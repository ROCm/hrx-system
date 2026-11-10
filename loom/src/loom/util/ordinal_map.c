// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/ordinal_map.h"

#include <string.h>

#include "iree/base/internal/math.h"

#define LOOM_ORDINAL_MAP_ORDINAL_MASK UINT32_C(0xFFFFFF)
#define LOOM_ORDINAL_MAP_DOMAIN_SHIFT 24

// Each branch skips common prefix bits and selects one of sixteen nibbles.
// Capacity doubles locally; insertion never moves more than sixteen children.
// All allocations for a branch are linear in its final live child count,
// including abandoned capacities. Every branch has at least two children, so
// total arena storage is linear in the number of reached identities.
typedef struct loom_ordinal_map_node_t {
  // Representative descendant key supplying the compressed prefix.
  uint32_t prefix;
  // Occupied nibble values, with children stored in bitmap rank order.
  uint16_t mask;
  // Shift of the four key bits selected by this branch, from zero through 24.
  uint8_t shift;
  // Allocated trailing child words, one of 2, 4, 8, or 16.
  uint8_t capacity;
  // Tagged leaves or aligned branch pointers in increasing nibble order.
  uint64_t children[];
} loom_ordinal_map_node_t;

static uint32_t loom_ordinal_map_key(uint8_t domain, uint32_t source_ordinal) {
  IREE_ASSERT(domain < 8);
  IREE_ASSERT(source_ordinal <= LOOM_ORDINAL_MAP_ORDINAL_MASK);
  return source_ordinal | ((uint32_t)domain << LOOM_ORDINAL_MAP_DOMAIN_SHIFT);
}

static uint64_t loom_ordinal_map_leaf(uint32_t key, uint32_t target_id) {
  return UINT64_C(1) | ((uint64_t)key << 1) | ((uint64_t)target_id << 32);
}

static uint32_t loom_ordinal_map_leaf_key(uint64_t leaf) {
  return (uint32_t)leaf >> 1;
}

static bool loom_ordinal_map_lookup_root(uint64_t reference, uint32_t key,
                                         uint32_t* out_target_id) {
  if (reference == 0) {
    return false;
  }
  while ((reference & 1) == 0) {
    const loom_ordinal_map_node_t* node =
        (const loom_ordinal_map_node_t*)(uintptr_t)reference;
    if (((key ^ node->prefix) >> (node->shift + 4)) != 0) {
      return false;
    }
    const uint32_t bit = UINT32_C(1) << ((key >> node->shift) & 15);
    if ((node->mask & bit) == 0) {
      return false;
    }
    reference =
        node->children[iree_math_count_ones_u32(node->mask & (bit - 1))];
  }
  if (loom_ordinal_map_leaf_key(reference) != key) {
    return false;
  }
  *out_target_id = (uint32_t)(reference >> 32);
  return true;
}

// Splits a leaf or compressed prefix at its first differing nibble. The caller
// retains the incoming edge so the new branch can splice without another walk.
static iree_status_t loom_ordinal_map_split(iree_arena_allocator_t* arena,
                                            uint32_t key, uint32_t target_id,
                                            uint32_t other_key,
                                            uint64_t* edge) {
  const uint32_t shift =
      (31 - iree_math_count_leading_zeros_u32(key ^ other_key)) & ~UINT32_C(3);
  const uint32_t nibble = (key >> shift) & 15;
  const uint32_t other_nibble = (other_key >> shift) & 15;
  loom_ordinal_map_node_t* node = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, sizeof(*node) + 2 * sizeof(node->children[0]), (void**)&node));
  *node = (loom_ordinal_map_node_t){
      .prefix = key,
      .mask =
          (uint16_t)((UINT32_C(1) << nibble) | (UINT32_C(1) << other_nibble)),
      .shift = (uint8_t)shift,
      .capacity = 2,
  };
  const uint32_t direction = nibble > other_nibble;
  node->children[direction] = loom_ordinal_map_leaf(key, target_id);
  node->children[direction ^ 1] = *edge;
  *edge = (uintptr_t)node;
  return iree_ok_status();
}

// Inserts an identity proven absent before entering this root. Bucket-array
// growth preserves that fact, and rehashing never introduces duplicate keys.
static iree_status_t loom_ordinal_map_insert_absent(
    iree_arena_allocator_t* arena, uint32_t key, uint32_t target_id,
    uint64_t* edge) {
  if (*edge == 0) {
    *edge = loom_ordinal_map_leaf(key, target_id);
    return iree_ok_status();
  }
  while ((*edge & 1) == 0) {
    loom_ordinal_map_node_t* node = (loom_ordinal_map_node_t*)(uintptr_t)*edge;
    if (((key ^ node->prefix) >> (node->shift + 4)) != 0) {
      return loom_ordinal_map_split(arena, key, target_id, node->prefix, edge);
    }
    const uint32_t bit = UINT32_C(1) << ((key >> node->shift) & 15);
    const uint32_t rank = iree_math_count_ones_u32(node->mask & (bit - 1));
    if ((node->mask & bit) != 0) {
      edge = &node->children[rank];
      continue;
    }
    const uint32_t child_count = iree_math_count_ones_u32(node->mask);
    if (child_count == node->capacity) {
      const uint32_t capacity = node->capacity * 2;
      loom_ordinal_map_node_t* replacement = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          arena,
          sizeof(*replacement) + capacity * sizeof(replacement->children[0]),
          (void**)&replacement));
      *replacement = *node;
      replacement->capacity = (uint8_t)capacity;
      memcpy(replacement->children, node->children,
             rank * sizeof(node->children[0]));
      memcpy(replacement->children + rank + 1, node->children + rank,
             (child_count - rank) * sizeof(node->children[0]));
      node = replacement;
      *edge = (uintptr_t)node;
    } else {
      memmove(node->children + rank + 1, node->children + rank,
              (child_count - rank) * sizeof(node->children[0]));
    }
    node->children[rank] = loom_ordinal_map_leaf(key, target_id);
    node->mask |= bit;
    return iree_ok_status();
  }
  return loom_ordinal_map_split(arena, key, target_id,
                                loom_ordinal_map_leaf_key(*edge), edge);
}

static uint32_t loom_ordinal_map_bucket(uint32_t key, uint32_t capacity) {
  return (key * UINT32_C(2654435769)) & (capacity - 1);
}

// Rehashing owns this traversal and visits each live leaf once. Recursion is
// bounded by the seven nibble positions, not by the number of reached facts.
static iree_status_t loom_ordinal_map_rehash_root(iree_arena_allocator_t* arena,
                                                  uint64_t* buckets,
                                                  uint32_t capacity,
                                                  uint64_t root) {
  if ((root & 1) != 0) {
    const uint32_t key = loom_ordinal_map_leaf_key(root);
    return loom_ordinal_map_insert_absent(
        arena, key, (uint32_t)(root >> 32),
        &buckets[loom_ordinal_map_bucket(key, capacity)]);
  }
  const loom_ordinal_map_node_t* node =
      (const loom_ordinal_map_node_t*)(uintptr_t)root;
  const uint32_t count = iree_math_count_ones_u32(node->mask);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = loom_ordinal_map_rehash_root(arena, buckets, capacity,
                                          node->children[i]);
  }
  return status;
}

static iree_status_t loom_ordinal_map_grow(loom_ordinal_map_t* projection) {
  const uint32_t capacity = iree_max(16u, projection->buckets.capacity * 2);
  uint64_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      projection->arena, capacity, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0, capacity * sizeof(*buckets));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < projection->buckets.capacity && iree_status_is_ok(status); ++i) {
    const uint64_t root = projection->buckets.values[i];
    if ((root & 1) != 0) {
      // Growth splits old buckets and never merges them, so a singleton's
      // destination is empty without probing or searching another root.
      const uint32_t key = loom_ordinal_map_leaf_key(root);
      buckets[loom_ordinal_map_bucket(key, capacity)] = root;
    } else if (root != 0) {
      status = loom_ordinal_map_rehash_root(projection->arena, buckets,
                                            capacity, root);
    }
  }
  if (iree_status_is_ok(status)) {
    projection->buckets.values = buckets;
    projection->buckets.capacity = capacity;
  }
  return status;
}

void loom_ordinal_map_initialize(iree_arena_allocator_t* arena,
                                 loom_ordinal_map_t* out_projection) {
  *out_projection = (loom_ordinal_map_t){
      .arena = arena,
      .buckets =
          {
              .values = out_projection->buckets.initial_values,
              .capacity = 2,
          },
  };
}

bool loom_ordinal_map_lookup(const loom_ordinal_map_t* projection,
                             uint8_t domain, uint32_t source_ordinal,
                             uint32_t* out_target_id) {
  const uint32_t key = loom_ordinal_map_key(domain, source_ordinal);
  return loom_ordinal_map_lookup_root(
      projection->buckets
          .values[loom_ordinal_map_bucket(key, projection->buckets.capacity)],
      key, out_target_id);
}

static iree_status_t loom_ordinal_map_insert_collision_or_grow(
    loom_ordinal_map_t* projection, uint32_t key, uint32_t target_id) {
  uint32_t previous_target_id = 0;
  if (loom_ordinal_map_lookup_root(
          projection->buckets.values[loom_ordinal_map_bucket(
              key, projection->buckets.capacity)],
          key, &previous_target_id)) {
    IREE_ASSERT(previous_target_id == target_id);
    return iree_ok_status();
  }
  if (projection->buckets.count == projection->buckets.capacity / 2) {
    IREE_RETURN_IF_ERROR(loom_ordinal_map_grow(projection));
  }
  iree_status_t status = loom_ordinal_map_insert_absent(
      projection->arena, key, target_id,
      &projection->buckets
           .values[loom_ordinal_map_bucket(key, projection->buckets.capacity)]);
  if (iree_status_is_ok(status)) {
    ++projection->buckets.count;
  }
  return status;
}

iree_status_t loom_ordinal_map_insert(loom_ordinal_map_t* projection,
                                      uint8_t domain, uint32_t source_ordinal,
                                      uint32_t target_id) {
  const uint32_t key = loom_ordinal_map_key(domain, source_ordinal);
  uint64_t* bucket =
      &projection->buckets
           .values[loom_ordinal_map_bucket(key, projection->buckets.capacity)];
  if (*bucket == 0 &&
      projection->buckets.count < projection->buckets.capacity / 2) {
    *bucket = loom_ordinal_map_leaf(key, target_id);
    ++projection->buckets.count;
    return iree_ok_status();
  }
  return loom_ordinal_map_insert_collision_or_grow(projection, key, target_id);
}
