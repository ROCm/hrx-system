// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/cfg_condition_projection_cache.h"

#include <string.h>

struct loom_cfg_condition_projection_cache_slot_t {
  // Stable edge ordinal, or UINT32_MAX for an empty slot.
  uint32_t edge;

  // Source construction-set ID.
  loom_condition_relation_set_id_t source;

  // Projected construction-set ID.
  loom_condition_relation_set_id_t target;
};

static uint32_t loom_cfg_condition_projection_hash(
    loom_cfg_edge_index_t edge, loom_condition_relation_set_id_t source) {
  uint64_t value = ((uint64_t)edge << 32) | source;
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  value ^= value >> 31;
  return (uint32_t)value;
}

bool loom_cfg_condition_projection_cache_lookup(
    const loom_cfg_condition_projection_cache_t* cache,
    loom_cfg_edge_index_t edge, loom_condition_relation_set_id_t source,
    loom_condition_relation_set_id_t* out_target) {
  if (cache->capacity == 0) {
    return false;
  }
  uint32_t slot =
      loom_cfg_condition_projection_hash(edge, source) & (cache->capacity - 1);
  while (cache->slots[slot].edge != UINT32_MAX) {
    if (cache->slots[slot].edge == edge &&
        cache->slots[slot].source == source) {
      *out_target = cache->slots[slot].target;
      return true;
    }
    slot = (slot + 1) & (cache->capacity - 1);
  }
  return false;
}

static iree_status_t loom_cfg_condition_projection_cache_reserve(
    loom_cfg_condition_projection_cache_t* cache) {
  uint32_t capacity = cache->capacity == 0 ? 64 : cache->capacity;
  while ((uint64_t)(cache->count + 1) * 4 > (uint64_t)capacity * 3) {
    if (capacity > UINT32_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "condition projection cache exceeds uint32_t");
    }
    capacity *= 2;
  }
  if (capacity == cache->capacity) {
    return iree_ok_status();
  }

  loom_cfg_condition_projection_cache_slot_t* slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      cache->arena, capacity, sizeof(*slots), (void**)&slots));
  memset(slots, 0xFF, capacity * sizeof(*slots));
  for (uint32_t i = 0; i < cache->capacity; ++i) {
    const loom_cfg_condition_projection_cache_slot_t entry = cache->slots[i];
    if (entry.edge == UINT32_MAX) {
      continue;
    }
    uint32_t slot =
        loom_cfg_condition_projection_hash(entry.edge, entry.source) &
        (capacity - 1);
    while (slots[slot].edge != UINT32_MAX) {
      slot = (slot + 1) & (capacity - 1);
    }
    slots[slot] = entry;
  }
  cache->slots = slots;
  cache->capacity = capacity;
  return iree_ok_status();
}

iree_status_t loom_cfg_condition_projection_cache_insert(
    loom_cfg_condition_projection_cache_t* cache, loom_cfg_edge_index_t edge,
    loom_condition_relation_set_id_t source,
    loom_condition_relation_set_id_t target) {
  IREE_RETURN_IF_ERROR(loom_cfg_condition_projection_cache_reserve(cache));
  uint32_t slot =
      loom_cfg_condition_projection_hash(edge, source) & (cache->capacity - 1);
  while (cache->slots[slot].edge != UINT32_MAX) {
    slot = (slot + 1) & (cache->capacity - 1);
  }
  cache->slots[slot] = (loom_cfg_condition_projection_cache_slot_t){
      .edge = edge,
      .source = source,
      .target = target,
  };
  ++cache->count;
  return iree_ok_status();
}
