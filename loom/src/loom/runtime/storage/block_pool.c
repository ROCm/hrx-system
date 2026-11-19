// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/block_pool.h"

#include <string.h>

enum {
  LOOM_BLOCK_OWNED,
  LOOM_BLOCK_FREE,
  LOOM_BLOCK_PREFERRED,
  LOOM_BLOCK_PREFERRED_FREE,
  LOOM_BLOCK_RELEASED,
  LOOM_BLOCK_SET_COUNT,
};

static uint64_t* loom_runtime_block_pool_set(
    const loom_runtime_block_pool_t* pool, uint32_t set) {
  return pool->sets ? pool->sets + set * pool->membership.word_count : NULL;
}

static void loom_runtime_block_pool_insert(loom_runtime_block_pool_t* pool,
                                           uint32_t set, uint32_t block) {
  loom_index_set_insert(&pool->membership,
                        loom_runtime_block_pool_set(pool, set), block);
}

static void loom_runtime_block_pool_erase(loom_runtime_block_pool_t* pool,
                                          uint32_t set, uint32_t block) {
  loom_index_set_erase(&pool->membership,
                       loom_runtime_block_pool_set(pool, set), block);
}

iree_status_t loom_runtime_block_pool_initialize(
    uint32_t capacity, iree_allocator_t host_allocator,
    loom_runtime_block_pool_t* out_pool) {
  *out_pool = (loom_runtime_block_pool_t){.host_allocator = host_allocator};
  return loom_runtime_block_pool_grow(out_pool, capacity);
}

void loom_runtime_block_pool_deinitialize(loom_runtime_block_pool_t* pool) {
  iree_allocator_free(pool->host_allocator, pool->references);
  memset(pool, 0, sizeof(*pool));
}

iree_status_t loom_runtime_block_pool_grow(loom_runtime_block_pool_t* pool,
                                           uint32_t capacity) {
  if (capacity == pool->capacity) {
    return iree_ok_status();
  }
  const loom_index_set_layout_t membership =
      loom_index_set_calculate_layout(capacity);
  iree_host_size_t storage_size = 0, references_offset = 0, sets_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD(capacity, uint32_t, &references_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          (iree_host_size_t)LOOM_BLOCK_SET_COUNT * membership.word_count,
          uint64_t, iree_alignof(uint64_t), &sets_offset)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator, storage_size,
                                             (void**)&storage));
  uint32_t* references = (uint32_t*)(storage + references_offset);
  uint64_t* sets = (uint64_t*)(storage + sets_offset);
  if (pool->capacity) {
    memcpy(references, pool->references, pool->capacity * sizeof(*references));
    for (uint32_t set = 0; set < LOOM_BLOCK_SET_COUNT; ++set) {
      const uint64_t* old_words = loom_runtime_block_pool_set(pool, set);
      uint64_t* words = sets + set * membership.word_count;
      for (uint8_t level = 0; level < pool->membership.level_count; ++level) {
        const uint32_t begin = pool->membership.level_starts[level];
        const uint32_t end = level + 1 < pool->membership.level_count
                                 ? pool->membership.level_starts[level + 1]
                                 : pool->membership.word_count;
        memcpy(words + membership.level_starts[level], old_words + begin,
               (end - begin) * sizeof(*words));
      }
      const bool nonempty = old_words[pool->membership.word_count - 1] != 0;
      for (uint8_t level = pool->membership.level_count;
           level < membership.level_count; ++level) {
        words[membership.level_starts[level]] = nonempty;
      }
    }
  }
  loom_index_set_insert_range(&membership,
                              sets + LOOM_BLOCK_FREE * membership.word_count,
                              pool->capacity, capacity - pool->capacity);
  iree_allocator_free(pool->host_allocator, pool->references);
  pool->references = references;
  pool->sets = sets;
  pool->available += capacity - pool->capacity;
  pool->capacity = capacity;
  pool->membership = membership;
  return iree_ok_status();
}

void loom_runtime_block_pool_acquire(loom_runtime_block_pool_t* pool,
                                     uint32_t count, uint32_t* out_blocks) {
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t block = loom_index_set_select(
        &pool->membership,
        loom_runtime_block_pool_set(pool, LOOM_BLOCK_PREFERRED_FREE), 0);
    if (block == UINT32_MAX) {
      block = loom_index_set_select(
          &pool->membership, loom_runtime_block_pool_set(pool, LOOM_BLOCK_FREE),
          0);
    }
    out_blocks[i] = block;
    pool->references[block] = 1;
    loom_runtime_block_pool_erase(pool, LOOM_BLOCK_FREE, block);
    loom_runtime_block_pool_erase(pool, LOOM_BLOCK_PREFERRED_FREE, block);
    loom_runtime_block_pool_erase(pool, LOOM_BLOCK_RELEASED, block);
    loom_runtime_block_pool_insert(pool, LOOM_BLOCK_OWNED, block);
  }
  pool->available -= count;
}

void loom_runtime_block_pool_retain(loom_runtime_block_pool_t* pool,
                                    uint32_t count, const uint32_t* blocks) {
  for (uint32_t i = 0; i < count; ++i) {
    ++pool->references[blocks[i]];
  }
}

void loom_runtime_block_pool_release(loom_runtime_block_pool_t* pool,
                                     uint32_t count, const uint32_t* blocks) {
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t block = blocks[i];
    if (--pool->references[block] == 0) {
      loom_runtime_block_pool_erase(pool, LOOM_BLOCK_OWNED, block);
      loom_runtime_block_pool_insert(pool, LOOM_BLOCK_FREE, block);
      loom_runtime_block_pool_insert(pool, LOOM_BLOCK_RELEASED, block);
      if (loom_runtime_block_pool_is_preferred(pool, block)) {
        loom_runtime_block_pool_insert(pool, LOOM_BLOCK_PREFERRED_FREE, block);
      }
      ++pool->available;
    }
  }
}

uint32_t loom_runtime_block_pool_find_released(
    const loom_runtime_block_pool_t* pool, uint32_t first) {
  if (first >= pool->capacity) {
    return UINT32_MAX;
  }
  const uint32_t block = loom_index_set_select(
      &pool->membership, loom_runtime_block_pool_set(pool, LOOM_BLOCK_RELEASED),
      first);
  return block >= first ? block : UINT32_MAX;
}

void loom_runtime_block_pool_acknowledge_releases(
    loom_runtime_block_pool_t* pool) {
  loom_index_set_clear(&pool->membership,
                       loom_runtime_block_pool_set(pool, LOOM_BLOCK_RELEASED));
}

bool loom_runtime_block_pool_is_preferred(const loom_runtime_block_pool_t* pool,
                                          uint32_t block) {
  return (loom_runtime_block_pool_set(pool, LOOM_BLOCK_PREFERRED)[block / 64] &
          (UINT64_C(1) << (block % 64))) != 0;
}

void loom_runtime_block_pool_set_preferred(loom_runtime_block_pool_t* pool,
                                           uint32_t block, bool preferred) {
  if (preferred) {
    loom_runtime_block_pool_insert(pool, LOOM_BLOCK_PREFERRED, block);
    if (!pool->references[block]) {
      loom_runtime_block_pool_insert(pool, LOOM_BLOCK_PREFERRED_FREE, block);
    }
  } else {
    loom_runtime_block_pool_erase(pool, LOOM_BLOCK_PREFERRED, block);
    loom_runtime_block_pool_erase(pool, LOOM_BLOCK_PREFERRED_FREE, block);
  }
}

bool loom_runtime_block_pool_any_owned(const loom_runtime_block_pool_t* pool,
                                       uint32_t first, uint32_t count) {
  return loom_runtime_block_pool_find_owned(pool, first, count) != UINT32_MAX;
}

uint32_t loom_runtime_block_pool_find_owned(
    const loom_runtime_block_pool_t* pool, uint32_t first, uint32_t count) {
  if (!count) {
    return UINT32_MAX;
  }
  const uint32_t index = loom_index_set_select(
      &pool->membership, loom_runtime_block_pool_set(pool, LOOM_BLOCK_OWNED),
      first);
  return index >= first && index - first < count ? index : UINT32_MAX;
}

uint32_t loom_runtime_block_pool_plan_compaction(
    const loom_runtime_block_pool_t* pool, uint32_t* destinations) {
  for (uint32_t i = 0; i < pool->capacity; ++i) {
    destinations[i] = pool->references[i] ? i : UINT32_MAX;
  }
  const uint32_t live = pool->capacity - pool->available;
  uint32_t high = pool->capacity;
  uint32_t moved = 0;
  for (uint32_t low = 0; low < live; ++low) {
    if (destinations[low] != UINT32_MAX) {
      continue;
    }
    do {
      --high;
    } while (destinations[high] == UINT32_MAX);
    destinations[high] = low;
    ++moved;
  }
  return moved;
}

void loom_runtime_block_pool_commit_compaction(loom_runtime_block_pool_t* pool,
                                               const uint32_t* destinations) {
  if (!pool->capacity) {
    return;
  }
  for (uint32_t i = 0; i < pool->capacity; ++i) {
    if (destinations[i] != UINT32_MAX && destinations[i] != i) {
      pool->references[destinations[i]] = pool->references[i];
      pool->references[i] = 0;
      loom_index_set_erase(&pool->membership,
                           loom_runtime_block_pool_set(pool, LOOM_BLOCK_OWNED),
                           i);
      loom_index_set_insert(&pool->membership,
                            loom_runtime_block_pool_set(pool, LOOM_BLOCK_OWNED),
                            destinations[i]);
    }
  }
  const uint32_t live = pool->capacity - pool->available;
  uint64_t* free = loom_runtime_block_pool_set(pool, LOOM_BLOCK_FREE);
  uint64_t* preferred_free =
      loom_runtime_block_pool_set(pool, LOOM_BLOCK_PREFERRED_FREE);
  memset(free, 0, pool->membership.word_count * sizeof(*free));
  memset(preferred_free, 0,
         pool->membership.word_count * sizeof(*preferred_free));
  memset(loom_runtime_block_pool_set(pool, LOOM_BLOCK_RELEASED), 0,
         pool->membership.word_count * sizeof(uint64_t));
  loom_index_set_insert_range(&pool->membership, free, live, pool->available);
  loom_index_set_insert_range(
      &pool->membership, loom_runtime_block_pool_set(pool, LOOM_BLOCK_RELEASED),
      live, pool->available);
  for (uint32_t block = live; block < pool->capacity; ++block) {
    if (loom_runtime_block_pool_is_preferred(pool, block)) {
      loom_runtime_block_pool_insert(pool, LOOM_BLOCK_PREFERRED_FREE, block);
    }
  }
}
