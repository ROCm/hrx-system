// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/block_pool.h"

#include <string.h>

static uint64_t* loom_runtime_block_pool_membership(
    const loom_runtime_block_pool_t* pool) {
  return pool->references ? (uint64_t*)(pool->references + pool->capacity)
                          : NULL;
}

iree_status_t loom_runtime_block_pool_initialize(
    uint32_t capacity, iree_allocator_t host_allocator,
    loom_runtime_block_pool_t* out_pool) {
  *out_pool = (loom_runtime_block_pool_t){.host_allocator = host_allocator};
  return loom_runtime_block_pool_grow(out_pool, capacity);
}

void loom_runtime_block_pool_deinitialize(loom_runtime_block_pool_t* pool) {
  iree_allocator_free(pool->host_allocator, pool->free_blocks);
  memset(pool, 0, sizeof(*pool));
}

iree_status_t loom_runtime_block_pool_grow(loom_runtime_block_pool_t* pool,
                                           uint32_t capacity) {
  if (capacity == pool->capacity) {
    return iree_ok_status();
  }
  const loom_index_set_layout_t membership =
      loom_index_set_calculate_layout(capacity);
  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_mul(capacity, 2 * sizeof(uint32_t),
                                  &storage_size) ||
      !iree_host_size_checked_mul_add(storage_size, membership.word_count,
                                      sizeof(uint64_t), &storage_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "block ownership metadata exceeds host size");
  }
  uint32_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator, storage_size,
                                             (void**)&storage));
  const uint32_t added = capacity - pool->capacity;
  for (uint32_t i = 0; i < added; ++i) {
    storage[i] = capacity - i - 1;
  }
  if (pool->capacity) {
    memcpy(storage + added, pool->free_blocks,
           pool->available * sizeof(*storage));
    memcpy(storage + capacity, pool->references,
           pool->capacity * sizeof(*storage));
    // Copy established summaries into their new offsets. Newly introduced
    // levels summarize the old root; no reference-count rescan is needed.
    const uint64_t* old_words = loom_runtime_block_pool_membership(pool);
    uint64_t* words = (uint64_t*)(storage + capacity + capacity);
    for (uint8_t level = 0; level < pool->membership.level_count; ++level) {
      const uint32_t begin = pool->membership.level_starts[level];
      const uint32_t end = level + 1 < pool->membership.level_count
                               ? pool->membership.level_starts[level + 1]
                               : pool->membership.word_count;
      memcpy(words + membership.level_starts[level], old_words + begin,
             (end - begin) * sizeof(*words));
    }
    for (uint8_t level = pool->membership.level_count;
         level < membership.level_count; ++level) {
      words[membership.level_starts[level]] = pool->available != pool->capacity;
    }
  }
  iree_allocator_free(pool->host_allocator, pool->free_blocks);
  pool->free_blocks = storage;
  pool->references = storage + capacity;
  pool->capacity = capacity;
  pool->available += added;
  pool->examined += added;
  pool->membership = membership;
  return iree_ok_status();
}

void loom_runtime_block_pool_acquire(loom_runtime_block_pool_t* pool,
                                     uint32_t count, uint32_t* out_blocks) {
  for (uint32_t i = 0; i < count; ++i) {
    out_blocks[i] = pool->free_blocks[--pool->available];
    pool->references[out_blocks[i]] = 1;
    loom_index_set_insert(&pool->membership,
                          loom_runtime_block_pool_membership(pool),
                          out_blocks[i]);
  }
  pool->examined = iree_min(pool->examined, pool->available);
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
    if (--pool->references[blocks[i]] == 0) {
      loom_index_set_erase(&pool->membership,
                           loom_runtime_block_pool_membership(pool), blocks[i]);
      pool->free_blocks[pool->available++] = blocks[i];
    }
  }
}

uint32_t loom_runtime_block_pool_released(const loom_runtime_block_pool_t* pool,
                                          const uint32_t** out_blocks) {
  const uint32_t count = pool->available - pool->examined;
  *out_blocks = count ? pool->free_blocks + pool->examined : NULL;
  return count;
}

void loom_runtime_block_pool_acknowledge_releases(
    loom_runtime_block_pool_t* pool) {
  pool->examined = pool->available;
}

bool loom_runtime_block_pool_any_owned(const loom_runtime_block_pool_t* pool,
                                       uint32_t first, uint32_t count) {
  if (!count) {
    return false;
  }
  const uint32_t index = loom_index_set_select(
      &pool->membership, loom_runtime_block_pool_membership(pool), first);
  return index >= first && index - first < count;
}

void loom_runtime_block_pool_prioritize(loom_runtime_block_pool_t* pool,
                                        bool (*preferred)(void* user_data,
                                                          uint32_t block),
                                        void* user_data) {
  if (pool->examined != pool->available) {
    pool->examined = 0;
  }
  uint32_t begin = 0, end = pool->available;
  while (begin < end) {
    const uint32_t block = pool->free_blocks[begin];
    if (preferred(user_data, block)) {
      pool->free_blocks[begin] = pool->free_blocks[--end];
      pool->free_blocks[end] = block;
    } else {
      ++begin;
    }
  }
}

uint32_t loom_runtime_block_pool_plan_compaction(
    const loom_runtime_block_pool_t* pool, uint32_t* destinations) {
  for (uint32_t i = 0; i < pool->capacity; ++i) {
    destinations[i] = i;
  }
  for (uint32_t i = 0; i < pool->available; ++i) {
    destinations[pool->free_blocks[i]] = UINT32_MAX;
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
  for (uint32_t i = 0; i < pool->capacity; ++i) {
    if (destinations[i] != UINT32_MAX && destinations[i] != i) {
      pool->references[destinations[i]] = pool->references[i];
      pool->references[i] = 0;
      loom_index_set_erase(&pool->membership,
                           loom_runtime_block_pool_membership(pool), i);
      loom_index_set_insert(&pool->membership,
                            loom_runtime_block_pool_membership(pool),
                            destinations[i]);
    }
  }
  for (uint32_t i = 0; i < pool->available; ++i) {
    pool->free_blocks[i] = pool->capacity - i - 1;
  }
  pool->examined = 0;
}
