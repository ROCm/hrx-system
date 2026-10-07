// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/block_pool.h"

#include <string.h>

iree_status_t loom_serve_block_pool_initialize(
    uint32_t capacity, iree_allocator_t allocator,
    loom_serve_block_pool_t* out_pool) {
  *out_pool = (loom_serve_block_pool_t){.allocator = allocator};
  if (!capacity) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, capacity, 2 * sizeof(*out_pool->free_blocks),
      (void**)&out_pool->free_blocks));
  out_pool->references = out_pool->free_blocks + capacity;
  out_pool->capacity = capacity;
  out_pool->available = capacity;
  for (uint32_t i = 0; i < capacity; ++i) {
    out_pool->free_blocks[i] = capacity - i - 1;
  }
  return iree_ok_status();
}

void loom_serve_block_pool_deinitialize(loom_serve_block_pool_t* pool) {
  iree_allocator_free(pool->allocator, pool->free_blocks);
  memset(pool, 0, sizeof(*pool));
}

void loom_serve_block_pool_acquire(loom_serve_block_pool_t* pool,
                                   uint32_t count, uint32_t* out_blocks) {
  for (uint32_t i = 0; i < count; ++i) {
    out_blocks[i] = pool->free_blocks[--pool->available];
    pool->references[out_blocks[i]] = 1;
  }
}

void loom_serve_block_pool_retain(loom_serve_block_pool_t* pool, uint32_t count,
                                  const uint32_t* blocks) {
  for (uint32_t i = 0; i < count; ++i) {
    ++pool->references[blocks[i]];
  }
}

void loom_serve_block_pool_release(loom_serve_block_pool_t* pool,
                                   uint32_t count, const uint32_t* blocks) {
  for (uint32_t i = 0; i < count; ++i) {
    if (--pool->references[blocks[i]] == 0) {
      pool->free_blocks[pool->available++] = blocks[i];
    }
  }
}

uint32_t loom_serve_block_pool_plan_compaction(
    const loom_serve_block_pool_t* pool, uint32_t* destinations) {
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

void loom_serve_block_pool_commit_compaction(loom_serve_block_pool_t* pool,
                                             const uint32_t* destinations) {
  for (uint32_t i = 0; i < pool->capacity; ++i) {
    if (destinations[i] != UINT32_MAX && destinations[i] != i) {
      pool->references[destinations[i]] = pool->references[i];
      pool->references[i] = 0;
    }
  }
  for (uint32_t i = 0; i < pool->available; ++i) {
    pool->free_blocks[i] = pool->capacity - i - 1;
  }
}
