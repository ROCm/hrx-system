// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/block_pool.h"

#include <string.h>

iree_status_t loom_serve_block_pool_initialize(
    uint32_t capacity, iree_allocator_t allocator,
    loom_serve_block_pool_t* out_pool) {
  *out_pool = (loom_serve_block_pool_t){.allocator = allocator};
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, capacity, sizeof(*out_pool->free_blocks),
      (void**)&out_pool->free_blocks));
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
  }
}

void loom_serve_block_pool_release(loom_serve_block_pool_t* pool,
                                   uint32_t count, const uint32_t* blocks) {
  for (uint32_t i = 0; i < count; ++i) {
    pool->free_blocks[pool->available++] = blocks[i];
  }
}
