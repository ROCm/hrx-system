// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_BLOCK_POOL_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_BLOCK_POOL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single-owner free physical block IDs. Storage geometry, logical mappings,
// admission credit, and execution retirement belong to the caller. Warm
// acquire/release only move IDs; they allocate no host or device memory.
typedef struct loom_serve_block_pool_t {
  // Allocator owning the fixed free-ID array.
  iree_allocator_t allocator;
  // Total number of physical block IDs in this pool.
  uint32_t capacity;
  // Number of unowned IDs at the front of free_blocks.
  uint32_t available;
  // Fixed free-ID stack; acquired entries beyond available are not read.
  uint32_t* free_blocks;
} loom_serve_block_pool_t;

// Creates capacity IDs in [0, capacity). Failure leaves an empty pool that can
// be deinitialized. The caller owns all backing storage addressed by these IDs.
iree_status_t loom_serve_block_pool_initialize(
    uint32_t capacity, iree_allocator_t allocator,
    loom_serve_block_pool_t* out_pool);

// Releases metadata after every consumer of acquired IDs has retired.
void loom_serve_block_pool_deinitialize(loom_serve_block_pool_t* pool);

// Acquires count IDs into caller-owned mapping storage. The caller has already
// established count <= available for the whole transaction before mutation.
void loom_serve_block_pool_acquire(loom_serve_block_pool_t* pool,
                                   uint32_t count, uint32_t* out_blocks);

// Returns distinct owned IDs after their last device use has retired. They may
// immediately be reassigned; mappings referring to them must be inaccessible.
void loom_serve_block_pool_release(loom_serve_block_pool_t* pool,
                                   uint32_t count, const uint32_t* blocks);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_BLOCK_POOL_H_
