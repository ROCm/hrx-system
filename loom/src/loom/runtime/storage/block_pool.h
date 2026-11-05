// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_BLOCK_POOL_H_
#define LOOM_RUNTIME_STORAGE_BLOCK_POOL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serialized-owner physical block IDs with shared lifetime references. Storage
// geometry, logical mappings, admission credit and execution retirement belong
// to the caller. Warm ownership changes allocate no host or device memory.
typedef struct loom_runtime_block_pool_t {
  // Allocator owning the growable free-ID and reference arrays.
  iree_allocator_t host_allocator;
  // Total number of physical block IDs in this pool.
  uint32_t capacity;
  // Number of unowned IDs at the front of free_blocks.
  uint32_t available;
  // Free-ID stack; acquired entries beyond available are not read.
  uint32_t* free_blocks;
  // References per ID; zero denotes an ID on the free stack. Borrows backing.
  uint32_t* references;
} loom_runtime_block_pool_t;

// Creates capacity IDs in [0, capacity), allocating nothing for zero capacity.
// Failure leaves an empty pool that can be deinitialized. The caller owns all
// backing storage addressed by these IDs.
iree_status_t loom_runtime_block_pool_initialize(
    uint32_t capacity, iree_allocator_t host_allocator,
    loom_runtime_block_pool_t* out_pool);

// Releases metadata after every consumer of acquired IDs has retired.
void loom_runtime_block_pool_deinitialize(loom_runtime_block_pool_t* pool);

// Appends IDs without changing live identities or references. Metadata growth
// occurs only on the serialized owner; allocation failure leaves it unchanged.
// The caller establishes capacity >= the current capacity and owns backing.
iree_status_t loom_runtime_block_pool_grow(loom_runtime_block_pool_t* pool,
                                           uint32_t capacity);

// Acquires count IDs into caller-owned mapping storage. The caller has already
// established count <= available for the whole transaction before mutation.
void loom_runtime_block_pool_acquire(loom_runtime_block_pool_t* pool,
                                     uint32_t count, uint32_t* out_blocks);

// Adds one owner to each already-owned ID. Ownership changes are serialized;
// the caller bounds the number of holders below UINT32_MAX.
void loom_runtime_block_pool_retain(loom_runtime_block_pool_t* pool,
                                    uint32_t count, const uint32_t* blocks);

// Drops one owner after its device use retires. Last release returns the ID
// for immediate reassignment; the released mapping must be inaccessible.
void loom_runtime_block_pool_release(loom_runtime_block_pool_t* pool,
                                     uint32_t count, const uint32_t* blocks);

// Cold placement maintenance: moves preferred free IDs to the acquisition end
// of the stack without changing any owned ID or reference. Visits each free ID
// once and allocates nothing. Relative free-ID order is unspecified afterward.
void loom_runtime_block_pool_prioritize(loom_runtime_block_pool_t* pool,
                                        bool (*preferred)(void* user_data,
                                                          uint32_t block),
                                        void* user_data);

static inline bool loom_runtime_block_pool_is_shared(
    const loom_runtime_block_pool_t* pool, uint32_t block) {
  return pool->references[block] > 1;
}

// Builds a cold maintenance map of capacity entries: UINT32_MAX for originally
// free IDs, otherwise the ID's destination in a compact live prefix. Every
// moved destination is originally free and disjoint from all sources, so the
// copies may execute concurrently. Returns the number of moved blocks without
// changing ownership. The caller excludes acquisitions/releases until commit.
uint32_t loom_runtime_block_pool_plan_compaction(
    const loom_runtime_block_pool_t* pool, uint32_t* destinations);

// Publishes the compact live prefix after copies retire and every logical map
// has been rewritten using the plan. No old-address consumers may remain.
void loom_runtime_block_pool_commit_compaction(loom_runtime_block_pool_t* pool,
                                               const uint32_t* destinations);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_STORAGE_BLOCK_POOL_H_
