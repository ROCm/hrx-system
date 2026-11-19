// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_BLOCK_POOL_H_
#define LOOM_RUNTIME_STORAGE_BLOCK_POOL_H_

#include "iree/base/api.h"
#include "loom/util/index_set.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serialized-owner physical block IDs with shared lifetime references. Storage
// geometry, logical mappings, admission credit and execution retirement belong
// to the caller. Warm ownership changes allocate no host or device memory.
typedef struct loom_runtime_block_pool_t {
  // Allocator owning the growable references and membership arrays.
  iree_allocator_t host_allocator;
  // Total number of physical block IDs in this pool.
  uint32_t capacity;
  // Number of unowned IDs available for acquisition.
  uint32_t available;
  // References per ID; zero denotes an unowned ID. Owns the coallocation.
  uint32_t* references;
  // Shared layout for owned, free, preferred, preferred-free, and released IDs.
  loom_index_set_layout_t membership;
  // Indexed ownership and placement facts, coallocated after references.
  uint64_t* sets;
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

// First unexamined released ID at or after first, or UINT32_MAX. Reacquisition
// removes a release; shared releases produce none. The owner establishes safe
// physical coverage before acknowledging, independently of logical IDs.
uint32_t loom_runtime_block_pool_find_released(
    const loom_runtime_block_pool_t* pool, uint32_t first);

// Records successful capture of the current releases. Deferred or failed
// capture leaves them pending; the owner excludes mutations while capturing.
// Empty release sets take constant time regardless of logical capacity.
void loom_runtime_block_pool_acknowledge_releases(
    loom_runtime_block_pool_t* pool);

// Whether any ID in the in-domain interval [first, first + count) has owners.
// Queries the retained membership summary without scanning reference counts.
bool loom_runtime_block_pool_any_owned(const loom_runtime_block_pool_t* pool,
                                       uint32_t first, uint32_t count);

// First owned ID in the interval, or UINT32_MAX when it is empty. Geometry
// consumers can retain the exact occupied boundary instead of searching again.
uint32_t loom_runtime_block_pool_find_owned(
    const loom_runtime_block_pool_t* pool, uint32_t first, uint32_t count);

// Retains producer-established placement preference independently of ownership.
// Preferred free IDs are acquired first; other free-ID order is unspecified.
// Updating one ID never scans the pool or allocates.
void loom_runtime_block_pool_set_preferred(loom_runtime_block_pool_t* pool,
                                           uint32_t block, bool preferred);
bool loom_runtime_block_pool_is_preferred(const loom_runtime_block_pool_t* pool,
                                          uint32_t block);

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
// All remaining free IDs become unexamined for backing reclamation.
void loom_runtime_block_pool_commit_compaction(loom_runtime_block_pool_t* pool,
                                               const uint32_t* destinations);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_STORAGE_BLOCK_POOL_H_
