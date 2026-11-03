// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_STORAGE_BACKING_DIRECTORY_H_
#define LOOM_SERVE_RUNTIME_STORAGE_BACKING_DIRECTORY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_buffer_t iree_hal_buffer_t;

// Mutable state for one touched slot in a logical virtual reservation.
typedef struct loom_serve_backing_entry_t {
  // Logical slot index within the virtual reservation.
  iree_host_size_t index;
  // Owned physical handle, or null for uncommitted storage.
  iree_hal_buffer_t* physical;
  // Whether physical is currently mapped into its virtual slot.
  bool mapped;
  // Whether a current retained range requires this slab.
  bool keep;
  // Missing slab selected by the current serialized commitment transaction.
  bool pending_commit;
  // Temporary union mark scoped to one serialized footprint query.
  bool counted;
  // Accepted asynchronous operation excludes another editor of this slot.
  bool in_flight;
  // Entry contains a live logical slot.
  bool occupied;
  // Vacant entry preserves a linear-probing chain until the next rehash.
  bool tombstone;
} loom_serve_backing_entry_t;

// Sparse slot directory whose storage scales with touched slots, not the
// reservation's logical extent. Mutations are serialized by its outer owner.
typedef struct loom_serve_backing_directory_t {
  // Allocator owning entries through deinitialization.
  iree_allocator_t host_allocator;
  // Number of live entries.
  iree_host_size_t count;
  // Vacant probe-chain entries awaiting reuse or the next rehash.
  iree_host_size_t tombstone_count;
  // Power-of-two entry capacity.
  iree_host_size_t capacity;
  // Owned linear-probing table.
  loom_serve_backing_entry_t* entries;
} loom_serve_backing_directory_t;

// Initializes an empty directory without allocating storage.
void loom_serve_backing_directory_initialize(
    iree_allocator_t host_allocator,
    loom_serve_backing_directory_t* out_directory);

// Frees directory storage. Entry payload ownership remains with the caller.
void loom_serve_backing_directory_deinitialize(
    loom_serve_backing_directory_t* directory);

// Finds an existing entry, returning NULL when the slot has never been touched.
loom_serve_backing_entry_t* loom_serve_backing_directory_find(
    loom_serve_backing_directory_t* directory, iree_host_size_t index);
const loom_serve_backing_entry_t* loom_serve_backing_directory_find_const(
    const loom_serve_backing_directory_t* directory, iree_host_size_t index);

// Finds or inserts an empty entry for index. Table growth is proportional to
// the live entry count and may fail without changing existing entries.
iree_status_t loom_serve_backing_directory_insert(
    loom_serve_backing_directory_t* directory, iree_host_size_t index,
    loom_serve_backing_entry_t** out_entry);

// Removes an entry while preserving its probe chain.
void loom_serve_backing_directory_remove(
    loom_serve_backing_directory_t* directory,
    loom_serve_backing_entry_t* entry);

// Removes marker-only entries with neither physical nor in-flight ownership.
void loom_serve_backing_directory_discard_empty(
    loom_serve_backing_directory_t* directory);

// Shrinks the table to the minimum capacity required by its live entries.
// Empty directories release all storage without allocating.
iree_status_t loom_serve_backing_directory_shrink(
    loom_serve_backing_directory_t* directory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_SERVE_RUNTIME_STORAGE_BACKING_DIRECTORY_H_
