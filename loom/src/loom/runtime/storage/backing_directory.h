// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_BACKING_DIRECTORY_H_
#define LOOM_RUNTIME_STORAGE_BACKING_DIRECTORY_H_

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "loom/runtime/storage/backing_pages.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One directory key covers one word of separately replaceable logical pages.
#define LOOM_RUNTIME_BACKING_GROUP_SIZE 64u

typedef struct loom_runtime_backing_group_t {
  // Logical group index within the virtual reservation.
  iree_host_size_t index;
  // Pages selected by the current unsubmitted commitment.
  uint64_t pending_commit;
  // Pages owned by an accepted mapping operation.
  uint64_t in_flight;
  // Pages with published physical ownership, including failed native edits
  // whose sources must survive until checked detach.
  uint64_t published;
  // Temporary union for the serialized footprint, retention, or trim visitor.
  uint64_t selected;
  // Pages with a captured physical lease, including unpublished admissions.
  uint64_t assigned;
  union {
    // Shared parent and source offset at the group origin, modulo the byte
    // address width. Adding an assigned page's displacement recovers its offset
    // even when the group's first page precedes the physical parent.
    loom_runtime_backing_slot_t origin;
    // Exact slots in assigned-bit order when physical offsets diverge.
    loom_runtime_backing_slot_t* fragments;
  } sources;
  // Zero selects the inline origin; otherwise the allocated fragment capacity.
  uint8_t fragment_capacity;
  // Root bucket contains a group.
  bool occupied;
  // Vacant root bucket preserves a probe chain.
  bool tombstone;
} loom_runtime_backing_group_t;

// Sparse table of touched virtual groups. The serialized owner may relocate
// these headers; asynchronous operations capture leases independently.
typedef struct loom_runtime_backing_directory_t {
  // Allocator owning group headers and fragmented source storage.
  iree_allocator_t host_allocator;
  // Number of live groups.
  iree_host_size_t count;
  // Vacant probe-chain headers awaiting reuse or the next rehash.
  iree_host_size_t tombstone_count;
  // Power-of-two root capacity in groups.
  iree_host_size_t capacity;
  // Owned linear-probing table.
  loom_runtime_backing_group_t* groups;
} loom_runtime_backing_directory_t;

// Logical pages from index through the end of its group or the supplied range.
static inline uint64_t loom_runtime_backing_group_mask(iree_host_size_t index,
                                                       iree_host_size_t end) {
  const int position = (int)(index % LOOM_RUNTIME_BACKING_GROUP_SIZE);
  const int count = (int)iree_min(
      end - index,
      (iree_host_size_t)LOOM_RUNTIME_BACKING_GROUP_SIZE - position);
  return iree_math_mask_low_bits_u64(UINT64_MAX, count) << position;
}

// Removes the lowest contiguous run from mask and returns its page count.
static inline iree_host_size_t loom_runtime_backing_take_run(
    uint64_t* mask, int* out_position) {
  const int position = iree_math_count_trailing_zeros_u64(*mask);
  const int length = iree_math_count_trailing_zeros_u64_width(
      ~(*mask >> position), 64 - position);
  *mask &= ~(iree_math_mask_low_bits_u64(UINT64_MAX, length) << position);
  *out_position = position;
  return length;
}

// Installs or clears consecutive leases within a single group. Assignment may
// allocate only when the group needs distinct physical source offsets.
iree_status_t loom_runtime_backing_group_assign(
    loom_runtime_backing_group_t* group, iree_allocator_t allocator,
    uint64_t mask, iree_device_size_t stride,
    loom_runtime_backing_slot_t first);
void loom_runtime_backing_group_clear(loom_runtime_backing_group_t* group,
                                      iree_allocator_t allocator, uint64_t mask,
                                      iree_device_size_t stride);

// Returns the lease of the selected logical page.
loom_runtime_backing_slot_t loom_runtime_backing_group_slot(
    const loom_runtime_backing_group_t* group, int position,
    iree_device_size_t stride);

// Extracts the first contiguous logical and physical interval of a selection.
iree_host_size_t loom_runtime_backing_group_take_lease(
    const loom_runtime_backing_group_t* group, uint64_t* mask,
    iree_device_size_t stride, int* out_position,
    loom_runtime_backing_slot_t* out_slot);

// Range cursor over existing groups. Dense ranges probe one key per word;
// sparse ranges visit only root buckets. Removing the current group is safe.
typedef struct loom_runtime_backing_cursor_t {
  // Borrowed owner, excluding table rehash during traversal.
  loom_runtime_backing_directory_t* directory;
  // First included logical page.
  iree_host_size_t begin;
  // Exclusive end page.
  iree_host_size_t end;
  // Next root bucket or logical group index.
  iree_host_size_t position;
  // Exclusive traversal limit.
  iree_host_size_t limit;
  // Walk root buckets when they are fewer than logical groups in the range.
  bool sparse;
} loom_runtime_backing_cursor_t;

loom_runtime_backing_cursor_t loom_runtime_backing_directory_iterate(
    loom_runtime_backing_directory_t* directory, iree_host_size_t begin,
    iree_host_size_t end);
loom_runtime_backing_group_t* loom_runtime_backing_directory_next(
    loom_runtime_backing_cursor_t* cursor, uint64_t* out_mask);

// Removes the group after its final ownership or temporary selection is gone.
void loom_runtime_backing_directory_prune(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_group_t* group);

// Initializes an empty directory without allocating storage.
void loom_runtime_backing_directory_initialize(
    iree_allocator_t host_allocator,
    loom_runtime_backing_directory_t* out_directory);

// Frees directory storage. Physical leases remain with the caller.
void loom_runtime_backing_directory_deinitialize(
    loom_runtime_backing_directory_t* directory);

// Finds an existing group, returning NULL when its key has never been touched.
loom_runtime_backing_group_t* loom_runtime_backing_directory_find(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index);
const loom_runtime_backing_group_t* loom_runtime_backing_directory_find_const(
    const loom_runtime_backing_directory_t* directory, iree_host_size_t index);

// Finds or inserts a group for the logical group index. Table growth is
// proportional to the live entry count and may fail without changing existing
// entries.
iree_status_t loom_runtime_backing_directory_insert(
    loom_runtime_backing_directory_t* directory, iree_host_size_t index,
    loom_runtime_backing_group_t** out_entry);

// Removes an empty group while preserving its probe chain.
void loom_runtime_backing_directory_remove(
    loom_runtime_backing_directory_t* directory,
    loom_runtime_backing_group_t* entry);

// Drops empty groups and releases an empty directory.
void loom_runtime_backing_directory_discard_empty(
    loom_runtime_backing_directory_t* directory);

// Shrinks the table to the minimum capacity required by its live entries.
// Empty directories release all storage without allocating.
iree_status_t loom_runtime_backing_directory_shrink(
    loom_runtime_backing_directory_t* directory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_RUNTIME_STORAGE_BACKING_DIRECTORY_H_
