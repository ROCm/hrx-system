// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_MEMORY_POOL_H_
#define LIBHRX_SRC_BINDING_HIP_MEMORY_POOL_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One tracked hipMalloc sub-allocation.
typedef struct iree_hip_pool_allocation_slot_t {
  // Base device address returned to the caller.
  uintptr_t address;
  // Aligned byte length of this sub-allocation.
  size_t size;
  // Byte length requested by the HIP caller.
  size_t requested_size;
  // Whether this slot currently represents a live allocation.
  bool live;
} iree_hip_pool_allocation_slot_t;

// Fixed-capacity allocation tracker used by the HIP device pool.
typedef struct iree_hip_pool_allocation_tracker_t {
  // Caller-owned slot storage.
  iree_hip_pool_allocation_slot_t* slots;
  // Total number of slots available.
  size_t capacity;
  // Number of currently live sub-allocations.
  size_t live_count;
  // Highest slot index ever initialized, plus one.
  size_t high_water_mark;
  // Search position where the next reusable slot scan starts.
  size_t free_hint;
} iree_hip_pool_allocation_tracker_t;

// Initializes |tracker| over zeroed or uninitialized caller-owned |slots|.
void iree_hip_pool_allocation_tracker_initialize(
    iree_hip_pool_allocation_tracker_t* tracker,
    iree_hip_pool_allocation_slot_t* slots, size_t capacity);

// Resets all tracked allocation state without freeing |slots|.
void iree_hip_pool_allocation_tracker_reset(
    iree_hip_pool_allocation_tracker_t* tracker);

// Tracks a new live allocation.
//
// Returns false when |tracker| has no reusable slot. |address| must be non-zero
// and |size| must be non-zero.
bool iree_hip_pool_allocation_tracker_insert(
    iree_hip_pool_allocation_tracker_t* tracker, uintptr_t address,
    size_t size);

// Tracks a new live allocation with separate physical and requested sizes.
//
// |size| is the aligned allocator extent used for overlap and free-memory
// accounting. |requested_size| is the HIP-visible extent returned by
// iree_hip_pool_allocation_tracker_find.
bool iree_hip_pool_allocation_tracker_insert_sized(
    iree_hip_pool_allocation_tracker_t* tracker, uintptr_t address,
    size_t size, size_t requested_size);

// Marks the allocation starting at |address| dead.
//
// Returns false when no live allocation starts at |address|. If |out_size| is
// non-NULL, receives the released allocation size on success.
bool iree_hip_pool_allocation_tracker_release(
    iree_hip_pool_allocation_tracker_t* tracker, uintptr_t address,
    size_t* out_size);

// Finds the live allocation containing |address|.
//
// Returns false when |address| is outside all live allocations. If non-NULL,
// |out_base| and |out_size| receive the containing allocation's base and
// HIP-visible requested size.
bool iree_hip_pool_allocation_tracker_find(
    const iree_hip_pool_allocation_tracker_t* tracker, uintptr_t address,
    uintptr_t* out_base, size_t* out_size);

// Finds the live allocation containing |address| and returns the physical
// aligned sub-allocation extent used by the slab allocator.
bool iree_hip_pool_allocation_tracker_find_physical(
    const iree_hip_pool_allocation_tracker_t* tracker, uintptr_t address,
    uintptr_t* out_base, size_t* out_size);

// Returns the sum of all live allocation sizes.
size_t iree_hip_pool_allocation_tracker_live_size(
    const iree_hip_pool_allocation_tracker_t* tracker);

// Aligns |value| up to |alignment| with overflow checking.
//
// |alignment| must be non-zero and a power of two.
bool iree_hip_pool_align_size(size_t value, size_t alignment,
                              size_t* out_aligned_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LIBHRX_SRC_BINDING_HIP_MEMORY_POOL_H_
