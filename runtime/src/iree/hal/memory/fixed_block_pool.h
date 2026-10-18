// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_FIXED_BLOCK_POOL_H_
#define IREE_HAL_MEMORY_FIXED_BLOCK_POOL_H_

#include "iree/base/api.h"
#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_fixed_block_pool_t
//===----------------------------------------------------------------------===//

// Options for creating a HAL pool that wraps
// iree_hal_memory_fixed_block_allocator_t.
typedef struct iree_hal_fixed_block_pool_options_t {
  // User-visible capacity of each block, excluding hidden guards.
  iree_device_size_t block_size;

  // Blocks requested per growth operation. Zero selects up to 64 blocks,
  // bounded by source capacity. The finite constructor requires zero and
  // derives its complete block count from the supplied range.
  uint32_t blocks_per_slab;

  // Inline history width per block and whole-range return. Zero selects the
  // fixed-block allocator default. Wider histories collapse to a pool timeline
  // point covering every original prerequisite before backing is returned.
  uint32_t frontier_capacity;

  // Required absolute block alignment; zero selects source requirements.
  iree_device_size_t alignment;

  // ASAN policy used to shape hidden backing ranges for reservations.
  iree_hal_asan_pool_options_t asan;

  // Logical byte budget for live reservations in this pool. 0 means unlimited.
  // Checked for the entire transaction before claiming any blocks. Can return
  // IREE_HAL_POOL_ACQUIRE_OVER_BUDGET without touching the allocator.
  iree_device_size_t budget_limit;

  // Optional named-memory trace identifier for logical reservations returned by
  // this pool. Empty uses a generic process-stable identifier.
  iree_string_view_t trace_name;
} iree_hal_fixed_block_pool_options_t;

// Initializes options to their defaults. The caller supplies block_size.
IREE_API_EXPORT void iree_hal_fixed_block_pool_options_initialize(
    iree_hal_fixed_block_pool_options_t* options);

// Resolves the ordinary backing request, including alignment and hidden guards,
// without acquiring memory. Useful for configuring a shared retention cache.
IREE_API_EXPORT iree_status_t iree_hal_fixed_block_pool_query_backing_request(
    iree_hal_pool_t* backing_pool,
    const iree_hal_fixed_block_pool_options_t* options,
    iree_hal_pool_reservation_request_t* out_request);

// Creates an initially empty fixed-block allocator retaining |backing_pool|.
// Growth acquires ordinary reservations with their exact history and captures
// prepared ranges. Whole unused ranges return directly to the parent; an
// explicit backing cache owns idle retention. Trim does not trim the parent
// and its floor does not change automatic idle return.
//
// Acquisition copies candidate frontiers and commits complete batches under
// short metadata locks. Eligibility queries and all host/native allocation run
// outside them. Release publishes raw block availability before briefly taking
// the acquisition and publication locks to detach empty backing. Parent calls
// run after unlocking, without requiring executor progress.
// Failed batches claim no capacity. Non-dominated recycled blocks are returned
// as NEEDS_WAIT only with ALLOW_WAIT_FRONTIER and no ready block available.
// Tainted histories never become queue waits or empty readiness.
//
// The device group and its progress owner outlive the pool. Final destruction
// requires quiescent callers and returned reservations. It never waits for
// device execution. Retained ranges with unrepresentable
// histories require caller-established quiescence before final destruction.
IREE_API_EXPORT iree_status_t iree_hal_fixed_block_pool_create(
    iree_hal_pool_t* backing_pool,
    const iree_hal_fixed_block_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool);

// Creates a finite pool retaining the supplied prepared buffer range. Offsets
// are relative to the source view; WHOLE_BUFFER uses its remaining extent.
// Alignment rounds the range inward. The pool never grows or releases its
// backing before destruction. Explicit allocation epochs remain caller-owned.
// blocks_per_slab must be zero: capacity and initial history are derived from
// the buffer range.
// Every untouched byte inherits the source's exact reuse prerequisite.
IREE_API_EXPORT iree_status_t iree_hal_fixed_block_pool_create_from_buffer(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length,
    const iree_hal_fixed_block_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_FIXED_BLOCK_POOL_H_
