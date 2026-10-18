// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_POOL_HISTORY_H_
#define IREE_HAL_MEMORY_POOL_HISTORY_H_

#include "iree/async/frontier_tracker.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

// Overflow-only dependency capture. Each backing range secures one release
// reserve. Publication consumes a reserve; partial acquisition replenishes it
// before exposing a proof. Native operations, callbacks, and allocation never
// run under this metadata mutex.
typedef struct iree_hal_pool_history_t {
  // Guards capture inputs, reserve ownership, and backing credits.
  iree_slim_mutex_t mutex;
  // Published before any free block can expose an open proof. Acquire checks
  // this after claiming blocks; zero means their proofs are already immutable.
  iree_atomic_int32_t capture_pending;
  // Retained timeline; NULL when the source never carries reuse history.
  iree_async_frontier_timeline_t* timeline;
  // Unused captures available without allocation on return.
  iree_async_frontier_join_t* reserves;
  // Mutable overflow inputs; immutable before becoming public.
  iree_async_frontier_join_t* open;
  // Number of unused and open captures, excluding published waits.
  uint32_t reserve_count;
  // Attached/prepared ranges with infallible return credits.
  uint32_t backing_count;
  // Compact per-block frontier capacity, resolved by the pool constructor.
  uint8_t frontier_capacity;
} iree_hal_pool_history_t;

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t iree_hal_pool_history_initialize(
    iree_async_frontier_tracker_t* tracker, uint8_t frontier_capacity,
    iree_hal_pool_history_t* out_history);
void iree_hal_pool_history_deinitialize(iree_hal_pool_history_t* history);

// Secures one backing's release credit and compacts a wide inherited history.
// A compact result borrows caller storage until copied into backing metadata;
// otherwise the original frontier is returned unchanged. Runs outside pool
// locks. Every successful preparation is paired with retire_backing, including
// failed slab construction after preparation.
iree_status_t iree_hal_pool_history_prepare_backing(
    iree_hal_pool_history_t* history,
    const iree_async_frontier_t* initial_frontier,
    iree_async_single_frontier_t* compact_storage,
    const iree_async_frontier_t** out_frontier);
// Seals inputs before parent return and consumes the departed backing's credit.
// Runs outside pool locks; never allocates or waits for dependency completion.
void iree_hal_pool_history_retire_backing(iree_hal_pool_history_t* history);

// Copies both inputs before replacing target with the current open proof.
// May run under a pool's mutation lock: only presecured metadata is touched.
void iree_hal_pool_history_collapse(void* user_data,
                                    iree_async_frontier_t* target,
                                    const iree_async_frontier_t* source);

// Seals any open proof before exposing acquisition results. The common path is
// one atomic read. Overflow may prepare replacement metadata outside all locks.
iree_status_t iree_hal_pool_history_seal_acquire(
    iree_hal_pool_history_t* history);
// Releases unused capture storage while preserving live backing return credits.
void iree_hal_pool_history_trim(iree_hal_pool_history_t* history);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_MEMORY_POOL_HISTORY_H_
