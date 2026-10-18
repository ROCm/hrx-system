// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_POOL_WAIT_H_
#define IREE_HAL_POOL_WAIT_H_

#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Cold capacity observation shared by synchronous allocators and queues.
// Successful allocation attempts need neither this object nor its operations.
// Allocation retries use ALLOW_WAIT_FRONTIER and handle returned dependencies
// separately; access completion alone does not publish a capacity notification.
typedef struct iree_hal_pool_wait_t iree_hal_pool_wait_t;

typedef struct iree_hal_pool_wait_callback_t {
  // Receives one owned terminal status after all admitted operations and
  // cancellation keys retire. OK requests an allocation retry; it reserves no
  // capacity and establishes no completion of previous buffer accesses.
  void(IREE_API_PTR* fn)(void* user_data, iree_status_t status);
  // Borrowed until the terminal callback finishes.
  void* user_data;
} iree_hal_pool_wait_callback_t;

// Captures the pool's immutable, distinct local and backing notifications.
// The pool and its progress owners must outlive the helper. Every source is
// submitted and cancelled through its own proactor, which must remain polling.
// Returns FAILED_PRECONDITION if the pool has no capacity source; such a pool
// must report allocation failure as a status instead of transient exhaustion.
// The helper can be reused for successive allocation attempts.
IREE_API_EXPORT iree_status_t iree_hal_pool_wait_allocate(
    iree_hal_pool_t* pool, iree_allocator_t host_allocator,
    iree_hal_pool_wait_t** out_wait);

// Observes every source before the caller retries its allocation transaction.
// Submits no asynchronous work. Each prepare must be followed by abort after a
// successful/terminal attempt, or commit when capacity is still unavailable.
IREE_API_EXPORT void iree_hal_pool_wait_prepare(iree_hal_pool_wait_t* wait);

// Ends a prepared observation without submitting work or invoking a callback.
IREE_API_EXPORT void iree_hal_pool_wait_abort(iree_hal_pool_wait_t* wait);

// Waits asynchronously for any captured source using the pre-retry tokens.
// Exactly one callback follows every commit, including admission failure,
// timeout and cancellation. The first terminal event cancels remaining waits
// without signalling their notifications, then joins all admitted ownership.
// Unexpected failures during retirement propagate with the terminal result.
//
// Callers with a retry loop convert |timeout| to an absolute deadline once and
// pass that same timeout to every commit. The callback must be non-null and may
// execute before this function returns. Wait storage may be destroyed or reused
// at the callback's final handoff; commit makes no further accesses afterward.
IREE_API_EXPORT void iree_hal_pool_wait_commit(
    iree_hal_pool_wait_t* wait, iree_timeout_t timeout,
    iree_hal_pool_wait_callback_t callback);

// Requests cancellation of the current committed round. Its terminal callback
// remains mandatory. The caller serializes this call with callback-owned
// destruction or reuse; a completed round has nothing left to cancel.
IREE_API_EXPORT void iree_hal_pool_wait_cancel(iree_hal_pool_wait_t* wait);

// Frees an idle helper. No prepared observation, committed round, or
// external cancellation call may still reference it.
IREE_API_EXPORT void iree_hal_pool_wait_free(iree_hal_pool_wait_t* wait);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_POOL_WAIT_H_
