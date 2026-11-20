// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Worker thread for the POSIX proactor.
//
// Workers execute blocking file operations and return their results to the
// poll owner. The poll owner handles descriptor readiness independently.
//
// Workers are self-contained structs with an initialize/deinitialize lifecycle.
// The proactor allocates workers as a trailing array and initializes each one.
// Each worker has a back-pointer to the owning proactor for queue access.

#ifndef IREE_ASYNC_PLATFORM_POSIX_WORKER_H_
#define IREE_ASYNC_PLATFORM_POSIX_WORKER_H_

#include "iree/async/util/completion_pool.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_async_proactor_posix_t;

// One blocking file worker. Its result storage outlives every publication.
typedef struct iree_async_posix_worker_t {
  // Owning proactor, including the file queue and shutdown flag.
  struct iree_async_proactor_posix_t* proactor;
  // Owned thread handle. Its final release joins the worker.
  iree_thread_t* thread;
  // Guaranteed result storage while the shared completion pool is full.
  iree_async_posix_completion_t reserved_completion;
  // Set by this worker before publication and cleared by the poll owner.
  iree_atomic_int32_t reserved_completion_pending;
  // Wakes this worker after the poll owner has taken its reserved result.
  iree_notification_t completion_consumed;
} iree_async_posix_worker_t;

// Returns the worker's reserved entry after the poll owner has taken its
// result.
void iree_async_posix_worker_release_completion(
    iree_async_posix_completion_t* completion);

// Initializes a worker and starts its thread.
// |out_worker| must point to caller-allocated storage (typically in the
// proactor's trailing data array).
// On failure, caller should still call deinitialize for cleanup.
iree_status_t iree_async_posix_worker_initialize(
    struct iree_async_proactor_posix_t* proactor,
    iree_thread_affinity_t affinity,
    const iree_thread_affinity_set_t* processor_set, iree_allocator_t allocator,
    iree_async_posix_worker_t* out_worker);

// Joins the worker and releases its completion storage after the proactor has
// signaled shutdown. Also cleans up a worker whose thread failed to start.
void iree_async_posix_worker_deinitialize(iree_async_posix_worker_t* worker);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_WORKER_H_
