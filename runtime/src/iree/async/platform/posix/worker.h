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

#include "iree/async/operation.h"
#include "iree/async/util/completion_pool.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_async_proactor_posix_t;

// Submitter-local chain of independent files. Each may have linked successors
// that remain owned by the poll owner until the preceding operation completes.
typedef struct iree_async_posix_file_batch_t {
  // First active file operation in this submission.
  iree_async_operation_t* head;
  // Last active file operation in this submission.
  iree_async_operation_t* tail;
  // Number of file heads available to wake idle workers.
  iree_host_size_t count;
} iree_async_posix_file_batch_t;

// FIFO of admitted file operations. The queue borrows their intrusive next
// pointers until a worker claims them. Linked continuations use linked_next.
typedef struct iree_async_posix_file_queue_t {
  // Protects both ends of the queue across submitters and workers.
  iree_slim_mutex_t mutex;
  // First operation available for a worker to claim.
  iree_async_operation_t* head;
  // Last operation, for splicing a whole submission in constant time.
  iree_async_operation_t* tail;
  // Signaled when files arrive or workers must exit.
  iree_notification_t notification;
} iree_async_posix_file_queue_t;

// Splices a nonempty, fully acquired batch onto the file queue and wakes
// enough workers to consume it. The submitter terminates the chain at its tail.
void iree_async_posix_file_queue_push(
    struct iree_async_proactor_posix_t* proactor,
    const iree_async_posix_file_batch_t* batch);

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
