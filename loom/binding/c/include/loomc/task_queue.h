// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TASK_QUEUE_H_
#define LOOMC_TASK_QUEUE_H_

#include "loomc/task.h"
#include "loomc/task_pool.h"

/// @file
/// Generic Loom task queue attached to a standard task pool.
///
/// A queue is one scheduling and ownership domain backed by one cooperative
/// process on the pool's shared worker executor. Multiple queues attached to
/// one pool can make progress independently at process drain boundaries. This
/// permits compiler, materialization, and application work to overlap while
/// retaining one worker population.

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque generic task queue.
typedef struct loomc_task_queue_t loomc_task_queue_t;

/// Allocates a task queue attached to `pool`.
///
/// @param pool Pool whose worker executor will drain the queue.
/// @param allocator Host allocator used for queue storage.
/// @param out_queue Receives the allocated queue on success.
/// @return OK after the queue is ready to accept tasks.
///
/// @ownership
/// The queue retains the pool's worker executor and may outlive the pool
/// handle. The caller owns `out_queue` and frees it with
/// `loomc_task_queue_free` after every object that may submit through its sink
/// has completed.
LOOMC_API_EXPORT loomc_status_t loomc_task_queue_allocate(
    const loomc_task_pool_t* pool, loomc_allocator_t allocator,
    loomc_task_queue_t** out_queue);

/// Returns a borrowed generic task sink backed by `queue`.
///
/// @param queue Queue accepting submitted work, or NULL.
/// @return Borrowed sink, or an empty sink for a NULL queue.
///
/// The sink remains valid until shutdown begins. Rejected submission leaves
/// task ownership with the caller according to the generic sink contract.
LOOMC_API_EXPORT loomc_task_sink_t
loomc_task_queue_sink(loomc_task_queue_t* queue);

/// Stops accepting new tasks and begins draining accepted work.
///
/// @param queue Queue to shut down.
/// @return OK after shutdown begins.
///
/// This operation is thread safe and idempotent. Tasks already executing
/// finish normally. Their attempts to submit additional work are rejected, so
/// owners of recursively expanding work await that work before shutting down
/// its sink.
LOOMC_API_EXPORT loomc_status_t
loomc_task_queue_shutdown(loomc_task_queue_t* queue);

/// Waits until every accepted task has drained and the queue process releases.
///
/// @param queue Queue whose shutdown has already begun.
/// @return OK after the queue process has released its executor state.
///
/// Shutdown must have begun before this call. The calling thread must not be a
/// task executing on `queue`, because a worker cannot wait for its own release.
LOOMC_API_EXPORT loomc_status_t
loomc_task_queue_await_shutdown(loomc_task_queue_t* queue);

/// Drains, awaits, and frees `queue`.
///
/// @param queue Queue to free. Passing NULL is allowed.
///
/// The calling thread must not be a task executing on `queue`, and no owner may
/// retain or submit through the queue sink after this call begins.
LOOMC_API_EXPORT void loomc_task_queue_free(loomc_task_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TASK_QUEUE_H_
