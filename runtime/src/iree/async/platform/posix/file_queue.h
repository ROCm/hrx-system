// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PLATFORM_POSIX_FILE_QUEUE_H_
#define IREE_ASYNC_PLATFORM_POSIX_FILE_QUEUE_H_

#include "iree/async/operation.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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
  // Set at shutdown before waking every worker.
  iree_atomic_int32_t stopping;
  // Signaled when files arrive or workers must exit.
  iree_notification_t notification;
} iree_async_posix_file_queue_t;

void iree_async_posix_file_queue_initialize(
    iree_async_posix_file_queue_t* queue);
void iree_async_posix_file_queue_deinitialize(
    iree_async_posix_file_queue_t* queue);

// Wakes all consumers and ends their waits. The owner joins each thread before
// deinitializing the queue.
void iree_async_posix_file_queue_stop(iree_async_posix_file_queue_t* queue);

// Splices a nonempty, terminated batch and wakes up to |worker_count|
// consumers.
void iree_async_posix_file_queue_push(
    iree_async_posix_file_queue_t* queue,
    const iree_async_posix_file_batch_t* batch, iree_host_size_t worker_count);

// Waits until a worker owns one operation or the queue stops (returns NULL).
iree_async_operation_t* iree_async_posix_file_queue_take(
    iree_async_posix_file_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // IREE_ASYNC_PLATFORM_POSIX_FILE_QUEUE_H_
