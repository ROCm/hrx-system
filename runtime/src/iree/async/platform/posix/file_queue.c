// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/file_queue.h"

void iree_async_posix_file_queue_initialize(
    iree_async_posix_file_queue_t* queue) {
  memset(queue, 0, sizeof(*queue));
  iree_slim_mutex_initialize(&queue->mutex);
  iree_notification_initialize(&queue->notification);
}

void iree_async_posix_file_queue_deinitialize(
    iree_async_posix_file_queue_t* queue) {
  iree_notification_deinitialize(&queue->notification);
  iree_slim_mutex_deinitialize(&queue->mutex);
}

void iree_async_posix_file_queue_stop(iree_async_posix_file_queue_t* queue) {
  iree_atomic_store(&queue->stopping, 1, iree_memory_order_release);
  iree_notification_post(&queue->notification, IREE_ALL_WAITERS);
}

void iree_async_posix_file_queue_push(
    iree_async_posix_file_queue_t* queue,
    const iree_async_posix_file_batch_t* batch, iree_host_size_t worker_count) {
  iree_slim_mutex_lock(&queue->mutex);
  if (queue->tail) {
    queue->tail->next = batch->head;
  } else {
    queue->head = batch->head;
  }
  queue->tail = batch->tail;
  iree_slim_mutex_unlock(&queue->mutex);
  iree_notification_post(&queue->notification,
                         (int32_t)iree_min(batch->count, worker_count));
}

static iree_async_operation_t* iree_async_posix_file_queue_pop(
    iree_async_posix_file_queue_t* queue) {
  iree_slim_mutex_lock(&queue->mutex);
  iree_async_operation_t* operation = queue->head;
  if (operation) {
    queue->head = operation->next;
    if (!queue->head) {
      queue->tail = NULL;
    }
    operation->next = NULL;
  }
  iree_slim_mutex_unlock(&queue->mutex);
  return operation;
}

iree_async_operation_t* iree_async_posix_file_queue_take(
    iree_async_posix_file_queue_t* queue) {
  for (;;) {
    // Take the wait token before checking either condition. A racing enqueue
    // or shutdown then prevents commit_wait from sleeping through its wake.
    iree_wait_token_t token =
        iree_notification_prepare_wait(&queue->notification);
    if (iree_atomic_load(&queue->stopping, iree_memory_order_acquire)) {
      iree_notification_cancel_wait(&queue->notification);
      return NULL;
    }
    iree_async_operation_t* operation = iree_async_posix_file_queue_pop(queue);
    if (!operation) {
      iree_notification_commit_wait(&queue->notification, token, /*spin_ns=*/0,
                                    IREE_TIME_INFINITE_FUTURE);
      continue;
    }
    iree_notification_cancel_wait(&queue->notification);
    return operation;
  }
}
