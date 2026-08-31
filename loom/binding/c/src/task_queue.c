// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/task_queue.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/threading/notification.h"
#include "iree/task/executor.h"
#include "loomc/iree.h"
#include "loomc/iree/task_pool.h"

typedef enum loomc_task_queue_state_e {
  // Accepts new work and keeps the process reusable for later submissions.
  LOOMC_TASK_QUEUE_STATE_ACCEPTING = 0,

  // Rejects new work and releases the process after the queue drains.
  LOOMC_TASK_QUEUE_STATE_DRAINING = 1,
} loomc_task_queue_state_t;

struct loomc_task_queue_t {
  // Host allocator used for queue storage.
  iree_allocator_t allocator;

  // Shared worker executor retained for the queue lifetime.
  iree_task_executor_t* executor;

  // Number of dense worker-local slots exposed to task callbacks.
  loomc_host_size_t worker_count;

  // Persistent cooperative process draining the ready queue.
  iree_task_process_t process;

  // Ready-queue state serialized across submitters and workers.
  struct {
    // Serializes queue pointers and shutdown state.
    iree_slim_mutex_t mutex;

    // First task awaiting execution.
    loomc_task_t* head;

    // Last task awaiting execution.
    loomc_task_t* tail;

    // Number of tasks linked from `head` through `tail`.
    loomc_host_size_t count;

    // Worker capacity published for the current non-empty queue wave.
    int32_t scheduled_capacity;

    // Current task acceptance and shutdown state.
    loomc_task_queue_state_t state;
  } ready;

  // Process-release synchronization used only during shutdown.
  struct {
    // Posted after the executor has made its final process access.
    iree_notification_t notification;

    // Whether the executor has made its final process access.
    iree_atomic_int32_t released;
  } shutdown;
};

static loomc_task_t* loomc_task_queue_pop_ready_locked(
    loomc_task_queue_t* queue) {
  loomc_task_t* task = queue->ready.head;
  if (task == NULL) return NULL;
  queue->ready.head = task->next;
  task->next = NULL;
  if (queue->ready.head == NULL) queue->ready.tail = NULL;
  if (--queue->ready.count == 0) queue->ready.scheduled_capacity = 0;
  return task;
}

static iree_status_t loomc_task_queue_process_drain(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result) {
  loomc_task_queue_t* queue = (loomc_task_queue_t*)process->user_data;

  iree_slim_mutex_lock(&queue->ready.mutex);
  loomc_task_t* task = loomc_task_queue_pop_ready_locked(queue);
  const bool completed =
      task == NULL && queue->ready.state == LOOMC_TASK_QUEUE_STATE_DRAINING;
  iree_slim_mutex_unlock(&queue->ready.mutex);

  if (task != NULL) {
    const loomc_host_size_t worker_ordinal =
        (loomc_host_size_t)worker_context->worker_index % queue->worker_count;
    loomc_task_execute(task, worker_ordinal);
    *out_result = (iree_task_process_drain_result_t){
        .completed = false,
        .did_work = true,
    };
  } else {
    *out_result = (iree_task_process_drain_result_t){
        .completed = completed,
        .did_work = false,
    };
  }
  return iree_ok_status();
}

static void loomc_task_queue_process_release(iree_task_process_t* process) {
  loomc_task_queue_t* queue = (loomc_task_queue_t*)process->user_data;
  iree_slim_mutex_lock(&queue->ready.mutex);
  iree_atomic_store(&queue->shutdown.released, 1, iree_memory_order_release);
  iree_notification_post(&queue->shutdown.notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&queue->ready.mutex);
}

static loomc_status_t loomc_task_queue_submit(void* user_data,
                                              loomc_task_t* task) {
  loomc_task_queue_t* queue = (loomc_task_queue_t*)user_data;
  bool schedule_process = false;
  iree_slim_mutex_lock(&queue->ready.mutex);
  if (queue->ready.state != LOOMC_TASK_QUEUE_STATE_ACCEPTING) {
    iree_slim_mutex_unlock(&queue->ready.mutex);
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "task queue is shutting down");
  }
  if (queue->ready.tail != NULL) {
    queue->ready.tail->next = task;
  } else {
    queue->ready.head = task;
  }
  queue->ready.tail = task;
  ++queue->ready.count;
  const int32_t desired_capacity =
      (int32_t)iree_min(queue->ready.count, queue->worker_count);
  if (desired_capacity > queue->ready.scheduled_capacity) {
    queue->ready.scheduled_capacity = desired_capacity;
    iree_atomic_store(&queue->process.wake_budget, desired_capacity,
                      iree_memory_order_relaxed);
    schedule_process = true;
  }
  iree_slim_mutex_unlock(&queue->ready.mutex);
  if (schedule_process) {
    iree_task_executor_schedule_process(queue->executor, &queue->process);
  }
  return loomc_ok_status();
}

static bool loomc_task_queue_released_condition(void* user_data) {
  const loomc_task_queue_t* queue = (const loomc_task_queue_t*)user_data;
  return iree_atomic_load(&queue->shutdown.released,
                          iree_memory_order_acquire) != 0;
}

static void loomc_task_queue_await_release(loomc_task_queue_t* queue) {
  iree_notification_await(&queue->shutdown.notification,
                          loomc_task_queue_released_condition, queue,
                          iree_infinite_timeout());

  // The condition may become true before the release callback finishes
  // posting. Crossing its mutex keeps notification storage alive until the
  // callback has made its final access.
  iree_slim_mutex_lock(&queue->ready.mutex);
  iree_slim_mutex_unlock(&queue->ready.mutex);
}

static void loomc_task_queue_request_shutdown(loomc_task_queue_t* queue) {
  bool schedule_process = false;
  int32_t wake_budget = 1;
  iree_slim_mutex_lock(&queue->ready.mutex);
  if (queue->ready.state == LOOMC_TASK_QUEUE_STATE_ACCEPTING) {
    queue->ready.state = LOOMC_TASK_QUEUE_STATE_DRAINING;
    wake_budget =
        (int32_t)iree_min(iree_max(queue->ready.count, 1), queue->worker_count);
    schedule_process = true;
  }
  iree_slim_mutex_unlock(&queue->ready.mutex);
  if (schedule_process) {
    iree_atomic_store(&queue->process.wake_budget, wake_budget,
                      iree_memory_order_relaxed);
    iree_task_executor_schedule_process(queue->executor, &queue->process);
  }
}

loomc_status_t loomc_task_queue_allocate(const loomc_task_pool_t* pool,
                                         loomc_allocator_t allocator,
                                         loomc_task_queue_t** out_queue) {
  if (pool == NULL || out_queue == NULL ||
      !loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "task pool, allocator, and out_queue are required");
  }
  *out_queue = NULL;
  iree_allocator_t iree_allocator = iree_allocator_from_loomc(allocator);
  loomc_task_queue_t* queue = NULL;
  iree_status_t status = iree_allocator_malloc_aligned(
      iree_allocator, sizeof(*queue), iree_alignof(loomc_task_queue_t),
      /*offset=*/0, (void**)&queue);
  if (!iree_status_is_ok(status)) return loomc_status_from_iree(status);
  memset(queue, 0, sizeof(*queue));
  queue->allocator = iree_allocator;
  queue->executor = iree_task_executor_from_loomc_task_pool(pool);
  iree_task_executor_retain(queue->executor);
  queue->worker_count = iree_task_executor_worker_count(queue->executor);
  queue->ready.state = LOOMC_TASK_QUEUE_STATE_ACCEPTING;
  iree_slim_mutex_initialize(&queue->ready.mutex);
  iree_notification_initialize(&queue->shutdown.notification);
  iree_atomic_store(&queue->shutdown.released, 0, iree_memory_order_relaxed);

  iree_task_process_initialize(loomc_task_queue_process_drain,
                               /*suspend_count=*/0,
                               /*wake_budget=*/(int32_t)queue->worker_count,
                               &queue->process);
  iree_task_process_set_flags(&queue->process,
                              IREE_TASK_PROCESS_FLAG_COMPUTE_SLOT);
  queue->process.release_fn = loomc_task_queue_process_release;
  queue->process.user_data = queue;
  *out_queue = queue;
  return loomc_ok_status();
}

loomc_task_sink_t loomc_task_queue_sink(loomc_task_queue_t* queue) {
  return queue != NULL ? (loomc_task_sink_t){
                             .submit = loomc_task_queue_submit,
                             .user_data = queue,
                         }
                       : (loomc_task_sink_t){0};
}

loomc_status_t loomc_task_queue_shutdown(loomc_task_queue_t* queue) {
  if (queue == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "task queue is required");
  }
  loomc_task_queue_request_shutdown(queue);
  return loomc_ok_status();
}

loomc_status_t loomc_task_queue_await_shutdown(loomc_task_queue_t* queue) {
  if (queue == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "task queue is required");
  }
  iree_slim_mutex_lock(&queue->ready.mutex);
  const bool shutdown_started =
      queue->ready.state != LOOMC_TASK_QUEUE_STATE_ACCEPTING;
  iree_slim_mutex_unlock(&queue->ready.mutex);
  if (!shutdown_started) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "task queue shutdown has not begun");
  }
  loomc_task_queue_await_release(queue);
  return loomc_ok_status();
}

void loomc_task_queue_free(loomc_task_queue_t* queue) {
  if (queue == NULL) return;
  loomc_task_queue_request_shutdown(queue);
  loomc_task_queue_await_release(queue);
  iree_task_executor_release(queue->executor);
  iree_notification_deinitialize(&queue->shutdown.notification);
  iree_slim_mutex_deinitialize(&queue->ready.mutex);
  iree_allocator_free_aligned(queue->allocator, queue);
}
