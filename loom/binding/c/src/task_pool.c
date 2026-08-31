// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/task_pool.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/threading/notification.h"
#include "iree/task/executor.h"
#include "iree/task/topology.h"
#include "loomc/iree.h"

#define LOOMC_TASK_POOL_DEFAULT_WORKER_COUNT 4
#define LOOMC_TASK_POOL_DEFAULT_WORKER_STACK_SIZE (2 * 1024 * 1024)

typedef enum loomc_task_pool_state_e {
  // Accepts new work and keeps the process reusable for later submissions.
  LOOMC_TASK_POOL_STATE_ACCEPTING = 0,

  // Rejects new work and releases the process after the shared queue drains.
  LOOMC_TASK_POOL_STATE_DRAINING = 1,
} loomc_task_pool_state_t;

struct loomc_task_pool_t {
  // Host allocator used for pool and executor storage.
  iree_allocator_t allocator;

  // Worker executor owned by the pool.
  iree_task_executor_t* executor;

  // Number of worker threads in the executor.
  loomc_host_size_t worker_count;

  // Persistent cooperative process draining the shared ready queue.
  iree_task_process_t process;

  // Ready-queue state serialized across submitters and workers.
  struct {
    // Serializes queue pointers and shutdown state.
    iree_slim_mutex_t mutex;

    // First task awaiting execution.
    loomc_task_t* head;

    // Last task awaiting execution.
    loomc_task_t* tail;

    // Number of tasks linked from |head| through |tail|.
    loomc_host_size_t count;

    // Worker capacity published for the current non-empty queue wave.
    int32_t scheduled_capacity;

    // Current task acceptance and shutdown state.
    loomc_task_pool_state_t state;
  } ready;

  // Process-release synchronization used only during shutdown.
  struct {
    // Posted after the executor has made its final process access.
    iree_notification_t notification;

    // Whether the executor has made its final process access.
    iree_atomic_int32_t released;
  } shutdown;
};

static loomc_task_t* loomc_task_pool_pop_ready_locked(loomc_task_pool_t* pool) {
  loomc_task_t* task = pool->ready.head;
  if (task == NULL) return NULL;
  pool->ready.head = task->next;
  task->next = NULL;
  if (pool->ready.head == NULL) pool->ready.tail = NULL;
  if (--pool->ready.count == 0) pool->ready.scheduled_capacity = 0;
  return task;
}

static iree_status_t loomc_task_pool_process_drain(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result) {
  loomc_task_pool_t* pool = (loomc_task_pool_t*)process->user_data;

  iree_slim_mutex_lock(&pool->ready.mutex);
  loomc_task_t* task = loomc_task_pool_pop_ready_locked(pool);
  const bool completed =
      task == NULL && pool->ready.state == LOOMC_TASK_POOL_STATE_DRAINING;
  iree_slim_mutex_unlock(&pool->ready.mutex);

  if (task != NULL) {
    loomc_task_execute(task, (loomc_host_size_t)worker_context->worker_index);
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

static void loomc_task_pool_process_release(iree_task_process_t* process) {
  loomc_task_pool_t* pool = (loomc_task_pool_t*)process->user_data;
  iree_slim_mutex_lock(&pool->ready.mutex);
  iree_atomic_store(&pool->shutdown.released, 1, iree_memory_order_release);
  iree_notification_post(&pool->shutdown.notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&pool->ready.mutex);
}

static loomc_status_t loomc_task_pool_submit(void* user_data,
                                             loomc_task_t* task) {
  loomc_task_pool_t* pool = (loomc_task_pool_t*)user_data;
  bool schedule_process = false;
  iree_slim_mutex_lock(&pool->ready.mutex);
  if (pool->ready.state != LOOMC_TASK_POOL_STATE_ACCEPTING) {
    iree_slim_mutex_unlock(&pool->ready.mutex);
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "task pool is shutting down");
  }
  if (pool->ready.tail != NULL) {
    pool->ready.tail->next = task;
  } else {
    pool->ready.head = task;
  }
  pool->ready.tail = task;
  ++pool->ready.count;
  const int32_t desired_capacity =
      (int32_t)iree_min(pool->ready.count, pool->worker_count);
  if (desired_capacity > pool->ready.scheduled_capacity) {
    pool->ready.scheduled_capacity = desired_capacity;
    iree_atomic_store(&pool->process.wake_budget, desired_capacity,
                      iree_memory_order_relaxed);
    schedule_process = true;
  }
  iree_slim_mutex_unlock(&pool->ready.mutex);
  if (schedule_process) {
    iree_task_executor_schedule_process(pool->executor, &pool->process);
  }
  return loomc_ok_status();
}

static bool loomc_task_pool_released_condition(void* user_data) {
  const loomc_task_pool_t* pool = (const loomc_task_pool_t*)user_data;
  return iree_atomic_load(&pool->shutdown.released,
                          iree_memory_order_acquire) != 0;
}

static void loomc_task_pool_await_release(loomc_task_pool_t* pool) {
  iree_notification_await(&pool->shutdown.notification,
                          loomc_task_pool_released_condition, pool,
                          iree_infinite_timeout());

  // The condition may become true before the release callback finishes
  // posting. Crossing its mutex keeps notification storage alive until the
  // callback has made its final access.
  iree_slim_mutex_lock(&pool->ready.mutex);
  iree_slim_mutex_unlock(&pool->ready.mutex);
}

static void loomc_task_pool_request_shutdown(loomc_task_pool_t* pool) {
  bool schedule_process = false;
  int32_t wake_budget = 1;
  iree_slim_mutex_lock(&pool->ready.mutex);
  if (pool->ready.state == LOOMC_TASK_POOL_STATE_ACCEPTING) {
    pool->ready.state = LOOMC_TASK_POOL_STATE_DRAINING;
    wake_budget =
        (int32_t)iree_min(iree_max(pool->ready.count, 1), pool->worker_count);
    schedule_process = true;
  }
  iree_slim_mutex_unlock(&pool->ready.mutex);
  if (schedule_process) {
    iree_atomic_store(&pool->process.wake_budget, wake_budget,
                      iree_memory_order_relaxed);
    iree_task_executor_schedule_process(pool->executor, &pool->process);
  }
}

loomc_status_t loomc_task_pool_allocate(
    const loomc_task_pool_options_t* options, loomc_allocator_t allocator,
    loomc_task_pool_t** out_pool) {
  if (out_pool == NULL || !loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator and out_pool are required");
  }
  *out_pool = NULL;
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_TASK_POOL_OPTIONS) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "task pool options have an unknown type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "task pool options structure_size is too small");
    }
    if (options->next != NULL) {
      return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                               "task pool option extensions are not supported");
    }
    if (options->max_worker_count > IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
      return loomc_make_status(
          LOOMC_STATUS_OUT_OF_RANGE,
          "task pool worker count exceeds executor limits");
    }
  }

  const loomc_host_size_t max_worker_count =
      options != NULL && options->max_worker_count != 0
          ? options->max_worker_count
          : LOOMC_TASK_POOL_DEFAULT_WORKER_COUNT;
  iree_allocator_t iree_allocator = iree_allocator_from_loomc(allocator);
  loomc_task_pool_t* pool = NULL;
  iree_status_t status = iree_allocator_malloc_aligned(
      iree_allocator, sizeof(*pool), iree_alignof(loomc_task_pool_t),
      /*offset=*/0, (void**)&pool);
  if (!iree_status_is_ok(status)) return loomc_status_from_iree(status);
  memset(pool, 0, sizeof(*pool));
  pool->allocator = iree_allocator;
  pool->ready.state = LOOMC_TASK_POOL_STATE_ACCEPTING;
  iree_slim_mutex_initialize(&pool->ready.mutex);
  iree_notification_initialize(&pool->shutdown.notification);
  iree_atomic_store(&pool->shutdown.released, 0, iree_memory_order_relaxed);

  iree_task_topology_t topology;
  status = iree_task_topology_initialize_from_physical_cores(
      IREE_TASK_TOPOLOGY_NODE_ID_ANY, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER, max_worker_count, &topology);
  if (iree_status_is_ok(status)) {
    if (iree_task_topology_group_count(&topology) == 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "host topology contains no available cores");
    } else {
      iree_task_executor_options_t executor_options;
      iree_task_executor_options_initialize(&executor_options);
      executor_options.worker_stack_size =
          options != NULL && options->worker_stack_size != 0
              ? options->worker_stack_size
              : LOOMC_TASK_POOL_DEFAULT_WORKER_STACK_SIZE;
      status = iree_task_executor_create(executor_options, &topology,
                                         iree_allocator, &pool->executor);
    }
    iree_task_topology_deinitialize(&topology);
  }
  if (!iree_status_is_ok(status)) {
    iree_notification_deinitialize(&pool->shutdown.notification);
    iree_slim_mutex_deinitialize(&pool->ready.mutex);
    iree_allocator_free_aligned(iree_allocator, pool);
    return loomc_status_from_iree(status);
  }

  pool->worker_count = iree_task_executor_worker_count(pool->executor);
  iree_task_process_initialize(loomc_task_pool_process_drain,
                               /*suspend_count=*/0,
                               /*wake_budget=*/(int32_t)pool->worker_count,
                               &pool->process);
  iree_task_process_set_flags(&pool->process,
                              IREE_TASK_PROCESS_FLAG_COMPUTE_SLOT);
  pool->process.release_fn = loomc_task_pool_process_release;
  pool->process.user_data = pool;
  *out_pool = pool;
  return loomc_ok_status();
}

loomc_host_size_t loomc_task_pool_worker_count(const loomc_task_pool_t* pool) {
  return pool != NULL ? pool->worker_count : 0;
}

loomc_task_sink_t loomc_task_pool_sink(loomc_task_pool_t* pool) {
  return pool != NULL ? (loomc_task_sink_t){
                            .submit = loomc_task_pool_submit,
                            .user_data = pool,
                        }
                      : (loomc_task_sink_t){0};
}

loomc_status_t loomc_task_pool_shutdown(loomc_task_pool_t* pool) {
  if (pool == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "task pool is required");
  }
  loomc_task_pool_request_shutdown(pool);
  return loomc_ok_status();
}

loomc_status_t loomc_task_pool_await_shutdown(loomc_task_pool_t* pool) {
  if (pool == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "task pool is required");
  }
  iree_slim_mutex_lock(&pool->ready.mutex);
  const bool shutdown_started =
      pool->ready.state != LOOMC_TASK_POOL_STATE_ACCEPTING;
  iree_slim_mutex_unlock(&pool->ready.mutex);
  if (!shutdown_started) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "task pool shutdown has not begun");
  }
  loomc_task_pool_await_release(pool);
  return loomc_ok_status();
}

void loomc_task_pool_free(loomc_task_pool_t* pool) {
  if (pool == NULL) return;
  loomc_task_pool_request_shutdown(pool);
  loomc_task_pool_await_release(pool);
  iree_task_executor_release(pool->executor);
  iree_notification_deinitialize(&pool->shutdown.notification);
  iree_slim_mutex_deinitialize(&pool->ready.mutex);
  iree_allocator_free_aligned(pool->allocator, pool);
}
