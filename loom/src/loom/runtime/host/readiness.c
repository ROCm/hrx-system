// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/host/readiness.h"

#include <string.h>

void loom_runtime_readiness_initialize(iree_task_executor_t* executor,
                                       iree_async_proactor_t* proactor,
                                       loom_runtime_readiness_t* readiness) {
  *readiness =
      (loom_runtime_readiness_t){.executor = executor, .proactor = proactor};
  iree_slim_mutex_initialize(&readiness->mutex);
}

void loom_runtime_readiness_deinitialize(loom_runtime_readiness_t* readiness) {
  iree_slim_mutex_deinitialize(&readiness->mutex);
}

void loom_runtime_readiness_set_observer(
    loom_runtime_readiness_t* readiness,
    loom_runtime_wake_callback_t observer) {
  iree_slim_mutex_lock(&readiness->mutex);
  readiness->observer = observer;
  iree_slim_mutex_unlock(&readiness->mutex);
}

void loom_runtime_readiness_notify(loom_runtime_readiness_t* readiness) {
  iree_slim_mutex_lock(&readiness->mutex);
  readiness->observer.wake(readiness->observer.user_data);
  iree_slim_mutex_unlock(&readiness->mutex);
}

static iree_status_t loom_runtime_readiness_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  result->completed = true;
  result->did_work = true;
  loom_runtime_readiness_task_t* task =
      iree_containerof(process, loom_runtime_readiness_task_t, process);
  return task->run(task->user_data);
}

static void loom_runtime_readiness_cpu_completed(iree_task_process_t* process,
                                                 iree_status_t status) {
  loom_runtime_readiness_t* readiness = process->user_data;
  loom_runtime_readiness_task_t* task =
      iree_containerof(process, loom_runtime_readiness_task_t, process);
  iree_slim_mutex_lock(&readiness->mutex);
  task->result.status = status;
  iree_slim_mutex_unlock(&readiness->mutex);
}

static void loom_runtime_readiness_cpu_released(iree_task_process_t* process) {
  loom_runtime_readiness_t* readiness = process->user_data;
  loom_runtime_readiness_task_t* task =
      iree_containerof(process, loom_runtime_readiness_task_t, process);
  iree_slim_mutex_lock(&readiness->mutex);
  task->result.ready = true;
  readiness->observer.wake(readiness->observer.user_data);
  iree_slim_mutex_unlock(&readiness->mutex);
}

void loom_runtime_readiness_schedule(loom_runtime_readiness_t* readiness,
                                     loom_runtime_cpu_work_fn_t run,
                                     void* user_data,
                                     loom_runtime_readiness_task_t* task) {
  task->run = run;
  task->user_data = user_data;
  iree_task_process_initialize(loom_runtime_readiness_run, 0, 1,
                               &task->process);
  task->process.user_data = readiness;
  task->process.completion_fn = loom_runtime_readiness_cpu_completed;
  task->process.release_fn = loom_runtime_readiness_cpu_released;
  iree_task_executor_schedule_process(readiness->executor, &task->process);
}

static void loom_runtime_readiness_wait_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  loom_runtime_readiness_t* readiness = user_data;
  loom_runtime_readiness_wait_t* wait =
      iree_containerof((iree_async_semaphore_wait_operation_t*)operation,
                       loom_runtime_readiness_wait_t, operation);
  iree_slim_mutex_lock(&readiness->mutex);
  wait->result.status = status;
  wait->result.ready = true;
  readiness->observer.wake(readiness->observer.user_data);
  iree_slim_mutex_unlock(&readiness->mutex);
}

iree_status_t loom_runtime_readiness_await(
    loom_runtime_readiness_t* readiness, iree_host_size_t count,
    iree_async_semaphore_t** semaphores, uint64_t* values,
    loom_runtime_readiness_wait_t* wait) {
  memset(&wait->operation, 0, sizeof(wait->operation));
  iree_async_operation_initialize(
      &wait->operation.base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE, loom_runtime_readiness_wait_completed,
      readiness);
  wait->operation.semaphores = semaphores;
  wait->operation.values = values;
  wait->operation.count = count;
  wait->operation.mode = IREE_ASYNC_WAIT_MODE_ALL;
  return iree_async_proactor_submit_one(readiness->proactor,
                                        &wait->operation.base);
}

bool loom_runtime_readiness_take(loom_runtime_readiness_t* readiness,
                                 loom_runtime_readiness_result_t* result,
                                 iree_status_t* out_status) {
  iree_slim_mutex_lock(&readiness->mutex);
  const bool ready = result->ready;
  if (ready) {
    *out_status = result->status;
    *result = (loom_runtime_readiness_result_t){0};
  }
  iree_slim_mutex_unlock(&readiness->mutex);
  return ready;
}
