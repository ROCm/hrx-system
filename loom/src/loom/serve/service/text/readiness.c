// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/readiness.h"

#include <string.h>

#include "iree/async/proactor.h"

void loom_serve_text_readiness_initialize(
    iree_task_executor_t* executor, iree_async_proactor_t* proactor,
    loom_serve_text_readiness_t* readiness) {
  *readiness = (loom_serve_text_readiness_t){.device = {.proactor = proactor},
                                             .cpu = {.executor = executor}};
  iree_slim_mutex_initialize(&readiness->mutex);
}

void loom_serve_text_readiness_deinitialize(
    loom_serve_text_readiness_t* readiness) {
  iree_slim_mutex_deinitialize(&readiness->mutex);
}

void loom_serve_text_readiness_set_observer(
    loom_serve_text_readiness_t* readiness,
    loom_runtime_wake_callback_t observer) {
  iree_slim_mutex_lock(&readiness->mutex);
  readiness->observer = observer;
  iree_slim_mutex_unlock(&readiness->mutex);
}

void loom_serve_text_readiness_notify(loom_serve_text_readiness_t* readiness) {
  iree_slim_mutex_lock(&readiness->mutex);
  readiness->observer.wake(readiness->observer.user_data);
  iree_slim_mutex_unlock(&readiness->mutex);
}

static void loom_serve_text_readiness_device_completed(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  loom_serve_text_readiness_t* readiness = user_data;
  iree_slim_mutex_lock(&readiness->mutex);
  readiness->device.result.status = status;
  readiness->device.result.ready = true;
  readiness->observer.wake(readiness->observer.user_data);
  iree_slim_mutex_unlock(&readiness->mutex);
}

iree_status_t loom_serve_text_readiness_await(
    loom_serve_text_readiness_t* readiness,
    const loom_runtime_execution_frontier_t* frontier) {
  iree_async_semaphore_wait_operation_t* wait = &readiness->device.operation;
  memset(wait, 0, sizeof(*wait));
  iree_async_operation_initialize(
      &wait->base, IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      loom_serve_text_readiness_device_completed, readiness);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    readiness->device.semaphores[i] =
        (iree_async_semaphore_t*)frontier->semaphores[i];
    readiness->device.values[i] = frontier->values[i];
  }
  wait->semaphores = readiness->device.semaphores;
  wait->values = readiness->device.values;
  wait->count = 2;
  wait->mode = IREE_ASYNC_WAIT_MODE_ALL;
  return iree_async_proactor_submit_one(readiness->device.proactor,
                                        &wait->base);
}

static iree_status_t loom_serve_text_readiness_run(
    iree_task_process_t* process, const iree_task_worker_context_t* context,
    iree_task_process_drain_result_t* result) {
  result->completed = true;
  result->did_work = true;
  loom_serve_text_readiness_t* readiness = process->user_data;
  return readiness->cpu.run(readiness->cpu.user_data);
}

static void loom_serve_text_readiness_cpu_completed(
    iree_task_process_t* process, iree_status_t status) {
  loom_serve_text_readiness_t* readiness = process->user_data;
  iree_slim_mutex_lock(&readiness->mutex);
  readiness->cpu.result.status = status;
  iree_slim_mutex_unlock(&readiness->mutex);
}

static void loom_serve_text_readiness_cpu_released(
    iree_task_process_t* process) {
  loom_serve_text_readiness_t* readiness = process->user_data;
  iree_slim_mutex_lock(&readiness->mutex);
  readiness->cpu.result.ready = true;
  readiness->observer.wake(readiness->observer.user_data);
  iree_slim_mutex_unlock(&readiness->mutex);
}

void loom_serve_text_readiness_schedule(loom_serve_text_readiness_t* readiness,
                                        loom_serve_text_cpu_work_fn_t run,
                                        void* user_data) {
  readiness->cpu.run = run;
  readiness->cpu.user_data = user_data;
  iree_task_process_initialize(loom_serve_text_readiness_run, 0, 1,
                               &readiness->cpu.process);
  readiness->cpu.process.user_data = readiness;
  readiness->cpu.process.completion_fn =
      loom_serve_text_readiness_cpu_completed;
  readiness->cpu.process.release_fn = loom_serve_text_readiness_cpu_released;
  iree_task_executor_schedule_process(readiness->cpu.executor,
                                      &readiness->cpu.process);
}

static bool loom_serve_text_readiness_take(
    loom_serve_text_readiness_t* readiness,
    loom_serve_text_readiness_result_t* result, iree_status_t* out_status) {
  iree_slim_mutex_lock(&readiness->mutex);
  const bool ready = result->ready;
  if (ready) {
    *out_status = result->status;
    *result = (loom_serve_text_readiness_result_t){0};
  }
  iree_slim_mutex_unlock(&readiness->mutex);
  return ready;
}

bool loom_serve_text_readiness_take_cpu(loom_serve_text_readiness_t* readiness,
                                        iree_status_t* out_status) {
  return loom_serve_text_readiness_take(readiness, &readiness->cpu.result,
                                        out_status);
}

bool loom_serve_text_readiness_take_device(
    loom_serve_text_readiness_t* readiness, iree_status_t* out_status) {
  return loom_serve_text_readiness_take(readiness, &readiness->device.result,
                                        out_status);
}
