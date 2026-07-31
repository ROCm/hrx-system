// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/loom_compile_pool.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/task/process.h"
#include "iree/task/topology.h"
#include "iree/task/tuning.h"

typedef struct qwen_loom_compile_batch_t {
  // Pool executing this batch.
  qwen_loom_compile_pool_t* pool;
  // Total jobs available to claim.
  iree_host_size_t job_count;
  // Callback invoked for each claimed job.
  qwen_loom_compile_job_fn_t job_fn;
  // Caller payload passed to |job_fn|.
  void* user_data;
  // Next job ordinal available to claim.
  iree_atomic_int64_t next_job_ordinal;
  // Number of jobs completed successfully.
  iree_atomic_int64_t completed_job_count;
  // Set once every active process drainer has exited.
  iree_atomic_int32_t released;
  // Notification posted after |released| becomes true.
  iree_notification_t release_notification;
  // First terminal status delivered by the task process.
  iree_status_t completion_status;
} qwen_loom_compile_batch_t;

iree_status_t qwen_loom_compile_pool_initialize(
    iree_host_size_t worker_count, iree_allocator_t host_allocator,
    qwen_loom_compile_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  memset(out_pool, 0, sizeof(*out_pool));
  if (worker_count == 0 ||
      worker_count > (iree_host_size_t)IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Loom compile worker count is outside executor "
                            "limits");
  }

  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(worker_count, &topology);

  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  // Compiler jobs run recursive analyses and lowering passes. Sanitizer
  // instrumentation can increase their stack use beyond runtime task defaults.
  options.worker_stack_size = 2 * 1024 * 1024;
  iree_status_t status = iree_task_executor_create(
      options, &topology, host_allocator, &out_pool->executor);
  iree_task_topology_deinitialize(&topology);
  if (iree_status_is_ok(status)) {
    out_pool->worker_count = worker_count;
  }
  return status;
}

void qwen_loom_compile_pool_deinitialize(qwen_loom_compile_pool_t* pool) {
  if (!pool) return;
  iree_task_executor_release(pool->executor);
  memset(pool, 0, sizeof(*pool));
}

static iree_task_process_drain_result_t
qwen_loom_compile_batch_completed_result(void) {
  return (iree_task_process_drain_result_t){
      .completed = true,
      .did_work = false,
  };
}

static iree_status_t qwen_loom_compile_batch_drain(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result) {
  qwen_loom_compile_batch_t* batch =
      (qwen_loom_compile_batch_t*)process->user_data;
  if (iree_task_process_has_error(process)) {
    *out_result = qwen_loom_compile_batch_completed_result();
    return iree_ok_status();
  }

  const int64_t job_ordinal = iree_atomic_fetch_add(&batch->next_job_ordinal, 1,
                                                    iree_memory_order_acq_rel);
  if (job_ordinal >= (int64_t)batch->job_count) {
    const int64_t completed_job_count = iree_atomic_load(
        &batch->completed_job_count, iree_memory_order_acquire);
    *out_result = (iree_task_process_drain_result_t){
        .completed = completed_job_count >= (int64_t)batch->job_count,
        .did_work = false,
    };
    return iree_ok_status();
  }

  const iree_host_size_t worker_ordinal = worker_context->worker_index;
  iree_status_t status = batch->job_fn(batch->user_data, worker_ordinal,
                                       (iree_host_size_t)job_ordinal);
  if (!iree_status_is_ok(status)) {
    *out_result = (iree_task_process_drain_result_t){
        .completed = true,
        .did_work = true,
    };
    return status;
  }

  const int64_t completed_job_count =
      iree_atomic_fetch_add(&batch->completed_job_count, 1,
                            iree_memory_order_acq_rel) +
      1;
  *out_result = (iree_task_process_drain_result_t){
      .completed = completed_job_count >= (int64_t)batch->job_count,
      .did_work = true,
  };
  return iree_ok_status();
}

static void qwen_loom_compile_batch_complete(iree_task_process_t* process,
                                             iree_status_t status) {
  qwen_loom_compile_batch_t* batch =
      (qwen_loom_compile_batch_t*)process->user_data;
  batch->completion_status = status;
}

static void qwen_loom_compile_batch_release(iree_task_process_t* process) {
  qwen_loom_compile_batch_t* batch =
      (qwen_loom_compile_batch_t*)process->user_data;
  iree_atomic_store(&batch->released, 1, iree_memory_order_release);
  iree_notification_post(&batch->release_notification, IREE_ALL_WAITERS);
}

static bool qwen_loom_compile_batch_is_released(void* user_data) {
  qwen_loom_compile_batch_t* batch = (qwen_loom_compile_batch_t*)user_data;
  return iree_atomic_load(&batch->released, iree_memory_order_acquire) != 0;
}

iree_status_t qwen_loom_compile_pool_run_batch(
    qwen_loom_compile_pool_t* pool, iree_host_size_t job_count,
    qwen_loom_compile_job_fn_t job_fn, void* user_data) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(job_fn);
  if (!pool->executor || pool->worker_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Loom compile pool is not initialized");
  }
  if (job_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Loom compile batch must contain at least one job");
  }
  if (job_count > (iree_host_size_t)INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Loom compile batch is too large");
  }

  qwen_loom_compile_batch_t batch = {
      .pool = pool,
      .job_count = job_count,
      .job_fn = job_fn,
      .user_data = user_data,
      .completion_status = iree_ok_status(),
  };
  iree_atomic_store(&batch.next_job_ordinal, 0, iree_memory_order_relaxed);
  iree_atomic_store(&batch.completed_job_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&batch.released, 0, iree_memory_order_relaxed);
  iree_notification_initialize(&batch.release_notification);

  iree_task_process_t process;
  iree_task_process_initialize(qwen_loom_compile_batch_drain,
                               /*suspend_count=*/0, (int32_t)pool->worker_count,
                               &process);
  iree_task_process_set_flags(&process, IREE_TASK_PROCESS_FLAG_COMPUTE_SLOT);
  process.completion_fn = qwen_loom_compile_batch_complete;
  process.release_fn = qwen_loom_compile_batch_release;
  process.user_data = &batch;

  iree_task_executor_schedule_process(pool->executor, &process);
  iree_notification_await(&batch.release_notification,
                          qwen_loom_compile_batch_is_released, &batch,
                          iree_infinite_timeout());
  iree_notification_deinitialize(&batch.release_notification);
  return batch.completion_status;
}
