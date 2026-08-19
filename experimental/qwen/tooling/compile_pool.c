// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/tooling/compile_pool.h"

#include <string.h>

#include "iree/task/executor.h"
#include "iree/task/topology.h"

struct qwen_tooling_compile_pool_t {
  // Allocator used to release this object.
  iree_allocator_t host_allocator;
  // Task executor used to drain compilation batches.
  iree_task_executor_t* executor;
  // Number of stable worker ordinals exposed to batch callbacks.
  iree_host_size_t worker_count;
};

static iree_task_process_drain_result_t
qwen_tooling_compile_batch_completed_result(void) {
  return (iree_task_process_drain_result_t){
      .completed = true,
      .did_work = false,
  };
}

static iree_status_t qwen_tooling_compile_batch_drain(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result) {
  qwen_tooling_compile_batch_t* batch =
      (qwen_tooling_compile_batch_t*)process->user_data;
  if (iree_task_process_has_error(process)) {
    *out_result = qwen_tooling_compile_batch_completed_result();
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

  const iree_host_size_t worker_ordinal =
      worker_context->worker_index % batch->pool->worker_count;
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

static void qwen_tooling_compile_batch_complete(iree_task_process_t* process,
                                                iree_status_t status) {
  qwen_tooling_compile_batch_t* batch =
      (qwen_tooling_compile_batch_t*)process->user_data;
  batch->completion_status = status;
}

static void qwen_tooling_compile_batch_release(iree_task_process_t* process) {
  qwen_tooling_compile_batch_t* batch =
      (qwen_tooling_compile_batch_t*)process->user_data;
  iree_atomic_store(&batch->released, 1, iree_memory_order_release);
  iree_notification_post(&batch->completion_notification, IREE_ALL_WAITERS);
}

static bool qwen_tooling_compile_batch_is_released(void* user_data) {
  qwen_tooling_compile_batch_t* batch =
      (qwen_tooling_compile_batch_t*)user_data;
  return iree_atomic_load(&batch->released, iree_memory_order_acquire) != 0;
}

iree_status_t qwen_tooling_compile_pool_create(
    iree_host_size_t worker_count, iree_allocator_t host_allocator,
    qwen_tooling_compile_pool_t** out_pool) {
  if (!out_pool || worker_count == 0 ||
      worker_count > (iree_host_size_t)IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "compile pool options are invalid");
  }
  *out_pool = NULL;

  qwen_tooling_compile_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  pool->host_allocator = host_allocator;

  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(worker_count, &topology);
  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  options.worker_stack_size = 2 * 1024 * 1024;
  iree_status_t status = iree_task_executor_create(
      options, &topology, host_allocator, &pool->executor);
  iree_task_topology_deinitialize(&topology);
  if (iree_status_is_ok(status)) {
    pool->worker_count = worker_count;
    *out_pool = pool;
  } else {
    iree_allocator_free(host_allocator, pool);
  }
  return status;
}

void qwen_tooling_compile_pool_release(qwen_tooling_compile_pool_t* pool) {
  if (!pool) return;
  const iree_allocator_t host_allocator = pool->host_allocator;
  iree_task_executor_release(pool->executor);
  iree_allocator_free(host_allocator, pool);
}

iree_host_size_t qwen_tooling_compile_pool_worker_count(
    const qwen_tooling_compile_pool_t* pool) {
  return pool ? pool->worker_count : 0;
}

iree_status_t qwen_tooling_compile_pool_submit(
    qwen_tooling_compile_pool_t* pool, iree_host_size_t job_count,
    qwen_tooling_compile_job_fn_t job_fn, void* user_data,
    qwen_tooling_compile_batch_t* out_batch) {
  if (!pool || !pool->executor || pool->worker_count == 0 || job_count == 0 ||
      job_count > (iree_host_size_t)INT64_MAX || !job_fn || !out_batch) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "compile batch options are invalid");
  }

  memset(out_batch, 0, sizeof(*out_batch));
  out_batch->pool = pool;
  out_batch->job_count = job_count;
  out_batch->job_fn = job_fn;
  out_batch->user_data = user_data;
  out_batch->completion_status = iree_ok_status();
  iree_atomic_store(&out_batch->next_job_ordinal, 0, iree_memory_order_relaxed);
  iree_atomic_store(&out_batch->completed_job_count, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&out_batch->released, 0, iree_memory_order_relaxed);
  iree_notification_initialize(&out_batch->completion_notification);

  const iree_host_size_t wake_budget = iree_min(pool->worker_count, job_count);
  iree_task_process_initialize(qwen_tooling_compile_batch_drain,
                               /*suspend_count=*/0, (int32_t)wake_budget,
                               &out_batch->process);
  iree_task_process_set_flags(&out_batch->process,
                              IREE_TASK_PROCESS_FLAG_COMPUTE_SLOT);
  out_batch->process.completion_fn = qwen_tooling_compile_batch_complete;
  out_batch->process.release_fn = qwen_tooling_compile_batch_release;
  out_batch->process.user_data = out_batch;
  iree_task_executor_schedule_process(pool->executor, &out_batch->process);
  return iree_ok_status();
}

iree_status_t qwen_tooling_compile_batch_wait(
    qwen_tooling_compile_batch_t* batch) {
  if (!batch || !batch->pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "compile batch was not submitted");
  }
  iree_notification_await(&batch->completion_notification,
                          qwen_tooling_compile_batch_is_released, batch,
                          iree_infinite_timeout());
  iree_notification_deinitialize(&batch->completion_notification);
  iree_status_t status = batch->completion_status;
  batch->completion_status = iree_ok_status();
  batch->pool = NULL;
  return status;
}
