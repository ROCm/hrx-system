// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compile_pool_prototype.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/thread.h"
#include "iree/task/process.h"
#include "iree/task/topology.h"
#include "iree/task/tuning.h"

typedef struct loomc_benchmark_compile_batch_t {
  // Pool that scheduled this batch.
  loomc_benchmark_compile_pool_t* pool;

  // Total jobs available to drain.
  iree_host_size_t job_count;

  // Callback invoked for each claimed job.
  loomc_benchmark_compile_job_fn_t job_fn;

  // Caller payload passed to `job_fn`.
  void* user_data;

  // Next job ordinal to claim.
  iree_atomic_int64_t next_job_ordinal;

  // Number of jobs completed successfully.
  iree_atomic_int64_t completed_job_count;

  // Set once every active drainer has left the process.
  iree_atomic_int32_t released;

  // First terminal status reported through the task process completion path.
  iree_status_t completion_status;
} loomc_benchmark_compile_batch_t;

void loomc_benchmark_compile_pool_initialize_empty(
    loomc_benchmark_compile_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  memset(out_pool, 0, sizeof(*out_pool));
}

iree_status_t loomc_benchmark_compile_pool_initialize_with_executor(
    iree_task_executor_t* executor, iree_host_size_t worker_count,
    loomc_benchmark_compile_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(executor);
  IREE_ASSERT_ARGUMENT(out_pool);
  if (worker_count == 0 ||
      worker_count > (iree_host_size_t)IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "worker count is outside task executor limits");
  }
  loomc_benchmark_compile_pool_initialize_empty(out_pool);
  iree_task_executor_retain(executor);
  out_pool->executor = executor;
  out_pool->worker_count = worker_count;
  return iree_ok_status();
}

iree_status_t loomc_benchmark_compile_pool_initialize_owning(
    iree_host_size_t worker_count, iree_allocator_t host_allocator,
    loomc_benchmark_compile_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  if (worker_count == 0 ||
      worker_count > (iree_host_size_t)IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "worker count is outside task executor limits");
  }

  loomc_benchmark_compile_pool_initialize_empty(out_pool);
  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(worker_count, &topology);

  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  // Compiler workers run recursive IR analyses and lowering code, and
  // sanitizer instrumentation can materially increase stack use. The task
  // system default is sized for runtime work items, not compiler jobs.
  options.worker_stack_size = 2 * 1024 * 1024;
  iree_task_executor_t* executor = NULL;
  iree_status_t status =
      iree_task_executor_create(options, &topology, host_allocator, &executor);
  iree_task_topology_deinitialize(&topology);

  if (iree_status_is_ok(status)) {
    out_pool->executor = executor;
    out_pool->worker_count = worker_count;
  }
  return status;
}

void loomc_benchmark_compile_pool_deinitialize(
    loomc_benchmark_compile_pool_t* pool) {
  if (pool == NULL || pool->executor == NULL) {
    return;
  }
  iree_task_executor_release(pool->executor);
  loomc_benchmark_compile_pool_initialize_empty(pool);
}

static iree_task_process_drain_result_t
loomc_benchmark_compile_batch_completed_result(void) {
  return (iree_task_process_drain_result_t){
      .completed = true,
      .did_work = false,
  };
}

static iree_status_t loomc_benchmark_compile_batch_drain(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result) {
  loomc_benchmark_compile_batch_t* batch =
      (loomc_benchmark_compile_batch_t*)process->user_data;
  if (iree_task_process_has_error(process)) {
    *out_result = loomc_benchmark_compile_batch_completed_result();
    return iree_ok_status();
  }

  int64_t job_ordinal = iree_atomic_fetch_add(&batch->next_job_ordinal, 1,
                                              iree_memory_order_acq_rel);
  if (job_ordinal >= (int64_t)batch->job_count) {
    int64_t completed_count = iree_atomic_load(&batch->completed_job_count,
                                               iree_memory_order_acquire);
    *out_result = (iree_task_process_drain_result_t){
        .completed = completed_count >= (int64_t)batch->job_count,
        .did_work = false,
    };
    return iree_ok_status();
  }

  iree_host_size_t worker_ordinal =
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

  int64_t completed_count =
      iree_atomic_fetch_add(&batch->completed_job_count, 1,
                            iree_memory_order_acq_rel) +
      1;
  *out_result = (iree_task_process_drain_result_t){
      .completed = completed_count >= (int64_t)batch->job_count,
      .did_work = true,
  };
  return iree_ok_status();
}

static void loomc_benchmark_compile_batch_complete(iree_task_process_t* process,
                                                   iree_status_t status) {
  loomc_benchmark_compile_batch_t* batch =
      (loomc_benchmark_compile_batch_t*)process->user_data;
  batch->completion_status = status;
}

static void loomc_benchmark_compile_batch_release(
    iree_task_process_t* process) {
  loomc_benchmark_compile_batch_t* batch =
      (loomc_benchmark_compile_batch_t*)process->user_data;
  iree_atomic_store(&batch->released, 1, iree_memory_order_release);
}

iree_status_t loomc_benchmark_compile_pool_run_batch(
    loomc_benchmark_compile_pool_t* pool, iree_host_size_t job_count,
    loomc_benchmark_compile_job_fn_t job_fn, void* user_data) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(job_fn);
  if (!pool->executor || pool->worker_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "compile pool has not been initialized");
  }
  if (job_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "compile batch must contain at least one job");
  }
  if (job_count > (iree_host_size_t)INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "compile batch is too large");
  }

  loomc_benchmark_compile_batch_t batch = {
      .pool = pool,
      .job_count = job_count,
      .job_fn = job_fn,
      .user_data = user_data,
      .completion_status = iree_ok_status(),
  };
  iree_atomic_store(&batch.next_job_ordinal, 0, iree_memory_order_relaxed);
  iree_atomic_store(&batch.completed_job_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&batch.released, 0, iree_memory_order_relaxed);

  iree_task_process_t process;
  iree_task_process_initialize(loomc_benchmark_compile_batch_drain,
                               /*suspend_count=*/0, (int32_t)pool->worker_count,
                               &process);
  iree_task_process_set_flags(&process, IREE_TASK_PROCESS_FLAG_COMPUTE_SLOT);
  process.completion_fn = loomc_benchmark_compile_batch_complete;
  process.release_fn = loomc_benchmark_compile_batch_release;
  process.user_data = &batch;

  iree_task_executor_schedule_process(pool->executor, &process);
  while (!iree_atomic_load(&batch.released, iree_memory_order_acquire)) {
    iree_thread_yield();
  }
  return batch.completion_status;
}
