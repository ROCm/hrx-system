// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_TOOLING_COMPILE_POOL_H_
#define EXPERIMENTAL_QWEN_TOOLING_COMPILE_POOL_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/task/process.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Fixed worker pool shared by Qwen compilation batches.
typedef struct qwen_tooling_compile_pool_t qwen_tooling_compile_pool_t;

// Compiles one independently owned job on a stable worker slot.
typedef iree_status_t (*qwen_tooling_compile_job_fn_t)(
    void* user_data, iree_host_size_t worker_ordinal,
    iree_host_size_t job_ordinal);

// One asynchronously executing batch owned by its submitting thread.
//
// The batch storage remains live from submission through wait. Only the
// submitting thread initializes and waits on a batch.
typedef struct qwen_tooling_compile_batch_t {
  // Pool draining this batch.
  qwen_tooling_compile_pool_t* pool;
  // Number of jobs available to claim.
  iree_host_size_t job_count;
  // Callback invoked once for each claimed job.
  qwen_tooling_compile_job_fn_t job_fn;
  // Caller payload passed to |job_fn|.
  void* user_data;
  // Executor process used to distribute jobs across workers.
  iree_task_process_t process;
  // Next job ordinal to claim.
  iree_atomic_int64_t next_job_ordinal;
  // Number of jobs completed successfully.
  iree_atomic_int64_t completed_job_count;
  // Set after all workers have released the process.
  iree_atomic_int32_t released;
  // Wakes the submitting thread after process release.
  iree_notification_t completion_notification;
  // Terminal process status transferred to the waiter.
  iree_status_t completion_status;
} qwen_tooling_compile_batch_t;

// Creates an owning pool with |worker_count| task workers.
iree_status_t qwen_tooling_compile_pool_create(
    iree_host_size_t worker_count, iree_allocator_t host_allocator,
    qwen_tooling_compile_pool_t** out_pool);

// Releases a pool after all submitted batches have completed.
void qwen_tooling_compile_pool_release(qwen_tooling_compile_pool_t* pool);

// Returns the number of stable worker ordinals in |pool|.
iree_host_size_t qwen_tooling_compile_pool_worker_count(
    const qwen_tooling_compile_pool_t* pool);

// Submits |job_count| callback invocations and returns immediately.
iree_status_t qwen_tooling_compile_pool_submit(
    qwen_tooling_compile_pool_t* pool, iree_host_size_t job_count,
    qwen_tooling_compile_job_fn_t job_fn, void* user_data,
    qwen_tooling_compile_batch_t* out_batch);

// Waits for a submitted batch and transfers its terminal status.
iree_status_t qwen_tooling_compile_batch_wait(
    qwen_tooling_compile_batch_t* batch);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_TOOLING_COMPILE_POOL_H_
