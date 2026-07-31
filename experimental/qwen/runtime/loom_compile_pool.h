// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_LOOM_COMPILE_POOL_H_
#define EXPERIMENTAL_QWEN_RUNTIME_LOOM_COMPILE_POOL_H_

#include "iree/base/api.h"
#include "iree/task/executor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Synchronous bounded worker pool for independent Loom compiler jobs.
typedef struct qwen_loom_compile_pool_t {
  // Task executor retained by the pool.
  iree_task_executor_t* executor;
  // Dense worker count exposed to job callbacks.
  iree_host_size_t worker_count;
} qwen_loom_compile_pool_t;

// Callback invoked once for each job in a batch.
//
// The worker ordinal is stable for the duration of the pool and may index
// worker-local compiler state. At most one callback runs for a worker ordinal
// at a time.
typedef iree_status_t (*qwen_loom_compile_job_fn_t)(
    void* user_data, iree_host_size_t worker_ordinal,
    iree_host_size_t job_ordinal);

// Initializes an owning pool with exactly |worker_count| task workers.
iree_status_t qwen_loom_compile_pool_initialize(
    iree_host_size_t worker_count, iree_allocator_t host_allocator,
    qwen_loom_compile_pool_t* out_pool);

// Deinitializes |pool| and joins its task workers.
void qwen_loom_compile_pool_deinitialize(qwen_loom_compile_pool_t* pool);

// Runs |job_count| jobs and waits until every active worker has left the batch.
//
// The first job failure cancels unclaimed work and is returned after in-flight
// callbacks finish. A pool runs one batch at a time.
iree_status_t qwen_loom_compile_pool_run_batch(
    qwen_loom_compile_pool_t* pool, iree_host_size_t job_count,
    qwen_loom_compile_job_fn_t job_fn, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_LOOM_COMPILE_POOL_H_
