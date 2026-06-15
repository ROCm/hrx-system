// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_BENCHMARK_UTIL_COMPILE_POOL_PROTOTYPE_H_
#define LOOMC_BENCHMARK_UTIL_COMPILE_POOL_PROTOTYPE_H_

#include "iree/base/api.h"
#include "iree/task/executor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct loomc_benchmark_compile_pool_t loomc_benchmark_compile_pool_t;

typedef iree_status_t (*loomc_benchmark_compile_job_fn_t)(
    void* user_data, iree_host_size_t worker_ordinal,
    iree_host_size_t job_ordinal);

struct loomc_benchmark_compile_pool_t {
  // Task executor used to drain compile batches.
  iree_task_executor_t* executor;

  // Dense worker-slot count available to benchmark scenarios.
  iree_host_size_t worker_count;
};

void loomc_benchmark_compile_pool_initialize_empty(
    loomc_benchmark_compile_pool_t* out_pool);

iree_status_t loomc_benchmark_compile_pool_initialize_with_executor(
    iree_task_executor_t* executor, iree_host_size_t worker_count,
    loomc_benchmark_compile_pool_t* out_pool);

iree_status_t loomc_benchmark_compile_pool_initialize_owning(
    iree_host_size_t worker_count, iree_allocator_t host_allocator,
    loomc_benchmark_compile_pool_t* out_pool);

void loomc_benchmark_compile_pool_deinitialize(
    loomc_benchmark_compile_pool_t* pool);

iree_status_t loomc_benchmark_compile_pool_run_batch(
    loomc_benchmark_compile_pool_t* pool, iree_host_size_t job_count,
    loomc_benchmark_compile_job_fn_t job_fn, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOMC_BENCHMARK_UTIL_COMPILE_POOL_PROTOTYPE_H_
