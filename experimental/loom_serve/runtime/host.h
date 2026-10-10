// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_HOST_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_HOST_H_

#include "iree/async/util/proactor_pool.h"
#include "iree/task/executor.h"
#include "loomc/task.h"
#include "loomc/task_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

// Application-owned CPU workers and asynchronous I/O infrastructure. Devices,
// compiler queues, media jobs and scheduler processes share this host; models
// never create private worker populations. Devices retain the host. Callers
// retire all attached work before releasing its final owner.
typedef struct loom_serve_host_t loom_serve_host_t;

// Creates up to worker_count physical-core workers with compiler-sized stacks.
// Worker count must be positive. Proactors are created lazily per NUMA node.
iree_status_t loom_serve_host_create(iree_host_size_t worker_count,
                                     loom_serve_host_t** out_host,
                                     iree_allocator_t allocator);

// Releases I/O infrastructure and joins the final executor owner. This is an
// outer application teardown operation, never a task or completion callback.
// All dependent operations have already retired. NULL is permitted.
void loom_serve_host_release(loom_serve_host_t* host);

// Retains shared host resources for a dependent owner.
void loom_serve_host_retain(loom_serve_host_t* host);

// Borrowed interfaces valid through final host release. The task-pool adapter
// uses exactly this executor; it does not create another worker population.
iree_task_executor_t* loom_serve_host_executor(const loom_serve_host_t* host);
loomc_task_pool_t* loom_serve_host_task_pool(const loom_serve_host_t* host);

// Shared compiler scheduling domain, borrowed through host release. Catalogs
// retire their own jobs, not this queue; only outer host teardown drains it.
// Worker ordinals match the shared task pool and executor.
loomc_task_sink_t loom_serve_host_compiler_sink(const loom_serve_host_t* host);

iree_async_proactor_pool_t* loom_serve_host_proactor_pool(
    const loom_serve_host_t* host);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_HOST_H_
