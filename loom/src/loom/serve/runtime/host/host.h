// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_HOST_HOST_H_
#define LOOM_SERVE_RUNTIME_HOST_HOST_H_

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/task/executor.h"
#include "loomc/target.h"
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

struct loomc_iree_hal_target_provider_t;

// Creates one immutable compiler target environment for an application host.
typedef loomc_status_t(
    LOOMC_API_PTR* loom_serve_target_environment_create_fn_t)(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

// Application-global compiler target composition. The environment factory and
// provider descriptors are process-lifetime functions/data from linked LoomC
// target leaves. The host composes the supplied packages with VM preparation
// support, owns the resulting environment, and copies the provider pointer
// list. Device-only hosts may omit this configuration.
typedef struct loom_serve_compiler_options_t {
  // Factory for the linked native compiler packages; VM support is added once.
  loom_serve_target_environment_create_fn_t target_environment_create;
  // Ordered HAL target routes borrowed for host creation.
  const struct loomc_iree_hal_target_provider_t* const* target_providers;
  // Number of entries in target_providers.
  iree_host_size_t target_provider_count;
} loom_serve_compiler_options_t;

// Creates up to worker_count physical-core workers with compiler-sized stacks.
// Worker count must be positive. Proactors are created lazily per NUMA node.
// The host always owns a task driver backed by this exact executor; that
// runtime capability is independent of optional native x86 compiler support.
// compiler_options may be NULL when the host will never compile model sources.
iree_status_t loom_serve_host_create(
    iree_host_size_t worker_count,
    const loom_serve_compiler_options_t* compiler_options,
    iree_allocator_t host_allocator, loom_serve_host_t** out_host);

// Releases I/O infrastructure and joins the final executor owner. This is an
// outer application teardown operation, never a task or completion callback.
// All dependent operations have already retired. NULL is permitted.
void loom_serve_host_release(loom_serve_host_t* host);

// Retains shared host resources for a dependent owner. NULL is permitted.
void loom_serve_host_retain(loom_serve_host_t* host);

// Borrowed interfaces valid through final host release. The task-pool adapter
// uses exactly this executor; it does not create another worker population.
iree_task_executor_t* loom_serve_host_executor(const loom_serve_host_t* host);
loomc_task_pool_t* loom_serve_host_task_pool(const loom_serve_host_t* host);
// Task driver sharing the host executor, borrowed through host release.
iree_hal_driver_t* loom_serve_host_task_driver(const loom_serve_host_t* host);

// Shared compiler scheduling domain, borrowed through host release. Catalogs
// retire their own jobs, not this queue; only outer host teardown drains it.
// Worker ordinals match the shared task pool and executor.
loomc_task_sink_t loom_serve_host_compiler_sink(const loom_serve_host_t* host);

// Shared native and VM compiler environment, or NULL for a device-only host.
// Borrowed through host release. All model source catalogs use this grammar.
loomc_target_environment_t* loom_serve_host_target_environment(
    const loom_serve_host_t* host);

// Ordered HAL target routes paired with the host environment. Returns NULL and
// zero routes for a device-only host. The returned array is borrowed through
// host release.
const struct loomc_iree_hal_target_provider_t* const*
loom_serve_host_target_providers(const loom_serve_host_t* host,
                                 iree_host_size_t* out_provider_count);

iree_async_proactor_pool_t* loom_serve_host_proactor_pool(
    const loom_serve_host_t* host);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_HOST_HOST_H_
