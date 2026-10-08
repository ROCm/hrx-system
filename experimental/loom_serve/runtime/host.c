// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/host.h"

#include "iree/base/threading/numa.h"
#include "iree/task/topology.h"
#include "loomc/iree.h"
#include "loomc/iree/task_pool.h"

struct loom_serve_host_t {
  // Application and dependent device owners of the shared host resources.
  iree_atomic_ref_count_t ref_count;
  // Allocation policy for the host and its shared services.
  iree_allocator_t allocator;
  // Worker executor shared by compiler and application processes.
  iree_task_executor_t* executor;
  // Loom task adapter retaining the same executor, with no private workers.
  loomc_task_pool_t* task_pool;
  // NUMA-local I/O services outliving every attached device.
  iree_async_proactor_pool_t* proactor_pool;
};

iree_status_t loom_serve_host_create(iree_host_size_t worker_count,
                                     loom_serve_host_t** out_host,
                                     iree_allocator_t allocator) {
  *out_host = NULL;
  if (!worker_count || worker_count > IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host worker count is outside executor capacity");
  }
  loom_serve_host_t* host = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*host), (void**)&host));
  host->allocator = allocator;
  iree_atomic_ref_count_init(&host->ref_count);
  iree_task_topology_t topology;
  iree_status_t status = iree_task_topology_initialize_from_physical_cores(
      IREE_TASK_TOPOLOGY_NODE_ID_ANY, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER, worker_count, &topology);
  if (iree_status_is_ok(status)) {
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    options.worker_stack_size = 2 * 1024 * 1024;
    status = iree_task_executor_create(options, &topology, allocator,
                                       &host->executor);
    iree_task_topology_deinitialize(&topology);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_task_pool_allocate_from_iree_executor(
        host->executor, loomc_allocator_from_iree(allocator),
        &host->task_pool));
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), NULL,
        iree_async_proactor_pool_options_default(), allocator,
        &host->proactor_pool);
  }
  if (iree_status_is_ok(status)) {
    *out_host = host;
  } else {
    loom_serve_host_release(host);
  }
  return status;
}

void loom_serve_host_retain(loom_serve_host_t* host) {
  iree_atomic_ref_count_inc(&host->ref_count);
}

void loom_serve_host_release(loom_serve_host_t* host) {
  if (!host || iree_atomic_ref_count_dec(&host->ref_count) != 1) {
    return;
  }
  iree_async_proactor_pool_release(host->proactor_pool);
  loomc_task_pool_free(host->task_pool);
  iree_task_executor_release(host->executor);
  iree_allocator_free(host->allocator, host);
}

iree_task_executor_t* loom_serve_host_executor(const loom_serve_host_t* host) {
  return host->executor;
}

loomc_task_pool_t* loom_serve_host_task_pool(const loom_serve_host_t* host) {
  return host->task_pool;
}

iree_async_proactor_pool_t* loom_serve_host_proactor_pool(
    const loom_serve_host_t* host) {
  return host->proactor_pool;
}
