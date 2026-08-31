// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/task_pool.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/task/executor.h"
#include "iree/task/topology.h"
#include "loomc/iree.h"
#include "loomc/iree/task_pool.h"

#define LOOMC_TASK_POOL_DEFAULT_WORKER_COUNT 4
#define LOOMC_TASK_POOL_DEFAULT_WORKER_STACK_SIZE (2 * 1024 * 1024)

struct loomc_task_pool_t {
  // Host allocator used for pool storage.
  iree_allocator_t allocator;

  // Shared worker executor retained for the pool lifetime.
  iree_task_executor_t* executor;
};

static loomc_status_t loomc_task_pool_allocate_with_executor(
    iree_task_executor_t* executor, loomc_allocator_t allocator,
    loomc_task_pool_t** out_pool) {
  if (iree_task_executor_worker_count(executor) == 0) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "task executor contains no workers");
  }
  iree_allocator_t iree_allocator = iree_allocator_from_loomc(allocator);
  loomc_task_pool_t* pool = NULL;
  iree_status_t status = iree_allocator_malloc_aligned(
      iree_allocator, sizeof(*pool), iree_alignof(loomc_task_pool_t),
      /*offset=*/0, (void**)&pool);
  if (!iree_status_is_ok(status)) return loomc_status_from_iree(status);
  memset(pool, 0, sizeof(*pool));
  pool->allocator = iree_allocator;
  iree_task_executor_retain(executor);
  pool->executor = executor;
  *out_pool = pool;
  return loomc_ok_status();
}

loomc_status_t loomc_task_pool_allocate(
    const loomc_task_pool_options_t* options, loomc_allocator_t allocator,
    loomc_task_pool_t** out_pool) {
  if (out_pool == NULL || !loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator and out_pool are required");
  }
  *out_pool = NULL;
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_TASK_POOL_OPTIONS) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "task pool options have an unknown type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "task pool options structure_size is too small");
    }
    if (options->next != NULL) {
      return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                               "task pool option extensions are not supported");
    }
    if (options->max_worker_count > IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
      return loomc_make_status(
          LOOMC_STATUS_OUT_OF_RANGE,
          "task pool worker count exceeds executor limits");
    }
  }

  const loomc_host_size_t max_worker_count =
      options != NULL && options->max_worker_count != 0
          ? options->max_worker_count
          : LOOMC_TASK_POOL_DEFAULT_WORKER_COUNT;
  iree_task_topology_t topology;
  iree_status_t status = iree_task_topology_initialize_from_physical_cores(
      IREE_TASK_TOPOLOGY_NODE_ID_ANY, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      IREE_TASK_TOPOLOGY_DISTRIBUTION_SCATTER, max_worker_count, &topology);

  iree_task_executor_t* executor = NULL;
  if (iree_status_is_ok(status)) {
    if (iree_task_topology_group_count(&topology) == 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "host topology contains no available cores");
    } else {
      iree_task_executor_options_t executor_options;
      iree_task_executor_options_initialize(&executor_options);
      executor_options.worker_stack_size =
          options != NULL && options->worker_stack_size != 0
              ? options->worker_stack_size
              : LOOMC_TASK_POOL_DEFAULT_WORKER_STACK_SIZE;
      status = iree_task_executor_create(executor_options, &topology,
                                         iree_allocator_from_loomc(allocator),
                                         &executor);
    }
    iree_task_topology_deinitialize(&topology);
  }
  if (!iree_status_is_ok(status)) return loomc_status_from_iree(status);

  loomc_status_t public_status =
      loomc_task_pool_allocate_with_executor(executor, allocator, out_pool);
  iree_task_executor_release(executor);
  return public_status;
}

loomc_host_size_t loomc_task_pool_worker_count(const loomc_task_pool_t* pool) {
  return pool != NULL ? iree_task_executor_worker_count(
                            iree_task_executor_from_loomc_task_pool(pool))
                      : 0;
}

void loomc_task_pool_free(loomc_task_pool_t* pool) {
  if (pool == NULL) return;
  iree_task_executor_release(pool->executor);
  iree_allocator_free_aligned(pool->allocator, pool);
}

loomc_status_t loomc_task_pool_allocate_from_iree_executor(
    iree_task_executor_t* executor, loomc_allocator_t allocator,
    loomc_task_pool_t** out_pool) {
  if (executor == NULL || out_pool == NULL ||
      !loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "executor, allocator, and out_pool are required");
  }
  *out_pool = NULL;
  return loomc_task_pool_allocate_with_executor(executor, allocator, out_pool);
}

iree_task_executor_t* iree_task_executor_from_loomc_task_pool(
    const loomc_task_pool_t* pool) {
  return pool != NULL ? pool->executor : NULL;
}
