// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_IREE_TASK_POOL_H_
#define LOOMC_IREE_TASK_POOL_H_

#include "iree/task/executor.h"
#include "loomc/task_pool.h"

/// @file
/// Optional adapters between Loom task pools and IREE task executors.
///
/// IREE-hosted applications use these adapters to share one worker executor
/// between Loom task queues, HAL materializers, and application-owned
/// cooperative processes. Core LoomC headers remain independent from IREE task
/// types.

#ifdef __cplusplus
extern "C" {
#endif

/// Allocates a Loom task pool retaining an existing IREE task executor.
///
/// @param executor Executor to retain for the returned pool.
/// @param allocator Host allocator used for pool storage.
/// @param out_pool Receives the allocated pool on success.
/// @return OK after retaining `executor` and allocating the pool.
///
/// @ownership
/// The caller retains its existing executor reference. The returned pool owns
/// one additional reference released by `loomc_task_pool_free`.
LOOMC_API_EXPORT loomc_status_t loomc_task_pool_allocate_from_iree_executor(
    iree_task_executor_t* executor, loomc_allocator_t allocator,
    loomc_task_pool_t** out_pool);

/// Returns the borrowed IREE task executor owned by `pool`.
///
/// @param pool Pool to query, or NULL.
/// @return Borrowed executor, or NULL for a NULL pool.
///
/// A consumer that may outlive `pool` retains the returned executor before the
/// pool is freed. Cooperative processes attached to the executor own their
/// process state and teardown independently from Loom task queues.
LOOMC_API_EXPORT iree_task_executor_t* iree_task_executor_from_loomc_task_pool(
    const loomc_task_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_IREE_TASK_POOL_H_
