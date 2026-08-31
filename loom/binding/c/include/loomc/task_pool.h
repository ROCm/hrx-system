// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TASK_POOL_H_
#define LOOMC_TASK_POOL_H_

#include "loomc/status.h"

/// @file
/// Optional standard worker population for concurrent application work.
///
/// A pool retains one worker executor but no work queue or scheduling policy.
/// Callers attach one `loomc_task_queue_t` for each independent Loom task
/// domain, while IREE-hosted applications may attach other cooperative
/// processes such as HAL executable or command-buffer materializers. This lets
/// compilation and materialization overlap without creating competing worker
/// populations or forcing unrelated work through one FIFO.
///
/// Applications with an existing scheduler use `loomc_task_sink_t` without
/// linking this package or the IREE task runtime.

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque standard task pool.
typedef struct loomc_task_pool_t loomc_task_pool_t;

/// Task-pool creation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_TASK_POOL_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_task_pool_options_t)`, and fill the requested fields.
typedef struct loomc_task_pool_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_TASK_POOL_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future task-pool options.
  const void* next;

  /// Maximum physical-core workers to create, or zero for the default.
  ///
  /// The default is four workers. The actual count may be lower when processor
  /// affinity, a container CPU set, or the host topology exposes fewer
  /// physical cores. Workers scatter across cache domains so independent
  /// compiler jobs use the available memory bandwidth. Latency-oriented hosts
  /// commonly request eight; autotuning and compile-report search may benefit
  /// from wider pools.
  loomc_host_size_t max_worker_count;

  /// Minimum stack size in bytes for each worker, or zero for the default.
  ///
  /// The 2 MiB default is sized for recursive compiler analysis and sanitizer
  /// instrumentation rather than only short runtime callbacks. Stack storage
  /// is virtual address space reserved by the host threading implementation;
  /// physical pages are committed as workers use them.
  loomc_host_size_t worker_stack_size;
} loomc_task_pool_options_t;

/// Allocates a standard task pool and starts its worker population.
///
/// @param options Pool configuration, or NULL for defaults.
/// @param allocator Host allocator used for pool and executor storage.
/// @param out_pool Receives the allocated pool on success.
/// @return OK when the pool and all worker threads were created.
///
/// @ownership
/// The caller owns `out_pool` and frees it with `loomc_task_pool_free`. Task
/// queues and native cooperative processes attached to the pool retain the
/// underlying worker executor independently and may outlive the pool handle.
LOOMC_API_EXPORT loomc_status_t loomc_task_pool_allocate(
    const loomc_task_pool_options_t* options, loomc_allocator_t allocator,
    loomc_task_pool_t** out_pool);

/// Returns the number of worker threads represented by `pool`.
///
/// @param pool Pool to inspect, or NULL.
/// @return Worker count, or zero for a NULL pool.
LOOMC_API_EXPORT loomc_host_size_t
loomc_task_pool_worker_count(const loomc_task_pool_t* pool);

/// Releases the pool's worker-executor reference and frees `pool`.
///
/// @param pool Pool to free. Passing NULL is allowed.
///
/// Attached task queues and native processes retain the worker executor until
/// their own teardown completes. Freeing the final executor owner joins the
/// worker threads before returning.
LOOMC_API_EXPORT void loomc_task_pool_free(loomc_task_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TASK_POOL_H_
