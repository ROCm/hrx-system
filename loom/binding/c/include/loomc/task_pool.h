// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TASK_POOL_H_
#define LOOMC_TASK_POOL_H_

#include "loomc/task.h"

/// @file
/// Optional standard worker pool for generic LoomC tasks.
///
/// The pool implements `loomc_task_sink_t` and can run arbitrary application
/// and compiler work in one persistent worker population. It owns scheduling
/// policy only: caches, compilers, workspaces, requests, products, and
/// application state remain separately composed by the caller. Applications
/// with an existing scheduler use the same task sink protocol without linking
/// this package or the IREE task runtime.

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
/// The caller owns `out_pool` and frees it with `loomc_task_pool_free` after
/// every object that may submit through its sink has completed.
LOOMC_API_EXPORT loomc_status_t loomc_task_pool_allocate(
    const loomc_task_pool_options_t* options, loomc_allocator_t allocator,
    loomc_task_pool_t** out_pool);

/// Returns the number of worker threads owned by `pool`.
///
/// @param pool Pool to inspect, or NULL.
/// @return Worker count, or zero for a NULL pool.
LOOMC_API_EXPORT loomc_host_size_t
loomc_task_pool_worker_count(const loomc_task_pool_t* pool);

/// Returns a borrowed generic task sink backed by `pool`.
///
/// @param pool Pool accepting submitted work, or NULL.
/// @return Borrowed sink, or an empty sink for a NULL pool.
///
/// The sink remains valid until shutdown begins. Rejected submission leaves
/// task ownership with the caller according to the generic sink contract.
LOOMC_API_EXPORT loomc_task_sink_t
loomc_task_pool_sink(loomc_task_pool_t* pool);

/// Stops accepting new tasks and begins draining accepted work.
///
/// @param pool Pool to shut down.
/// @return OK after shutdown begins.
///
/// This operation is thread safe and idempotent. Tasks already executing
/// finish normally. Their attempts to submit additional work are rejected, so
/// owners of recursively expanding work await that work before shutting down
/// its sink.
LOOMC_API_EXPORT loomc_status_t
loomc_task_pool_shutdown(loomc_task_pool_t* pool);

/// Waits until every accepted task and every pool worker has released `pool`.
///
/// @param pool Pool whose shutdown has already begun.
/// @return OK after every worker releases the pool.
///
/// Shutdown must have begun before this call. The calling thread must not be a
/// task executing on `pool`, because a worker cannot wait for its own release.
LOOMC_API_EXPORT loomc_status_t
loomc_task_pool_await_shutdown(loomc_task_pool_t* pool);

/// Drains, awaits, and frees `pool`.
///
/// @param pool Pool to free. Passing NULL is allowed.
///
/// The calling thread must not be a task executing on `pool`, and no owner may
/// retain or submit through the pool sink after this call begins.
LOOMC_API_EXPORT void loomc_task_pool_free(loomc_task_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TASK_POOL_H_
