// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TASK_H_
#define LOOMC_TASK_H_

#include "loomc/status.h"

/// @file
/// Generic caller-scheduled work.
///
/// A task is a queueable base embedded in application or LoomC work records.
/// Task sinks provide the ownership boundary needed to compose LoomC with a
/// caller's event loop, deterministic queue, custom scheduler, or the optional
/// standard task pool. The protocol carries no compiler, product, cache,
/// thread-pool, or IREE task state.

#ifdef __cplusplus
extern "C" {
#endif

/// Base embedded at offset zero in one caller-defined work record.
///
/// A task has one ownership reference and is not reference counted. Successful
/// submission transfers that reference to the sink, which executes the task
/// exactly once and then destroys it exactly once. Failed submission leaves
/// destruction to the caller.
typedef struct loomc_task_t loomc_task_t;

/// Stateless callbacks for one task representation.
typedef struct loomc_task_vtable_t {
  /// Executes `task` exactly once without destroying its storage.
  ///
  /// `worker_ordinal` identifies the sink execution lane running this task.
  /// Ordinals are dense and mutually exclusive: two task callbacks never run
  /// concurrently with the same ordinal. Concrete tasks may use the ordinal to
  /// index caller-owned worker-local state.
  ///
  /// The callback terminalizes every outcome owned by the task before
  /// returning.
  void(LOOMC_API_PTR* execute)(loomc_task_t* task,
                               loomc_host_size_t worker_ordinal);

  /// Destroys `task` storage after execution or rejected-submission cleanup.
  void(LOOMC_API_PTR* destroy)(loomc_task_t* task);
} loomc_task_vtable_t;

struct loomc_task_t {
  /// Static callbacks implementing the concrete task.
  const loomc_task_vtable_t* vtable;

  /// Sink-owned intrusive link while the accepted task is queued.
  ///
  /// Task initialization sets this to NULL. A sink may use it only between
  /// accepting the task and invoking its terminal callback, and must clear it
  /// before execution. Exclusive task ownership permits one queue membership
  /// at a time.
  loomc_task_t* next;
};

/// Initializes a caller-allocated task base.
///
/// @param vtable Static callback table that outlives `out_task`.
/// @param out_task Task base to initialize.
///
/// Both arguments and all vtable callbacks are required. The initialized task
/// must be submitted or destroyed exactly once before its storage is reused.
LOOMC_API_EXPORT void loomc_task_initialize(const loomc_task_vtable_t* vtable,
                                            loomc_task_t* out_task);

/// Executes and then destroys an accepted task.
///
/// @param task Task ownership reference to consume.
/// @param worker_ordinal Dense mutually exclusive execution-lane ordinal.
///
/// Queue implementations call this exactly once for each task they accept.
/// The vtable is captured before execution so the destroy callback may reclaim
/// task storage.
LOOMC_API_EXPORT void loomc_task_execute(loomc_task_t* task,
                                         loomc_host_size_t worker_ordinal);

/// Destroys a task without executing it.
///
/// @param task Task ownership reference to consume.
///
/// Callers use this after a rejected submission they choose not to retry.
LOOMC_API_EXPORT void loomc_task_destroy(loomc_task_t* task);

/// Accepts ownership of one generic task.
///
/// @param user_data Sink-owned state.
/// @param task Task offered for submission.
/// @return OK after accepting ownership, or a non-OK status without accepting
/// ownership.
///
/// An accepted task is eventually executed exactly once and then destroyed
/// exactly once, possibly before this callback returns. A rejected task is not
/// executed or destroyed by the sink.
typedef loomc_status_t(LOOMC_API_PTR* loomc_task_submit_fn_t)(
    void* user_data, loomc_task_t* task);

/// Borrowed destination for generic tasks.
typedef struct loomc_task_sink_t {
  /// Callback accepting one task.
  loomc_task_submit_fn_t submit;

  /// Opaque state passed to `submit`.
  void* user_data;
} loomc_task_sink_t;

/// Submits a task and transfers ownership on success.
///
/// @param sink Borrowed task destination.
/// @param task Task ownership reference offered to `sink`.
/// @return The sink submission status.
///
/// This validates the public protocol before entering the sink callback and
/// never accesses `task` after an OK return because an inline sink may already
/// have destroyed it.
LOOMC_API_EXPORT loomc_status_t loomc_task_sink_submit(loomc_task_sink_t sink,
                                                       loomc_task_t* task);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TASK_H_
