// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_HOST_READINESS_H_
#define LOOM_RUNTIME_HOST_READINESS_H_

#include "iree/async/operations/semaphore.h"
#include "iree/async/proactor.h"
#include "iree/base/threading/mutex.h"
#include "iree/task/executor.h"
#include "loom/runtime/wake.h"

#ifdef __cplusplus
extern "C" {
#endif

// A finite CPU borrower. The result becomes observable only after the task
// process releases the application storage borrowed by this call.
typedef iree_status_t (*loom_runtime_cpu_work_fn_t)(void* user_data);

// One outcome published by a final CPU release or asynchronous completion.
// Only the owning readiness domain may publish or consume these fields.
typedef struct loom_runtime_readiness_result_t {
  // Owned callback outcome, transferred to the serialized consumer.
  iree_status_t status;
  // Publication includes the producer's final borrowed application access.
  bool ready;
} loom_runtime_readiness_result_t;

// Publication domain for independently owned tasks and waits. The serialized
// consumer owns their kinds, payloads, and storage; callbacks only publish and
// wake. An operation can be reused after its result is consumed. The domain
// imposes no operation count or ordering between independent results.
typedef struct loom_runtime_readiness_t {
  // Joins producer publication with consumption and observer detachment.
  iree_slim_mutex_t mutex;
  // Borrowed nonblocking wake, alive through accepted callback release.
  loom_runtime_wake_callback_t observer;
  // Borrowed shared workers, outliving every submitted task.
  iree_task_executor_t* executor;
  // Borrowed shared I/O owner, outliving every submitted wait.
  iree_async_proactor_t* proactor;
} loom_runtime_readiness_t;

// Caller-owned CPU operation, stable through consumption of its result.
typedef struct loom_runtime_readiness_task_t {
  // Finite process whose final release publishes readiness.
  iree_task_process_t process;
  // Work installed before scheduling and borrowed through process release.
  loom_runtime_cpu_work_fn_t run;
  // Application payload borrowed by run until process release.
  void* user_data;
  // Completion stores status; final release publishes readiness.
  loom_runtime_readiness_result_t result;
} loom_runtime_readiness_task_t;

// Caller-owned ALL wait. Dependency arrays remain stable through consumption
// of the result. Dependency count is determined by the submitting producer.
typedef struct loom_runtime_readiness_wait_t {
  // Intrusive operation recycled after its final callback outcome is consumed.
  iree_async_semaphore_wait_operation_t operation;
  // Terminal readiness outcome, independent of other accepted operations.
  loom_runtime_readiness_result_t result;
} loom_runtime_readiness_wait_t;

void loom_runtime_readiness_initialize(iree_task_executor_t* executor,
                                       iree_async_proactor_t* proactor,
                                       loom_runtime_readiness_t* readiness);

// All accepted results have been consumed before domain destruction.
void loom_runtime_readiness_deinitialize(loom_runtime_readiness_t* readiness);

// Installs the observer before scheduling. Detachment follows consumption of
// accepted results and joins the final observer access under the domain lock.
void loom_runtime_readiness_set_observer(loom_runtime_readiness_t* readiness,
                                         loom_runtime_wake_callback_t observer);

// Makes the consumer runnable without accepting another task or wait.
void loom_runtime_readiness_notify(loom_runtime_readiness_t* readiness);

// Runs a finite borrower on shared workers. task starts zero-initialized or
// has had its previous result consumed. Scheduling transfers no payload owner;
// the caller preserves borrowed state until final release is consumed.
void loom_runtime_readiness_schedule(loom_runtime_readiness_t* readiness,
                                     loom_runtime_cpu_work_fn_t run,
                                     void* user_data,
                                     loom_runtime_readiness_task_t* task);

// Accepts one ALL wait over caller-owned dependency arrays. Failure owes no
// callback. Success borrows the arrays and semaphores until the result is
// consumed. wait starts zero-initialized or has had its previous result taken.
// Failed readiness does not imply retirement of the work being observed.
iree_status_t loom_runtime_readiness_await(loom_runtime_readiness_t* readiness,
                                           iree_host_size_t count,
                                           iree_async_semaphore_t** semaphores,
                                           uint64_t* values,
                                           loom_runtime_readiness_wait_t* wait);

// False leaves out_status unchanged. True transfers the terminal outcome and
// authorizes reuse of the operation storage and its borrowed callback state.
// Device resources remain governed by the observed execution's retirement.
// The result belongs to an operation submitted through this domain.
bool loom_runtime_readiness_take(loom_runtime_readiness_t* readiness,
                                 loom_runtime_readiness_result_t* result,
                                 iree_status_t* out_status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_HOST_READINESS_H_
