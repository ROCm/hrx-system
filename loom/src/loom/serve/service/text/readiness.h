// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_READINESS_H_
#define LOOM_SERVE_SERVICE_TEXT_READINESS_H_

#include "iree/async/operations/semaphore.h"
#include "iree/base/threading/mutex.h"
#include "iree/task/executor.h"
#include "loom/runtime/wake.h"
#include "loom/serve/runtime/execution/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// One finite CPU borrower. Return transfers its outcome to the serialized
// owner after the task process releases all borrowed application storage.
typedef iree_status_t (*loom_serve_text_cpu_work_fn_t)(void* user_data);

typedef struct loom_serve_text_readiness_result_t {
  // Owned callback outcome, consumed by the serialized service owner.
  iree_status_t status;
  // Publication is complete, including the producer's final borrowed access.
  bool ready;
} loom_serve_text_readiness_result_t;

// Paired publication channels for one finite CPU job and one model frontier.
// The service owns operation kinds and consumes CPU release before mutating
// model/VM state, even when device readiness arrived earlier. Scheduling and
// take operations run on the serialized owner; callbacks only publish/wake.
typedef struct loom_serve_text_readiness_t {
  // Joins producer publication with owner consumption and observer detachment.
  iree_slim_mutex_t mutex;
  // Borrowed nonblocking owner wake, alive through accepted callback release.
  loom_runtime_wake_callback_t observer;
  // One accepted device wait, independent of the finite CPU borrower.
  struct {
    // Borrowed shared proactor, outliving this publication channel.
    iree_async_proactor_t* proactor;
    // Reused only after the previous callback outcome is consumed.
    iree_async_semaphore_wait_operation_t operation;
    // Borrowed toll-free model work and feedback semaphore handles.
    iree_async_semaphore_t* semaphores[2];
    // Captured values independent of later model submissions.
    uint64_t values[2];
    // Callback-published outcome protected by mutex.
    loom_serve_text_readiness_result_t result;
  } device;
  // One task process, reused after the owner consumes its released outcome.
  struct {
    // Borrowed shared host workers, outliving this publication channel.
    iree_task_executor_t* executor;
    // Finite process whose release publishes readiness, not just completion.
    iree_task_process_t process;
    // Borrowed work function installed before scheduling the process.
    loom_serve_text_cpu_work_fn_t run;
    // Borrowed application state, released before readiness is published.
    void* user_data;
    // Completion stores status; release publishes ready under mutex.
    loom_serve_text_readiness_result_t result;
  } cpu;
} loom_serve_text_readiness_t;

void loom_serve_text_readiness_initialize(
    iree_task_executor_t* executor, iree_async_proactor_t* proactor,
    loom_serve_text_readiness_t* readiness);

// All accepted CPU/device results have been taken before destruction.
void loom_serve_text_readiness_deinitialize(
    loom_serve_text_readiness_t* readiness);

// Installs the observer before scheduling; detachment follows consumption of
// all accepted callbacks and joins their final observer access under mutex.
void loom_serve_text_readiness_set_observer(
    loom_serve_text_readiness_t* readiness,
    loom_runtime_wake_callback_t observer);

// Makes the owner runnable without submitting another CPU/device operation.
void loom_serve_text_readiness_notify(loom_serve_text_readiness_t* readiness);

// Accepts one ALL wait. Failure owes no callback; success retains the borrowed
// frontier semaphores until take_device consumes its terminal outcome.
iree_status_t loom_serve_text_readiness_await(
    loom_serve_text_readiness_t* readiness,
    const loom_serve_execution_frontier_t* frontier);

// Runs one finite borrower on shared workers. A prior CPU job has been taken.
void loom_serve_text_readiness_schedule(loom_serve_text_readiness_t* readiness,
                                        loom_serve_text_cpu_work_fn_t run,
                                        void* user_data);

// False leaves out_status unchanged. True transfers the owned terminal result
// and permits reuse of that channel; it does not consume the other channel.
bool loom_serve_text_readiness_take_cpu(loom_serve_text_readiness_t* readiness,
                                        iree_status_t* out_status);
bool loom_serve_text_readiness_take_device(
    loom_serve_text_readiness_t* readiness, iree_status_t* out_status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_READINESS_H_
