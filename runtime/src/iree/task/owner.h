// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TASK_OWNER_H_
#define IREE_TASK_OWNER_H_

#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/task/executor.h"

#ifdef __cplusplus
extern "C" {
#endif

// One bounded, exclusive compute-process turn. The application owns its status
// and retirement policy: a failed operation may still have accepted borrowers.
// completed is published only after wake sources detach and those borrowers
// retire. Neither this callback nor external calls wait or reenter the owner.
typedef void (*iree_task_owner_advance_fn_t)(
    void* user_data, const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result);

// An externally enterable serialized process using an existing executor. The
// application keeps this storage, executor, and callback state alive through
// final process release and every external call. No worker or message queue is
// allocated. Fields are implementation storage, not caller synchronization.
typedef struct iree_task_owner_t {
  // Cooperative compute process; user_data is the application callback payload.
  iree_task_process_t process;
  // Borrowed executor, outliving process release and external calls.
  iree_task_executor_t* executor;
  // Immutable application progress callback.
  iree_task_owner_advance_fn_t advance;
  // Serializes worker turns and short external calls. Workers only try-lock.
  iree_slim_mutex_t mutex;
  // Completion observed under mutex, closing the gap before executor
  // retirement.
  bool completed;
  // Joins the final release callback independently of logical completion.
  struct {
    // Protects released and the release callback's final notification access.
    iree_slim_mutex_t mutex;
    // Application-side join; executor workers never wait here.
    iree_notification_t notification;
    // The executor has relinquished all process/drain storage.
    bool released;
  } retirement;
} iree_task_owner_t;

// Initializes inert storage. Install wake sources through call, or before any
// activation, then wake. A call itself schedules a turn after releasing access.
void iree_task_owner_initialize(iree_task_executor_t* executor,
                                iree_task_owner_advance_fn_t advance,
                                void* user_data, iree_task_owner_t* out_owner);

// Releases synchronization storage after await and all external calls/wakes
// have joined, or before the owner has ever been activated.
void iree_task_owner_deinitialize(iree_task_owner_t* owner);

// Thread-safe nonblocking activation. A late wake of a completed process is
// harmless while owner storage is still alive; it does not retain that storage.
void iree_task_owner_wake(iree_task_owner_t* owner);

// Runs a short callback on the calling thread under the same exclusion as
// advance. May wait for the current turn, never for asynchronous work. The
// callback owns any result/status in its payload. It must not perform blocking
// I/O, wait, reenter, or destroy the owner. Already-owned code calls its typed
// methods directly. Unlock schedules a turn even after partial work or error,
// closing a wake lost when a worker could not enter during this call.
void iree_task_owner_call(iree_task_owner_t* owner, void (*call)(void*),
                          void* user_data);

// Joins final process release, not just logical completion. Only an external
// application thread waits here. No timeout grants permission to free a live
// owner; its application must first request and finish actual retirement.
void iree_task_owner_await(iree_task_owner_t* owner);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TASK_OWNER_H_
