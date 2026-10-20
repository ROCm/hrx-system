// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_EXECUTION_RECONFIGURATION_H_
#define LOOM_SERVE_SERVICE_EXECUTION_RECONFIGURATION_H_

#include "loom/serve/runtime/execution/scheduler.h"
#include "loom/serve/service/reconfiguration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Model-specific execution replacement over unchanged preparation, control and
// retained resources. All callbacks are present or the whole capability is
// zero.
typedef struct loom_serve_execution_update_t {
  // Borrowed model, alive through every job and candidate destruction.
  void* self;
  // Accepts independent compilation on shared workers. Success owes completion
  // once (possibly before return); rejection owes none. Inputs remain borrowed
  // through completion. Compiling never mutates the model's active execution.
  iree_status_t (*compile)(void* self, const loom_serve_sources_t* sources,
                           loom_serve_variant_completion_fn_t completion,
                           void* user_data);
  // Owner-only atomic publication at the service's semantic boundary. NULL
  // restores original execution. Failure keeps the previous execution usable.
  iree_status_t (*select)(void* self, void* value);
  // Destroys an unborrowed candidate on the owner; NULL denotes original code.
  loom_serve_variant_dispose_fn_t dispose;
} loom_serve_execution_update_t;

typedef struct loom_serve_execution_reconfiguration_t
    loom_serve_execution_reconfiguration_t;

// Bounded choices and preferred-choice bookkeeping. The service determines
// quiescence; the pool itself imposes no active/default choice on consumers.
// Compilation uses background host admission through actual producer release.
// Zero capacity or an absent update capability returns NULL.
iree_status_t loom_serve_execution_reconfiguration_allocate(
    loom_serve_execution_update_t update,
    loom_serve_execution_scheduler_t* scheduler, iree_host_size_t capacity,
    iree_allocator_t host_allocator,
    loom_serve_execution_reconfiguration_t** out_reconfiguration);

// pending() is false before destruction. Candidate references are released
// before model retirement. NULL is accepted by all operations below.
void loom_serve_execution_reconfiguration_free(
    loom_serve_execution_reconfiguration_t* reconfiguration);
void loom_serve_execution_reconfiguration_set_observer(
    loom_serve_execution_reconfiguration_t* reconfiguration,
    loom_serve_wake_callback_t observer);
bool loom_serve_execution_reconfiguration_pending(
    const loom_serve_execution_reconfiguration_t* reconfiguration);

// Collects compiler results independently of inference. Selects only when the
// service declares quiescence. Stopping cancels unstarted compilation and
// selection, and joins accepted producers without binding anything. Failures
// stay with management, not inference.
bool loom_serve_execution_reconfiguration_advance(
    loom_serve_execution_reconfiguration_t* reconfiguration, bool quiescent,
    bool stopping);
loom_serve_reconfiguration_t loom_serve_execution_reconfiguration_interface(
    loom_serve_execution_reconfiguration_t* reconfiguration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_EXECUTION_RECONFIGURATION_H_
