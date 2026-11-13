// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_EXECUTION_RECONFIGURATION_H_
#define LOOM_SERVICE_EXECUTION_RECONFIGURATION_H_

#include "loom/service/reconfiguration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Model-specific execution replacement over unchanged preparation, control and
// retained resources. All callbacks are present or the whole capability is
// zero.
typedef struct loom_service_execution_update_t {
  // Borrowed model, alive through every job and candidate destruction.
  void* self;
  // Accepts independent compilation on shared workers. Success owes completion
  // once (possibly before return); rejection owes none. Inputs remain borrowed
  // through completion. Compiling never mutates the model's active execution.
  iree_status_t (*compile)(void* self, const loom_runtime_sources_t* sources,
                           loom_runtime_variant_completion_fn_t completion,
                           void* user_data);
  // Owner-only atomic publication at the service's semantic boundary. NULL
  // restores original execution. Failure keeps the previous execution usable.
  iree_status_t (*select)(void* self, void* value);
  // Destroys an unborrowed candidate on the owner; NULL denotes original code.
  loom_runtime_variant_dispose_fn_t dispose;
} loom_service_execution_update_t;

typedef struct loom_service_execution_reconfiguration_t
    loom_service_execution_reconfiguration_t;

// Bounded choices and preferred-choice bookkeeping. The service determines
// quiescence; the pool itself imposes no active/default choice on consumers.
// Zero capacity or an absent update capability returns NULL.
iree_status_t loom_service_execution_reconfiguration_allocate(
    loom_service_execution_update_t update, iree_host_size_t capacity,
    iree_allocator_t host_allocator,
    loom_service_execution_reconfiguration_t** out_reconfiguration);

// pending() is false before destruction. Candidate references are released
// before model retirement. NULL is accepted by all operations below.
void loom_service_execution_reconfiguration_free(
    loom_service_execution_reconfiguration_t* reconfiguration);
void loom_service_execution_reconfiguration_set_observer(
    loom_service_execution_reconfiguration_t* reconfiguration,
    loom_runtime_wake_callback_t observer);
bool loom_service_execution_reconfiguration_pending(
    const loom_service_execution_reconfiguration_t* reconfiguration);

// Collects compiler results independently of inference. Selects only when the
// service declares quiescence. Stopping cancels selection and joins producers
// without binding anything. Failures stay with management, not inference.
bool loom_service_execution_reconfiguration_advance(
    loom_service_execution_reconfiguration_t* reconfiguration, bool quiescent,
    bool stopping);
loom_service_reconfiguration_t loom_service_execution_reconfiguration_interface(
    loom_service_execution_reconfiguration_t* reconfiguration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_EXECUTION_RECONFIGURATION_H_
