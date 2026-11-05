// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_CALL_H_
#define LOOM_SERVE_SERVICE_FINITE_CALL_H_

#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/service/finite/consumer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_finite_call_t loom_serve_finite_call_t;

// Independently retained output for one request in a packed device cohort.
typedef struct loom_serve_finite_call_output_t {
  // Borrowed until call_close, including after service/model removal. Import
  // result_ready before reading device rows. The device group/pool outlives it.
  loom_serve_finite_result_t* result;
  // Original cohort association, unaffected by cancellation of other requests.
  iree_host_size_t request_index;
  // Number of rows belonging to this request, in its original input order.
  iree_host_size_t row_count;
} loom_serve_finite_call_output_t;

// Admission implementation: moves an owning input builder on success, leaving
// it initialized and empty. Failure preserves input. The caller and service
// have independent ownership; only consumer.end relinquishes service ownership.
iree_status_t loom_serve_finite_call_allocate(
    iree_string_builder_t* body, iree_allocator_t host_allocator,
    loom_serve_finite_call_t** out_call,
    loom_serve_finite_consumer_t* out_consumer);

// Immutable service input, alive through consumer.end.
iree_string_view_t loom_serve_finite_call_body(
    const loom_serve_finite_call_t* call);

// All mutable operations and observations use the service's serialized owner.
// Wake callbacks only notify an external event loop; they never reenter it.
// Installation notices an already-published result or completed call. Clearing
// the observer on the owner joins its last access without another semaphore.
void loom_serve_finite_call_set_observer(loom_serve_finite_call_t* call,
                                         loom_serve_wake_callback_t observer);

// Available after submission, before host readiness. Successful device
// dependencies authorize device access; is_complete separately reports the
// owner's collected request completion. Model removal also joins its accepted
// device captures. A published result remains borrowed until close on failure.
const loom_serve_finite_call_output_t* loom_serve_finite_call_output(
    const loom_serve_finite_call_t* call);
bool loom_serve_finite_call_is_complete(const loom_serve_finite_call_t* call);
// Owned terminal status copy. Query after is_complete; querying does not clear
// it. Actual device capture retirement remains an outer model/group obligation.
iree_status_t loom_serve_finite_call_status(
    const loom_serve_finite_call_t* call);

// Consumes caller ownership and drops interest without waiting. Accepted CPU
// borrowers preserve input through release; queue captures retain independent
// output storage. Device output can outlive service/model removal, but callers
// release it before joining retirement of its enclosing resource group. NULL
// is valid.
void loom_serve_finite_call_close(loom_serve_finite_call_t* call);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_FINITE_CALL_H_
