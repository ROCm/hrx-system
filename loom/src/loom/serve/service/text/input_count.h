// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_INPUT_COUNT_H_
#define LOOM_SERVE_SERVICE_TEXT_INPUT_COUNT_H_

#include "loom/serve/service/text/chat.h"
#include "loom/serve/service/text/telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

// A stateless input query owns only its transport claim, never a session, KV
// reservation, parameter pin, or projected media. The claim retains the body.
typedef struct loom_serve_text_input_count_request_t {
  // Claimed connection, relinquished once by completion or cancellation.
  loom_serve_http_connection_t* connection;
  // Immutable JSON borrowing the claimed request storage.
  iree_string_view_t body;
  // Original receipt retained across CPU queueing and completion.
  loom_serve_text_receipt_t receipt;
} loom_serve_text_input_count_request_t;

// The serialized service lends the head to its existing finite CPU channel.
// Appending or removing other entries cannot move the borrowed head. Task
// release precedes completion, cancellation of the head, and destruction.
typedef struct loom_serve_text_input_count_t {
  // Allocator owning the queue and temporary source preparation storage.
  iree_allocator_t host_allocator;
  // Bounded FIFO storage allocated once with the service.
  loom_serve_text_input_count_request_t* requests;
  // Number of retained claims, including an accepted head borrower.
  iree_host_size_t count;
  // Maximum claims, independent of execution rows and KV capacity.
  iree_host_size_t capacity;
  // CPU result consumed only after readiness publishes task release.
  iree_host_size_t input_tokens;
} loom_serve_text_input_count_t;

iree_status_t loom_serve_text_input_count_initialize(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    loom_serve_text_input_count_t* out_queue);

// All claims and accepted CPU borrows have retired.
void loom_serve_text_input_count_deinitialize(
    loom_serve_text_input_count_t* queue);

// The owner has validated the route and JSON content type. Takes the claim,
// including immediate rejection of stateful headers or a full queue.
iree_status_t loom_serve_text_input_count_enqueue(
    loom_serve_text_input_count_t* queue,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request, loom_serve_text_receipt_t receipt,
    loom_serve_text_telemetry_t* telemetry);

// Finite CPU work with the model's exclusive source invocation loan. Counts
// full fresh input, bounded by context_capacity but without output credit.
// No request-local VM, row selection, or image projection is performed.
iree_status_t loom_serve_text_input_count_run(
    loom_serve_text_input_count_t* queue,
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t name,
    iree_host_size_t default_max_tokens, iree_host_size_t context_capacity);

// Consumes the released head's outcome and relinquishes its claim. Telemetry
// counts the utility request, not inference tokens that were never executed.
// Unexpected source failures propagate even after client cancellation or stop.
iree_status_t loom_serve_text_input_count_complete(
    loom_serve_text_input_count_t* queue, bool stopping, iree_status_t status,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch);

// Cancels failed unborrowed claims, or all claims when stopping. The owner
// identifies whether the head has an outstanding CPU loan. Returns progress.
bool loom_serve_text_input_count_cancel(loom_serve_text_input_count_t* queue,
                                        bool borrowed, bool stopping,
                                        loom_serve_text_telemetry_t* telemetry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_INPUT_COUNT_H_
