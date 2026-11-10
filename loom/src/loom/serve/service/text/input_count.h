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

// Owner-only delivery after the counting worker releases its input borrow.
// No callback submits model work or waits for the consumer.
typedef struct loom_serve_text_input_count_consumer_vtable_t {
  // Monotonic caller interest; closing cannot revoke an accepted CPU borrow.
  bool (*is_open)(void* self);
  // Consumes a completed or locally rejected receipt. Status is borrowed; OK
  // carries input_tokens. Scratch is available only for this call. Delivery
  // owns final accounting, including a peer lost during send. Only response
  // allocation may fail; receipt ownership is consumed on every return.
  iree_status_t (*end)(void* self, iree_host_size_t input_tokens,
                       const iree_status_t status,
                       loom_serve_text_telemetry_t* telemetry,
                       loom_serve_text_receipt_t receipt,
                       iree_string_builder_t* scratch);
  // Consumes cancelled or failed delivery without allocating a response.
  void (*abort)(void* self, loom_service_telemetry_outcome_t outcome,
                const iree_status_t status,
                loom_serve_text_telemetry_t* telemetry,
                loom_serve_text_receipt_t receipt);
} loom_serve_text_input_count_consumer_vtable_t;

typedef struct loom_serve_text_input_count_consumer_t {
  // Owned receipt retaining the request body through end or abort.
  void* self;
  // Immutable process-lifetime operations shared by all receipts of this kind.
  const loom_serve_text_input_count_consumer_vtable_t* vtable;
} loom_serve_text_input_count_consumer_t;

// A stateless input query owns only its delivery receipt, never a session, KV
// reservation, parameter pin, or projected media. The receipt retains the body.
typedef struct loom_serve_text_input_count_request_t {
  // Delivery relinquished once after all accepted input borrowers retire.
  loom_serve_text_input_count_consumer_t consumer;
  // Immutable JSON borrowing the receipt's request storage.
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

// Transfers a receipt into an available queue slot. The owner has checked
// capacity; input validation runs on the existing CPU loan after acceptance.
void loom_serve_text_input_count_enqueue(
    loom_serve_text_input_count_t* queue, iree_string_view_t body,
    loom_serve_text_input_count_consumer_t consumer,
    loom_serve_text_receipt_t receipt);

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

// Independent caller/service ownership of one native count and its result.
// All operations use the serialized service owner. Closing a pending call
// cancels interest; accepted input borrowers retain the body until retirement.
typedef struct loom_serve_text_input_count_call_t
    loom_serve_text_input_count_call_t;

// Allocates a delivery receipt and moves the owning body only on success.
// The service has checked admission capacity before creating the call. Fixed
// borrowed and sizing-only builders cannot transfer ownership.
iree_status_t loom_serve_text_input_count_call_allocate(
    iree_string_builder_t* body, iree_allocator_t host_allocator,
    loom_serve_text_input_count_call_t** out_call,
    loom_serve_text_input_count_consumer_t* out_consumer);

// Immutable input borrowed until the accepted delivery ends or aborts.
iree_string_view_t loom_serve_text_input_count_call_body(
    const loom_serve_text_input_count_call_t* call);

// Completion follows input retirement. Status returns an owned copy; OK while
// pending does not imply success. Result is valid after successful completion
// and survives service/model removal until the caller closes the call.
bool loom_serve_text_input_count_call_is_complete(
    const loom_serve_text_input_count_call_t* call);
iree_status_t loom_serve_text_input_count_call_status(
    const loom_serve_text_input_count_call_t* call);
iree_host_size_t loom_serve_text_input_count_call_result(
    const loom_serve_text_input_count_call_t* call);

// Consumes caller ownership; accepted work keeps its independent input loan.
void loom_serve_text_input_count_call_close(
    loom_serve_text_input_count_call_t* call);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_INPUT_COUNT_H_
