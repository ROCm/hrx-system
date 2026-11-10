// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/input_count.h"

#include <string.h>

#include "loom/service/telemetry/record.h"

iree_status_t loom_serve_text_input_count_initialize(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    loom_serve_text_input_count_t* out_queue) {
  *out_queue = (loom_serve_text_input_count_t){.host_allocator = host_allocator,
                                               .capacity = capacity};
  return iree_allocator_malloc_array(host_allocator, capacity,
                                     sizeof(*out_queue->requests),
                                     (void**)&out_queue->requests);
}

void loom_serve_text_input_count_deinitialize(
    loom_serve_text_input_count_t* queue) {
  iree_allocator_free(queue->host_allocator, queue->requests);
  memset(queue, 0, sizeof(*queue));
}

static void loom_serve_text_input_count_remove(
    loom_serve_text_input_count_t* queue, iree_host_size_t index) {
  --queue->count;
  memmove(queue->requests + index, queue->requests + index + 1,
          (queue->count - index) * sizeof(*queue->requests));
}

void loom_serve_text_input_count_enqueue(
    loom_serve_text_input_count_t* queue, iree_string_view_t body,
    loom_serve_text_input_count_consumer_t consumer,
    loom_serve_text_receipt_t receipt) {
  queue->requests[queue->count++] = (loom_serve_text_input_count_request_t){
      .consumer = consumer, .body = body, .receipt = receipt};
}

iree_status_t loom_serve_text_input_count_run(
    loom_serve_text_input_count_t* queue,
    const loom_serve_text_chat_policy_t* policy, iree_string_view_t name,
    iree_host_size_t default_max_tokens, iree_host_size_t context_capacity) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, name.data, name.size);
  const loom_serve_text_input_count_request_t request = queue->requests[0];
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, request.receipt.serial);
  queue->input_tokens = 0;
  loom_serve_text_chat_t chat = {0};
  iree_status_t status = loom_serve_text_chat_initialize(
      policy, name, request.body, request.receipt.serial, default_max_tokens,
      queue->host_allocator, &chat);
  loom_serve_text_media_span_t* media_spans = NULL;
  iree_host_size_t media_span_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_prepare_input(
        &chat, 0, LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,
        LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH, context_capacity, 0,
        queue->host_allocator, NULL, &queue->input_tokens, &media_spans,
        &media_span_count);
  }
  iree_allocator_free(queue->host_allocator, media_spans);
  loom_serve_text_chat_deinitialize(&chat);
  IREE_RETURN_AND_END_ZONE(z0, status);
}

iree_status_t loom_serve_text_input_count_complete(
    loom_serve_text_input_count_t* queue, bool stopping, iree_status_t status,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch) {
  const loom_serve_text_input_count_request_t request = queue->requests[0];
  loom_serve_text_input_count_remove(queue, 0);
  const bool request_error = iree_status_is_invalid_argument(status) ||
                             iree_status_is_out_of_range(status) ||
                             iree_status_is_unimplemented(status) ||
                             iree_status_is_resource_exhausted(status);
  const loom_serve_text_input_count_consumer_t consumer = request.consumer;
  if (!iree_status_is_ok(status) && !request_error) {
    // A closed caller cannot hide a source failure from the model owner.
    consumer.vtable->abort(consumer.self, LOOM_SERVICE_TELEMETRY_FAILED, status,
                           telemetry, request.receipt);
    return status;
  }
  if (stopping || !consumer.vtable->is_open(consumer.self)) {
    const bool failed = !iree_status_is_ok(telemetry->failure);
    consumer.vtable->abort(consumer.self,
                           failed ? LOOM_SERVICE_TELEMETRY_FAILED
                                  : LOOM_SERVICE_TELEMETRY_CANCELLED,
                           failed ? telemetry->failure : status, telemetry,
                           request.receipt);
    iree_status_free(status);
    return iree_ok_status();
  }
  iree_status_t delivery_status =
      consumer.vtable->end(consumer.self, queue->input_tokens, status,
                           telemetry, request.receipt, scratch);
  iree_status_free(status);
  return delivery_status;
}

bool loom_serve_text_input_count_cancel(
    loom_serve_text_input_count_t* queue, bool borrowed, bool stopping,
    loom_serve_text_telemetry_t* telemetry) {
  bool progress = false;
  for (iree_host_size_t i = borrowed ? 1 : 0; i < queue->count;) {
    const loom_serve_text_input_count_request_t request = queue->requests[i];
    if (stopping || !request.consumer.vtable->is_open(request.consumer.self)) {
      request.consumer.vtable->abort(request.consumer.self,
                                     !iree_status_is_ok(telemetry->failure)
                                         ? LOOM_SERVICE_TELEMETRY_FAILED
                                         : LOOM_SERVICE_TELEMETRY_CANCELLED,
                                     telemetry->failure, telemetry,
                                     request.receipt);
      loom_serve_text_input_count_remove(queue, i);
      progress = true;
    } else {
      ++i;
    }
  }
  return progress;
}

struct loom_serve_text_input_count_call_t {
  // Allocator owning this receipt; input retains its original allocator.
  iree_allocator_t host_allocator;
  // Immutable input ownership consumed only after the worker loan retires.
  iree_string_builder_t body;
  // Caller ownership and monotonic interest, consumed by close.
  bool caller_open;
  // Accepted service ownership, consumed after input retirement.
  bool service_open;
  // Owned terminal failure, independent of service/model lifetime.
  iree_status_t status;
  // Scalar successful result, published with service detachment.
  iree_host_size_t input_tokens;
};

static void loom_serve_text_input_count_call_release(
    loom_serve_text_input_count_call_t* call) {
  if (!call->caller_open && !call->service_open) {
    iree_status_free(call->status);
    iree_allocator_free(call->host_allocator, call);
  }
}

static bool loom_serve_text_input_count_call_open(void* self) {
  return ((const loom_serve_text_input_count_call_t*)self)->caller_open;
}

static void loom_serve_text_input_count_call_retire(
    void* self, loom_service_telemetry_outcome_t outcome,
    const iree_status_t status, loom_serve_text_telemetry_t* telemetry,
    loom_serve_text_receipt_t receipt) {
  loom_serve_text_input_count_call_t* call = self;
  loom_serve_text_telemetry_end(telemetry, receipt, outcome, 0, status);
  call->status = outcome == LOOM_SERVICE_TELEMETRY_CANCELLED
                     ? iree_status_from_code(IREE_STATUS_CANCELLED)
                     : iree_status_clone(status);
  iree_string_builder_deinitialize(&call->body);
  call->service_open = false;
  loom_serve_text_input_count_call_release(call);
}

static iree_status_t loom_serve_text_input_count_call_end(
    void* self, iree_host_size_t input_tokens, const iree_status_t status,
    loom_serve_text_telemetry_t* telemetry, loom_serve_text_receipt_t receipt,
    iree_string_builder_t* scratch) {
  loom_serve_text_input_count_call_t* call = self;
  if (iree_status_is_ok(status)) {
    call->input_tokens = input_tokens;
  }
  loom_serve_text_input_count_call_retire(self,
                                          iree_status_is_ok(status)
                                              ? LOOM_SERVICE_TELEMETRY_COMPLETED
                                              : LOOM_SERVICE_TELEMETRY_REJECTED,
                                          status, telemetry, receipt);
  return iree_ok_status();
}

iree_status_t loom_serve_text_input_count_call_allocate(
    iree_string_builder_t* body, iree_allocator_t host_allocator,
    loom_serve_text_input_count_call_t** out_call,
    loom_serve_text_input_count_consumer_t* out_consumer) {
  *out_call = NULL;
  *out_consumer = (loom_serve_text_input_count_consumer_t){0};
  if (iree_allocator_is_null(body->allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native input counting requires an owning builder");
  }
  loom_serve_text_input_count_call_t* call = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*call), (void**)&call));
  call->host_allocator = host_allocator;
  call->body = *body;
  call->caller_open = true;
  call->service_open = true;
  iree_string_builder_initialize(body->allocator, body);
  static const loom_serve_text_input_count_consumer_vtable_t vtable = {
      .is_open = loom_serve_text_input_count_call_open,
      .end = loom_serve_text_input_count_call_end,
      .abort = loom_serve_text_input_count_call_retire};
  *out_call = call;
  *out_consumer =
      (loom_serve_text_input_count_consumer_t){.self = call, .vtable = &vtable};
  return iree_ok_status();
}

iree_string_view_t loom_serve_text_input_count_call_body(
    const loom_serve_text_input_count_call_t* call) {
  return iree_string_builder_view(&call->body);
}

bool loom_serve_text_input_count_call_is_complete(
    const loom_serve_text_input_count_call_t* call) {
  return !call->service_open;
}

iree_status_t loom_serve_text_input_count_call_status(
    const loom_serve_text_input_count_call_t* call) {
  return iree_status_clone(call->status);
}

iree_host_size_t loom_serve_text_input_count_call_result(
    const loom_serve_text_input_count_call_t* call) {
  return call->input_tokens;
}

void loom_serve_text_input_count_call_close(
    loom_serve_text_input_count_call_t* call) {
  if (call) {
    call->caller_open = false;
    loom_serve_text_input_count_call_release(call);
  }
}
