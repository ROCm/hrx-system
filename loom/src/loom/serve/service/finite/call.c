// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/call.h"

struct loom_serve_finite_call_t {
  // Allocator owning this receipt.
  iree_allocator_t host_allocator;
  // Moved source input, alive through the last service borrower.
  iree_string_builder_t body;
  // Caller interest and ownership, consumed by close.
  bool caller_open;
  // Execution ownership, consumed only after the final CPU/source borrower.
  bool service_open;
  // Optional nonblocking caller notification, invoked on the owner.
  loom_serve_wake_callback_t observer;
  // Owned final status, independent of the model's lifetime.
  iree_status_t status;
  // Retained device output, independent of source and command storage.
  loom_serve_finite_call_output_t output;
};

static void loom_serve_finite_call_release(loom_serve_finite_call_t* call) {
  if (!call->caller_open && !call->service_open) {
    loom_serve_finite_result_release(call->output.result);
    iree_status_free(call->status);
    iree_allocator_free(call->host_allocator, call);
  }
}

static bool loom_serve_finite_call_open(void* self) {
  return ((loom_serve_finite_call_t*)self)->caller_open;
}

static void loom_serve_finite_call_publish(void* self,
                                           loom_serve_finite_result_t* result,
                                           iree_host_size_t request_index,
                                           iree_host_size_t row_count) {
  loom_serve_finite_call_t* call = self;
  loom_serve_finite_result_retain(result);
  call->output = (loom_serve_finite_call_output_t){
      .result = result,
      .request_index = request_index,
      .row_count = row_count,
  };
  if (call->observer.wake) {
    call->observer.wake(call->observer.user_data);
  }
}

static iree_status_t loom_serve_finite_call_end(
    void* self, const loom_serve_finite_protocol_t* protocol,
    const iree_status_t status, bool rejected, iree_string_builder_t* output,
    bool* out_delivered) {
  loom_serve_finite_call_t* call = self;
  *out_delivered = call->caller_open;
  call->status = iree_status_clone(status);
  call->service_open = false;
  iree_string_builder_deinitialize(&call->body);
  if (call->observer.wake) {
    call->observer.wake(call->observer.user_data);
  }
  loom_serve_finite_call_release(call);
  return iree_ok_status();
}

iree_status_t loom_serve_finite_call_allocate(
    iree_string_builder_t* body, iree_allocator_t host_allocator,
    loom_serve_finite_call_t** out_call,
    loom_serve_finite_consumer_t* out_consumer) {
  *out_call = NULL;
  *out_consumer = (loom_serve_finite_consumer_t){0};
  if (iree_allocator_is_null(body->allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native finite input requires an owning builder");
  }
  loom_serve_finite_call_t* call = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*call), (void**)&call));
  call->host_allocator = host_allocator;
  call->body = *body;
  iree_string_builder_initialize(body->allocator, body);
  call->caller_open = true;
  call->service_open = true;
  static const loom_serve_finite_consumer_vtable_t vtable = {
      .is_open = loom_serve_finite_call_open,
      .publish = loom_serve_finite_call_publish,
      .end = loom_serve_finite_call_end,
  };
  *out_consumer =
      (loom_serve_finite_consumer_t){.self = call, .vtable = &vtable};
  *out_call = call;
  return iree_ok_status();
}

iree_string_view_t loom_serve_finite_call_body(
    const loom_serve_finite_call_t* call) {
  return iree_string_builder_view(&call->body);
}

void loom_serve_finite_call_set_observer(loom_serve_finite_call_t* call,
                                         loom_serve_wake_callback_t observer) {
  call->observer = observer;
  if (observer.wake && (call->output.result || !call->service_open)) {
    observer.wake(observer.user_data);
  }
}

const loom_serve_finite_call_output_t* loom_serve_finite_call_output(
    const loom_serve_finite_call_t* call) {
  return call->output.result ? &call->output : NULL;
}

bool loom_serve_finite_call_is_complete(const loom_serve_finite_call_t* call) {
  return !call->service_open;
}

iree_status_t loom_serve_finite_call_status(
    const loom_serve_finite_call_t* call) {
  return iree_status_clone(call->status);
}

void loom_serve_finite_call_close(loom_serve_finite_call_t* call) {
  if (!call) {
    return;
  }
  call->caller_open = false;
  call->observer = (loom_serve_wake_callback_t){0};
  // No caller can borrow these views after close. Accepted device consumers
  // retain their own captures; only input must await the service's end.
  loom_serve_finite_result_release(call->output.result);
  call->output.result = NULL;
  loom_serve_finite_call_release(call);
}
