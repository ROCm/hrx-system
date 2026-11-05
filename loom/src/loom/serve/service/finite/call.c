// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/call.h"

struct loom_serve_finite_call_t {
  // Allocator owning this receipt.
  iree_allocator_t host_allocator;
  // Moved source input through the last service borrower, then moved host
  // response through caller close. Device calls release input at completion.
  iree_string_builder_t storage;
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
    iree_string_builder_deinitialize(&call->storage);
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
  iree_string_builder_deinitialize(&call->storage);
  if (call->caller_open) {
    call->status = iree_status_clone(status);
    if (iree_status_is_ok(status)) {
      call->storage = *output;
      iree_string_builder_initialize(output->allocator, output);
    }
  }
  call->service_open = false;
  if (call->observer.wake) {
    call->observer.wake(call->observer.user_data);
  }
  loom_serve_finite_call_release(call);
  return iree_ok_status();
}

static iree_status_t loom_serve_finite_call_encode(
    void* self, const loom_serve_finite_protocol_t* protocol,
    const loom_serve_finite_request_t* request, iree_string_view_t name,
    const iree_const_byte_span_t* rows, uint64_t input_tokens,
    iree_string_builder_t* output) {
  iree_string_builder_reset(output);
  return protocol->response(request, name, rows, input_tokens, output);
}

iree_status_t loom_serve_finite_call_allocate(
    iree_string_builder_t* body,
    loom_serve_finite_output_format_t output_format,
    iree_allocator_t host_allocator, loom_serve_finite_call_t** out_call,
    loom_serve_finite_consumer_t* out_consumer) {
  *out_call = NULL;
  *out_consumer = (loom_serve_finite_consumer_t){0};
  if (output_format != LOOM_SERVE_FINITE_OUTPUT_DEVICE &&
      output_format != LOOM_SERVE_FINITE_OUTPUT_JSON) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported finite output format");
  }
  if (iree_allocator_is_null(body->allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native finite input requires an owning builder");
  }
  loom_serve_finite_call_t* call = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*call), (void**)&call));
  call->host_allocator = host_allocator;
  call->storage = *body;
  iree_string_builder_initialize(body->allocator, body);
  call->caller_open = true;
  call->service_open = true;
  static const loom_serve_finite_consumer_vtable_t device_vtable = {
      .is_open = loom_serve_finite_call_open,
      .publish = loom_serve_finite_call_publish,
      .end = loom_serve_finite_call_end,
  };
  static const loom_serve_finite_consumer_vtable_t json_vtable = {
      .is_open = loom_serve_finite_call_open,
      .encode = loom_serve_finite_call_encode,
      .end = loom_serve_finite_call_end,
  };
  *out_consumer = (loom_serve_finite_consumer_t){
      .self = call,
      .vtable = output_format == LOOM_SERVE_FINITE_OUTPUT_DEVICE
                    ? &device_vtable
                    : &json_vtable,
  };
  *out_call = call;
  return iree_ok_status();
}

iree_string_view_t loom_serve_finite_call_body(
    const loom_serve_finite_call_t* call) {
  return iree_string_builder_view(&call->storage);
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

iree_string_view_t loom_serve_finite_call_response(
    const loom_serve_finite_call_t* call) {
  return call->service_open ? iree_string_view_empty()
                            : iree_string_builder_view(&call->storage);
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
