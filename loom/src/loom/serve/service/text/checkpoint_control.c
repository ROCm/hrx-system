// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/checkpoint_control.h"

#include <string.h>

#include "loom/serve/service/text/session.h"
#include "loom/serve/service/text/telemetry.h"

struct loom_serve_text_checkpoint_control_t {
  // Allocator owning this receipt and its trailing identifier bytes.
  iree_allocator_t host_allocator;
  // Copied operation with identifiers borrowing this allocation's tail.
  loom_serve_text_checkpoint_request_t request;
  // Caller interest and ownership, consumed only by close.
  bool caller_open;
  // Accepted operation ownership, consumed after every borrower retires.
  bool service_open;
  // Owned terminal failure, independent of registration lifetime.
  iree_status_t status;
  // Copied successful result, published with service detachment.
  loom_serve_text_checkpoint_result_t result;
};

iree_status_t loom_serve_text_checkpoint_request_validate(
    const loom_serve_text_checkpoint_request_t* request) {
  if ((unsigned)request->action > LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported checkpoint operation");
  }
  if (!request->name.size ||
      !loom_serve_text_session_name_valid(request->name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "checkpoint name must be 1-64 ASCII identifier "
                            "characters");
  }
  if (request->action == LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN) {
    if (!request->session.size ||
        !loom_serve_text_session_name_valid(request->session)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pin requires a nonempty session identifier");
    }
  } else if (request->session.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "only pin accepts a source session");
  }
  return iree_ok_status();
}

static void loom_serve_text_checkpoint_control_release(
    loom_serve_text_checkpoint_control_t* control) {
  if (!control->caller_open && !control->service_open) {
    iree_status_free(control->status);
    iree_allocator_free(control->host_allocator, control);
  }
}

static bool loom_serve_text_checkpoint_control_open(void* self) {
  const loom_serve_text_checkpoint_control_t* control = self;
  return control->caller_open;
}

static void loom_serve_text_checkpoint_control_retire(
    void* self, loom_serve_telemetry_outcome_t outcome,
    const iree_status_t status, loom_serve_text_telemetry_t* telemetry,
    const loom_serve_text_receipt_t* receipt) {
  loom_serve_text_checkpoint_control_t* control = self;
  loom_serve_text_telemetry_end(telemetry, *receipt, outcome, 0, status);
  control->status = iree_status_clone(status);
  control->service_open = false;
  loom_serve_text_checkpoint_control_release(control);
}

static iree_status_t loom_serve_text_checkpoint_control_end(
    void* self, const loom_serve_text_checkpoint_request_t* request,
    const loom_serve_text_checkpoint_result_t* result,
    const iree_status_t failure, loom_serve_text_telemetry_t* telemetry,
    const loom_serve_text_receipt_t* receipt, iree_string_builder_t* scratch) {
  loom_serve_text_checkpoint_control_t* control = self;
  if (result) {
    control->result = *result;
  }
  loom_serve_text_checkpoint_control_retire(
      self,
      result ? LOOM_SERVE_TELEMETRY_COMPLETED : LOOM_SERVE_TELEMETRY_REJECTED,
      failure, telemetry, receipt);
  return iree_ok_status();
}

iree_status_t loom_serve_text_checkpoint_control_allocate(
    const loom_serve_text_checkpoint_request_t* request,
    iree_allocator_t host_allocator,
    loom_serve_text_checkpoint_control_t** out_control,
    loom_serve_text_checkpoint_consumer_t* out_consumer) {
  *out_control = NULL;
  *out_consumer = (loom_serve_text_checkpoint_consumer_t){0};
  loom_serve_text_checkpoint_control_t* control = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator,
      sizeof(*control) + request->name.size + request->session.size,
      (void**)&control));
  control->host_allocator = host_allocator;
  control->caller_open = true;
  control->service_open = true;
  char* storage = (char*)(control + 1);
  memcpy(storage, request->name.data, request->name.size);
  if (request->session.size) {
    memcpy(storage + request->name.size, request->session.data,
           request->session.size);
  }
  control->request = (loom_serve_text_checkpoint_request_t){
      .action = request->action,
      .name = iree_make_string_view(storage, request->name.size),
      .session = iree_make_string_view(storage + request->name.size,
                                       request->session.size)};
  *out_control = control;
  *out_consumer = (loom_serve_text_checkpoint_consumer_t){
      .self = control,
      .is_open = loom_serve_text_checkpoint_control_open,
      .end = loom_serve_text_checkpoint_control_end,
      .abort = loom_serve_text_checkpoint_control_retire};
  return iree_ok_status();
}

const loom_serve_text_checkpoint_request_t*
loom_serve_text_checkpoint_control_request(
    const loom_serve_text_checkpoint_control_t* control) {
  return &control->request;
}

bool loom_serve_text_checkpoint_control_is_complete(
    const loom_serve_text_checkpoint_control_t* control) {
  return !control->service_open;
}

iree_status_t loom_serve_text_checkpoint_control_status(
    const loom_serve_text_checkpoint_control_t* control) {
  return iree_status_clone(control->status);
}

const loom_serve_text_checkpoint_result_t*
loom_serve_text_checkpoint_control_result(
    const loom_serve_text_checkpoint_control_t* control) {
  return !control->service_open && iree_status_is_ok(control->status)
             ? &control->result
             : NULL;
}

void loom_serve_text_checkpoint_control_close(
    loom_serve_text_checkpoint_control_t* control) {
  if (control) {
    control->caller_open = false;
    loom_serve_text_checkpoint_control_release(control);
  }
}
