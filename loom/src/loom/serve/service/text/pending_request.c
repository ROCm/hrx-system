// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/pending_request.h"

#include "loom/serve/service/text/schedule.h"

iree_status_t loom_serve_text_pending_request_prepare(
    loom_serve_text_pending_request_t* request, loom_serve_text_model_t* model,
    loom_serve_text_input_prefix_t prefix,
    const loom_serve_text_pool_usage_t* geometry, int32_t* scratch,
    iree_allocator_t host_allocator) {
  const iree_host_size_t capacity =
      loom_serve_text_model_context_capacity(model);
  if (loom_serve_text_model_mode(model) == LOOM_SERVE_TEXT_MODE_ENCODE) {
    IREE_RETURN_IF_ERROR(loom_serve_text_sequence_prepare(
        &request->sequence, model, request->body, request->receipt.serial,
        prefix, request->checkpoint != NULL, scratch, host_allocator,
        &request->input));
    const iree_host_size_t extent =
        request->input.retained_count + request->input.count;
    const iree_host_size_t reservation =
        iree_host_size_ceil_div(extent, geometry->block_size) *
        geometry->block_size;
    if (geometry->capacity && reservation > geometry->capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "known sequence exceeds shared KV capacity");
    }
    request->reservation = reservation;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_input_prepare(
      &request->chat, prefix, request->checkpoint != NULL, capacity, scratch,
      &request->input, host_allocator));
  const iree_host_size_t reservation = loom_serve_text_request_reservation(
      capacity, request->input.retained_count + request->input.count,
      request->chat.max_tokens, loom_serve_text_model_draft_depth(model),
      geometry->block_size);
  if (geometry->capacity && reservation > geometry->capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "request needs %zu page-rounded KV positions including completion "
        "and speculative credit, but the shared pool holds %zu; compact "
        "history, lower max_tokens, or increase pool_capacity",
        reservation, geometry->capacity);
  }
  request->reservation = reservation;
  return iree_ok_status();
}

// Preserves queue identity and checkpoint accounting through removal. Status
// already produced by optional image work transfers to the response owner.
static iree_status_t loom_serve_text_pending_request_release(
    loom_serve_text_pending_request_t* request,
    iree_allocator_t host_allocator) {
  iree_status_t status =
      loom_serve_text_media_request_take_status(request->media);
  loom_serve_text_media_request_free(request->media);
  request->media = NULL;
  loom_serve_text_chat_deinitialize(&request->chat);
  loom_serve_text_sequence_deinitialize(&request->sequence);
  loom_serve_text_input_deinitialize(&request->input, host_allocator);
  return status;
}

void loom_serve_text_pending_request_cancel(
    loom_serve_text_pending_request_t* request,
    loom_serve_text_telemetry_t* telemetry, iree_allocator_t host_allocator) {
  iree_status_t status =
      loom_serve_text_pending_request_release(request, host_allocator);
  if (!iree_status_is_ok(status)) {
    loom_serve_text_telemetry_diagnose(telemetry, request->receipt.serial,
                                       status);
  }
  const iree_status_code_t failure_code = iree_status_code(telemetry->failure);
  loom_serve_text_telemetry_end(telemetry, request->receipt,
                                failure_code ? LOOM_SERVE_TELEMETRY_FAILED
                                             : LOOM_SERVE_TELEMETRY_CANCELLED,
                                0, telemetry->failure);
  request->consumer.vtable->end(
      request->consumer.self, false,
      failure_code ? failure_code : IREE_STATUS_CANCELLED);
}

iree_status_t loom_serve_text_pending_request_reject(
    loom_serve_text_pending_request_t* request,
    loom_serve_text_telemetry_t* telemetry, iree_status_t failure,
    iree_allocator_t host_allocator) {
  failure = iree_status_join(failure, loom_serve_text_pending_request_release(
                                          request, host_allocator));
  loom_serve_text_telemetry_end(telemetry, request->receipt,
                                LOOM_SERVE_TELEMETRY_REJECTED, 0, failure);
  return request->consumer.vtable->reject(request->consumer.self, failure);
}
