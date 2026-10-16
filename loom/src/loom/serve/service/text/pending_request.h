// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_PENDING_REQUEST_H_
#define LOOM_SERVE_SERVICE_TEXT_PENDING_REQUEST_H_

#include "loom/serve/service/text/checkpoint_registry.h"
#include "loom/serve/service/text/consumer.h"
#include "loom/serve/service/text/input.h"
#include "loom/serve/service/text/media_projector.h"
#include "loom/serve/service/text/telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

// A queued request owns no device reservation. Only the selected attempt may
// borrow an idle row/cache; body/name/tool views borrow its consumer-owned
// immutable input storage.
typedef struct loom_serve_text_pending_request_t {
  // Input lifetime and output delivery, moved at admission or consumed on end.
  loom_serve_text_consumer_t consumer;
  // Raw JSON retained until source rendering completes.
  iree_string_view_t body;
  // Canonical prompt/tool descriptors; null policy means not yet rendered.
  loom_serve_text_chat_t chat;
  // Compact prepared tokens independent of any reusable execution row.
  loom_serve_text_input_t input;
  // Owned projected image occurrences, transferred to the admitted session.
  loom_serve_text_media_request_t* media;
  // Page-rounded completion requirement for the prepared history.
  iree_host_size_t reservation;
  // Cumulative CPU preparation time, excluding unchanged admission retries.
  iree_duration_t prepare_duration;
  // Validated local session key borrowing immutable consumer input.
  iree_string_view_t name;
  // Borrowed explicit endpoint through admission or queue cancellation.
  loom_serve_text_checkpoint_entry_t* checkpoint;
  // Identity and time retained through rejection, cancellation or admission.
  loom_serve_text_receipt_t receipt;
} loom_serve_text_pending_request_t;

// Tokenizes an already-rendered chat under an exclusive source invocation loan.
// Model configuration and pool geometry are immutable, captured by the owner;
// this call never reads concurrently changing pool usage or acquires device
// pins. The selected prefix remains borrowed until admission completes.
iree_status_t loom_serve_text_pending_request_prepare(
    loom_serve_text_pending_request_t* request, loom_serve_text_model_t* model,
    loom_serve_text_input_prefix_t prefix,
    const loom_serve_text_pool_usage_t* geometry, int32_t* scratch,
    iree_allocator_t host_allocator);

// Terminal request ownership, after CPU borrows retire and projector jobs have
// detached. Both paths release host/image state before the consumer receipt.
// Queue ordering and checkpoint pending counts remain with the pending queue.
void loom_serve_text_pending_request_cancel(
    loom_serve_text_pending_request_t* request,
    loom_serve_text_telemetry_t* telemetry, iree_allocator_t host_allocator);
iree_status_t loom_serve_text_pending_request_reject(
    loom_serve_text_pending_request_t* request,
    loom_serve_text_telemetry_t* telemetry, iree_status_t failure,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_PENDING_REQUEST_H_
