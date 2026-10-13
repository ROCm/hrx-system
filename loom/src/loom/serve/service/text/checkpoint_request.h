// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_CHECKPOINT_REQUEST_H_
#define LOOM_SERVE_SERVICE_TEXT_CHECKPOINT_REQUEST_H_

#include "loom/serve/service/text/checkpoint_registry.h"
#include "loom/serve/service/text/session.h"
#include "loom/serve/service/text/telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_text_checkpoint_action_e {
  LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
  LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND,
  LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
} loom_serve_text_checkpoint_action_t;

// One control claim owned by the serialized service owner. A queued request
// owns only its HTTP storage. After submission, checkpoint borrows a registry
// entry through exact device readiness; even peer failure cannot retire that
// borrow early. No callback mutates this record.
typedef struct loom_serve_text_checkpoint_request_t {
  // Owned HTTP claim; NULL when the request has been relinquished.
  loom_serve_http_connection_t* connection;
  // Borrowed immutable request storage, retained by connection.
  const loom_serve_http_request_t* request;
  // Original observation identity, retained across queueing and submission.
  loom_serve_text_receipt_t receipt;
  // Validated operation, meaningful after successful parse.
  loom_serve_text_checkpoint_action_t action;
  // Target suffix before parse; validated identifier afterward. Borrows claim.
  iree_string_view_t name;
  // Registry entry borrowed by accepted device work; NULL before submission.
  loom_serve_text_checkpoint_entry_t* checkpoint;
} loom_serve_text_checkpoint_request_t;

// Resolved source facts from the service's live row/cache/admission owners.
// These are borrowed only during submit. A queued generation can make an idle
// row busy; the checkpoint handler has no access to that admission queue.
typedef struct loom_serve_text_checkpoint_source_t {
  // Matching live session, or NULL when no row retains the requested history.
  const loom_serve_text_session_t* session;
  // Matching detached immutable endpoint, or NULL when none is retained.
  const loom_serve_text_checkpoint_entry_t* cached;
  // An active or queued generation currently borrows the named session.
  bool busy;
} loom_serve_text_checkpoint_source_t;

// Validates external operation/body/identifiers at an unborrowed model
// boundary. name initially contains the suffix after /v1/checkpoints/. Returns
// the source session identifier for pins, empty for other actions. Failure
// leaves the claim owned by the caller for rejection; no registry/model state
// is touched.
iree_status_t loom_serve_text_checkpoint_request_parse(
    loom_serve_text_checkpoint_request_t* control,
    iree_string_view_t* out_session_name);

// Handles a parsed control using source facts resolved by the service owner.
// Immediate replies/rejections relinquish and clear control. An accepted pin
// or suspension leaves checkpoint non-NULL and returns its exact frontier;
// the caller registers readiness and excludes model mutation until completion.
// Sets trim_pending when a release makes storage reclaimable; never clears it.
// Scratch, telemetry and registry remain caller-owned. No readiness wait
// occurs.
iree_status_t loom_serve_text_checkpoint_request_submit(
    loom_serve_text_checkpoint_request_t* control,
    loom_serve_text_checkpoint_registry_t* registry,
    loom_serve_text_checkpoint_source_t source, bool suspension_supported,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch,
    loom_serve_execution_frontier_t* out_frontier, bool* trim_pending,
    iree_allocator_t host_allocator);

// Publishes successfully completed native state before replying or cancelling
// the HTTP claim. Suspension makes storage reclaimable. Called exactly once
// after successful ALL readiness, including when the peer has disconnected.
iree_status_t loom_serve_text_checkpoint_request_complete(
    loom_serve_text_checkpoint_request_t* control, bool stopping,
    loom_serve_text_telemetry_t* telemetry, iree_string_builder_t* scratch,
    bool* trim_pending);

// Relinquishes a queued or retired control claim with terminal accounting.
// Accepted work must have retired first. The registry/model, not this HTTP
// owner, retains native storage after failed device readiness.
void loom_serve_text_checkpoint_request_abort(
    loom_serve_text_checkpoint_request_t* control,
    loom_serve_text_telemetry_t* telemetry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_CHECKPOINT_REQUEST_H_
