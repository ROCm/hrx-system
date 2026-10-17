// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_CHECKPOINT_CONTROL_H_
#define LOOM_SERVE_SERVICE_TEXT_CHECKPOINT_CONTROL_H_

#include "loom/serve/telemetry/record.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_serve_text_receipt_t;
struct loom_serve_text_telemetry_t;

typedef enum loom_serve_text_checkpoint_action_e {
  LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
  LOOM_SERVE_TEXT_CHECKPOINT_ACTION_SUSPEND,
  LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
} loom_serve_text_checkpoint_action_t;

// Native admission copies identifiers only on acceptance. Named checkpoint
// ownership is independent of the returned control: closing a control never
// implicitly releases a pin. All calls use the serialized serving owner.
typedef struct loom_serve_text_checkpoint_request_t {
  // Operation over this model's immutable retained history.
  loom_serve_text_checkpoint_action_t action;
  // Nonempty checkpoint identifier, borrowed until admission returns.
  iree_string_view_t name;
  // Nonempty source session for PIN; empty for SUSPEND and RELEASE.
  iree_string_view_t session;
} loom_serve_text_checkpoint_request_t;

// Scalar observation after successful retirement, without a native state
// borrow.
typedef struct loom_serve_text_checkpoint_result_t {
  // Exact consumed token frontier of the pinned, suspended, or released state.
  iree_host_size_t position;
  // Retained host snapshot bytes; zero for a resident or released checkpoint.
  iree_host_size_t host_bytes;
} loom_serve_text_checkpoint_result_t;

typedef struct loom_serve_text_checkpoint_control_t
    loom_serve_text_checkpoint_control_t;

// One service-owned delivery receipt. Its input views stay alive through end or
// abort, including accepted device work after caller close. All operations run
// on the serialized serving owner; there are no CPU borrowers or blocking
// calls.
typedef struct loom_serve_text_checkpoint_consumer_t {
  // Owned recipient, consumed exactly once by end or abort.
  void* self;
  // Monotonic caller interest, independent of accepted native side effects.
  bool (*is_open)(void* self);
  // Consumes successful or locally rejected delivery after retirement. A
  // nonnull result denotes success; otherwise failure is a borrowed rejection.
  // Ends delivery accounting, including a transport peer lost during send.
  // Scratch is borrowed only at this retired model boundary. Only response
  // allocation may fail; ownership is consumed on every return.
  iree_status_t (*end)(void* self,
                       const loom_serve_text_checkpoint_request_t* request,
                       const loom_serve_text_checkpoint_result_t* result,
                       const iree_status_t failure,
                       struct loom_serve_text_telemetry_t* telemetry,
                       const struct loom_serve_text_receipt_t* receipt,
                       iree_string_builder_t* scratch);
  // Consumes a cancelled or failed receipt without allocating a response.
  // Native storage belongs to model retirement, never to this recipient.
  void (*abort)(void* self, loom_serve_telemetry_outcome_t outcome,
                const iree_status_t status,
                struct loom_serve_text_telemetry_t* telemetry,
                const struct loom_serve_text_receipt_t* receipt);
} loom_serve_text_checkpoint_consumer_t;

// Validates external operation and identifiers before service admission. No
// model lookup or allocation occurs; source availability is checked at submit.
iree_status_t loom_serve_text_checkpoint_request_validate(
    const loom_serve_text_checkpoint_request_t* request);

// Allocates independent caller/service owners and copies validated identifiers.
// This does not accept model work. Service admission owns request validation
// and queue capacity before construction; allocation is the only failure here.
iree_status_t loom_serve_text_checkpoint_control_allocate(
    const loom_serve_text_checkpoint_request_t* request,
    iree_allocator_t host_allocator,
    loom_serve_text_checkpoint_control_t** out_control,
    loom_serve_text_checkpoint_consumer_t* out_consumer);

// Immutable copied input. Caller views remain valid until close; accepted
// service views remain valid until the consumer's end or abort.
const loom_serve_text_checkpoint_request_t*
loom_serve_text_checkpoint_control_request(
    const loom_serve_text_checkpoint_control_t* control);

// Completion follows service receipt retirement. Failure is an owned copy; OK
// while pending does not imply success. A result exists only after successful
// completion and remains valid until close, independently of model removal.
bool loom_serve_text_checkpoint_control_is_complete(
    const loom_serve_text_checkpoint_control_t* control);
iree_status_t loom_serve_text_checkpoint_control_status(
    const loom_serve_text_checkpoint_control_t* control);
const loom_serve_text_checkpoint_result_t*
loom_serve_text_checkpoint_control_result(
    const loom_serve_text_checkpoint_control_t* control);

// Consumes caller ownership immediately. Unsubmitted work is cancelled;
// submitted pin/suspend effects still publish after successful readiness.
// The service retains the control through receipt retirement; the model owns
// accepted native storage through physical retirement. No wait or implicit
// checkpoint release occurs, even after successful completion.
void loom_serve_text_checkpoint_control_close(
    loom_serve_text_checkpoint_control_t* control);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_CHECKPOINT_CONTROL_H_
