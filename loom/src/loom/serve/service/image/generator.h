// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_IMAGE_GENERATOR_H_
#define LOOM_SERVE_SERVICE_IMAGE_GENERATOR_H_

#include "iree/tokenizer/tokenizer.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/parameter/residency.h"
#include "loom/serve/service/execution_reconfiguration.h"
#include "loom/serve/service/image/request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copied source-selected invocation facts, valid after successful preparation.
// No pointer into the source VM or its request payload escapes in this value.
typedef struct loom_serve_image_prepared_t {
  // Retained command index selected by the source program.
  uint64_t stage;
  // Source-declared stage tag; its units and interpretation are model-defined.
  int64_t tag;
  // Bytes of opaque prepared input consumed by this invocation.
  uint64_t input_bytes;
} loom_serve_image_prepared_t;

enum loom_serve_image_submit_flag_bits_e {
  // At a retired part boundary, release invocation resources without issuing
  // more model work. The returned terminal frontier still requires retirement.
  LOOM_SERVE_IMAGE_SUBMIT_FLAG_CANCEL = 1u << 0,
};
typedef uint32_t loom_serve_image_submit_flags_t;

// Finite-image invocation with explicit CPU and device phases. The scheduler
// owns admission, submission and successful completion; prepare runs as a CPU
// task with exclusive model access. None of these functions waits for device
// or I/O completion. Source input rejection precedes residency acquisition;
// other failure is terminal and preserves accepted model ownership until
// outer-thread model destruction. One invocation is active at a time.
typedef struct loom_serve_image_generator_t {
  // Borrowed model owner, outliving the service and every accepted task/wait.
  void* self;
  // CPU source preparation, with no shared admission-policy mutation. Returns
  // INVALID_ARGUMENT for rejected input with no retained payload; other errors
  // are terminal. Success retains a payload until discard or completion and
  // returns copied invocation facts through out_prepared.
  iree_status_t (*prepare)(void* self,
                           const loom_serve_image_request_t* request,
                           loom_serve_image_prepared_t* out_prepared);
  // Installs the serialized owner's nonblocking wake before preparation. The
  // owner detaches it only after every accepted compilation has retired.
  void (*set_observer)(void* self, loom_serve_wake_callback_t observer);
  // Resolves prepared code before residency admission. A miss progresses
  // asynchronously; cancellation joins accepted producers before completion.
  iree_status_t (*advance_prepared_request)(void* self, bool cancel,
                                            bool* out_complete);
  // Releases prepared CPU payload and withdraws any unsubmitted admission
  // interest. Runs on the owner, never concurrently with other services.
  void (*discard_prepared_request)(void* self);
  // New-work admission on the shared owner; policy deferral differs from
  // physical denial. ACQUIRED pins and enqueues parameter activation.
  iree_status_t (*try_admit)(void* self,
                             loom_serve_residency_admission_t* out_admission);
  // Enqueues one source-controlled part, returning borrowed completion handles.
  // The first call follows admission; each subsequent call follows successful
  // retirement of both previous frontiers. A terminal part authorizes complete
  // only after its frontiers retire. CANCEL is valid between submitted parts.
  iree_status_t (*submit)(void* self, loom_serve_image_submit_flags_t flags,
                          loom_serve_execution_frontier_t* out_frontier,
                          bool* out_complete);
  // Consumes successful completion and releases model pins. Borrowed F32 CHW
  // RGB in [-1,1] remains live until the next invocation or model destruction.
  iree_const_byte_span_t (*complete)(void* self);
  // Optional owner-side cumulative parameter activity. This query reads only
  // activation-owner data, even while CPU preparation borrows the invocation.
  // NULL means the generator does not expose weight activity, not zero work.
  loom_serve_weight_activity_t (*weight_activity)(void* self);
  // Optional independent execution compiler over retained preparation/state.
  // Selection affects subsequent preparations. An active request pins its
  // generation through all parts; CPU preparation serializes with selection.
  loom_serve_execution_update_t execution_update;
  // Optional immutable input tokenizer, live through the model lifetime. Raw
  // HTTP utilities borrow it independently of the serialized invocation.
  const iree_tokenizer_t* tokenizer;
  // Borrowed admission-policy owner identity shared with other generators and
  // modalities. Equal non-NULL identities can release each other's pressure;
  // NULL declares no shared capacity. Outlives this generator and its service.
  const void* capacity_domain;
} loom_serve_image_generator_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_IMAGE_GENERATOR_H_
