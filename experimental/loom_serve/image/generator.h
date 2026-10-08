// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_IMAGE_GENERATOR_H_
#define EXPERIMENTAL_LOOM_SERVE_IMAGE_GENERATOR_H_

#include "experimental/loom_serve/image/request.h"
#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/runtime/residency.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

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
  // are terminal. Success retains a payload until discard or completion.
  iree_status_t (*prepare)(void* self,
                           const loom_serve_image_request_t* request);
  // Releases prepared CPU payload and withdraws any unsubmitted admission
  // interest. Runs on the owner, never concurrently with other services.
  void (*discard_prepared_request)(void* self);
  // New-work admission on the shared owner; policy deferral differs from
  // physical denial. ACQUIRED pins and enqueues parameter activation.
  iree_status_t (*try_admit)(void* self,
                             loom_serve_residency_admission_t* out_admission);
  // Enqueues consumption and feedback, returning borrowed completion handles.
  iree_status_t (*submit)(void* self,
                          loom_serve_execution_frontier_t* out_frontier);
  // Consumes successful completion and releases model pins. Borrowed F32 CHW
  // RGB in [-1,1] remains live until the next invocation or model destruction.
  iree_const_byte_span_t (*complete)(void* self);
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

#endif  // EXPERIMENTAL_LOOM_SERVE_IMAGE_GENERATOR_H_
