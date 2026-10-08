// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_IMAGE_GENERATOR_H_
#define EXPERIMENTAL_LOOM_SERVE_IMAGE_GENERATOR_H_

#include "experimental/loom_serve/image/request.h"
#include "experimental/loom_serve/runtime/execution.h"

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
  // Releases prepared CPU payload before submit, without any residency change.
  void (*discard_prepared_request)(void* self);
  // Pins and enqueues parameter activation; false means insufficient capacity.
  iree_status_t (*try_acquire)(void* self, bool* out_admitted);
  // Enqueues consumption and feedback, returning borrowed completion handles.
  iree_status_t (*submit)(void* self,
                          loom_serve_execution_frontier_t* out_frontier);
  // Consumes successful completion and releases model pins. Borrowed F32 CHW
  // RGB in [-1,1] remains live until the next invocation or model destruction.
  iree_const_byte_span_t (*complete)(void* self);
} loom_serve_image_generator_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_IMAGE_GENERATOR_H_
