// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_IMAGE_SERVICE_H_
#define EXPERIMENTAL_LOOM_SERVE_IMAGE_SERVICE_H_

#include "experimental/loom_serve/http/server.h"
#include "experimental/loom_serve/image/request.h"
#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/runtime/host.h"

#ifdef __cplusplus
extern "C" {
#endif

// Finite-image invocation with explicit CPU and device phases. The scheduler
// owns admission, submission and successful completion; prepare runs as a CPU
// task with exclusive model access. None of these functions waits for device
// or I/O completion. Failure is terminal and preserves accepted model ownership
// until outer-thread model destruction. One invocation is active at a time.
typedef struct loom_serve_image_generator_t {
  // Borrowed model owner, outliving service_run and every accepted task/wait.
  void* self;
  // Pins and enqueues parameter activation; false means insufficient capacity.
  iree_status_t (*try_acquire)(void* self, bool* out_admitted);
  // CPU source preparation, with no shared admission-policy mutation.
  iree_status_t (*prepare)(void* self,
                           const loom_serve_image_request_t* request);
  // Enqueues consumption and feedback, returning borrowed completion handles.
  iree_status_t (*submit)(void* self,
                          loom_serve_execution_frontier_t* out_frontier);
  // Consumes successful completion and releases model pins. Borrowed F32 CHW
  // RGB in [-1,1] remains live until the next invocation or model destruction.
  iree_const_byte_span_t (*complete)(void* self);
} loom_serve_image_generator_t;

typedef struct loom_serve_image_service_options_t {
  // Borrowed model identifier exposed by discovery and accepted in requests.
  iree_string_view_t model;
  // Output pixel width of the retained generator.
  uint32_t width;
  // Output pixel height of the retained generator.
  uint32_t height;
  // Whether the retained generator accepts an adapter strength other than one.
  bool adapter_enabled;
  // Positive bound on requests queued behind the single active image.
  iree_host_size_t pending_capacity;
  // Periodic JSONL reporting interval in nanoseconds; zero disables it.
  iree_duration_t heartbeat_interval;
} loom_serve_image_service_options_t;

// Runs bounded image admission over a borrowed local HTTP transport until
// shutdown or terminal failure. POST /v1/images/generations returns one base64
// PNG; GET /healthz and /v1/models remain responsive while an image runs.
// Queued requests own only decoded host input. Peer reset removes queued work
// or discards an active result at the complete-image boundary. Slow readers own
// copied response bytes, never model output or workspace. Return joins the
// shared tasks and asynchronous waits and relinquishes all connection views
// before either borrowed owner can be destroyed. Only this outer application
// call joins; scheduler workers never wait. JSONL events omit prompts.
iree_status_t loom_serve_image_service_run(
    loom_serve_host_t* host, loom_serve_image_generator_t generator,
    loom_serve_http_server_t* server,
    const loom_serve_image_service_options_t* options,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_IMAGE_SERVICE_H_
