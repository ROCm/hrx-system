// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_IMAGE_SERVICE_H_
#define EXPERIMENTAL_LOOM_SERVE_IMAGE_SERVICE_H_

#include "experimental/loom_serve/http_server.h"
#include "experimental/loom_serve/image_request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Coarse finite-image invocation. Calls are serialized on one worker; the
// implementation retires all work borrowing request storage before returning.
// Successful RGB is little-endian F32 CHW in [-1,1], borrowed until the next
// call. The service encodes it before allowing another call. Failure after
// request admission is terminal, and stops the service after retiring
// ownership.
typedef struct loom_serve_image_generator_t {
  // Borrowed model owner, outliving service_run and its joined worker.
  void* self;
  // Complete native image generation, with no HTTP or connection ownership.
  iree_status_t (*generate)(void* self,
                            const loom_serve_image_request_t* request,
                            iree_const_byte_span_t* out_rgb);
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
} loom_serve_image_service_options_t;

// Runs bounded image admission over a borrowed local HTTP transport until
// shutdown or terminal failure. POST /v1/images/generations returns one base64
// PNG; GET /healthz and /v1/models remain responsive while an image runs.
// Queued requests own only decoded host input. Peer reset removes queued work
// or discards an active result at the complete-image boundary. Slow readers own
// copied response bytes, never model output or workspace. Return joins the
// worker and relinquishes all connection views before either borrowed owner
// can be destroyed. JSONL lifecycle events omit prompts.
iree_status_t loom_serve_image_service_run(
    loom_serve_image_generator_t generator, loom_serve_http_server_t* server,
    const loom_serve_image_service_options_t* options,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_IMAGE_SERVICE_H_
