// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_IMAGE_SERVICE_H_
#define EXPERIMENTAL_LOOM_SERVE_IMAGE_SERVICE_H_

#include "experimental/loom_serve/image/generator.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/service/service.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_image_service_options_t {
  // Borrowed deployment route name, outliving the service. Source identity
  // remains owned by the generator independently of this HTTP selector.
  iree_string_view_t name;
  // Output pixel width of the retained generator.
  uint32_t width;
  // Output pixel height of the retained generator.
  uint32_t height;
  // Whether the retained generator accepts an adapter strength other than one.
  bool adapter_enabled;
  // Positive bound on requests queued behind the single active image.
  iree_host_size_t pending_capacity;
  // Periodic telemetry snapshot interval; zero disables periodic records,
  // not counters or lifecycle records. Consumers pull JSONL through /logz.
  iree_duration_t heartbeat_interval;
} loom_serve_image_service_options_t;

typedef struct loom_serve_image_service_t loom_serve_image_service_t;

// Creates bounded image admission without accepting asynchronous work.
// Model calls are serialized by the router; finite preparation and encoding
// share the host executor. The generator and deployment name outlive this
// object. A supplied telemetry stream is borrowed through destruction and
// preserves the registration's history; NULL creates a service-owned stream.
iree_status_t loom_serve_image_service_create(
    loom_serve_host_t* host, loom_serve_image_generator_t generator,
    const loom_serve_image_service_options_t* options,
    loom_serve_telemetry_stream_t* stream,
    loom_serve_image_service_t** out_service, iree_allocator_t host_allocator);

// Borrows callbacks for the common HTTP owner. Prepared requests wait for
// shared admission without retaining device pins; waiting alone is not activity
// that can release shared capacity. Successful model readiness releases pins
// before CPU encoding, whose borrowed output remains exclusive until
// retirement. POST /v1/images/generations returns one base64 PNG selected by
// its model field. Peer reset removes queued work or discards an active result
// at the complete-image boundary. Slow readers own copied response bytes, never
// model output or workspace. Bounded observation records omit prompts and
// never wait on log sinks.
loom_serve_http_service_t loom_serve_image_service_interface(
    loom_serve_image_service_t* service);

// Relinquishes host requests after CPU/device callbacks and timer receipts
// retire. Failed readiness leaves device ownership for outer model destruction;
// this function does not drain or destroy the model. NULL is accepted.
void loom_serve_image_service_destroy(loom_serve_image_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_IMAGE_SERVICE_H_
