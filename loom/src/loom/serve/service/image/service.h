// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_IMAGE_SERVICE_H_
#define LOOM_SERVE_SERVICE_IMAGE_SERVICE_H_

#include "loom/runtime/host/host.h"
#include "loom/runtime/service_control.h"
#include "loom/serve/service/image/generator.h"
#include "loom/serve/service/service.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_image_service_options_t {
  // Borrowed deployment route name, outliving the service. Source identity
  // remains owned by the generator independently of this HTTP selector.
  iree_string_view_t name;
  // Default output pixel width for requests omitting size.
  uint32_t width;
  // Default output pixel height for requests omitting size.
  uint32_t height;
  // Positive additional host request credit beyond the first admitted image.
  // Queued and active cohort items share the total pending_capacity + 1 bound.
  iree_host_size_t pending_capacity;
  // Retained execution choices including original/pending/failed candidates;
  // zero disables reconfiguration independently of request admission.
  iree_host_size_t variant_capacity;
  // Periodic telemetry snapshot interval; zero disables periodic records,
  // not counters or lifecycle records. Consumers pull JSONL through /logz.
  iree_duration_t heartbeat_interval;
} loom_serve_image_service_options_t;

typedef struct loom_serve_image_service_t loom_serve_image_service_t;

// Creates bounded image admission without accepting asynchronous work.
// Model calls are serialized by the router; request/cohort preparation and
// encoding share the host executor. The generator and deployment name outlive
// this object. A supplied telemetry stream is borrowed through destruction and
// preserves the registration's history; NULL creates a service-owned stream.
// The supplied control has one single-choice offer and 1 + variant_capacity
// operation slots, each with one device use. Its storage and summary publisher
// outlive this engine and the generator's actual retirement.
iree_status_t loom_serve_image_service_allocate(
    loom_runtime_host_t* host, loom_serve_image_generator_t generator,
    const loom_serve_image_service_options_t* options,
    loom_runtime_telemetry_stream_t* stream,
    loom_runtime_service_control_binding_t scheduling,
    iree_allocator_t host_allocator, loom_serve_image_service_t** out_service);

// Consumes accepted readiness and cleanup without granting another source unit.
// Preparation ends at a prepared cohort; execution spans its source-controlled
// parts and final encoders. Both remain progressable with no scheduling grants.
iree_status_t loom_serve_image_service_progress(
    loom_serve_image_service_t* service, loom_runtime_service_flags_t flags,
    bool* out_progress);

// Accepts one current choice without waiting. A stale choice succeeds with a
// zero operation; a failed start can return both an error and an accepted key.
iree_status_t loom_serve_image_service_try_schedule(
    loom_serve_image_service_t* service, loom_runtime_service_offer_key_t offer,
    uint64_t choice_ordinal, loom_runtime_service_flags_t flags,
    loom_runtime_service_operation_key_t* out_operation);

// Borrows callbacks for the common HTTP owner. Prepared requests wait for
// shared admission without retaining device pins; waiting alone is not activity
// that can release shared capacity. Successful model readiness releases pins
// before CPU encoding, whose borrowed output remains exclusive until
// retirement. POST /v1/images/generations returns one base64 PNG selected by
// its model field. Compatible prepared requests share one command cohort.
// Peer reset removes queued work but preserves captured slots until retirement;
// only loss of all peers stops submissions after the active part retires.
// One PNG encoder borrows each surviving RGB range in turn. Slow readers own
// copied response bytes, never model output or workspace. Bounded observation
// records omit prompts and never wait on log sinks.
loom_serve_service_t loom_serve_image_service_interface(
    loom_serve_image_service_t* service);

// Relinquishes host requests after CPU/device callbacks and timer receipts
// retire. Failed readiness leaves device ownership for outer model destruction;
// this function does not drain or destroy the model. NULL is accepted.
void loom_serve_image_service_free(loom_serve_image_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_IMAGE_SERVICE_H_
