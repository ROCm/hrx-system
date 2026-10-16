// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_RECOGNITION_SERVICE_H_
#define LOOM_SERVE_SERVICE_RECOGNITION_SERVICE_H_

#include "loom/serve/service/recognition/stream.h"
#include "loom/serve/service/service.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_recognition_service_t
    loom_serve_recognition_service_t;

typedef struct loom_serve_recognition_service_options_t {
  // Deployment name borrowed through adapter retirement.
  iree_string_view_t name;
  // Host-only queue capacity beyond the source's resident streams.
  iree_host_size_t pending_capacity;
  // Maximum copied source options for each stream.
  iree_host_size_t request_limit;
  // Maximum retained transcript document per stream.
  iree_host_size_t document_limit;
  // Decoded frames staged per upload, independent of its total duration.
  iree_host_size_t decode_frames;
} loom_serve_recognition_service_options_t;

// Common native/HTTP owner over continuing-input stream execution. Its bounded
// request ledger owns admission, drain accounting, and terminal observations.
// Model work and PCM conversion share the host executor. HTTP upload bytes
// remain borrowed from a claimed connection through every accepted conversion
// task, including disconnect and shutdown. Native callers retain final
// documents independently of this owner. The generator/model outlives this
// owner.
iree_status_t loom_serve_recognition_service_allocate(
    loom_serve_host_t* host, loom_serve_recognition_generator_t generator,
    const loom_serve_recognition_service_options_t* options,
    loom_serve_telemetry_stream_t* telemetry, iree_allocator_t host_allocator,
    loom_serve_recognition_service_t** out_service);

loom_serve_service_t loom_serve_recognition_service_interface(
    loom_serve_recognition_service_t* service);

// Releases after STOPPING has drained accepted CPU/device callbacks and every
// connection claim. Never waits for model execution. NULL is accepted.
void loom_serve_recognition_service_free(
    loom_serve_recognition_service_t* service);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_RECOGNITION_SERVICE_H_
