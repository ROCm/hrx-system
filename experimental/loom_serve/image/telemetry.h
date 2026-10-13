// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_IMAGE_TELEMETRY_H_
#define EXPERIMENTAL_LOOM_SERVE_IMAGE_TELEMETRY_H_

#include "experimental/loom_serve/image/generator.h"
#include "loom/serve/telemetry/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Host-observed work for one image, copied only after the CPU borrower releases
// it. These durations are not GPU timestamps or individual kernel timings.
typedef struct loom_serve_image_work_t {
  // Time waiting for the exclusive generator, before CPU preparation.
  uint64_t queue_ns;
  // CPU source-input preparation time.
  uint64_t prepare_ns;
  // Time awaiting parameter admission after CPU preparation.
  uint64_t admission_ns;
  // Host device-submission time, summed across command parts.
  uint64_t submit_ns;
  // Submission return through readiness collection, summed across parts.
  uint64_t execution_ns;
  // CPU PNG encoding time.
  uint64_t png_ns;
  // CPU JSON/base64 serialization time.
  uint64_t json_ns;
  // Encoded PNG bytes, excluding JSON/base64 and HTTP overhead.
  uint64_t png_bytes;
} loom_serve_image_work_t;

typedef struct loom_serve_image_totals_t {
  // Successful model completions, including results for disconnected peers.
  uint64_t generated_images;
  // Pixels in successfully completed model output, not bytes or tokens.
  uint64_t generated_pixels;
  // Sum of completed preparation, device and encoding work across requests.
  loom_serve_image_work_t work;
} loom_serve_image_totals_t;

extern const loom_serve_telemetry_event_t LOOM_SERVE_IMAGE_STARTED;
extern const loom_serve_telemetry_event_t LOOM_SERVE_IMAGE_SUBMITTED;
// Successful device completion and pin release, before CPU output encoding.
extern const loom_serve_telemetry_event_t LOOM_SERVE_IMAGE_GENERATED;
extern const loom_serve_telemetry_event_t LOOM_SERVE_IMAGE_SHUTDOWN;

// Owner-only publication; payloads own their bytes and format only on readers.
void loom_serve_image_record_prepared(
    loom_serve_telemetry_stream_t* stream, uint64_t request,
    const loom_serve_image_prepared_t* prepared);
void loom_serve_image_record_work(loom_serve_telemetry_stream_t* stream,
                                  uint64_t request,
                                  const loom_serve_image_work_t* work);
void loom_serve_image_record_totals(loom_serve_telemetry_stream_t* stream,
                                    const loom_serve_image_totals_t* totals);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_IMAGE_TELEMETRY_H_
