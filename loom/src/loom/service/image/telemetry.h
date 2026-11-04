// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_IMAGE_TELEMETRY_H_
#define LOOM_SERVICE_IMAGE_TELEMETRY_H_

#include "loom/runtime/telemetry/sample.h"
#include "loom/runtime/telemetry/stream.h"
#include "loom/service/image/preparation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Host-observed work for one image, copied only after the CPU borrower releases
// it. Shared cohort work contributes to each member's latency, not additional
// work. These durations are not GPU timestamps or individual kernel timings.
typedef struct loom_service_image_work_t {
  // Receipt through cohort selection, excluding this request's CPU preparation.
  uint64_t queue_ns;
  // CPU source-input preparation plus shared cohort capture time.
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
} loom_service_image_work_t;

typedef struct loom_service_image_totals_t {
  // Successful model completions, including results for disconnected peers.
  uint64_t generated_images;
  // Pixels in successfully completed model output, not bytes or tokens.
  uint64_t generated_pixels;
  // Completed work; shared cohort preparation/admission/device time counts
  // once. Queue time sums per-request waiting, and encoding counts each result
  // once.
  loom_service_image_work_t work;
} loom_service_image_totals_t;

// Samples cumulative completed work on the serialized service owner. Shared
// cohort work is counted once; no in-flight CPU job is inspected.
iree_status_t loom_service_image_sample_totals(
    const loom_service_image_totals_t* totals,
    loom_runtime_telemetry_sample_sink_t sink);

extern const loom_runtime_telemetry_event_t LOOM_SERVICE_IMAGE_STARTED;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_IMAGE_SUBMITTED;
// Successful device completion and pin release, before CPU output encoding.
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_IMAGE_GENERATED;
extern const loom_runtime_telemetry_event_t LOOM_SERVICE_IMAGE_SHUTDOWN;

// Owner-only publication; payloads own their bytes and format only on readers.
void loom_service_image_record_prepared(
    loom_runtime_telemetry_stream_t* stream, uint64_t request,
    const loom_service_image_prepared_t* prepared);
void loom_service_image_record_work(loom_runtime_telemetry_stream_t* stream,
                                    uint64_t request,
                                    const loom_service_image_work_t* work);
void loom_service_image_record_totals(
    loom_runtime_telemetry_stream_t* stream,
    const loom_service_image_totals_t* totals);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVICE_IMAGE_TELEMETRY_H_
