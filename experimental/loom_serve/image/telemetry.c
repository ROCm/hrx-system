// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/image/telemetry.h"

#include <inttypes.h>
#include <string.h>

static iree_status_t image_work_format(const loom_serve_image_work_t* work,
                                       iree_string_builder_t* output) {
  return iree_string_builder_append_format(
      output,
      ",\"queue_ns\":%" PRIu64 ",\"prepare_ns\":%" PRIu64
      ",\"admission_ns\":%" PRIu64 ",\"submit_ns\":%" PRIu64
      ",\"execution_ns\":%" PRIu64 ",\"png_ns\":%" PRIu64
      ",\"json_ns\":%" PRIu64 ",\"png_bytes\":%" PRIu64,
      work->queue_ns, work->prepare_ns, work->admission_ns, work->submit_ns,
      work->execution_ns, work->png_ns, work->json_ns, work->png_bytes);
}

static iree_status_t image_request_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_image_work_t work;
  static_assert(sizeof(work) <= sizeof(record->data), "record capacity");
  memcpy(&work, record->data.words, sizeof(work));
  return image_work_format(&work, output);
}

static iree_status_t image_totals_format(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  loom_serve_image_totals_t totals;
  static_assert(sizeof(totals) <= sizeof(record->data), "record capacity");
  memcpy(&totals, record->data.words, sizeof(totals));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"generated_images\":%" PRIu64 ",\"generated_pixels\":%" PRIu64,
      totals.generated_images, totals.generated_pixels));
  return image_work_format(&totals.work, output);
}

const loom_serve_telemetry_event_t LOOM_SERVE_IMAGE_STARTED = {"image_started",
                                                               NULL};
const loom_serve_telemetry_event_t LOOM_SERVE_IMAGE_SUBMITTED = {
    "image_submitted", NULL};
const loom_serve_telemetry_event_t LOOM_SERVE_IMAGE_SHUTDOWN = {
    "image_shutdown", NULL};
static const loom_serve_telemetry_event_t image_request_event = {
    "image_request_stats", image_request_format};
static const loom_serve_telemetry_event_t image_totals_event = {
    "image_work", image_totals_format};

void loom_serve_image_record_work(loom_serve_telemetry_stream_t* stream,
                                  uint64_t request,
                                  const loom_serve_image_work_t* work) {
  loom_serve_telemetry_record_t record = {.time_ns = iree_time_now(),
                                          .request = request,
                                          .event = &image_request_event};
  memcpy(record.data.words, work, sizeof(*work));
  loom_serve_telemetry_stream_append(stream, &record);
}

void loom_serve_image_record_totals(loom_serve_telemetry_stream_t* stream,
                                    const loom_serve_image_totals_t* totals) {
  loom_serve_telemetry_record_t record = {.time_ns = iree_time_now(),
                                          .event = &image_totals_event};
  memcpy(record.data.words, totals, sizeof(*totals));
  loom_serve_telemetry_stream_append(stream, &record);
}
