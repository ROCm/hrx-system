// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/image/telemetry.h"

#include <string.h>

#include "loom/runtime/telemetry/field.h"

iree_status_t loom_service_image_sample_totals(
    const loom_service_image_totals_t* totals,
    loom_runtime_telemetry_sample_sink_t sink) {
  static const loom_runtime_telemetry_metric_t fields[] = {
      {"generated_images", "images", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"generated_pixels", "pixels", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"queue", "nanoseconds", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"prepare", "nanoseconds", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"admission", "nanoseconds", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"submit", "nanoseconds", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"execution", "nanoseconds", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"png", "nanoseconds", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"json", "nanoseconds", LOOM_RUNTIME_TELEMETRY_COUNTER},
      {"png_bytes", "bytes", LOOM_RUNTIME_TELEMETRY_COUNTER},
  };
  const uint64_t values[] = {
      totals->generated_images,  totals->generated_pixels,
      totals->work.queue_ns,     totals->work.prepare_ns,
      totals->work.admission_ns, totals->work.submit_ns,
      totals->work.execution_ns, totals->work.png_ns,
      totals->work.json_ns,      totals->work.png_bytes,
  };
  static_assert(IREE_ARRAYSIZE(fields) == IREE_ARRAYSIZE(values),
                "one value per image field");
  const loom_runtime_telemetry_topic_t topic = {"image", 1,
                                                IREE_ARRAYSIZE(fields), fields};
  return sink.emit(sink.self, &topic, values);
}

static iree_status_t loom_service_image_prepared_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_image_prepared_t prepared;
  static_assert(sizeof(prepared) <= sizeof(record->data), "record capacity");
  memcpy(&prepared, record->data.words, sizeof(prepared));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("stage", prepared.stage),
      loom_runtime_telemetry_field_int64("tag", prepared.tag),
      loom_runtime_telemetry_field_uint64("input_bytes", prepared.input_bytes),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

enum { IMAGE_WORK_FIELD_COUNT = 8 };

static void loom_service_image_work_read_fields(
    const loom_service_image_work_t* work,
    loom_runtime_telemetry_field_t fields[IMAGE_WORK_FIELD_COUNT]) {
  fields[0] = loom_runtime_telemetry_field_uint64("queue_ns", work->queue_ns);
  fields[1] =
      loom_runtime_telemetry_field_uint64("prepare_ns", work->prepare_ns);
  fields[2] =
      loom_runtime_telemetry_field_uint64("admission_ns", work->admission_ns);
  fields[3] = loom_runtime_telemetry_field_uint64("submit_ns", work->submit_ns);
  fields[4] =
      loom_runtime_telemetry_field_uint64("execution_ns", work->execution_ns);
  fields[5] = loom_runtime_telemetry_field_uint64("png_ns", work->png_ns);
  fields[6] = loom_runtime_telemetry_field_uint64("json_ns", work->json_ns);
  fields[7] = loom_runtime_telemetry_field_uint64("png_bytes", work->png_bytes);
}

static iree_status_t loom_service_image_request_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_image_work_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  loom_runtime_telemetry_field_t fields[IMAGE_WORK_FIELD_COUNT];
  loom_service_image_work_read_fields(&value, fields);
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_image_totals_fields(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_image_totals_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  loom_runtime_telemetry_field_t fields[2 + IMAGE_WORK_FIELD_COUNT] = {
      loom_runtime_telemetry_field_uint64("generated_images",
                                          value.generated_images),
      loom_runtime_telemetry_field_uint64("generated_pixels",
                                          value.generated_pixels),
  };
  loom_service_image_work_read_fields(&value.work, fields + 2);
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

const loom_runtime_telemetry_event_t LOOM_SERVICE_IMAGE_STARTED = {
    "image_started", NULL};
const loom_runtime_telemetry_event_t LOOM_SERVICE_IMAGE_SUBMITTED = {
    "image_submitted", NULL};
const loom_runtime_telemetry_event_t LOOM_SERVICE_IMAGE_GENERATED = {
    "image_generated", NULL};
const loom_runtime_telemetry_event_t LOOM_SERVICE_IMAGE_SHUTDOWN = {
    "image_shutdown", NULL};
static const loom_runtime_telemetry_event_t image_prepared_event = {
    .name = "image_prepared",
    .read_fields = loom_service_image_prepared_fields};
static const loom_runtime_telemetry_event_t image_request_event = {
    "image_request_stats", loom_service_image_request_fields};
static const loom_runtime_telemetry_event_t image_totals_event = {
    "image_work", loom_service_image_totals_fields};

void loom_service_image_record_prepared(
    loom_runtime_telemetry_stream_t* stream, uint64_t request,
    const loom_service_image_prepared_t* prepared) {
  loom_runtime_telemetry_record_t record = {.time_ns = iree_time_now(),
                                            .request = request,
                                            .event = &image_prepared_event};
  memcpy(record.data.words, prepared, sizeof(*prepared));
  loom_runtime_telemetry_stream_append(stream, &record);
}

void loom_service_image_record_work(loom_runtime_telemetry_stream_t* stream,
                                    uint64_t request,
                                    const loom_service_image_work_t* work) {
  loom_runtime_telemetry_record_t record = {.time_ns = iree_time_now(),
                                            .request = request,
                                            .event = &image_request_event};
  memcpy(record.data.words, work, sizeof(*work));
  loom_runtime_telemetry_stream_append(stream, &record);
}

void loom_service_image_record_totals(
    loom_runtime_telemetry_stream_t* stream,
    const loom_service_image_totals_t* totals) {
  loom_runtime_telemetry_record_t record = {.time_ns = iree_time_now(),
                                            .event = &image_totals_event};
  memcpy(record.data.words, totals, sizeof(*totals));
  loom_runtime_telemetry_stream_append(stream, &record);
}
