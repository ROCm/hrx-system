// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/telemetry/record.h"

#include <string.h>

const char* loom_service_telemetry_outcome_name(
    loom_service_telemetry_outcome_t outcome) {
  static const char* const outcomes[] = {"completed", "rejected", "cancelled",
                                         "failed"};
  return outcomes[outcome];
}

void loom_service_telemetry_count_outcome(
    loom_service_telemetry_counters_t* counters,
    loom_service_telemetry_outcome_t outcome) {
  switch (outcome) {
    case LOOM_SERVICE_TELEMETRY_COMPLETED:
      ++counters->completed;
      break;
    case LOOM_SERVICE_TELEMETRY_REJECTED:
      ++counters->rejected;
      break;
    case LOOM_SERVICE_TELEMETRY_CANCELLED:
      ++counters->cancelled;
      break;
    case LOOM_SERVICE_TELEMETRY_FAILED:
      ++counters->failed;
      break;
  }
}

void loom_service_telemetry_diagnostic_capture(
    const iree_status_t status,
    loom_service_telemetry_diagnostic_t* out_diagnostic) {
  memset(out_diagnostic, 0, sizeof(*out_diagnostic));
  out_diagnostic->code = iree_status_code(status);
  const iree_string_view_t message = iree_status_message(status);
  out_diagnostic->length =
      iree_min(message.size, sizeof(out_diagnostic->message));
  out_diagnostic->truncated = message.size > out_diagnostic->length;
  if (out_diagnostic->length) {
    memcpy(out_diagnostic->message, message.data, out_diagnostic->length);
  }
}

void loom_service_telemetry_counters_read_fields(
    const loom_service_telemetry_counters_t* counters,
    loom_runtime_telemetry_field_t
        fields[LOOM_SERVICE_TELEMETRY_COUNTER_FIELD_COUNT]) {
  static const char* const names[] = {
      "requests", "completed",    "rejected",      "cancelled",
      "failed",   "input_tokens", "output_tokens", "batches"};
  const uint64_t values[] = {counters->requests,      counters->completed,
                             counters->rejected,      counters->cancelled,
                             counters->failed,        counters->input_tokens,
                             counters->output_tokens, counters->batches};
  static_assert(
      IREE_ARRAYSIZE(names) == LOOM_SERVICE_TELEMETRY_COUNTER_FIELD_COUNT,
      "one name per counter");
  static_assert(
      IREE_ARRAYSIZE(values) == LOOM_SERVICE_TELEMETRY_COUNTER_FIELD_COUNT,
      "one value per counter");
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    fields[i] = loom_runtime_telemetry_field_uint64(names[i], values[i]);
  }
}

static iree_status_t loom_service_telemetry_record_snapshot(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_telemetry_snapshot_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  loom_runtime_telemetry_field_t
      counters[LOOM_SERVICE_TELEMETRY_COUNTER_FIELD_COUNT];
  loom_service_telemetry_counters_read_fields(&value.counters, counters);
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("queued", value.queued),
      loom_runtime_telemetry_field_uint64("active", value.active),
      loom_runtime_telemetry_field_string("phase",
                                          iree_make_cstring_view(value.phase)),
      loom_runtime_telemetry_field_object("counters", IREE_ARRAYSIZE(counters),
                                          counters),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_telemetry_record_request_end(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_telemetry_end_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_service_telemetry_diagnostic_t* diagnostic = &value.diagnostic;
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_string(
          "outcome", iree_make_cstring_view(loom_service_telemetry_outcome_name(
                         (loom_service_telemetry_outcome_t)value.outcome))),
      loom_runtime_telemetry_field_uint64("duration_ns", value.duration_ns),
      loom_runtime_telemetry_field_uint64("input_tokens", value.input_tokens),
      loom_runtime_telemetry_field_uint64("status", diagnostic->code),
      loom_runtime_telemetry_field_bool("message_truncated",
                                        diagnostic->truncated != 0),
      loom_runtime_telemetry_field_string(
          "message",
          iree_make_string_view(diagnostic->message, diagnostic->length)),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_telemetry_record_first_output(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_telemetry_first_output_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("duration_ns", value.duration_ns),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_telemetry_record_batch(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_telemetry_batch_record_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("batch", value.id),
      loom_runtime_telemetry_field_uint64("requests", value.requests),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_service_telemetry_record_error(
    const loom_runtime_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_service_telemetry_diagnostic_t value;
  static_assert(sizeof(value) <= sizeof(record->data), "record capacity");
  memcpy(&value, record->data.words, sizeof(value));
  const loom_service_telemetry_diagnostic_t* diagnostic = &value;
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("status", diagnostic->code),
      loom_runtime_telemetry_field_bool("message_truncated",
                                        diagnostic->truncated != 0),
      loom_runtime_telemetry_field_string(
          "message",
          iree_make_string_view(diagnostic->message, diagnostic->length)),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_STREAM_BEGIN = {
    "stream_begin", NULL};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_SNAPSHOT = {
    "snapshot", loom_service_telemetry_record_snapshot};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_REQUEST_BEGIN = {
    "request_begin", NULL};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_FIRST_OUTPUT = {
    "first_output", loom_service_telemetry_record_first_output};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_REQUEST_END = {
    .name = "request_end",
    .read_fields = loom_service_telemetry_record_request_end};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_BATCH_SUBMIT = {
    "batch_submit", loom_service_telemetry_record_batch};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_BATCH_END = {
    "batch_end", loom_service_telemetry_record_batch};
const loom_runtime_telemetry_event_t LOOM_SERVICE_TELEMETRY_ERROR = {
    "error", loom_service_telemetry_record_error};
