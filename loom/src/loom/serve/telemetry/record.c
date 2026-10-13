// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/telemetry/record.h"

#include <inttypes.h>
#include <string.h>

#include "loom/util/json.h"

void loom_serve_telemetry_count_outcome(
    loom_serve_telemetry_counters_t* counters,
    loom_serve_telemetry_outcome_t outcome) {
  switch (outcome) {
    case LOOM_SERVE_TELEMETRY_COMPLETED:
      ++counters->completed;
      break;
    case LOOM_SERVE_TELEMETRY_REJECTED:
      ++counters->rejected;
      break;
    case LOOM_SERVE_TELEMETRY_CANCELLED:
      ++counters->cancelled;
      break;
    case LOOM_SERVE_TELEMETRY_FAILED:
      ++counters->failed;
      break;
  }
}

static iree_status_t record_quote(iree_string_builder_t* output,
                                  iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

void loom_serve_telemetry_diagnostic_capture(
    const iree_status_t status,
    loom_serve_telemetry_diagnostic_t* out_diagnostic) {
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

iree_status_t loom_serve_telemetry_counters_format(
    const loom_serve_telemetry_counters_t* counters,
    iree_string_builder_t* output) {
  return iree_string_builder_append_format(
      output,
      "{\"requests\":%" PRIu64 ",\"completed\":%" PRIu64
      ",\"rejected\":%" PRIu64 ",\"cancelled\":%" PRIu64 ",\"failed\":%" PRIu64
      ",\"input_tokens\":%" PRIu64 ",\"output_tokens\":%" PRIu64
      ",\"batches\":%" PRIu64 "}",
      counters->requests, counters->completed, counters->rejected,
      counters->cancelled, counters->failed, counters->input_tokens,
      counters->output_tokens, counters->batches);
}

iree_status_t loom_serve_telemetry_diagnostic_format(
    const loom_serve_telemetry_diagnostic_t* diagnostic,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"status\":%" PRIu64 ",\"message_truncated\":%s,\"message\":",
      diagnostic->code, diagnostic->truncated ? "true" : "false"));
  return record_quote(
      output, iree_make_string_view(diagnostic->message, diagnostic->length));
}

static iree_status_t record_snapshot(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"queued\":%" PRIu64 ",\"active\":%" PRIu64 ",\"phase\":",
      record->data.snapshot.queued, record->data.snapshot.active));
  IREE_RETURN_IF_ERROR(record_quote(
      output, iree_make_cstring_view(record->data.snapshot.phase)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"counters\":"));
  return loom_serve_telemetry_counters_format(&record->data.snapshot.counters,
                                              output);
}

static iree_status_t record_request_end(
    const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  static const char* const outcomes[] = {"completed", "rejected", "cancelled",
                                         "failed"};
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"outcome\":\"%s\",\"duration_ns\":%" PRIu64
      ",\"input_tokens\":%" PRIu64,
      outcomes[record->data.end.outcome], record->data.end.duration_ns,
      record->data.end.input_tokens));
  return loom_serve_telemetry_diagnostic_format(&record->data.end.diagnostic,
                                                output);
}

static iree_status_t record_batch(const loom_serve_telemetry_record_t* record,
                                  iree_string_builder_t* output) {
  return iree_string_builder_append_format(
      output, ",\"batch\":%" PRIu64 ",\"requests\":%" PRIu64,
      record->data.batch.id, record->data.batch.requests);
}

static iree_status_t record_error(const loom_serve_telemetry_record_t* record,
                                  iree_string_builder_t* output) {
  return loom_serve_telemetry_diagnostic_format(&record->data.error, output);
}

const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_STREAM_BEGIN = {
    "stream_begin", NULL};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_SNAPSHOT = {
    "snapshot", record_snapshot};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_REQUEST_BEGIN = {
    "request_begin", NULL};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_REQUEST_END = {
    "request_end", record_request_end};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_BATCH_SUBMIT = {
    "batch_submit", record_batch};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_BATCH_END = {
    "batch_end", record_batch};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_ERROR = {"error",
                                                                 record_error};

iree_status_t loom_serve_telemetry_record_format(
    uint64_t position, const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "{\"event\":"));
  IREE_RETURN_IF_ERROR(
      record_quote(output, iree_make_cstring_view(record->event->name)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"sequence\":%" PRIu64 ",\"time_ns\":%" PRId64 ",\"request\":%" PRIu64,
      position, record->time_ns, record->request));
  if (record->event->format) {
    IREE_RETURN_IF_ERROR(record->event->format(record, output));
  }
  return iree_string_builder_append_cstring(output, "}\n");
}
