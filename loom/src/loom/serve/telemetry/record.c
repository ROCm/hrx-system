// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/telemetry/record.h"

#include <inttypes.h>
#include <string.h>

#include "loom/util/json.h"

const char* loom_serve_telemetry_outcome_name(
    loom_serve_telemetry_outcome_t outcome) {
  static const char* const outcomes[] = {"completed", "rejected", "cancelled",
                                         "failed"};
  return outcomes[outcome];
}

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

static iree_status_t loom_serve_telemetry_record_quote(
    iree_string_builder_t* output, iree_string_view_t value) {
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

void loom_serve_telemetry_counters_read_fields(
    const loom_serve_telemetry_counters_t* counters,
    loom_runtime_telemetry_field_t
        fields[LOOM_SERVE_TELEMETRY_COUNTER_FIELD_COUNT]) {
  static const char* const names[] = {
      "requests", "completed",    "rejected",      "cancelled",
      "failed",   "input_tokens", "output_tokens", "batches"};
  const uint64_t values[] = {counters->requests,      counters->completed,
                             counters->rejected,      counters->cancelled,
                             counters->failed,        counters->input_tokens,
                             counters->output_tokens, counters->batches};
  static_assert(
      IREE_ARRAYSIZE(names) == LOOM_SERVE_TELEMETRY_COUNTER_FIELD_COUNT,
      "one name per counter");
  static_assert(
      IREE_ARRAYSIZE(values) == LOOM_SERVE_TELEMETRY_COUNTER_FIELD_COUNT,
      "one value per counter");
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    fields[i] = loom_runtime_telemetry_field_uint64(names[i], values[i]);
  }
}

static iree_status_t loom_serve_telemetry_record_snapshot(
    const loom_serve_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  loom_runtime_telemetry_field_t
      counters[LOOM_SERVE_TELEMETRY_COUNTER_FIELD_COUNT];
  loom_serve_telemetry_counters_read_fields(&record->data.snapshot.counters,
                                            counters);
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("queued",
                                          record->data.snapshot.queued),
      loom_runtime_telemetry_field_uint64("active",
                                          record->data.snapshot.active),
      loom_runtime_telemetry_field_string(
          "phase", iree_make_cstring_view(record->data.snapshot.phase)),
      loom_runtime_telemetry_field_object("counters", IREE_ARRAYSIZE(counters),
                                          counters),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_serve_telemetry_record_request_end(
    const loom_serve_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  const loom_serve_telemetry_diagnostic_t* diagnostic =
      &record->data.end.diagnostic;
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_string(
          "outcome",
          iree_make_cstring_view(loom_serve_telemetry_outcome_name(
              (loom_serve_telemetry_outcome_t)record->data.end.outcome))),
      loom_runtime_telemetry_field_uint64("duration_ns",
                                          record->data.end.duration_ns),
      loom_runtime_telemetry_field_uint64("input_tokens",
                                          record->data.end.input_tokens),
      loom_runtime_telemetry_field_uint64("status", diagnostic->code),
      loom_runtime_telemetry_field_bool("message_truncated",
                                        diagnostic->truncated != 0),
      loom_runtime_telemetry_field_string(
          "message",
          iree_make_string_view(diagnostic->message, diagnostic->length)),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_serve_telemetry_record_first_output(
    const loom_serve_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64(
          "duration_ns", record->data.first_output.duration_ns),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_serve_telemetry_record_batch(
    const loom_serve_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  const loom_runtime_telemetry_field_t fields[] = {
      loom_runtime_telemetry_field_uint64("batch", record->data.batch.id),
      loom_runtime_telemetry_field_uint64("requests",
                                          record->data.batch.requests),
  };
  return sink.emit(sink.self, IREE_ARRAYSIZE(fields), fields);
}

static iree_status_t loom_serve_telemetry_record_error(
    const loom_serve_telemetry_record_t* record,
    loom_runtime_telemetry_field_sink_t sink) {
  const loom_serve_telemetry_diagnostic_t* diagnostic = &record->data.error;
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

const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_STREAM_BEGIN = {
    "stream_begin", NULL};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_SNAPSHOT = {
    "snapshot", loom_serve_telemetry_record_snapshot};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_REQUEST_BEGIN = {
    "request_begin", NULL};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_FIRST_OUTPUT = {
    "first_output", loom_serve_telemetry_record_first_output};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_REQUEST_END = {
    .name = "request_end",
    .read_fields = loom_serve_telemetry_record_request_end};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_BATCH_SUBMIT = {
    "batch_submit", loom_serve_telemetry_record_batch};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_BATCH_END = {
    "batch_end", loom_serve_telemetry_record_batch};
const loom_serve_telemetry_event_t LOOM_SERVE_TELEMETRY_ERROR = {
    "error", loom_serve_telemetry_record_error};

static iree_status_t loom_serve_telemetry_record_write_fields(
    void* self, iree_host_size_t count,
    const loom_runtime_telemetry_field_t* fields) {
  loom_json_object_writer_t* output = self;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const loom_runtime_telemetry_field_t* field = &fields[i];
    const iree_string_view_t name = iree_make_cstring_view(field->name);
    switch (field->kind) {
      case LOOM_RUNTIME_TELEMETRY_FIELD_NULL:
        status = loom_json_object_write_null_field(output, name);
        break;
      case LOOM_RUNTIME_TELEMETRY_FIELD_UINT64:
        status = loom_json_object_write_uint64_field(output, name,
                                                     field->value.uint64_value);
        break;
      case LOOM_RUNTIME_TELEMETRY_FIELD_INT64:
        status = loom_json_object_write_int64_field(output, name,
                                                    field->value.int64_value);
        break;
      case LOOM_RUNTIME_TELEMETRY_FIELD_BOOL:
        status = loom_json_object_write_bool_field(output, name,
                                                   field->value.bool_value);
        break;
      case LOOM_RUNTIME_TELEMETRY_FIELD_STRING:
        status = loom_json_object_write_string_field(output, name,
                                                     field->value.string_value);
        break;
      case LOOM_RUNTIME_TELEMETRY_FIELD_OBJECT: {
        status = loom_json_object_begin_field(output, name);
        loom_json_object_writer_t child;
        if (iree_status_is_ok(status)) {
          status = loom_json_object_begin(output->stream, &child);
        }
        if (iree_status_is_ok(status)) {
          status = loom_serve_telemetry_record_write_fields(
              &child, field->value.object_value.count,
              field->value.object_value.fields);
        }
        if (iree_status_is_ok(status)) {
          status = loom_json_object_end(&child);
        }
        break;
      }
    }
  }
  return status;
}

iree_status_t loom_serve_telemetry_counters_format(
    const loom_serve_telemetry_counters_t* counters,
    iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  loom_runtime_telemetry_field_t
      fields[LOOM_SERVE_TELEMETRY_COUNTER_FIELD_COUNT];
  loom_serve_telemetry_counters_read_fields(counters, fields);
  IREE_RETURN_IF_ERROR(loom_serve_telemetry_record_write_fields(
      &object, IREE_ARRAYSIZE(fields), fields));
  return loom_json_object_end(&object);
}

iree_status_t loom_serve_telemetry_record_format(
    uint64_t position, const loom_serve_telemetry_record_t* record,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "{\"event\":"));
  IREE_RETURN_IF_ERROR(loom_serve_telemetry_record_quote(
      output, iree_make_cstring_view(record->event->name)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"sequence\":%" PRIu64 ",\"time_ns\":%" PRId64 ",\"request\":%" PRIu64,
      position, record->time_ns, record->request));
  if (record->event->read_fields) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(output, &stream);
    loom_json_object_writer_t object = {.stream = &stream, .field_count = 4};
    IREE_RETURN_IF_ERROR(record->event->read_fields(
        record, (loom_runtime_telemetry_field_sink_t){
                    &object, loom_serve_telemetry_record_write_fields}));
  }
  return iree_string_builder_append_cstring(output, "}\n");
}
