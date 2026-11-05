// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/telemetry/record.h"

#include <inttypes.h>
#include <string.h>

#include "loom/util/json.h"

static iree_status_t loom_serve_telemetry_record_quote(
    iree_string_builder_t* output, iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

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
    const loom_service_telemetry_counters_t* counters,
    iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  loom_runtime_telemetry_field_t
      fields[LOOM_SERVICE_TELEMETRY_COUNTER_FIELD_COUNT];
  loom_service_telemetry_counters_read_fields(counters, fields);
  IREE_RETURN_IF_ERROR(loom_serve_telemetry_record_write_fields(
      &object, IREE_ARRAYSIZE(fields), fields));
  return loom_json_object_end(&object);
}

iree_status_t loom_serve_telemetry_record_format(
    uint64_t position, const loom_runtime_telemetry_record_t* record,
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
