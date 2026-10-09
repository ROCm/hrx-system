// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/logs.h"

#include <inttypes.h>

#include "loom/util/json.h"

static iree_status_t logs_quote(iree_string_builder_t* output,
                                iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

static bool logs_cursor(iree_string_view_t value, uint64_t fields[3]) {
  for (iree_host_size_t i = 0; i < 3; ++i) {
    fields[i] = 0;
    iree_host_size_t length = 0;
    while (length < value.size && value.data[length] != ':') {
      const unsigned digit = (unsigned)(value.data[length] - '0');
      if (digit > 9 || fields[i] > (UINT64_MAX - digit) / 10) {
        return false;
      }
      fields[i] = fields[i] * 10 + digit;
      ++length;
    }
    if (!length || (i == 2 ? length != value.size : length == value.size)) {
      return false;
    }
    value = iree_string_view_substr(value, length + (i != 2),
                                    IREE_STRING_VIEW_NPOS);
  }
  return true;
}

iree_status_t loom_serve_http_logs(
    const loom_serve_telemetry_stream_t* stream,
    const loom_serve_telemetry_counters_t* counters, iree_string_view_t model,
    iree_string_view_t cursor, iree_string_builder_t* output) {
  const loom_serve_telemetry_identity_t identity =
      loom_serve_telemetry_stream_identity(stream);
  const uint64_t head = loom_serve_telemetry_stream_head(stream);
  const uint64_t capacity = loom_serve_telemetry_stream_capacity(stream);
  const uint64_t oldest = head > capacity ? head - capacity : 0;
  uint64_t position = 0;
  if (cursor.size) {
    uint64_t fields[3];
    if (!logs_cursor(cursor, fields)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "malformed telemetry cursor");
    }
    if (fields[0] != identity.created_ns || fields[1] != identity.ordinal) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "telemetry stream incarnation changed");
    }
    position = fields[2];
    if (position > head) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "telemetry cursor is ahead of publication");
    }
  }
  iree_string_builder_reset(output);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output,
      "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"
      "Cache-Control: no-store\r\nConnection: close\r\n\r\n"
      "{\"event\":\"stream\",\"scope\":"));
  IREE_RETURN_IF_ERROR(
      logs_quote(output, model.size ? IREE_SV("model") : IREE_SV("server")));
  if (model.size) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"model\":"));
    IREE_RETURN_IF_ERROR(logs_quote(output, model));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"counters_available\":%s", counters ? "true" : "false"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"identity\":\"%" PRIu64 ":%" PRIu64 "\",\"oldest\":%" PRIu64
      ",\"head\":%" PRIu64 ",\"capacity\":%" PRIu64 ",\"partial\":%s}\n",
      identity.created_ns, identity.ordinal, oldest, head, capacity,
      (position || oldest) ? "true" : "false"));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < 64 && position < head && iree_status_is_ok(status); ++i) {
    const uint64_t previous = position;
    loom_serve_telemetry_record_t record;
    const loom_serve_telemetry_read_t result =
        loom_serve_telemetry_stream_read(stream, &position, &record);
    if (result == LOOM_SERVE_TELEMETRY_READ_RECORD) {
      status = loom_serve_telemetry_record_format(previous, &record, output);
    } else if (result == LOOM_SERVE_TELEMETRY_READ_GAP) {
      status = iree_string_builder_append_format(
          output,
          "{\"event\":\"gap\",\"from\":%" PRIu64 ",\"to\":%" PRIu64 "}\n",
          previous, position);
    } else {
      break;
    }
  }
  if (iree_status_is_ok(status) && counters) {
    status = iree_string_builder_append_format(
        output, "{\"event\":\"counters\",\"time_ns\":%" PRId64 ",\"values\":",
        iree_time_now());
  }
  if (iree_status_is_ok(status) && counters) {
    status = loom_serve_telemetry_counters_format(counters, output);
  }
  if (iree_status_is_ok(status) && counters) {
    status = iree_string_builder_append_cstring(output, "}\n");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        output,
        "{\"event\":\"cursor\",\"value\":\"%" PRIu64 ":%" PRIu64 ":%" PRIu64
        "\",\"next\":%" PRIu64 "}\n",
        identity.created_ns, identity.ordinal, position, position);
  }
  return status;
}
