// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/inspection.h"

#include <inttypes.h>

#include "loom/util/json.h"

bool loom_serve_http_inspection_offset(iree_string_view_t value,
                                       iree_host_size_t* out_offset) {
  *out_offset = 0;
  if (!value.size) {
    return false;
  }
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const unsigned digit = (unsigned)(value.data[i] - '0');
    if (digit > 9 || *out_offset > (SIZE_MAX - digit) / 10) {
      return false;
    }
    *out_offset = *out_offset * 10 + digit;
  }
  return true;
}

static iree_status_t inspection_quote(iree_string_builder_t* output,
                                      iree_string_view_t value) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, value);
}

static iree_status_t inspection_identity(iree_string_builder_t* output,
                                         const char* name, uint64_t id) {
  if (!id) {
    return iree_string_builder_append_format(output, ",\"%s\":null", name);
  }
  return iree_string_builder_append_format(output, ",\"%s\":%" PRIu64, name,
                                           id);
}

static iree_status_t inspection_request(
    const loom_serve_http_request_snapshot_t* state, iree_time_t now,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "{\"id\":%" PRIu64 ",\"phase\":", state->id));
  IREE_RETURN_IF_ERROR(
      inspection_quote(output, iree_make_cstring_view(state->phase)));
  IREE_RETURN_IF_ERROR(inspection_identity(output, "session", state->session));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"age_seconds\":%.9f,\"counters\":",
      (now - state->arrival_time) / 1e9));
  IREE_RETURN_IF_ERROR(
      loom_serve_telemetry_counters_format(&state->counters, output));
  return iree_string_builder_append_cstring(output, "}");
}

static iree_status_t inspection_session(
    const loom_serve_http_session_snapshot_t* state,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "{\"id\":%" PRIu64 ",\"name\":", state->id));
  IREE_RETURN_IF_ERROR(inspection_quote(output, state->name));
  IREE_RETURN_IF_ERROR(inspection_identity(output, "request", state->request));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"storage\":"));
  IREE_RETURN_IF_ERROR(
      inspection_quote(output, iree_make_cstring_view(state->storage)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"position\":%" PRIu64 ",\"resident_tokens\":%" PRIu64
      ",\"reserved_tokens\":%" PRIu64 ",\"counters\":",
      state->position, state->resident_tokens, state->reserved_tokens));
  IREE_RETURN_IF_ERROR(
      loom_serve_telemetry_counters_format(&state->counters, output));
  return iree_string_builder_append_cstring(output, "}");
}

iree_status_t loom_serve_http_inspection_format(
    const loom_serve_http_inspection_t* inspection, void* self,
    loom_serve_http_inspection_kind_t kind, iree_host_size_t offset,
    iree_time_t now, iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"data\":["));
  bool complete = false;
  bool emitted = false;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t count = 0; count < 64 && iree_status_is_ok(status);
       ++count, ++offset) {
    loom_serve_http_request_snapshot_t request = {0};
    loom_serve_http_session_snapshot_t session = {0};
    if (!(kind == LOOM_SERVE_HTTP_INSPECT_REQUESTS
              ? inspection->request_at(self, offset, &request)
              : inspection->session_at(self, offset, &session))) {
      complete = true;
      break;
    }
    if (!request.id && !session.id) {
      continue;
    }
    if (emitted) {
      status = iree_string_builder_append_cstring(output, ",");
    }
    if (iree_status_is_ok(status)) {
      status = kind == LOOM_SERVE_HTTP_INSPECT_REQUESTS
                   ? inspection_request(&request, now, output)
                   : inspection_session(&session, output);
    }
    emitted = true;
  }
  if (iree_status_is_ok(status)) {
    status = complete ? iree_string_builder_append_cstring(
                            output, "],\"next_offset\":null")
                      : iree_string_builder_append_format(
                            output, "],\"next_offset\":%zu", offset);
  }
  return status;
}
