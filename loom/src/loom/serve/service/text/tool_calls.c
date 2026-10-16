// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/tool_calls.h"

#include "loom/util/json.h"

typedef struct loom_serve_text_tool_call_range_t {
  // Identifier byte offset in the shared storage.
  iree_host_size_t id;
  // Name byte offset, also the exclusive end of the identifier.
  iree_host_size_t name;
  // Argument byte offset, also the exclusive end of the name.
  iree_host_size_t arguments;
  // Exclusive end of the argument document.
  iree_host_size_t end;
} loom_serve_text_tool_call_range_t;

void loom_serve_text_tool_calls_initialize(
    iree_allocator_t host_allocator, loom_serve_text_tool_calls_t* out_calls) {
  *out_calls = (loom_serve_text_tool_calls_t){0};
  iree_string_builder_initialize(host_allocator, &out_calls->storage);
}

void loom_serve_text_tool_calls_deinitialize(
    loom_serve_text_tool_calls_t* calls) {
  iree_allocator_free(calls->storage.allocator, calls->ranges);
  iree_string_builder_deinitialize(&calls->storage);
  *calls = (loom_serve_text_tool_calls_t){0};
}

void loom_serve_text_tool_calls_reset(loom_serve_text_tool_calls_t* calls) {
  calls->count = 0;
  iree_string_builder_reset(&calls->storage);
}

iree_status_t loom_serve_text_tool_calls_append(
    loom_serve_text_tool_calls_t* calls, uint64_t request_id,
    iree_string_view_t name, iree_string_view_t arguments) {
  if (calls->count == calls->capacity) {
    const iree_host_size_t capacity = calls->capacity ? calls->capacity * 2 : 4;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        calls->storage.allocator, capacity * sizeof(*calls->ranges),
        (void**)&calls->ranges));
    calls->capacity = capacity;
  }
  loom_serve_text_tool_call_range_t range = {
      .id = iree_string_builder_size(&calls->storage)};
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &calls->storage, "call_%" PRIu64 "_%zu", request_id, calls->count));
  range.name = iree_string_builder_size(&calls->storage);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(&calls->storage, name));
  range.arguments = iree_string_builder_size(&calls->storage);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(&calls->storage, arguments));
  range.end = iree_string_builder_size(&calls->storage);
  calls->ranges[calls->count++] = range;
  return iree_ok_status();
}

loom_serve_text_tool_call_t loom_serve_text_tool_calls_at(
    const loom_serve_text_tool_calls_t* calls, iree_host_size_t index) {
  const loom_serve_text_tool_call_range_t range = calls->ranges[index];
  const char* data = iree_string_builder_buffer(&calls->storage);
  return (loom_serve_text_tool_call_t){
      .id = iree_make_string_view(data + range.id, range.name - range.id),
      .name = iree_make_string_view(data + range.name,
                                    range.arguments - range.name),
      .arguments = iree_make_string_view(data + range.arguments,
                                         range.end - range.arguments)};
}

iree_status_t loom_serve_text_tool_calls_write_json(
    const loom_serve_text_tool_calls_t* calls, bool indexed,
    iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  iree_status_t status = iree_string_builder_append_cstring(output, "[");
  for (iree_host_size_t i = 0; i < calls->count && iree_status_is_ok(status);
       ++i) {
    const loom_serve_text_tool_call_t call =
        loom_serve_text_tool_calls_at(calls, i);
    status = iree_string_builder_append_format(output, "%s{", i ? "," : "");
    if (iree_status_is_ok(status) && indexed) {
      status = iree_string_builder_append_format(output, "\"index\":%zu,", i);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_format(
          output,
          "\"id\":\"%.*s\",\"type\":\"function\",\"function\":{\"name\":",
          (int)call.id.size, call.id.data);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_write_escaped_string(&stream, call.name);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(output, ",\"arguments\":");
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_write_escaped_string(&stream, call.arguments);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(output, "}}");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "]");
  }
  return status;
}
