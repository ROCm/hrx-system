// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/voice_http.h"

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"

iree_status_t loom_serve_speech_voice_http_parse(
    const loom_serve_http_request_t* request, iree_string_view_t model,
    iree_allocator_t host_allocator,
    loom_serve_speech_voice_request_t* out_request) {
  *out_request = (loom_serve_speech_voice_request_t){0};
  if (!request->multipart) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "voice upload requires multipart/form-data with model, id, and file");
  }
  const loom_serve_http_part_list_t parts =
      loom_serve_http_multipart_parts(request->multipart);
  static const char* const names[] = {"model", "id", "file"};
  const loom_serve_http_part_t* fields[3] = {0};
  for (iree_host_size_t i = 0; i < parts.count; ++i) {
    const loom_serve_http_part_t* part = &parts.values[i];
    iree_host_size_t field = 0;
    for (; field < IREE_ARRAYSIZE(names); ++field) {
      if (iree_string_view_equal(part->name,
                                 iree_make_cstring_view(names[field]))) {
        break;
      }
    }
    if (field == IREE_ARRAYSIZE(names) || fields[field] ||
        (field != 2 && part->filename.data)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown, duplicate, or file-valued voice field '%.*s'",
          (int)part->name.size, part->name.data);
    }
    fields[field] = part;
  }
  if (!fields[0] || !fields[1] || !fields[2] ||
      !iree_string_view_equal(fields[0]->contents, model)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "voice upload requires the selected model, id, and file");
  }
  return loom_serve_speech_voice_request_initialize(
      fields[1]->contents, fields[2]->content_type,
      iree_make_const_byte_span(fields[2]->contents.data,
                                fields[2]->contents.size),
      host_allocator, out_request);
}

static iree_status_t loom_serve_speech_voice_remove_field(
    void* self, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  iree_string_view_t* fields = self;
  char name[5];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_json_unescape_string(key, sizeof(name), name, &length);
  if (iree_status_is_resource_exhausted(status)) {
    iree_status_free(status);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown voice removal field");
  }
  IREE_RETURN_IF_ERROR(status);
  const iree_string_view_t decoded = iree_make_string_view(name, length);
  const int index = iree_string_view_equal(decoded, IREE_SV("model")) ? 0
                    : iree_string_view_equal(decoded, IREE_SV("id"))  ? 1
                                                                      : -1;
  if (index < 0 || type != IREE_JSON_VALUE_TYPE_STRING || fields[index].data) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "voice removal requires unique model and id strings");
  }
  fields[index] = value;
  return iree_ok_status();
}

static iree_status_t loom_serve_speech_voice_string(
    iree_string_view_t raw, iree_string_builder_t* output) {
  char* data = NULL;
  iree_host_size_t capacity = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      output, raw.size, &data, &capacity));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(raw, capacity, data, &length));
  iree_string_builder_commit_append(output, length);
  return iree_ok_status();
}

iree_status_t loom_serve_speech_voice_http_remove_parse(
    const loom_serve_http_request_t* request, iree_string_view_t model,
    iree_string_builder_t* out_id) {
  iree_string_view_t content_type;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("content-type"),
                                             &content_type) ||
      !(iree_string_view_equal_case(content_type,
                                    IREE_SV("application/json")) ||
        iree_string_view_equal_case(
            content_type, IREE_SV("application/json; charset=utf-8"))) ||
      !iree_unicode_utf8_validate(request->body)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "voice removal requires UTF-8 application/json");
  }
  iree_string_view_t cursor = request->body;
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing voice removal JSON");
  }
  iree_string_view_t fields[2] = {0};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_speech_voice_remove_field, fields));
  if (!fields[0].size || !fields[1].size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "voice removal requires model and id");
  }
  iree_string_builder_t name;
  iree_string_builder_initialize(out_id->allocator, &name);
  iree_status_t status = loom_serve_speech_voice_string(fields[0], &name);
  if (iree_status_is_ok(status) &&
      !iree_string_view_equal(iree_string_builder_view(&name), model)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "voice removal model does not match the selected service");
  }
  iree_string_builder_deinitialize(&name);
  if (iree_status_is_ok(status)) {
    status = loom_serve_speech_voice_string(fields[1], out_id);
  }
  if (iree_status_is_ok(status) &&
      iree_string_view_find_char(iree_string_builder_view(out_id), '\0', 0) !=
          IREE_STRING_VIEW_NPOS) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "voice ID contains a NUL byte");
  }
  return status;
}
