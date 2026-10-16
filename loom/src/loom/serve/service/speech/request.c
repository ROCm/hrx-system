// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/request.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"

typedef struct loom_serve_speech_field_t {
  // Literal decoded protocol key.
  const char* name;
  // Accepted JSON value types as bit positions.
  uint32_t types;
  // Raw value view, with NULL data denoting absence.
  iree_string_view_t value;
  // Actual JSON type, set only when value is present.
  iree_json_value_type_t type;
} loom_serve_speech_field_t;

typedef struct loom_serve_speech_fields_t {
  // Number of recognized fields in values.
  iree_host_size_t count;
  // Mutable parser results, borrowed during object enumeration.
  loom_serve_speech_field_t* values;
} loom_serve_speech_fields_t;

#define SPEECH_STRING (1u << IREE_JSON_VALUE_TYPE_STRING)
#define SPEECH_NUMBER (1u << IREE_JSON_VALUE_TYPE_NUMBER)
#define SPEECH_OBJECT (1u << IREE_JSON_VALUE_TYPE_OBJECT)

static iree_status_t loom_serve_speech_request_validate(
    iree_string_view_t input, iree_string_view_t voice,
    iree_string_view_t instructions, float speed) {
  if (!iree_unicode_utf8_validate(input) ||
      !iree_unicode_utf8_validate(voice) ||
      !iree_unicode_utf8_validate(instructions)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech strings must be UTF-8");
  }
  if (!input.size || !voice.size ||
      iree_string_view_find_char(voice, '\0', 0) != IREE_STRING_VIEW_NPOS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech requires nonempty input and a nonempty "
                            "voice identifier without NUL bytes");
  }
  if (!isfinite(speed) || speed < 0.25f || speed > 4.0f) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech speed must be between 0.25 and 4");
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_speech_field_visit(void* user_data,
                                                   iree_string_view_t key,
                                                   iree_json_value_type_t type,
                                                   iree_string_view_t value) {
  loom_serve_speech_fields_t* fields = user_data;
  // Every recognized protocol key fits here, including after JSON unescaping.
  char decoded[16];
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_json_unescape_string(key, sizeof(decoded), decoded, &length);
  if (iree_status_is_resource_exhausted(status)) {
    iree_status_free(status);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported speech field name");
  }
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate(status, IREE_SV("invalid speech field name"));
  }
  key = iree_make_string_view(decoded, length);
  for (iree_host_size_t i = 0; i < fields->count; ++i) {
    loom_serve_speech_field_t* field = &fields->values[i];
    if (iree_string_view_equal(key, iree_make_cstring_view(field->name))) {
      if (field->value.data || !(field->types & (1u << type))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate or wrongly typed speech field '%s'",
                                field->name);
      }
      field->value = value;
      field->type = type;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported speech field '%.*s'", (int)key.size,
                          key.data);
}

static iree_status_t loom_serve_speech_decode_string(
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

static iree_status_t loom_serve_speech_match_string(
    iree_string_view_t raw, iree_string_view_t expected, const char* diagnostic,
    iree_allocator_t allocator) {
  iree_string_builder_t decoded;
  iree_string_builder_initialize(allocator, &decoded);
  iree_status_t status = loom_serve_speech_decode_string(raw, &decoded);
  if (iree_status_is_ok(status) &&
      !iree_string_view_equal(iree_string_builder_view(&decoded), expected)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", diagnostic);
  }
  iree_string_builder_deinitialize(&decoded);
  return status;
}

static iree_status_t loom_serve_speech_request_decode(
    iree_string_view_t body, iree_string_view_t model,
    iree_allocator_t host_allocator, loom_serve_speech_request_t* request) {
  if (!iree_unicode_utf8_validate(body)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech JSON must be UTF-8");
  }
  iree_string_view_t cursor = body;
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing speech JSON");
  }
  enum { INPUT, VOICE, INSTRUCTIONS, MODEL, SPEED, SEED, FORMAT, STREAM };
  loom_serve_speech_field_t values[] = {
      {"input", SPEECH_STRING},
      {"voice", SPEECH_STRING | SPEECH_OBJECT},
      {"instructions", SPEECH_STRING},
      {"model", SPEECH_STRING},
      {"speed", SPEECH_NUMBER},
      {"seed", SPEECH_NUMBER | SPEECH_STRING},
      {"response_format", SPEECH_STRING},
      {"stream_format", SPEECH_STRING}};
  loom_serve_speech_fields_t fields = {IREE_ARRAYSIZE(values), values};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_speech_field_visit, &fields));
  if (!values[INPUT].value.data || !values[VOICE].value.data ||
      !values[MODEL].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech requires input, voice and model");
  }
  IREE_RETURN_IF_ERROR(loom_serve_speech_match_string(
      values[MODEL].value, model, "speech model does not match this deployment",
      host_allocator));
  if (!values[FORMAT].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "select response_format='pcm'");
  }
  IREE_RETURN_IF_ERROR(loom_serve_speech_match_string(
      values[FORMAT].value, IREE_SV("pcm"), "select response_format='pcm'",
      host_allocator));
  if (values[STREAM].value.data) {
    IREE_RETURN_IF_ERROR(loom_serve_speech_match_string(
        values[STREAM].value, IREE_SV("audio"), "select stream_format='audio'",
        host_allocator));
  }
  if (values[SPEED].value.data &&
      !iree_string_view_atof(values[SPEED].value, &request->speed)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech speed must be between 0.25 and 4");
  }
  if (values[SEED].value.data) {
    iree_string_builder_t decoded;
    iree_string_builder_initialize(host_allocator, &decoded);
    iree_string_view_t seed = values[SEED].value;
    iree_status_t status = iree_ok_status();
    if (values[SEED].type == IREE_JSON_VALUE_TYPE_STRING) {
      status = loom_serve_speech_decode_string(seed, &decoded);
      seed = iree_string_builder_view(&decoded);
    }
    if (iree_status_is_ok(status)) {
      status = iree_json_parse_uint64(seed, &request->seed);
    }
    iree_string_builder_deinitialize(&decoded);
    IREE_RETURN_IF_ERROR(status);
  }
  iree_string_view_t voice = values[VOICE].value;
  if (values[VOICE].type == IREE_JSON_VALUE_TYPE_OBJECT) {
    loom_serve_speech_field_t id = {"id", SPEECH_STRING};
    loom_serve_speech_fields_t voice_fields = {1, &id};
    IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
        voice, loom_serve_speech_field_visit, &voice_fields));
    if (!id.value.data) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "custom voice requires id");
    }
    voice = id.value;
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_speech_decode_string(values[INPUT].value, &request->input));
  IREE_RETURN_IF_ERROR(loom_serve_speech_decode_string(voice, &request->voice));
  if (values[INSTRUCTIONS].value.data) {
    IREE_RETURN_IF_ERROR(loom_serve_speech_decode_string(
        values[INSTRUCTIONS].value, &request->instructions));
  }
  return loom_serve_speech_request_validate(
      iree_string_builder_view(&request->input),
      iree_string_builder_view(&request->voice),
      iree_string_builder_view(&request->instructions), request->speed);
}

void loom_serve_speech_request_deinitialize(
    loom_serve_speech_request_t* request) {
  iree_string_builder_deinitialize(&request->input);
  iree_string_builder_deinitialize(&request->voice);
  iree_string_builder_deinitialize(&request->instructions);
  memset(request, 0, sizeof(*request));
}

iree_status_t loom_serve_speech_request_initialize_text(
    iree_string_view_t input, iree_string_view_t voice,
    iree_string_view_t instructions, uint64_t seed, float speed,
    iree_allocator_t host_allocator, loom_serve_speech_request_t* out_request) {
  memset(out_request, 0, sizeof(*out_request));
  IREE_RETURN_IF_ERROR(
      loom_serve_speech_request_validate(input, voice, instructions, speed));
  iree_string_builder_initialize(host_allocator, &out_request->input);
  iree_string_builder_initialize(host_allocator, &out_request->voice);
  iree_string_builder_initialize(host_allocator, &out_request->instructions);
  out_request->seed = seed;
  out_request->speed = speed;
  iree_status_t status =
      iree_string_builder_append_string(&out_request->input, input);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&out_request->voice, voice);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&out_request->instructions,
                                               instructions);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_speech_request_deinitialize(out_request);
  }
  return status;
}

iree_status_t loom_serve_speech_request_initialize(
    iree_string_view_t body, iree_string_view_t model,
    iree_allocator_t host_allocator, loom_serve_speech_request_t* out_request) {
  memset(out_request, 0, sizeof(*out_request));
  iree_string_builder_initialize(host_allocator, &out_request->input);
  iree_string_builder_initialize(host_allocator, &out_request->voice);
  iree_string_builder_initialize(host_allocator, &out_request->instructions);
  out_request->speed = 1.0f;
  iree_status_t status = loom_serve_speech_request_decode(
      body, model, host_allocator, out_request);
  if (!iree_status_is_ok(status)) {
    loom_serve_speech_request_deinitialize(out_request);
  }
  return status;
}
