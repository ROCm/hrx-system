// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/image/request.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "loom/serve/service/image/output.h"

#define IMAGE_STRING (1u << IREE_JSON_VALUE_TYPE_STRING)
#define IMAGE_NUMBER (1u << IREE_JSON_VALUE_TYPE_NUMBER)
#define IMAGE_ARRAY (1u << IREE_JSON_VALUE_TYPE_ARRAY)

typedef struct loom_serve_image_field_t {
  // Literal protocol key, without JSON escapes.
  const char* name;
  // Accepted JSON types as bit positions.
  uint32_t types;
  // Borrowed raw JSON value; NULL data means absent.
  iree_string_view_t value;
} loom_serve_image_field_t;

typedef struct loom_serve_image_fields_t {
  // Accepted fields for this object, mutated once during enumeration.
  loom_serve_image_field_t* values;
  // Number of entries in values.
  iree_host_size_t count;
} loom_serve_image_fields_t;

static iree_status_t loom_serve_image_field_visit(void* user_data,
                                                  iree_string_view_t key,
                                                  iree_json_value_type_t type,
                                                  iree_string_view_t value) {
  loom_serve_image_fields_t* fields = user_data;
  for (iree_host_size_t i = 0; i < fields->count; ++i) {
    loom_serve_image_field_t* field = &fields->values[i];
    if (iree_string_view_equal(key, iree_make_cstring_view(field->name))) {
      if (field->value.data || !(field->types & (1u << type))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate or wrongly typed image field '%s'",
                                field->name);
      }
      field->value = value;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported image field '%.*s'", (int)key.size,
                          key.data);
}

static iree_status_t loom_serve_image_lora_visit(void* user_data,
                                                 iree_host_size_t index,
                                                 iree_json_value_type_t type,
                                                 iree_string_view_t value) {
  iree_string_builder_t* selections = user_data;
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "each LoRA selection must be an object");
  }
  loom_serve_image_field_t values[] = {{"id", IMAGE_STRING},
                                       {"scale", IMAGE_NUMBER}};
  loom_serve_image_fields_t fields = {values, IREE_ARRAYSIZE(values)};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      value, loom_serve_image_field_visit, &fields));
  if (!values[0].value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA selection requires a nonempty id");
  }
  float scale = 1.0f;
  if (values[1].value.data &&
      (!iree_string_view_atof(values[1].value, &scale) || !isfinite(scale))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA scale must be a finite F32 number");
  }
  const iree_host_size_t header_length = 16;
  const iree_host_size_t padding = 7;
  const iree_host_size_t maximum_length = (iree_host_size_t)iree_min(
      (uint64_t)IREE_HOST_SIZE_MAX, (uint64_t)INT64_MAX);
  if (values[0].value.size > maximum_length - header_length - padding ||
      selections->size >
          maximum_length - header_length - padding - values[0].value.size) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LoRA selections exceed source buffer storage");
  }
  char* record = NULL;
  iree_host_size_t capacity = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      selections, header_length + values[0].value.size + padding, &record,
      &capacity));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(values[0].value, capacity - header_length,
                                record + header_length, &length));
  if (!iree_unicode_utf8_validate(
          iree_make_string_view(record + header_length, length))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "decoded LoRA id must be UTF-8");
  }
  const iree_host_size_t record_length =
      header_length + iree_host_align(length, 8);
  memset(record, 0, header_length);
  iree_unaligned_store_le_u64(record, length);
  iree_unaligned_store_le_f32(record + 8, scale);
  memset(record + header_length + length, 0,
         record_length - header_length - length);
  iree_string_builder_commit_append(selections, record_length);
  return iree_ok_status();
}

static iree_status_t loom_serve_image_request_decode(
    iree_string_view_t body, iree_string_view_t model, uint32_t width,
    uint32_t height, loom_serve_image_request_t* request) {
  if (!iree_unicode_utf8_validate(body)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image JSON must be UTF-8");
  }
  iree_string_view_t cursor = body;
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing image JSON");
  }
  loom_serve_image_field_t fields[] = {
      {"prompt", IMAGE_STRING},         {"seed", IMAGE_STRING | IMAGE_NUMBER},
      {"loras", IMAGE_ARRAY},           {"model", IMAGE_STRING},
      {"size", IMAGE_STRING},           {"n", IMAGE_NUMBER},
      {"response_format", IMAGE_STRING}};
  loom_serve_image_fields_t object_fields = {fields, IREE_ARRAYSIZE(fields)};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_image_field_visit, &object_fields));
  if (!fields[0].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image prompt is required");
  }
  if (fields[1].value.data) {
    IREE_RETURN_IF_ERROR(
        iree_json_parse_uint64(fields[1].value, &request->seed));
  }
  if (fields[2].value.data) {
    IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
        fields[2].value, loom_serve_image_lora_visit, &request->loras));
  }
  if (fields[3].value.data && !iree_string_view_equal(fields[3].value, model)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "select model '%.*s'",
                            (int)model.size, model.data);
  }
  request->width = width;
  request->height = height;
  if (fields[4].value.data) {
    const iree_string_view_t size = fields[4].value;
    iree_string_view_t dimensions[2];
    iree_string_view_split(size, 'x', &dimensions[0], &dimensions[1]);
    uint32_t values[2] = {0};
    for (iree_host_size_t i = 0; i < 2; ++i) {
      for (iree_host_size_t j = 0; j < dimensions[i].size; ++j) {
        const unsigned digit = (unsigned)(dimensions[i].data[j] - '0');
        if (digit > 9 || values[i] > (UINT32_MAX - digit) / 10) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "size requires positive decimal WIDTHxHEIGHT");
        }
        values[i] = values[i] * 10 + digit;
      }
    }
    request->width = values[0];
    request->height = values[1];
  }
  iree_host_size_t encoded_length = 0;
  IREE_RETURN_IF_ERROR(loom_serve_image_rgba8_png_size(
      request->width, request->height, &encoded_length));
  uint64_t count = 1;
  if (fields[5].value.data) {
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(fields[5].value, &count));
  }
  if (count != 1 ||
      (fields[6].value.data &&
       !iree_string_view_equal(fields[6].value, IREE_SV("b64_json")))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "select n=1 and response_format='b64_json'");
  }
  char* decoded = NULL;
  iree_host_size_t capacity = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      &request->prompt, fields[0].value.size, &decoded, &capacity));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(fields[0].value, capacity, decoded, &length));
  iree_string_builder_commit_append(&request->prompt, length);
  if (!iree_unicode_utf8_validate(iree_string_builder_view(&request->prompt))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "decoded prompt must be UTF-8");
  }
  return iree_ok_status();
}

void loom_serve_image_request_deinitialize(
    loom_serve_image_request_t* request) {
  iree_string_builder_deinitialize(&request->prompt);
  iree_string_builder_deinitialize(&request->loras);
  memset(request, 0, sizeof(*request));
}

iree_status_t loom_serve_image_request_initialize(
    iree_string_view_t body, iree_string_view_t model, uint32_t width,
    uint32_t height, iree_allocator_t host_allocator,
    loom_serve_image_request_t* out_request) {
  memset(out_request, 0, sizeof(*out_request));
  iree_string_builder_initialize(host_allocator, &out_request->prompt);
  iree_string_builder_initialize(host_allocator, &out_request->loras);
  iree_status_t status =
      loom_serve_image_request_decode(body, model, width, height, out_request);
  if (!iree_status_is_ok(status)) {
    loom_serve_image_request_deinitialize(out_request);
  }
  return status;
}
