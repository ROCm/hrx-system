// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/request.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/math.h"

static const char id4_ideogram4_qwen_prompt_prefix[] = "<|im_start|>user\n";
static const char id4_ideogram4_qwen_prompt_suffix[] =
    "<|im_end|>\n<|im_start|>assistant\n";

enum {
  ID4_IDEOGRAM4_DIT_TEXT_INDICATOR = 0,
  ID4_IDEOGRAM4_DIT_IMAGE_INDICATOR = 1,
  ID4_IDEOGRAM4_DIT_IMAGE_POSITION_OFFSET = 65536,
  ID4_IDEOGRAM4_DIT_MROPE_SECTION_HEIGHT = 20,
  ID4_IDEOGRAM4_DIT_MROPE_SECTION_WIDTH = 20,
};

static const float id4_ideogram4_dit_mrope_theta = 5000000.0f;

static iree_status_t id4_ideogram4_request_count_member(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  (void)key;
  (void)value;
  iree_host_size_t* count = (iree_host_size_t*)user_data;
  ++*count;
  return iree_ok_status();
}

typedef struct id4_ideogram4_request_schema_validation_t {
  // Number of top-level object members seen by the validator.
  iree_host_size_t member_count;
  // True when the top-level prompt payload member was seen.
  bool has_prompt;
  // True when the top-level generation metadata member was seen.
  bool has_generation;
} id4_ideogram4_request_schema_validation_t;

typedef struct id4_ideogram4_request_generation_lookup_t {
  // True when a top-level generation member is present.
  bool has_generation;
  // Raw top-level generation member value.
  iree_string_view_t generation;
} id4_ideogram4_request_generation_lookup_t;

static iree_status_t id4_ideogram4_request_find_generation_member(
    void* user_data, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  (void)type;
  id4_ideogram4_request_generation_lookup_t* lookup =
      (id4_ideogram4_request_generation_lookup_t*)user_data;
  if (!iree_string_view_equal(key, IREE_SV("generation"))) {
    return iree_ok_status();
  }
  if (lookup->has_generation) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "full Ideogram 4 request has duplicate generation metadata");
  }
  lookup->has_generation = true;
  lookup->generation = value;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_validate_full_schema_member(
    void* user_data, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  (void)value;
  id4_ideogram4_request_schema_validation_t* validation =
      (id4_ideogram4_request_schema_validation_t*)user_data;
  ++validation->member_count;
  if (iree_string_view_equal(key, IREE_SV("prompt"))) {
    validation->has_prompt = true;
    switch (type) {
      case IREE_JSON_VALUE_TYPE_STRING:
      case IREE_JSON_VALUE_TYPE_OBJECT:
        return iree_ok_status();
      default:
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "full Ideogram 4 request prompt must be a string or object");
    }
  }
  if (iree_string_view_equal(key, IREE_SV("generation"))) {
    validation->has_generation = true;
    if (type == IREE_JSON_VALUE_TYPE_OBJECT) return iree_ok_status();
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "full Ideogram 4 request generation metadata must "
                            "be an object");
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "full Ideogram 4 request has unsupported top-level member `%.*s`",
      (int)key.size, key.data);
}

static iree_status_t id4_ideogram4_request_validate_qwen_encode_options(
    const id4_ideogram4_qwen_lowering_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering options structure size "
                            "%" PRIhsz " is smaller than expected %" PRIhsz,
                            options->structure_size, sizeof(*options));
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen request lowering extension structures are not supported");
  }
  if (!options->tokenizer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering tokenizer is required");
  }
  if (!options->request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering request is required");
  }
  if (iree_string_view_is_empty(options->request->qwen_prompt)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering wrapped prompt is empty");
  }
  if (options->max_token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering max token count is zero");
  }
  if (options->vocab_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering vocab size is zero");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_validate_lowering_options(
    const id4_ideogram4_qwen_lowering_options_t* options) {
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_request_validate_qwen_encode_options(options));
  if (options->token_capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering token capacity is zero");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_copy_string(
    iree_string_view_t value, iree_allocator_t host_allocator,
    iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, value.size, (void**)&storage));
  memcpy(storage, value.data, value.size);
  *out_value = iree_make_string_view(storage, value.size);
  return iree_ok_status();
}

enum { ID4_IDEOGRAM4_REQUEST_JSON_MAX_DEPTH = 128 };

static iree_status_t id4_ideogram4_request_append_compact_json_value(
    iree_string_view_t* value, iree_string_builder_t* builder,
    iree_host_size_t depth);

static iree_status_t id4_ideogram4_request_append_compact_json_string(
    iree_string_builder_t* builder, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\""));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, value));
  return iree_string_builder_append_cstring(builder, "\"");
}

static iree_status_t id4_ideogram4_request_append_compact_json_object(
    iree_string_view_t* object, iree_string_builder_t* builder,
    iree_host_size_t depth) {
  if (depth >= ID4_IDEOGRAM4_REQUEST_JSON_MAX_DEPTH) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Ideogram 4 prompt JSON nesting is too deep");
  }
  if (!iree_string_view_consume_prefix_char(object, '{')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 prompt JSON object is missing `{`");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "{"));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(object));
  bool first_member = true;
  while (!iree_string_view_is_empty(*object) &&
         !iree_string_view_starts_with_char(*object, '}')) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_json_consume_string(object, &key));
    IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(object));
    if (!iree_string_view_consume_prefix_char(object, ':')) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 prompt JSON object member is "
                              "missing `:`");
    }
    if (first_member) {
      first_member = false;
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_request_append_compact_json_string(builder, key));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ":"));
    IREE_RETURN_IF_ERROR(id4_ideogram4_request_append_compact_json_value(
        object, builder, depth + 1));
    IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(object));
    if (!iree_string_view_consume_prefix_char(object, ',')) break;
    IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(object));
  }
  if (!iree_string_view_consume_prefix_char(object, '}')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 prompt JSON object is missing `}`");
  }
  return iree_string_builder_append_cstring(builder, "}");
}

static iree_status_t id4_ideogram4_request_append_compact_json_array(
    iree_string_view_t* array, iree_string_builder_t* builder,
    iree_host_size_t depth) {
  if (depth >= ID4_IDEOGRAM4_REQUEST_JSON_MAX_DEPTH) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Ideogram 4 prompt JSON nesting is too deep");
  }
  if (!iree_string_view_consume_prefix_char(array, '[')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 prompt JSON array is missing `[`");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "["));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(array));
  bool first_element = true;
  while (!iree_string_view_is_empty(*array) &&
         !iree_string_view_starts_with_char(*array, ']')) {
    if (first_element) {
      first_element = false;
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(id4_ideogram4_request_append_compact_json_value(
        array, builder, depth + 1));
    IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(array));
    if (!iree_string_view_consume_prefix_char(array, ',')) break;
    IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(array));
  }
  if (!iree_string_view_consume_prefix_char(array, ']')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 prompt JSON array is missing `]`");
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t id4_ideogram4_request_append_compact_json_value(
    iree_string_view_t* value, iree_string_builder_t* builder,
    iree_host_size_t depth) {
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(value));
  if (iree_string_view_is_empty(*value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 prompt JSON value is empty");
  }
  switch (value->data[0]) {
    case '{':
      return id4_ideogram4_request_append_compact_json_object(value, builder,
                                                              depth);
    case '[':
      return id4_ideogram4_request_append_compact_json_array(value, builder,
                                                             depth);
    case '"': {
      iree_string_view_t string_value = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(iree_json_consume_string(value, &string_value));
      return id4_ideogram4_request_append_compact_json_string(builder,
                                                              string_value);
    }
    case 't': {
      iree_string_view_t keyword = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          iree_json_consume_keyword(value, IREE_SV("true"), &keyword));
      return iree_string_builder_append_string(builder, keyword);
    }
    case 'f': {
      iree_string_view_t keyword = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          iree_json_consume_keyword(value, IREE_SV("false"), &keyword));
      return iree_string_builder_append_string(builder, keyword);
    }
    case 'n': {
      iree_string_view_t keyword = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          iree_json_consume_keyword(value, IREE_SV("null"), &keyword));
      return iree_string_builder_append_string(builder, keyword);
    }
    default: {
      iree_string_view_t number = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(iree_json_consume_number(value, &number));
      return iree_string_builder_append_string(builder, number);
    }
  }
}

static iree_status_t id4_ideogram4_request_copy_prompt_payload(
    iree_string_view_t value, iree_allocator_t host_allocator,
    iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  if (!iree_string_view_starts_with_char(value, '{') &&
      !iree_string_view_starts_with_char(value, '[')) {
    return id4_ideogram4_request_copy_string(value, host_allocator, out_value);
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_string_view_t remaining = value;
  iree_status_t status = id4_ideogram4_request_append_compact_json_value(
      &remaining, &builder, /*depth=*/0);
  if (iree_status_is_ok(status)) {
    status = iree_json_consume_insignificant(&remaining);
  }
  if (iree_status_is_ok(status) && !iree_string_view_is_empty(remaining)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "trailing data after Ideogram 4 prompt payload");
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t prompt_size = iree_string_builder_size(&builder);
    char* storage = iree_string_builder_take_storage(&builder);
    *out_value = iree_make_string_view(storage, prompt_size);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t id4_ideogram4_request_lookup_required(
    iree_string_view_t object, iree_string_view_t key,
    iree_string_view_t* out_value) {
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(object, key, out_value));
  if (iree_string_view_is_empty(*out_value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 request field `%.*s` must not be empty",
                            (int)key.size, key.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_parse_required_u32(
    iree_string_view_t object, iree_string_view_t key, uint32_t* out_value) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_request_lookup_required(object, key, &value));
  uint64_t raw_value = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(value, &raw_value));
  if (raw_value == 0 || raw_value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 request field `%.*s` value %" PRIu64
                            " is outside the accepted uint32 range",
                            (int)key.size, key.data, raw_value);
  }
  *out_value = (uint32_t)raw_value;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_parse_required_u64(
    iree_string_view_t object, iree_string_view_t key, uint64_t* out_value) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_request_lookup_required(object, key, &value));
  return iree_json_parse_uint64(value, out_value);
}

static iree_status_t id4_ideogram4_request_parse_required_f32_positive(
    iree_string_view_t object, iree_string_view_t key, float* out_value) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_request_lookup_required(object, key, &value));
  double raw_value = 0.0;
  IREE_RETURN_IF_ERROR(iree_json_parse_double(value, &raw_value));
  if (!isfinite(raw_value) || raw_value <= 0.0 || raw_value > FLT_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 request field `%.*s` value %g is not a positive f32",
        (int)key.size, key.data, raw_value);
  }
  *out_value = (float)raw_value;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_validate_generation(
    const id4_ideogram4_request_generation_t* generation) {
  if (generation->latent_width == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 request generation latent width is zero");
  }
  if (generation->latent_height == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 request generation latent height is zero");
  }
  if (generation->denoise_step_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 request generation denoise step count is zero");
  }
  if (!isfinite(generation->guidance_scale) ||
      generation->guidance_scale <= 0.0f) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 request generation guidance scale is not positive");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_parse_generation(
    iree_string_view_t generation,
    id4_ideogram4_request_generation_t* out_generation) {
  memset(out_generation, 0, sizeof(*out_generation));
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_parse_required_u32(
      generation, IREE_SV("latent_width"), &out_generation->latent_width));
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_parse_required_u32(
      generation, IREE_SV("latent_height"), &out_generation->latent_height));
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_parse_required_u32(
      generation, IREE_SV("denoise_steps"),
      &out_generation->denoise_step_count));
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_parse_required_u64(
      generation, IREE_SV("seed"), &out_generation->seed));
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_parse_required_f32_positive(
      generation, IREE_SV("guidance_scale"), &out_generation->guidance_scale));
  return id4_ideogram4_request_validate_generation(out_generation);
}

static iree_status_t id4_ideogram4_request_wrap_qwen_prompt(
    iree_string_view_t prompt_payload, iree_allocator_t host_allocator,
    iree_string_view_t* out_qwen_prompt) {
  *out_qwen_prompt = iree_string_view_empty();
  const iree_host_size_t prefix_length =
      sizeof(id4_ideogram4_qwen_prompt_prefix) - 1;
  const iree_host_size_t suffix_length =
      sizeof(id4_ideogram4_qwen_prompt_suffix) - 1;
  if (prompt_payload.size > IREE_HOST_SIZE_MAX - prefix_length ||
      prompt_payload.size + prefix_length >
          IREE_HOST_SIZE_MAX - suffix_length) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Ideogram 4 Qwen prompt is too large");
  }
  const iree_host_size_t prompt_length =
      prefix_length + prompt_payload.size + suffix_length;
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, prompt_length, (void**)&storage));
  memcpy(storage, id4_ideogram4_qwen_prompt_prefix, prefix_length);
  memcpy(storage + prefix_length, prompt_payload.data, prompt_payload.size);
  memcpy(storage + prefix_length + prompt_payload.size,
         id4_ideogram4_qwen_prompt_suffix, suffix_length);
  *out_qwen_prompt = iree_make_string_view(storage, prompt_length);
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_select_prompt_payload(
    iree_string_view_t object, iree_string_view_t* out_prompt_payload,
    iree_string_view_t* out_generation) {
  *out_prompt_payload = object;
  *out_generation = iree_string_view_empty();

  id4_ideogram4_request_generation_lookup_t lookup;
  memset(&lookup, 0, sizeof(lookup));
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, id4_ideogram4_request_find_generation_member, &lookup));
  if (!lookup.has_generation) return iree_ok_status();
  *out_generation = lookup.generation;

  id4_ideogram4_request_schema_validation_t validation;
  memset(&validation, 0, sizeof(validation));
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, id4_ideogram4_request_validate_full_schema_member, &validation));
  if (!validation.has_prompt || !validation.has_generation ||
      validation.member_count != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "full Ideogram 4 requests must contain exactly `prompt` and "
        "`generation` members");
  }
  return id4_ideogram4_request_lookup_required(object, IREE_SV("prompt"),
                                               out_prompt_payload);
}

iree_status_t id4_ideogram4_request_parse_json(
    iree_string_view_t json, iree_allocator_t host_allocator,
    id4_ideogram4_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  memset(out_request, 0, sizeof(*out_request));

  iree_string_view_t remaining = json;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&remaining));
  iree_string_view_t object = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&remaining, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&remaining));
  if (!iree_string_view_is_empty(remaining)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing data after Ideogram 4 request object");
  }

  iree_host_size_t member_count = 0;
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object(
      object, id4_ideogram4_request_count_member, &member_count));
  if (member_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 request object is empty");
  }

  iree_string_view_t prompt_payload = iree_string_view_empty();
  iree_string_view_t generation = iree_string_view_empty();
  iree_status_t status = id4_ideogram4_request_select_prompt_payload(
      object, &prompt_payload, &generation);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_request_copy_prompt_payload(
        prompt_payload, host_allocator, &out_request->prompt_payload);
  }
  if (iree_status_is_ok(status) && !iree_string_view_is_empty(generation)) {
    status = id4_ideogram4_request_parse_generation(generation,
                                                    &out_request->generation);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_request_wrap_qwen_prompt(
        out_request->prompt_payload, host_allocator, &out_request->qwen_prompt);
  }
  if (iree_status_is_ok(status) && !iree_string_view_is_empty(generation)) {
    out_request->flags |= ID4_IDEOGRAM4_REQUEST_FLAG_HAS_GENERATION;
  }
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_request_deinitialize(out_request, host_allocator);
  }
  return status;
}

iree_status_t id4_ideogram4_request_initialize_text(
    iree_string_view_t prompt_payload,
    const id4_ideogram4_request_generation_t* generation,
    iree_allocator_t host_allocator, id4_ideogram4_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  memset(out_request, 0, sizeof(*out_request));
  if (iree_string_view_is_empty(prompt_payload)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 request prompt must not be empty");
  }
  if (generation) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_request_validate_generation(generation));
  }

  iree_status_t status = id4_ideogram4_request_copy_string(
      prompt_payload, host_allocator, &out_request->prompt_payload);
  if (iree_status_is_ok(status) && generation) {
    out_request->generation = *generation;
    out_request->flags |= ID4_IDEOGRAM4_REQUEST_FLAG_HAS_GENERATION;
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_request_wrap_qwen_prompt(
        out_request->prompt_payload, host_allocator, &out_request->qwen_prompt);
  }
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_request_deinitialize(out_request, host_allocator);
  }
  return status;
}

void id4_ideogram4_request_deinitialize(id4_ideogram4_request_t* request,
                                        iree_allocator_t host_allocator) {
  if (!request) return;
  iree_allocator_free(host_allocator, (void*)request->qwen_prompt.data);
  iree_allocator_free(host_allocator, (void*)request->prompt_payload.data);
  memset(request, 0, sizeof(*request));
}

static iree_status_t id4_ideogram4_request_allocate_qwen_inputs(
    uint32_t token_count, uint32_t token_capacity,
    iree_allocator_t host_allocator, id4_ideogram4_qwen_inputs_t* out_inputs) {
  out_inputs->token_count = token_count;
  out_inputs->token_capacity = token_capacity;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, token_count, sizeof(out_inputs->token_ids[0]),
      (void**)&out_inputs->token_ids));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, token_count, sizeof(out_inputs->token_weights[0]),
      (void**)&out_inputs->token_weights));
  iree_host_size_t attention_element_count = 0;
  if (!iree_host_size_checked_mul(token_capacity,
                                  (iree_host_size_t)token_capacity,
                                  &attention_element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen attention mask element count overflow");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(host_allocator, attention_element_count,
                                  sizeof(out_inputs->attention_mask[0]),
                                  (void**)&out_inputs->attention_mask));
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_encode_qwen_tokens(
    const id4_ideogram4_qwen_lowering_options_t* options,
    iree_allocator_t host_allocator,
    iree_tokenizer_token_id_t** out_token_storage, uint32_t* out_token_count) {
  *out_token_storage = NULL;
  *out_token_count = 0;

  iree_tokenizer_token_id_t* token_storage = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, options->max_token_count, sizeof(token_storage[0]),
      (void**)&token_storage);
  iree_host_size_t token_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_encode(
        options->tokenizer, options->request->qwen_prompt,
        options->tokenizer_flags,
        iree_tokenizer_make_token_output(token_storage, NULL, NULL,
                                         options->max_token_count),
        host_allocator, &token_count);
  }
  if (iree_status_is_ok(status) && token_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 request prompt produced no tokens");
  }
  if (iree_status_is_ok(status) && token_count > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Ideogram 4 request token count %" PRIhsz
                              " exceeds uint32 range",
                              token_count);
  }
  for (iree_host_size_t i = 0; i < token_count && iree_status_is_ok(status);
       ++i) {
    const iree_tokenizer_token_id_t token_id = token_storage[i];
    if (token_id < 0 || (uint32_t)token_id >= options->vocab_size) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "Ideogram 4 request token at index %" PRIhsz
                                " has id %" PRId32
                                " outside Qwen vocab size %" PRIu32,
                                i, token_id, options->vocab_size);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_token_storage = token_storage;
    *out_token_count = (uint32_t)token_count;
  } else {
    iree_allocator_free(host_allocator, token_storage);
  }
  return status;
}

iree_status_t id4_ideogram4_request_lower_qwen_inputs(
    const id4_ideogram4_qwen_lowering_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_qwen_inputs_t* out_inputs) {
  IREE_ASSERT_ARGUMENT(out_inputs);
  memset(out_inputs, 0, sizeof(*out_inputs));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_request_validate_lowering_options(options));

  iree_tokenizer_token_id_t* token_storage = NULL;
  uint32_t token_count = 0;
  iree_status_t status = id4_ideogram4_request_encode_qwen_tokens(
      options, host_allocator, &token_storage, &token_count);
  if (iree_status_is_ok(status) && token_count > options->token_capacity) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Ideogram 4 request token count %" PRIu32
                              " exceeds planned Qwen token capacity %" PRIu32,
                              token_count, options->token_capacity);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_request_allocate_qwen_inputs(
        token_count, options->token_capacity, host_allocator, out_inputs);
  }
  if (iree_status_is_ok(status)) {
    memcpy(out_inputs->token_ids, token_storage,
           token_count * sizeof(out_inputs->token_ids[0]));
    for (iree_host_size_t i = 0; i < token_count; ++i) {
      out_inputs->token_weights[i] = 1.0f;
    }
    const float future_token_mask = -FLT_MAX / 4.0f;
    iree_host_size_t attention_element_count = 0;
    if (!iree_host_size_checked_mul(options->token_capacity,
                                    (iree_host_size_t)options->token_capacity,
                                    &attention_element_count)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "Qwen attention mask element count overflow");
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < attention_element_count; ++i) {
        out_inputs->attention_mask[i] = future_token_mask;
      }
      for (iree_host_size_t query = 0; query < token_count; ++query) {
        for (iree_host_size_t key = 0; key < token_count; ++key) {
          out_inputs->attention_mask
              [query * (iree_host_size_t)options->token_capacity + key] =
              key <= query ? 0.0f : future_token_mask;
        }
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_qwen_inputs_deinitialize(out_inputs, host_allocator);
  }
  iree_allocator_free(host_allocator, token_storage);
  return status;
}

iree_status_t id4_ideogram4_request_count_qwen_tokens(
    const id4_ideogram4_qwen_lowering_options_t* options,
    iree_allocator_t host_allocator, uint32_t* out_token_count) {
  IREE_ASSERT_ARGUMENT(out_token_count);
  *out_token_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_request_validate_qwen_encode_options(options));

  iree_tokenizer_token_id_t* token_storage = NULL;
  iree_status_t status = id4_ideogram4_request_encode_qwen_tokens(
      options, host_allocator, &token_storage, out_token_count);
  iree_allocator_free(host_allocator, token_storage);
  return status;
}

void id4_ideogram4_qwen_inputs_deinitialize(id4_ideogram4_qwen_inputs_t* inputs,
                                            iree_allocator_t host_allocator) {
  if (!inputs) return;
  iree_allocator_free(host_allocator, inputs->attention_mask);
  iree_allocator_free(host_allocator, inputs->token_weights);
  iree_allocator_free(host_allocator, inputs->token_ids);
  memset(inputs, 0, sizeof(*inputs));
}

typedef enum id4_ideogram4_request_dit_branch_kind_e {
  ID4_IDEOGRAM4_REQUEST_DIT_BRANCH_CONDITIONED = 0,
  ID4_IDEOGRAM4_REQUEST_DIT_BRANCH_UNCONDITIONED = 1,
} id4_ideogram4_request_dit_branch_kind_t;

static iree_status_t id4_ideogram4_request_checked_mul_u32(
    uint32_t lhs, uint32_t rhs, iree_string_view_t label,
    uint32_t* out_result) {
  if (lhs != 0 && rhs > UINT32_MAX / lhs) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 %.*s count overflows uint32",
                            (int)label.size, label.data);
  }
  *out_result = lhs * rhs;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_checked_add_u32(
    uint32_t lhs, uint32_t rhs, iree_string_view_t label,
    uint32_t* out_result) {
  if (rhs > UINT32_MAX - lhs) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 %.*s count overflows uint32",
                            (int)label.size, label.data);
  }
  *out_result = lhs + rhs;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_validate_dit_lowering_options(
    const id4_ideogram4_dit_lowering_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DiT request lowering options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DiT request lowering options structure size "
                            "%" PRIhsz " is smaller than expected %" PRIhsz,
                            options->structure_size, sizeof(*options));
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "DiT request lowering extension structures are not supported");
  }
  if (!options->generation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DiT request lowering generation is required");
  }
  if (options->generation->latent_width == 0 ||
      options->generation->latent_height == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DiT request lowering latent shape is empty");
  }
  if (options->text_token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DiT request lowering text token count is zero");
  }
  if (options->attention_head_size == 0 ||
      (options->attention_head_size % 2) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "DiT request lowering attention head size must be nonzero and even");
  }
  const uint32_t half_size = options->attention_head_size / 2;
  const uint32_t required_height_half_size =
      ID4_IDEOGRAM4_DIT_MROPE_SECTION_HEIGHT * 3;
  const uint32_t required_width_half_size =
      ID4_IDEOGRAM4_DIT_MROPE_SECTION_WIDTH * 3;
  const uint32_t required_half_size =
      required_height_half_size > required_width_half_size
          ? required_height_half_size
          : required_width_half_size;
  if (half_size < required_half_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DiT request lowering attention head size %" PRIu32
                            " is too small for Ideogram 4 MRoPE sections",
                            options->attention_head_size);
  }
  return iree_ok_status();
}

static void id4_ideogram4_dit_branch_inputs_deinitialize(
    id4_ideogram4_dit_branch_inputs_t* inputs,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, inputs->position_embedding);
  iree_allocator_free(host_allocator, inputs->image_indicator);
  memset(inputs, 0, sizeof(*inputs));
}

static iree_status_t id4_ideogram4_request_calculate_dit_byte_length(
    iree_host_size_t element_count, iree_host_size_t element_size,
    iree_string_view_t label, iree_host_size_t* out_byte_length) {
  if (!iree_host_size_checked_mul(element_count, element_size,
                                  out_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 %.*s byte length overflows host size",
                            (int)label.size, label.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_allocate_dit_branch_inputs(
    uint32_t token_count, uint32_t attention_head_size,
    iree_allocator_t host_allocator,
    id4_ideogram4_dit_branch_inputs_t* out_inputs) {
  out_inputs->token_count = token_count;
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_calculate_dit_byte_length(
      token_count, sizeof(out_inputs->image_indicator[0]),
      IREE_SV("DiT image indicator"),
      &out_inputs->image_indicator_byte_length));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, token_count, sizeof(out_inputs->image_indicator[0]),
      (void**)&out_inputs->image_indicator));

  const uint32_t half_size = attention_head_size / 2;
  iree_host_size_t position_element_count = 0;
  if (!iree_host_size_checked_mul(4, half_size, &position_element_count) ||
      !iree_host_size_checked_mul(position_element_count, token_count,
                                  &position_element_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram 4 DiT position embedding element count overflows host size");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_calculate_dit_byte_length(
      position_element_count, sizeof(out_inputs->position_embedding[0]),
      IREE_SV("DiT position embedding"),
      &out_inputs->position_embedding_byte_length));
  return iree_allocator_malloc_array(host_allocator, position_element_count,
                                     sizeof(out_inputs->position_embedding[0]),
                                     (void**)&out_inputs->position_embedding);
}

static uint32_t id4_ideogram4_request_dit_position_axis(uint32_t half_channel) {
  if (half_channel < ID4_IDEOGRAM4_DIT_MROPE_SECTION_HEIGHT * 3 &&
      (half_channel % 3) == 1) {
    return 1;
  }
  if (half_channel < ID4_IDEOGRAM4_DIT_MROPE_SECTION_WIDTH * 3 &&
      (half_channel % 3) == 2) {
    return 2;
  }
  return 0;
}

static void id4_ideogram4_request_calculate_dit_position(
    id4_ideogram4_request_dit_branch_kind_t branch_kind,
    const id4_ideogram4_request_generation_t* generation,
    uint32_t text_token_count, uint32_t token_ordinal,
    uint32_t out_position[3]) {
  if (branch_kind == ID4_IDEOGRAM4_REQUEST_DIT_BRANCH_CONDITIONED &&
      token_ordinal < text_token_count) {
    out_position[0] = token_ordinal;
    out_position[1] = token_ordinal;
    out_position[2] = token_ordinal;
    return;
  }

  const uint32_t image_ordinal =
      branch_kind == ID4_IDEOGRAM4_REQUEST_DIT_BRANCH_CONDITIONED
          ? token_ordinal - text_token_count
          : token_ordinal;
  const uint32_t image_x = image_ordinal % generation->latent_width;
  const uint32_t image_y = image_ordinal / generation->latent_width;
  out_position[0] = ID4_IDEOGRAM4_DIT_IMAGE_POSITION_OFFSET;
  out_position[1] = ID4_IDEOGRAM4_DIT_IMAGE_POSITION_OFFSET + image_y;
  out_position[2] = ID4_IDEOGRAM4_DIT_IMAGE_POSITION_OFFSET + image_x;
}

static iree_host_size_t id4_ideogram4_request_dit_position_embedding_offset(
    uint32_t token_count, uint32_t half_size, uint32_t outer_ordinal,
    uint32_t inner_ordinal, uint32_t half_channel, uint32_t token_ordinal) {
  return ((((iree_host_size_t)outer_ordinal * 2 + inner_ordinal) * half_size +
           half_channel) *
          token_count) +
         token_ordinal;
}

static float id4_ideogram4_request_dit_mrope_inv_frequency(
    uint32_t half_channel, uint32_t attention_head_size) {
  const float exponent =
      (2.0f * (float)half_channel) / (float)attention_head_size;
  const float inv_frequency =
      1.0f / powf(id4_ideogram4_dit_mrope_theta, exponent);
  // The official BF16 model stores this non-persistent MRoPE buffer in BF16
  // before widening it to F32 for position multiplication.
  return iree_math_bf16_to_f32(iree_math_f32_to_bf16(inv_frequency));
}

static void id4_ideogram4_request_fill_dit_branch_inputs(
    id4_ideogram4_request_dit_branch_kind_t branch_kind,
    const id4_ideogram4_dit_lowering_options_t* options,
    id4_ideogram4_dit_branch_inputs_t* inputs) {
  const uint32_t text_token_count =
      branch_kind == ID4_IDEOGRAM4_REQUEST_DIT_BRANCH_CONDITIONED
          ? options->text_token_count
          : 0;
  for (uint32_t i = 0; i < inputs->token_count; ++i) {
    inputs->image_indicator[i] = i < text_token_count
                                     ? ID4_IDEOGRAM4_DIT_TEXT_INDICATOR
                                     : ID4_IDEOGRAM4_DIT_IMAGE_INDICATOR;
  }

  const uint32_t half_size = options->attention_head_size / 2;
  for (uint32_t half_channel = 0; half_channel < half_size; ++half_channel) {
    const uint32_t axis = id4_ideogram4_request_dit_position_axis(half_channel);
    const float inv_frequency = id4_ideogram4_request_dit_mrope_inv_frequency(
        half_channel, options->attention_head_size);
    for (uint32_t token_ordinal = 0; token_ordinal < inputs->token_count;
         ++token_ordinal) {
      uint32_t position[3] = {0, 0, 0};
      id4_ideogram4_request_calculate_dit_position(
          branch_kind, options->generation, options->text_token_count,
          token_ordinal, position);
      const float frequency = (float)position[axis] * inv_frequency;
      const float cos_value = cosf(frequency);
      const float sin_value = sinf(frequency);
      inputs->position_embedding
          [id4_ideogram4_request_dit_position_embedding_offset(
              inputs->token_count, half_size, 0, 0, half_channel,
              token_ordinal)] = cos_value;
      inputs->position_embedding
          [id4_ideogram4_request_dit_position_embedding_offset(
              inputs->token_count, half_size, 0, 1, half_channel,
              token_ordinal)] = sin_value;
      inputs->position_embedding
          [id4_ideogram4_request_dit_position_embedding_offset(
              inputs->token_count, half_size, 1, 0, half_channel,
              token_ordinal)] = -sin_value;
      inputs->position_embedding
          [id4_ideogram4_request_dit_position_embedding_offset(
              inputs->token_count, half_size, 1, 1, half_channel,
              token_ordinal)] = cos_value;
    }
  }
}

iree_status_t id4_ideogram4_request_lower_dit_inputs(
    const id4_ideogram4_dit_lowering_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_dit_inputs_t* out_inputs) {
  IREE_ASSERT_ARGUMENT(out_inputs);
  memset(out_inputs, 0, sizeof(*out_inputs));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_request_validate_dit_lowering_options(options));

  uint32_t image_token_count = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_checked_mul_u32(
      options->generation->latent_width, options->generation->latent_height,
      IREE_SV("DiT image token"), &image_token_count));
  uint32_t conditioned_token_count = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_checked_add_u32(
      options->text_token_count, image_token_count,
      IREE_SV("DiT conditioned token"), &conditioned_token_count));

  iree_status_t status = id4_ideogram4_request_allocate_dit_branch_inputs(
      conditioned_token_count, options->attention_head_size, host_allocator,
      &out_inputs->conditioned);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_request_allocate_dit_branch_inputs(
        image_token_count, options->attention_head_size, host_allocator,
        &out_inputs->unconditioned);
  }
  if (iree_status_is_ok(status)) {
    out_inputs->text_token_count = options->text_token_count;
    out_inputs->image_token_count = image_token_count;
    id4_ideogram4_request_fill_dit_branch_inputs(
        ID4_IDEOGRAM4_REQUEST_DIT_BRANCH_CONDITIONED, options,
        &out_inputs->conditioned);
    id4_ideogram4_request_fill_dit_branch_inputs(
        ID4_IDEOGRAM4_REQUEST_DIT_BRANCH_UNCONDITIONED, options,
        &out_inputs->unconditioned);
  } else {
    id4_ideogram4_dit_inputs_deinitialize(out_inputs, host_allocator);
  }
  return status;
}

void id4_ideogram4_dit_inputs_deinitialize(id4_ideogram4_dit_inputs_t* inputs,
                                           iree_allocator_t host_allocator) {
  if (!inputs) return;
  id4_ideogram4_dit_branch_inputs_deinitialize(&inputs->unconditioned,
                                               host_allocator);
  id4_ideogram4_dit_branch_inputs_deinitialize(&inputs->conditioned,
                                               host_allocator);
  memset(inputs, 0, sizeof(*inputs));
}

static iree_status_t id4_ideogram4_request_validate_denoise_generation(
    const id4_ideogram4_request_generation_t* generation) {
  if (!generation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation metadata is required");
  }
  if (generation->denoise_step_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 denoise step count is zero");
  }
  if (!isfinite(generation->guidance_scale) ||
      generation->guidance_scale <= 0.0f ||
      generation->guidance_scale > FLT_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 guidance scale is invalid");
  }
  return iree_ok_status();
}

static float id4_ideogram4_request_denoise_step_sigma(uint32_t step_ordinal,
                                                      uint32_t step_count) {
  if (step_ordinal == step_count) return 0.0f;
  if (step_count == 1) return 1.0f;
  const float max_timestep = 999.0f;
  const float timestep_delta = max_timestep / (float)(step_count - 1);
  const float scheduler_t = max_timestep - timestep_delta * (float)step_ordinal;
  return (scheduler_t + 1.0f) / 1000.0f;
}

static void id4_ideogram4_request_initialize_denoise_step(
    const id4_ideogram4_request_generation_t* generation, uint32_t step_ordinal,
    id4_ideogram4_denoise_step_t* out_step) {
  memset(out_step, 0, sizeof(*out_step));
  const float sigma = id4_ideogram4_request_denoise_step_sigma(
      step_ordinal, generation->denoise_step_count);
  const float next_sigma = id4_ideogram4_request_denoise_step_sigma(
      step_ordinal + 1, generation->denoise_step_count);
  out_step->timestep = 1000.0f - sigma * 1000.0f;
  out_step->scalings[0] = 1.0f;
  out_step->scalings[1] = -sigma;
  out_step->scalings[2] = 1.0f;
  out_step->sigmas[0] = sigma;
  out_step->sigmas[1] = next_sigma;
  out_step->guidance[0] = generation->guidance_scale;
}

iree_status_t id4_ideogram4_request_generation_lower_denoise_schedule(
    const id4_ideogram4_request_generation_t* generation,
    iree_allocator_t host_allocator,
    id4_ideogram4_denoise_schedule_t* out_schedule) {
  IREE_ASSERT_ARGUMENT(out_schedule);
  memset(out_schedule, 0, sizeof(*out_schedule));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_request_validate_denoise_generation(generation));

  id4_ideogram4_denoise_step_t* steps = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, generation->denoise_step_count, sizeof(steps[0]),
      (void**)&steps);
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < generation->denoise_step_count; ++i) {
      id4_ideogram4_request_initialize_denoise_step(generation, i, &steps[i]);
    }
    out_schedule->step_count = generation->denoise_step_count;
    out_schedule->steps = steps;
  }
  return status;
}

void id4_ideogram4_denoise_schedule_deinitialize(
    id4_ideogram4_denoise_schedule_t* schedule,
    iree_allocator_t host_allocator) {
  if (!schedule) return;
  iree_allocator_free(host_allocator, schedule->steps);
  memset(schedule, 0, sizeof(*schedule));
}
