// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/tokenization.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "loom/util/json.h"

typedef struct tokenization_field_t {
  // Protocol key, without escapes.
  const char* name;
  // Permitted JSON types as bit positions.
  uint32_t types;
  // Borrowed JSON value; NULL data means the field has not occurred.
  iree_string_view_t value;
} tokenization_field_t;

typedef struct tokenization_fields_t {
  // Mutable field table consumed exactly once at the HTTP input boundary.
  tokenization_field_t* values;
  // Number of fields accepted by this route.
  iree_host_size_t count;
} tokenization_fields_t;

static iree_status_t tokenization_field_visit(void* user_data,
                                              iree_string_view_t key,
                                              iree_json_value_type_t type,
                                              iree_string_view_t value) {
  tokenization_fields_t* fields = user_data;
  for (iree_host_size_t i = 0; i < fields->count; ++i) {
    tokenization_field_t* field = &fields->values[i];
    if (!iree_string_view_equal(key, iree_make_cstring_view(field->name))) {
      continue;
    }
    if (field->value.data || !(field->types & (1u << type))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate or wrongly typed field '%s'",
                              field->name);
    }
    field->value = value;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported tokenization field '%.*s'",
                          (int)key.size, key.data);
}

static iree_status_t tokenization_parse(iree_string_view_t body,
                                        tokenization_fields_t fields) {
  if (!iree_unicode_utf8_validate(body)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tokenization JSON must be UTF-8");
  }
  iree_string_view_t cursor = body, object;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing tokenization JSON");
  }
  return iree_json_enumerate_object_typed(object, tokenization_field_visit,
                                          &fields);
}

static iree_status_t tokenization_append_ids(iree_string_builder_t* output,
                                             iree_host_size_t count,
                                             const int32_t* ids) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "{\"tokens\":["));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = iree_string_builder_append_format(output, "%s%" PRId32,
                                               i ? "," : "", ids[i]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "]}");
  }
  return status;
}

// Pull at most one token beyond the public limit. Finalize has a separate
// non-retryable pending bound; allocating that bound preserves exact-limit
// behavior even when all input was consumed before the final tokens appeared.
static iree_status_t tokenization_encode(const iree_tokenizer_t* tokenizer,
                                         iree_string_view_t text,
                                         iree_tokenizer_encode_flags_t flags,
                                         iree_string_builder_t* output,
                                         iree_allocator_t allocator) {
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  const iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_oneshot_size(text.size);
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, state_size + transform_size, (void**)&storage));
  const iree_host_size_t capacity = LOOM_SERVE_HTTP_TOKENIZATION_MAX_TOKENS + 1;
  int32_t* tokens = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      allocator, capacity, sizeof(*tokens), (void**)&tokens);
  iree_tokenizer_encode_state_t* state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_encode_state_initialize(
        tokenizer, iree_make_byte_span(storage, state_size),
        iree_make_byte_span(storage + state_size, transform_size),
        (iree_tokenizer_offset_run_list_t){0},
        IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START | flags, &state);
  }
  iree_host_size_t count = 0;
  while (iree_status_is_ok(status) && text.size && count < capacity) {
    iree_host_size_t consumed = 0, produced = 0;
    status = iree_tokenizer_encode_state_feed(
        state, text,
        iree_tokenizer_make_token_output(tokens + count, NULL, NULL,
                                         capacity - count),
        &consumed, &produced);
    text = iree_string_view_substr(text, consumed, IREE_STRING_VIEW_NPOS);
    count += produced;
  }
  if (iree_status_is_ok(status) && count == capacity) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tokenization exceeds %u tokens",
                              LOOM_SERVE_HTTP_TOKENIZATION_MAX_TOKENS);
  }
  int32_t* pending = NULL;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t bound =
        iree_max(1, iree_tokenizer_encode_state_pending_token_bound(state));
    status = iree_allocator_malloc_array(allocator, bound, sizeof(*pending),
                                         (void**)&pending);
    iree_host_size_t produced = 0;
    if (iree_status_is_ok(status)) {
      status = iree_tokenizer_encode_state_finalize(
          state, iree_tokenizer_make_token_output(pending, NULL, NULL, bound),
          &produced);
    }
    if (iree_status_is_ok(status)) {
      if (produced > LOOM_SERVE_HTTP_TOKENIZATION_MAX_TOKENS - count) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "tokenization exceeds %u tokens",
                                  LOOM_SERVE_HTTP_TOKENIZATION_MAX_TOKENS);
      } else {
        memcpy(tokens + count, pending, produced * sizeof(*tokens));
        count += produced;
        status = tokenization_append_ids(output, count, tokens);
      }
    }
  }
  if (state) {
    iree_tokenizer_encode_state_deinitialize(state);
  }
  iree_allocator_free(allocator, pending);
  iree_allocator_free(allocator, tokens);
  iree_allocator_free(allocator, storage);
  return status;
}

iree_status_t loom_serve_http_tokenize(const iree_tokenizer_t* tokenizer,
                                       iree_string_view_t body,
                                       iree_string_builder_t* output,
                                       iree_allocator_t allocator) {
  const uint32_t string_type = 1u << IREE_JSON_VALUE_TYPE_STRING;
  const uint32_t boolean_types =
      (1u << IREE_JSON_VALUE_TYPE_TRUE) | (1u << IREE_JSON_VALUE_TYPE_FALSE);
  tokenization_field_t fields[] = {{"model", string_type},
                                   {"content", string_type},
                                   {"add_special", boolean_types},
                                   {"parse_special", boolean_types},
                                   {"with_pieces", boolean_types}};
  IREE_RETURN_IF_ERROR(tokenization_parse(
      body, (tokenization_fields_t){fields, IREE_ARRAYSIZE(fields)}));
  if (!fields[1].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "content string is required");
  }
  bool add_special = false, parse_special = true, with_pieces = false;
  if (fields[2].value.data) {
    IREE_RETURN_IF_ERROR(iree_json_parse_bool(fields[2].value, &add_special));
  }
  if (fields[3].value.data) {
    IREE_RETURN_IF_ERROR(iree_json_parse_bool(fields[3].value, &parse_special));
  }
  if (fields[4].value.data) {
    IREE_RETURN_IF_ERROR(iree_json_parse_bool(fields[4].value, &with_pieces));
  }
  if (with_pieces) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "raw token pieces are not supported");
  }
  char* text = NULL;
  // JSON unescaping cannot expand its source span. Allocate no more than the
  // declared text bound even when the transport admits a larger escaped body.
  const iree_host_size_t capacity = iree_min(
      fields[1].value.size, LOOM_SERVE_HTTP_TOKENIZATION_MAX_TEXT_BYTES + 1);
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, iree_max(1, capacity), (void**)&text));
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_json_unescape_string(fields[1].value, capacity, text, &length);
  if (iree_status_is_resource_exhausted(status)) {
    status = iree_status_join(
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "content exceeds %u decoded bytes",
                         LOOM_SERVE_HTTP_TOKENIZATION_MAX_TEXT_BYTES),
        status);
  }
  if (iree_status_is_ok(status) &&
      length > LOOM_SERVE_HTTP_TOKENIZATION_MAX_TEXT_BYTES) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "content exceeds %u decoded bytes",
                              LOOM_SERVE_HTTP_TOKENIZATION_MAX_TEXT_BYTES);
  }
  const iree_string_view_t value = iree_make_string_view(text, length);
  if (iree_status_is_ok(status) && !iree_unicode_utf8_validate(value)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "decoded content must be UTF-8");
  }
  if (iree_status_is_ok(status)) {
    const iree_tokenizer_encode_flags_t flags =
        (add_special ? IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS : 0) |
        (parse_special ? 0
                       : IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING);
    status = tokenization_encode(tokenizer, value, flags, output, allocator);
  }
  iree_allocator_free(allocator, text);
  return status;
}

typedef struct tokenization_ids_t {
  // Borrowed immutable vocabulary for external ID validation.
  const iree_tokenizer_vocab_t* vocab;
  // Writable token array, sized to the validated JSON array length.
  int32_t* values;
} tokenization_ids_t;

static iree_status_t tokenization_id_visit(void* user_data,
                                           iree_host_size_t index,
                                           iree_json_value_type_t type,
                                           iree_string_view_t value) {
  tokenization_ids_t* ids = user_data;
  uint64_t id = 0;
  if (type != IREE_JSON_VALUE_TYPE_NUMBER) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "token IDs must be integers");
  }
  iree_status_t status = iree_json_parse_uint64(value, &id);
  if (iree_status_is_out_of_range(status)) {
    status = iree_status_join(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "token ID is outside the integer range"),
        status);
  }
  IREE_RETURN_IF_ERROR(status);
  if (id > INT32_MAX || id >= iree_tokenizer_vocab_capacity(ids->vocab) ||
      iree_any_bit_set(
          iree_tokenizer_vocab_token_attrs(ids->vocab, (int32_t)id),
          IREE_TOKENIZER_TOKEN_ATTR_UNUSED)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "token ID %" PRIu64 " is not in the vocabulary",
                            id);
  }
  ids->values[index] = (int32_t)id;
  return iree_ok_status();
}

static iree_status_t tokenization_decode(const iree_tokenizer_t* tokenizer,
                                         iree_tokenizer_token_id_list_t tokens,
                                         iree_string_builder_t* output,
                                         iree_allocator_t allocator) {
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_decode_state_calculate_size(tokenizer, &state_size));
  const iree_host_size_t capacity =
      LOOM_SERVE_HTTP_TOKENIZATION_MAX_TEXT_BYTES + 1;
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, state_size + capacity,
                                             (void**)&storage));
  char* text = (char*)storage + state_size;
  iree_tokenizer_decode_state_t* state = NULL;
  iree_status_t status = iree_tokenizer_decode_state_initialize(
      tokenizer, IREE_TOKENIZER_DECODE_FLAG_NONE,
      iree_make_byte_span(storage, state_size), &state);
  iree_host_size_t length = 0;
  while (iree_status_is_ok(status) && tokens.count) {
    iree_host_size_t consumed = 0, produced = 0;
    status = iree_tokenizer_decode_state_feed(
        state, tokens,
        iree_make_mutable_string_view(text + length, capacity - length),
        &consumed, &produced);
    tokens.values += consumed;
    tokens.count -= consumed;
    length += produced;
    if (iree_status_is_ok(status) && !consumed && !produced) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "decoded content exceeds %u bytes",
                                LOOM_SERVE_HTTP_TOKENIZATION_MAX_TEXT_BYTES);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t produced = 0;
    status = iree_tokenizer_decode_state_finalize(
        state, iree_make_mutable_string_view(text + length, capacity - length),
        &produced);
    length += produced;
    // Finalize owns no allocations; RESOURCE_EXHAUSTED here means its output
    // does not fit, not failure to allocate the independent decoder state.
    if (iree_status_is_resource_exhausted(status)) {
      status = iree_status_join(
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "decoded content exceeds %u bytes",
                           LOOM_SERVE_HTTP_TOKENIZATION_MAX_TEXT_BYTES),
          status);
    }
  }
  if (iree_status_is_ok(status) &&
      length > LOOM_SERVE_HTTP_TOKENIZATION_MAX_TEXT_BYTES) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "decoded content exceeds %u bytes",
                              LOOM_SERVE_HTTP_TOKENIZATION_MAX_TEXT_BYTES);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "{\"content\":");
  }
  if (iree_status_is_ok(status)) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(output, &stream);
    status = loom_json_write_escaped_string(
        &stream, iree_make_string_view(text, length));
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "}");
  }
  if (state) {
    iree_tokenizer_decode_state_deinitialize(state);
  }
  iree_allocator_free(allocator, storage);
  return status;
}

iree_status_t loom_serve_http_detokenize(const iree_tokenizer_t* tokenizer,
                                         iree_string_view_t body,
                                         iree_string_builder_t* output,
                                         iree_allocator_t allocator) {
  tokenization_field_t fields[] = {
      {"model", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"tokens", 1u << IREE_JSON_VALUE_TYPE_ARRAY}};
  IREE_RETURN_IF_ERROR(tokenization_parse(
      body, (tokenization_fields_t){fields, IREE_ARRAYSIZE(fields)}));
  if (!fields[1].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tokens array is required");
  }
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(fields[1].value, &count));
  if (count > LOOM_SERVE_HTTP_TOKENIZATION_MAX_TOKENS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "detokenization exceeds %u tokens",
                            LOOM_SERVE_HTTP_TOKENIZATION_MAX_TOKENS);
  }
  tokenization_ids_t ids = {.vocab = iree_tokenizer_vocab(tokenizer)};
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, iree_max(1, count), sizeof(*ids.values), (void**)&ids.values));
  iree_status_t status = iree_json_enumerate_array_typed(
      fields[1].value, tokenization_id_visit, &ids);
  if (iree_status_is_ok(status)) {
    status = tokenization_decode(
        tokenizer, iree_tokenizer_make_token_id_list(ids.values, count), output,
        allocator);
  }
  iree_allocator_free(allocator, ids.values);
  return status;
}
