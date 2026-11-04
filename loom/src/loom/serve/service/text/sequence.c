// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/sequence.h"

#include <string.h>

#include "iree/base/internal/base64.h"
#include "iree/base/internal/csprng.h"
#include "iree/base/internal/json.h"

typedef struct loom_serve_text_sequence_fields_t {
  // Borrowed values for model, session, position, token_ids, and options.
  iree_string_view_t values[5];
} loom_serve_text_sequence_fields_t;

static iree_status_t loom_serve_text_sequence_field(void* user_data,
                                                    iree_string_view_t key,
                                                    iree_json_value_type_t type,
                                                    iree_string_view_t value) {
  loom_serve_text_sequence_fields_t* fields = user_data;
  static const char* const names[] = {"model", "session", "position",
                                      "token_ids", "options"};
  static const iree_json_value_type_t types[] = {
      IREE_JSON_VALUE_TYPE_STRING, IREE_JSON_VALUE_TYPE_STRING,
      IREE_JSON_VALUE_TYPE_NUMBER, IREE_JSON_VALUE_TYPE_ARRAY,
      IREE_JSON_VALUE_TYPE_OBJECT};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(names); ++i) {
    if (iree_string_view_equal(key, iree_make_cstring_view(names[i]))) {
      if (fields->values[i].data || type != types[i]) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "duplicate or wrongly typed sequence field '%s'", names[i]);
      }
      fields->values[i] = value;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported sequence field '%.*s'", (int)key.size,
                          key.data);
}

iree_status_t loom_serve_text_sequence_parse(
    iree_string_view_t body, loom_serve_text_sequence_t* out_sequence) {
  *out_sequence = (loom_serve_text_sequence_t){0};
  iree_string_view_t remaining = body;
  iree_string_view_t object;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&remaining));
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&remaining, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&remaining));
  if (remaining.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected content after sequence request");
  }
  loom_serve_text_sequence_fields_t fields = {0};
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_text_sequence_field, &fields));
  if (!fields.values[2].data || !fields.values[3].data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sequence requires position and token_ids");
  }
  uint64_t position = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(fields.values[2], &position));
  if (position > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sequence position exceeds host address space");
  }
  out_sequence->position = (iree_host_size_t)position;
  out_sequence->token_ids = fields.values[3];
  out_sequence->append = fields.values[1].data != NULL;
  if (out_sequence->append) {
    iree_host_size_t length = 0;
    IREE_RETURN_IF_ERROR(iree_json_unescape_string(
        fields.values[1], sizeof(out_sequence->handle) - 1,
        out_sequence->handle, &length));
    if (length != 32) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "session must be a server-issued sequence handle");
    }
    for (iree_host_size_t i = 0; i < length; ++i) {
      const char c = out_sequence->handle[i];
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid sequence handle");
      }
    }
  } else {
    uint8_t identity[16];
    IREE_RETURN_IF_ERROR(
        iree_csprng_fill(iree_make_byte_span(identity, sizeof(identity))));
    static const char hex[] = "0123456789abcdef";
    for (iree_host_size_t i = 0; i < sizeof(identity); ++i) {
      out_sequence->handle[2 * i] = hex[identity[i] >> 4];
      out_sequence->handle[2 * i + 1] = hex[identity[i] & 15];
    }
  }
  return iree_ok_status();
}

void loom_serve_text_sequence_deinitialize(
    loom_serve_text_sequence_t* sequence) {
  iree_vm_buffer_release(sequence->options);
  memset(sequence, 0, sizeof(*sequence));
}

typedef struct loom_serve_text_sequence_tokens_t {
  // Remaining model context available to the append.
  iree_host_size_t capacity;
  // Number of validated IDs written so far.
  iree_host_size_t count;
  // Native token bytes growing with actual input.
  iree_string_builder_t storage;
} loom_serve_text_sequence_tokens_t;

static iree_status_t loom_serve_text_sequence_token(void* user_data,
                                                    iree_host_size_t index,
                                                    iree_json_value_type_t type,
                                                    iree_string_view_t value) {
  loom_serve_text_sequence_tokens_t* tokens = user_data;
  int64_t token = 0;
  if (type != IREE_JSON_VALUE_TYPE_NUMBER) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "token_ids requires nonnegative int32 IDs");
  }
  IREE_RETURN_IF_ERROR(iree_json_parse_int64(value, &token));
  if (token < 0 || token > INT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "token ID is outside the nonnegative int32 range");
  }
  if (tokens->count == tokens->capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sequence exceeds remaining context capacity");
  }
  char* output = NULL;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_inline(
      &tokens->storage, sizeof(int32_t), &output));
  const int32_t token_id = (int32_t)token;
  memcpy(output, &token_id, sizeof(token_id));
  ++tokens->count;
  return iree_ok_status();
}

iree_status_t loom_serve_text_sequence_prepare(
    loom_serve_text_sequence_t* sequence, loom_serve_text_model_t* model,
    iree_string_view_t body, uint64_t request_id,
    loom_serve_text_input_prefix_t prefix, bool checkpoint,
    iree_allocator_t host_allocator, loom_serve_text_input_t* out_input) {
  const bool retained = sequence->append || checkpoint;
  if (retained && !prefix.completion) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "sequence is no longer retained; attach full history");
  }
  const iree_host_size_t position = retained ? prefix.position : 0;
  if (position != sequence->position) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected sequence position %zu but retained %zu",
                            sequence->position, position);
  }
  if (out_input->token_storage) {
    return iree_ok_status();
  }
  const iree_host_size_t capacity =
      loom_serve_text_model_context_capacity(model);
  loom_serve_text_sequence_tokens_t tokens = {.capacity = capacity - position};
  iree_string_builder_initialize(host_allocator, &tokens.storage);
  iree_status_t status = iree_json_enumerate_array_typed(
      sequence->token_ids, loom_serve_text_sequence_token, &tokens);
  if (iree_status_is_ok(status) && !tokens.count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "token_ids must not be empty");
  }
  iree_vm_buffer_t* owned = NULL;
  if (iree_status_is_ok(status)) {
    const iree_string_view_t bytes = iree_string_builder_view(&tokens.storage);
    status =
        iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                             iree_make_const_byte_span(bytes.data, bytes.size),
                             iree_alignof(int32_t), host_allocator, &owned);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_model_prepare_request(model, body, request_id,
                                                   owned, &sequence->options);
  }
  if (iree_status_is_ok(status)) {
    *out_input =
        (loom_serve_text_input_t){.token_storage = owned,
                                  .tokens = iree_vm_buffer_const_data(owned),
                                  .count = tokens.count,
                                  .retained_count = position};
  } else {
    iree_vm_buffer_release(owned);
  }
  iree_string_builder_deinitialize(&tokens.storage);
  return status;
}

iree_status_t loom_serve_text_sequence_chunk(
    iree_string_view_t handle, iree_host_size_t position,
    iree_host_size_t count, const int32_t* tokens, iree_const_byte_span_t bytes,
    bool final, iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "event: chunk\ndata: {\"session\":\"%.*s\",\"position\":%zu,"
      "\"next_position\":%zu,\"token_ids\":[",
      (int)handle.size, handle.data, position, position + count));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = iree_string_builder_append_format(output, "%s%d", i ? "," : "",
                                               tokens[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output, "],\"encoding\":\"base64\",\"data\":\""));
  char* target = NULL;
  iree_host_size_t capacity = 0, length = 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
      output, iree_base64_encoded_size(bytes.data_length), &target, &capacity));
  IREE_RETURN_IF_ERROR(iree_base64_encode(
      bytes, iree_make_mutable_string_view(target, capacity), &length));
  iree_string_builder_commit_append(output, length);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\"}\n\n"));
  if (final) {
    return iree_string_builder_append_format(
        output,
        "event: done\ndata: {\"session\":\"%.*s\",\"position\":%zu}\n\n",
        (int)handle.size, handle.data, position + count);
  }
  return iree_ok_status();
}
