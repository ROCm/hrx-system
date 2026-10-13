// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/embedding/protocol.h"

#include <inttypes.h>
#include <math.h>
#include <string.h>

#include "iree/base/internal/base64.h"
#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "loom/util/json.h"

typedef struct embedding_field_t {
  // Literal public field name.
  const char* name;
  // Allowed JSON types as bit positions.
  uint32_t types;
  // Actual JSON type received at the boundary.
  iree_json_value_type_t type;
  // Borrowed raw value; NULL denotes absence.
  iree_string_view_t value;
} embedding_field_t;

enum { EMBEDDING_FIELD_COUNT = 5 };

static iree_status_t embedding_field(void* user_data, iree_string_view_t key,
                                     iree_json_value_type_t type,
                                     iree_string_view_t value) {
  embedding_field_t* fields = user_data;
  for (iree_host_size_t i = 0; i < EMBEDDING_FIELD_COUNT; ++i) {
    embedding_field_t* field = &fields[i];
    if (!iree_string_view_equal(key, iree_make_cstring_view(field->name))) {
      continue;
    }
    if (field->value.data || !(field->types & (1u << type))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate or wrongly typed field '%s'",
                              field->name);
    }
    field->type = type;
    field->value = value;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported embeddings field '%.*s'", (int)key.size,
                          key.data);
}

typedef struct embedding_inputs_t {
  // Packed metadata, with offsets relative to payload until final assembly.
  iree_string_builder_t records;
  // Decoded strings or LE i32 IDs with eight-byte-aligned beginnings.
  iree_string_builder_t payload;
  // Number of complete input records.
  iree_host_size_t count;
  // Homogeneous top-level array element type, selected by its first element.
  iree_json_value_type_t type;
} embedding_inputs_t;

static iree_status_t embedding_token(void* user_data, iree_host_size_t index,
                                     iree_json_value_type_t type,
                                     iree_string_view_t value) {
  (void)index;
  if (type != IREE_JSON_VALUE_TYPE_NUMBER) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "token IDs must be nonnegative i32 integers");
  }
  uint64_t id = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(value, &id));
  if (id > INT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "token ID exceeds the i32 range");
  }
  uint8_t bytes[4];
  iree_unaligned_store_le_u32(bytes, (uint32_t)id);
  return iree_string_builder_append_string(
      user_data, iree_make_string_view((const char*)bytes, sizeof(bytes)));
}

static iree_status_t embedding_record(embedding_inputs_t* inputs, uint64_t kind,
                                      iree_host_size_t offset) {
  const iree_host_size_t end = iree_string_builder_size(&inputs->payload);
  uint8_t record[24];
  iree_unaligned_store_le_u64(record, kind);
  iree_unaligned_store_le_u64(record + 8, offset);
  iree_unaligned_store_le_u64(record + 16, end - offset);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      &inputs->records,
      iree_make_string_view((const char*)record, sizeof(record))));
  ++inputs->count;
  const char zeros[8] = {0};
  return iree_string_builder_append_string(
      &inputs->payload, iree_make_string_view(zeros, (8 - end % 8) % 8));
}

static iree_status_t embedding_sequence(embedding_inputs_t* inputs,
                                        iree_json_value_type_t type,
                                        iree_string_view_t value) {
  const iree_host_size_t offset = iree_string_builder_size(&inputs->payload);
  if (type == IREE_JSON_VALUE_TYPE_STRING) {
    char* target = NULL;
    iree_host_size_t capacity = 0, length = 0;
    IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
        &inputs->payload, value.size, &target, &capacity));
    IREE_RETURN_IF_ERROR(
        iree_json_unescape_string(value, capacity, target, &length));
    if (!iree_unicode_utf8_validate(iree_make_string_view(target, length))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "embedding input must decode to UTF-8");
    }
    iree_string_builder_commit_append(&inputs->payload, length);
  } else {
    IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(value, embedding_token,
                                                         &inputs->payload));
  }
  return embedding_record(inputs, type == IREE_JSON_VALUE_TYPE_STRING ? 0 : 1,
                          offset);
}

static iree_status_t embedding_element(void* user_data, iree_host_size_t index,
                                       iree_json_value_type_t type,
                                       iree_string_view_t value) {
  embedding_inputs_t* inputs = user_data;
  if (!index) {
    inputs->type = type;
  }
  if (type != inputs->type || (type != IREE_JSON_VALUE_TYPE_STRING &&
                               type != IREE_JSON_VALUE_TYPE_NUMBER &&
                               type != IREE_JSON_VALUE_TYPE_ARRAY)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "input requires homogeneous strings, IDs or ID arrays");
  }
  return type == IREE_JSON_VALUE_TYPE_NUMBER
             ? embedding_token(&inputs->payload, index, type, value)
             : embedding_sequence(inputs, type, value);
}

static iree_status_t embedding_parse(iree_string_view_t body,
                                     loom_serve_embedding_request_t* request,
                                     embedding_inputs_t* inputs,
                                     iree_allocator_t allocator) {
  if (!iree_unicode_utf8_validate(body)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "JSON must be UTF-8");
  }
  iree_string_view_t cursor = body, object;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing request JSON");
  }
  const uint32_t string_type = 1u << IREE_JSON_VALUE_TYPE_STRING;
  embedding_field_t fields[EMBEDDING_FIELD_COUNT] = {
      {.name = "model", .types = string_type},
      {.name = "input",
       .types = string_type | (1u << IREE_JSON_VALUE_TYPE_ARRAY)},
      {.name = "dimensions", .types = 1u << IREE_JSON_VALUE_TYPE_NUMBER},
      {.name = "encoding_format", .types = string_type},
      {.name = "user", .types = string_type}};
  IREE_RETURN_IF_ERROR(
      iree_json_enumerate_object_typed(object, embedding_field, fields));
  if (!fields[0].value.data || !fields[1].value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model and input are required");
  }
  if (fields[2].value.data) {
    IREE_RETURN_IF_ERROR(
        iree_json_parse_uint64(fields[2].value, &request->dimensions));
    if (!request->dimensions || request->dimensions > INT64_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dimensions must be a positive i64");
    }
  }
  if (fields[3].value.data) {
    char encoding[6];
    iree_host_size_t length = 0;
    IREE_RETURN_IF_ERROR(
        iree_json_unescape_string(fields[3].value, 0, NULL, &length));
    if (length > sizeof(encoding)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "encoding_format must be float or base64");
    }
    IREE_RETURN_IF_ERROR(iree_json_unescape_string(
        fields[3].value, sizeof(encoding), encoding, &length));
    const iree_string_view_t format = iree_make_string_view(encoding, length);
    if (iree_string_view_equal(format, IREE_SV("base64"))) {
      request->encoding = LOOM_SERVE_EMBEDDING_ENCODING_BASE64;
    } else if (!iree_string_view_equal(format, IREE_SV("float"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "encoding_format must be float or base64");
    }
  }
  if (fields[1].type == IREE_JSON_VALUE_TYPE_STRING) {
    IREE_RETURN_IF_ERROR(
        embedding_sequence(inputs, fields[1].type, fields[1].value));
  } else {
    IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
        fields[1].value, embedding_element, inputs));
    if (inputs->type == IREE_JSON_VALUE_TYPE_NUMBER) {
      IREE_RETURN_IF_ERROR(embedding_record(inputs, 1, 0));
    }
    if (!inputs->count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "input batch must not be empty");
    }
  }
  const iree_string_view_t records = iree_string_builder_view(&inputs->records);
  const iree_string_view_t payload = iree_string_builder_view(&inputs->payload);
  const iree_host_size_t header = 16 + records.size;
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_buffer_create(header + payload.size, 8, allocator, &buffer));
  iree_byte_span_t bytes;
  iree_status_t status =
      iree_vm_buffer_map_write(buffer, 0, header + payload.size, &bytes);
  if (iree_status_is_ok(status)) {
    iree_unaligned_store_le_u64(bytes.data, request->dimensions);
    iree_unaligned_store_le_u64(bytes.data + 8, inputs->count);
    memcpy(bytes.data + 16, records.data, records.size);
    memcpy(bytes.data + header, payload.data, payload.size);
    for (iree_host_size_t i = 0; i < inputs->count; ++i) {
      uint8_t* offset = bytes.data + 16 + i * 24 + 8;
      iree_unaligned_store_le_u64(offset,
                                  header + iree_unaligned_load_le_u64(offset));
    }
    status = iree_vm_buffer_subspan(buffer, 0, bytes.data_length,
                                    IREE_VM_BUFFER_ACCESS_FLAG_READ, allocator,
                                    &request->base.input);
    request->base.count = inputs->count;
  }
  iree_vm_buffer_release(buffer);
  return status;
}

void loom_serve_embedding_request_deinitialize(
    loom_serve_embedding_request_t* request) {
  iree_vm_buffer_release(request->base.input);
  memset(request, 0, sizeof(*request));
}

iree_status_t loom_serve_embedding_request_initialize(
    iree_string_view_t body, loom_serve_embedding_request_t* out_request,
    iree_allocator_t allocator) {
  memset(out_request, 0, sizeof(*out_request));
  embedding_inputs_t inputs = {.type = IREE_JSON_VALUE_TYPE_NULL};
  iree_string_builder_initialize(allocator, &inputs.records);
  iree_string_builder_initialize(allocator, &inputs.payload);
  iree_status_t status = embedding_parse(body, out_request, &inputs, allocator);
  iree_string_builder_deinitialize(&inputs.records);
  iree_string_builder_deinitialize(&inputs.payload);
  if (!iree_status_is_ok(status)) {
    loom_serve_embedding_request_deinitialize(out_request);
  }
  return status;
}

static iree_status_t embedding_vector(iree_const_byte_span_t bytes,
                                      loom_serve_embedding_encoding_t encoding,
                                      iree_string_builder_t* output) {
  // Both wire encodings obey the same finite-value contract.
  for (iree_host_size_t i = 0; i < bytes.data_length; i += 4) {
    const uint32_t bits = iree_unaligned_load_le_u32(bytes.data + i);
    float value;
    memcpy(&value, &bits, sizeof(value));
    if (!isfinite(value)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "source produced a nonfinite embedding");
    }
  }
  if (encoding == LOOM_SERVE_EMBEDDING_ENCODING_BASE64) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\""));
    char* target = NULL;
    iree_host_size_t capacity = 0, length = 0;
    IREE_RETURN_IF_ERROR(iree_string_builder_reserve_for_append(
        output, iree_base64_encoded_size(bytes.data_length), &target,
        &capacity));
    IREE_RETURN_IF_ERROR(iree_base64_encode(
        bytes, iree_make_mutable_string_view(target, capacity), &length));
    iree_string_builder_commit_append(output, length);
    return iree_string_builder_append_cstring(output, "\"");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "["));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < bytes.data_length && iree_status_is_ok(status); i += 4) {
    const uint32_t bits = iree_unaligned_load_le_u32(bytes.data + i);
    float value;
    memcpy(&value, &bits, sizeof(value));
    status = iree_string_builder_append_format(output, "%s%.9g", i ? "," : "",
                                               (double)value);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "]");
  }
  return status;
}

iree_status_t loom_serve_embedding_response(
    const loom_serve_embedding_request_t* request, iree_string_view_t model,
    const iree_const_byte_span_t* rows, uint64_t input_tokens,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      output, "{\"object\":\"list\",\"model\":"));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, model));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"data\":["));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < request->base.count && iree_status_is_ok(status); ++i) {
    if (request->dimensions && rows[i].data_length / 4 != request->dimensions) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "source ignored requested embedding dimensions");
    } else {
      status = iree_string_builder_append_format(
          output, "%s{\"object\":\"embedding\",\"index\":%zu,\"embedding\":",
          i ? "," : "", i);
    }
    if (iree_status_is_ok(status)) {
      status = embedding_vector(rows[i], request->encoding, output);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(output, "}");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        output,
        "],\"usage\":{\"prompt_tokens\":%" PRIu64 ",\"total_tokens\":%" PRIu64
        "}}",
        input_tokens, input_tokens);
  }
  return status;
}

static iree_status_t embedding_initialize(iree_string_view_t body,
                                          loom_serve_finite_request_t* request,
                                          iree_allocator_t allocator) {
  return loom_serve_embedding_request_initialize(
      body, (loom_serve_embedding_request_t*)request, allocator);
}

static void embedding_deinitialize(loom_serve_finite_request_t* request) {
  loom_serve_embedding_request_deinitialize(
      (loom_serve_embedding_request_t*)request);
}

static iree_status_t embedding_response(
    const loom_serve_finite_request_t* request, iree_string_view_t model,
    const iree_const_byte_span_t* rows, uint64_t input_tokens,
    iree_string_builder_t* output) {
  return loom_serve_embedding_response(
      (const loom_serve_embedding_request_t*)request, model, rows, input_tokens,
      output);
}

const loom_serve_finite_protocol_t* loom_serve_embedding_protocol(void) {
  static const loom_serve_finite_protocol_t protocol = {
      .name = "embedding",
      .endpoint = "/v1/embeddings",
      .output_modality = IREE_SVL("embedding"),
      .invalid_status = "400 Bad Request",
      .request_size = sizeof(loom_serve_embedding_request_t),
      .initialize = embedding_initialize,
      .deinitialize = embedding_deinitialize,
      .response = embedding_response};
  return &protocol;
}
