// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON serialization utilities for loom output streams.
//
// Provides a streaming JSON escape adapter per RFC 8259: wraps an inner
// loom_output_stream_t and escapes all incoming text before forwarding.
// No intermediate buffer — each fragment is escaped character by character,
// flushing literal runs as contiguous writes.
//
// Escapes:
//   " → \"     \ → \\     \n → \n     \r → \r     \t → \t
//   \b → \b    \f → \f    0x00-0x1F → \uNNNN
//   U+2028 (LINE SEPARATOR)      → \u2028
//   U+2029 (PARAGRAPH SEPARATOR) → \u2029
//
// All other valid UTF-8 (CJK, emoji, etc.) passes through unchanged. Malformed
// UTF-8 bytes are escaped as \ufffd so user-source excerpts from lexer
// diagnostics still produce valid JSON.
//
// Usage:
//
//   loom_output_stream_t stream;
//   loom_output_stream_for_builder(&builder, &stream);
//   // Write a quoted+escaped string value:
//   loom_json_write_escaped_string(&stream, iree_make_cstring_view("hello\n"));
//   // → "hello\n"
//
//   // Or use the raw escape adapter for streaming content:
//   loom_json_escape_stream_t escape_data;
//   loom_output_stream_t escape_stream;
//   loom_json_escape_stream_init(&stream, &escape_data, &escape_stream);
//   loom_output_stream_write_cstring(&stream, "\"");
//   loom_output_stream_write(&escape_stream, some_text);  // escaped
//   loom_output_stream_write_cstring(&stream, "\"");

#ifndef LOOM_UTIL_JSON_H_
#define LOOM_UTIL_JSON_H_

#include "iree/base/api.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// JSON-escaping stream adapter. Wraps an inner stream and escapes all
// incoming text per RFC 8259. Stack-allocated — no heap, no lifetime
// management.
typedef struct loom_json_escape_stream_t {
  loom_output_stream_t* inner;
} loom_json_escape_stream_t;

// Initializes a JSON-escaping stream that wraps |inner|. The resulting
// |out_stream| can be passed to any function expecting
// loom_output_stream_t* — all writes are escaped before reaching |inner|.
void loom_json_escape_stream_init(loom_output_stream_t* inner,
                                  loom_json_escape_stream_t* escape_data,
                                  loom_output_stream_t* out_stream);

// Writes a JSON-escaped string value (with surrounding quotes).
iree_status_t loom_json_write_escaped_string(loom_output_stream_t* stream,
                                             iree_string_view_t value);

// Streaming state for a JSON object. The writer emits directly to the output
// stream and owns field separator placement.
typedef struct loom_json_object_writer_t {
  // Output stream receiving the serialized object.
  loom_output_stream_t* stream;
  // Number of fields written to the object.
  iree_host_size_t field_count;
} loom_json_object_writer_t;

// Begins a JSON object and initializes |out_writer| to append fields to it.
iree_status_t loom_json_object_begin(loom_output_stream_t* stream,
                                     loom_json_object_writer_t* out_writer);

// Ends a JSON object previously begun with loom_json_object_begin.
iree_status_t loom_json_object_end(loom_json_object_writer_t* writer);

// Begins a named field in |writer|. The caller must write exactly one JSON
// value to writer->stream before beginning the next field or ending the object.
iree_status_t loom_json_object_begin_field(loom_json_object_writer_t* writer,
                                           iree_string_view_t name);

// Writes required object fields of the indicated type.
iree_status_t loom_json_object_write_string_field(
    loom_json_object_writer_t* writer, iree_string_view_t name,
    iree_string_view_t value);
// Writes a string field only when |value| is non-empty.
iree_status_t loom_json_object_write_string_field_if_nonempty(
    loom_json_object_writer_t* writer, iree_string_view_t name,
    iree_string_view_t value);
iree_status_t loom_json_object_write_uint32_field(
    loom_json_object_writer_t* writer, iree_string_view_t name, uint32_t value);
iree_status_t loom_json_object_write_uint64_field(
    loom_json_object_writer_t* writer, iree_string_view_t name, uint64_t value);
iree_status_t loom_json_object_write_int32_field(
    loom_json_object_writer_t* writer, iree_string_view_t name, int32_t value);
iree_status_t loom_json_object_write_int64_field(
    loom_json_object_writer_t* writer, iree_string_view_t name, int64_t value);
iree_status_t loom_json_object_write_bool_field(
    loom_json_object_writer_t* writer, iree_string_view_t name, bool value);
iree_status_t loom_json_object_write_host_size_field(
    loom_json_object_writer_t* writer, iree_string_view_t name,
    iree_host_size_t value);

// Writes a named field with an explicit JSON null value.
iree_status_t loom_json_object_write_null_field(
    loom_json_object_writer_t* writer, iree_string_view_t name);

// Streaming state for a JSON array. The writer emits directly to the output
// stream and owns element separator placement.
typedef struct loom_json_array_writer_t {
  // Output stream receiving the serialized array.
  loom_output_stream_t* stream;
  // Number of elements written to the array.
  iree_host_size_t element_count;
} loom_json_array_writer_t;

// Begins a JSON array and initializes |out_writer| to append elements to it.
iree_status_t loom_json_array_begin(loom_output_stream_t* stream,
                                    loom_json_array_writer_t* out_writer);

// Ends a JSON array previously begun with loom_json_array_begin.
iree_status_t loom_json_array_end(loom_json_array_writer_t* writer);

// Begins an element in |writer|. The caller must write exactly one JSON value
// to writer->stream before beginning the next element or ending the array.
iree_status_t loom_json_array_begin_element(loom_json_array_writer_t* writer);

// Writes array elements of the indicated type.
iree_status_t loom_json_array_write_string_element(
    loom_json_array_writer_t* writer, iree_string_view_t value);
iree_status_t loom_json_array_write_uint32_element(
    loom_json_array_writer_t* writer, uint32_t value);
iree_status_t loom_json_array_write_uint64_element(
    loom_json_array_writer_t* writer, uint64_t value);
iree_status_t loom_json_array_write_int32_element(
    loom_json_array_writer_t* writer, int32_t value);
iree_status_t loom_json_array_write_int64_element(
    loom_json_array_writer_t* writer, int64_t value);
iree_status_t loom_json_array_write_bool_element(
    loom_json_array_writer_t* writer, bool value);
iree_status_t loom_json_array_write_host_size_element(
    loom_json_array_writer_t* writer, iree_host_size_t value);

// Writes an explicit JSON null array element.
iree_status_t loom_json_array_write_null_element(
    loom_json_array_writer_t* writer);

// Deferred comma-separated JSON values serialized while their source data is
// live. The list owns the serialized array body but excludes the surrounding
// brackets so callers can embed it in a larger document later.
typedef struct loom_json_value_list_t {
  // Serialized values separated by the shared array writer.
  iree_string_builder_t body;
  // Number of complete serialized values in |body|.
  iree_host_size_t count;
} loom_json_value_list_t;

// Initializes an empty deferred value list using |allocator| for storage.
void loom_json_value_list_initialize(iree_allocator_t allocator,
                                     loom_json_value_list_t* out_list);

// Releases storage owned by |list|.
void loom_json_value_list_deinitialize(loom_json_value_list_t* list);

// Begins the next value and returns a stream receiving its serialized form.
// The caller must write exactly one complete JSON value to |out_stream|.
iree_status_t loom_json_value_list_begin_value(
    loom_json_value_list_t* list, loom_output_stream_t* out_stream);

// Returns the comma-separated serialized values without array brackets.
iree_string_view_t loom_json_value_list_body(
    const loom_json_value_list_t* list);

// Writes |list| as a complete JSON array.
iree_status_t loom_json_value_list_write_array(
    const loom_json_value_list_t* list, loom_output_stream_t* stream);

// Writes a borrowed comma-separated value list as a complete JSON array.
iree_status_t loom_json_write_value_list_array(iree_string_view_t body,
                                               loom_output_stream_t* stream);

// Writes an IREE status value as {"code":N,"name":"...","message":"..."}.
// The message field is omitted when |message| is empty.
iree_status_t loom_json_write_status_object(loom_output_stream_t* stream,
                                            iree_status_code_t code,
                                            iree_string_view_t message);

// Writes a JSON-escaped C string value (with surrounding quotes).
// Prefer loom_json_write_escaped_string with a known-length string_view.
static inline iree_status_t loom_json_write_escaped_cstring(
    loom_output_stream_t* stream, const char* value) {
  return loom_json_write_escaped_string(stream, iree_make_cstring_view(value));
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_JSON_H_
