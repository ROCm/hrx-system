// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JSON string escaping utilities for loom output streams.
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
