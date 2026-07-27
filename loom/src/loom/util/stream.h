// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Zero-allocation write-only output stream.
//
// All output (types, attributes, keywords, JSON fragments, diagnostics)
// flows through a single write callback. The stream tracks byte offsets
// for location capture. No intermediate buffers, no heap allocations in
// the write path.
//
// Built-in adapters wrap iree_string_builder_t and FILE*. Custom
// adapters implement the write callback for other destinations
// (network sockets, memory-mapped files, etc.).

#ifndef LOOM_UTIL_STREAM_H_
#define LOOM_UTIL_STREAM_H_

#include <stdarg.h>
#include <stdio.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Write-only output stream. The write callback receives text fragments
// of arbitrary length (including single characters). The offset field
// tracks the total bytes written for location capture.
typedef struct loom_output_stream_t {
  iree_status_t (*write)(void* user_data, iree_string_view_t text);
  void* user_data;
  iree_host_size_t offset;
} loom_output_stream_t;

// Writes text through the stream, updating the byte offset.
static inline iree_status_t loom_output_stream_write(
    loom_output_stream_t* stream, iree_string_view_t text) {
  if (text.size == 0) return iree_ok_status();
  iree_status_t status = stream->write(stream->user_data, text);
  if (iree_status_is_ok(status)) {
    stream->offset += text.size;
  }
  return status;
}

// Writes a single character through the stream.
static inline iree_status_t loom_output_stream_write_char(
    loom_output_stream_t* stream, char c) {
  return loom_output_stream_write(stream, iree_make_string_view(&c, 1));
}

// Writes a NUL-terminated C string through the stream.
static inline iree_status_t loom_output_stream_write_cstring(
    loom_output_stream_t* stream, const char* text) {
  return loom_output_stream_write(stream, iree_make_cstring_view(text));
}

// Writes formatted text through the stream. Uses iree_vfctprintf with a
// small stack buffer for batching — no heap allocation, no length limit.
iree_status_t loom_output_stream_write_format(loom_output_stream_t* stream,
                                              const char* format, ...);

// Initializes a stream that appends to an iree_string_builder_t.
void loom_output_stream_for_builder(iree_string_builder_t* builder,
                                    loom_output_stream_t* out_stream);

// Initializes a stream that writes to a FILE*.
void loom_output_stream_for_file(FILE* file, loom_output_stream_t* out_stream);

// Initializes a null stream that discards all output. The offset
// still tracks bytes for location-only computation.
void loom_output_stream_null(loom_output_stream_t* out_stream);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_STREAM_H_
