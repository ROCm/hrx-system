// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/json.h"

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// JSON-escaping stream adapter
//===----------------------------------------------------------------------===//

// Flushes a literal run [run_start, cursor) to the inner stream.
static inline iree_status_t loom_json_flush_run(loom_output_stream_t* inner,
                                                const char* run_start,
                                                const char* cursor) {
  if (cursor > run_start) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        inner, iree_make_string_view(run_start,
                                     (iree_host_size_t)(cursor - run_start))));
  }
  return iree_ok_status();
}

// Write callback that JSON-escapes all incoming text per RFC 8259.
//
// Walks the input byte by byte, accumulating literal runs and flushing escape
// sequences. Valid multi-byte UTF-8 passes through unchanged except for U+2028
// and U+2029, which are escaped for JavaScript/HTML safety. Malformed UTF-8
// bytes are escaped as U+FFFD replacement characters.
static iree_status_t loom_json_escape_write(void* user_data,
                                            iree_string_view_t text) {
  loom_output_stream_t* inner = ((loom_json_escape_stream_t*)user_data)->inner;
  const char* run_start = text.data;
  const char* end = text.data + text.size;
  for (const char* cursor = text.data; cursor < end; ++cursor) {
    unsigned char byte = (unsigned char)*cursor;
    if (byte >= 0x80) {
      iree_host_size_t sequence_length =
          iree_unicode_utf8_sequence_length(byte);
      iree_host_size_t remaining = (iree_host_size_t)(end - cursor);
      if (sequence_length > remaining ||
          !iree_unicode_utf8_is_valid_sequence((const uint8_t*)cursor,
                                               sequence_length)) {
        IREE_RETURN_IF_ERROR(loom_json_flush_run(inner, run_start, cursor));
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(inner, "\\ufffd"));
        run_start = cursor + 1;
        continue;
      }
      if (sequence_length == 3 && byte == 0xE2 &&
          (unsigned char)cursor[1] == 0x80 &&
          ((unsigned char)cursor[2] == 0xA8 ||
           (unsigned char)cursor[2] == 0xA9)) {
        IREE_RETURN_IF_ERROR(loom_json_flush_run(inner, run_start, cursor));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
            inner, ((unsigned char)cursor[2] == 0xA8) ? "\\u2028" : "\\u2029"));
        cursor += 2;
        run_start = cursor + 1;
      } else {
        cursor += (iree_host_size_t)(sequence_length - 1);
      }
      continue;
    }

    const char* escape = NULL;
    switch (*cursor) {
      case '"':
        escape = "\\\"";
        break;
      case '\\':
        escape = "\\\\";
        break;
      case '\b':
        escape = "\\b";
        break;
      case '\f':
        escape = "\\f";
        break;
      case '\n':
        escape = "\\n";
        break;
      case '\r':
        escape = "\\r";
        break;
      case '\t':
        escape = "\\t";
        break;
      default:
        if ((unsigned char)*cursor < 0x20) {
          // ASCII control character — emit as \uNNNN.
          IREE_RETURN_IF_ERROR(loom_json_flush_run(inner, run_start, cursor));
          IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
              inner, "\\u%04x", (unsigned)(unsigned char)*cursor));
          run_start = cursor + 1;
        }
        // Printable ASCII bytes pass through unchanged.
        continue;
    }
    // Flush the literal run before this escaped character.
    IREE_RETURN_IF_ERROR(loom_json_flush_run(inner, run_start, cursor));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(inner, escape));
    run_start = cursor + 1;
  }
  // Flush any trailing literal run.
  IREE_RETURN_IF_ERROR(loom_json_flush_run(inner, run_start, end));
  return iree_ok_status();
}

void loom_json_escape_stream_init(loom_output_stream_t* inner,
                                  loom_json_escape_stream_t* escape_data,
                                  loom_output_stream_t* out_stream) {
  escape_data->inner = inner;
  out_stream->write = loom_json_escape_write;
  out_stream->user_data = escape_data;
  out_stream->offset = 0;
}

//===----------------------------------------------------------------------===//
// Convenience: quoted + escaped string writes
//===----------------------------------------------------------------------===//

iree_status_t loom_json_write_escaped_string(loom_output_stream_t* stream,
                                             iree_string_view_t value) {
  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&escape_stream, value));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  return iree_ok_status();
}
