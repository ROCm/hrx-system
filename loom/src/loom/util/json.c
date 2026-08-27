// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/json.h"

#include <inttypes.h>

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

//===----------------------------------------------------------------------===//
// Streaming JSON containers
//===----------------------------------------------------------------------===//

static iree_status_t loom_json_write_separator(loom_output_stream_t* stream,
                                               iree_host_size_t value_count) {
  return value_count == 0 ? iree_ok_status()
                          : loom_output_stream_write_char(stream, ',');
}

iree_status_t loom_json_object_begin(loom_output_stream_t* stream,
                                     loom_json_object_writer_t* out_writer) {
  out_writer->stream = stream;
  out_writer->field_count = 0;
  return loom_output_stream_write_char(stream, '{');
}

iree_status_t loom_json_object_end(loom_json_object_writer_t* writer) {
  return loom_output_stream_write_char(writer->stream, '}');
}

iree_status_t loom_json_object_begin_field(loom_json_object_writer_t* writer,
                                           iree_string_view_t name) {
  IREE_RETURN_IF_ERROR(
      loom_json_write_separator(writer->stream, writer->field_count));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(writer->stream, name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(writer->stream, ':'));
  ++writer->field_count;
  return iree_ok_status();
}

iree_status_t loom_json_object_write_string_field(
    loom_json_object_writer_t* writer, iree_string_view_t name,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(writer, name));
  return loom_json_write_escaped_string(writer->stream, value);
}

iree_status_t loom_json_object_write_string_field_if_nonempty(
    loom_json_object_writer_t* writer, iree_string_view_t name,
    iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) return iree_ok_status();
  return loom_json_object_write_string_field(writer, name, value);
}

iree_status_t loom_json_object_write_uint32_field(
    loom_json_object_writer_t* writer, iree_string_view_t name,
    uint32_t value) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(writer, name));
  return loom_output_stream_write_format(writer->stream, "%" PRIu32, value);
}

iree_status_t loom_json_object_write_uint64_field(
    loom_json_object_writer_t* writer, iree_string_view_t name,
    uint64_t value) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(writer, name));
  return loom_output_stream_write_format(writer->stream, "%" PRIu64, value);
}

iree_status_t loom_json_object_write_int32_field(
    loom_json_object_writer_t* writer, iree_string_view_t name, int32_t value) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(writer, name));
  return loom_output_stream_write_format(writer->stream, "%" PRId32, value);
}

iree_status_t loom_json_object_write_int64_field(
    loom_json_object_writer_t* writer, iree_string_view_t name, int64_t value) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(writer, name));
  return loom_output_stream_write_format(writer->stream, "%" PRId64, value);
}

iree_status_t loom_json_object_write_bool_field(
    loom_json_object_writer_t* writer, iree_string_view_t name, bool value) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(writer, name));
  return loom_output_stream_write_cstring(writer->stream,
                                          value ? "true" : "false");
}

iree_status_t loom_json_object_write_host_size_field(
    loom_json_object_writer_t* writer, iree_string_view_t name,
    iree_host_size_t value) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(writer, name));
  return loom_output_stream_write_format(writer->stream, "%" PRIhsz, value);
}

iree_status_t loom_json_object_write_null_field(
    loom_json_object_writer_t* writer, iree_string_view_t name) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(writer, name));
  return loom_output_stream_write_cstring(writer->stream, "null");
}

iree_status_t loom_json_array_begin(loom_output_stream_t* stream,
                                    loom_json_array_writer_t* out_writer) {
  out_writer->stream = stream;
  out_writer->element_count = 0;
  return loom_output_stream_write_char(stream, '[');
}

iree_status_t loom_json_array_end(loom_json_array_writer_t* writer) {
  return loom_output_stream_write_char(writer->stream, ']');
}

iree_status_t loom_json_array_begin_element(loom_json_array_writer_t* writer) {
  IREE_RETURN_IF_ERROR(
      loom_json_write_separator(writer->stream, writer->element_count));
  ++writer->element_count;
  return iree_ok_status();
}

iree_status_t loom_json_array_write_string_element(
    loom_json_array_writer_t* writer, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(writer));
  return loom_json_write_escaped_string(writer->stream, value);
}

iree_status_t loom_json_array_write_uint32_element(
    loom_json_array_writer_t* writer, uint32_t value) {
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(writer));
  return loom_output_stream_write_format(writer->stream, "%" PRIu32, value);
}

iree_status_t loom_json_array_write_uint64_element(
    loom_json_array_writer_t* writer, uint64_t value) {
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(writer));
  return loom_output_stream_write_format(writer->stream, "%" PRIu64, value);
}

iree_status_t loom_json_array_write_int32_element(
    loom_json_array_writer_t* writer, int32_t value) {
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(writer));
  return loom_output_stream_write_format(writer->stream, "%" PRId32, value);
}

iree_status_t loom_json_array_write_int64_element(
    loom_json_array_writer_t* writer, int64_t value) {
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(writer));
  return loom_output_stream_write_format(writer->stream, "%" PRId64, value);
}

iree_status_t loom_json_array_write_bool_element(
    loom_json_array_writer_t* writer, bool value) {
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(writer));
  return loom_output_stream_write_cstring(writer->stream,
                                          value ? "true" : "false");
}

iree_status_t loom_json_array_write_host_size_element(
    loom_json_array_writer_t* writer, iree_host_size_t value) {
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(writer));
  return loom_output_stream_write_format(writer->stream, "%" PRIhsz, value);
}

iree_status_t loom_json_array_write_null_element(
    loom_json_array_writer_t* writer) {
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(writer));
  return loom_output_stream_write_cstring(writer->stream, "null");
}

void loom_json_value_list_initialize(iree_allocator_t allocator,
                                     loom_json_value_list_t* out_list) {
  *out_list = (loom_json_value_list_t){0};
  iree_string_builder_initialize(allocator, &out_list->body);
}

void loom_json_value_list_deinitialize(loom_json_value_list_t* list) {
  iree_string_builder_deinitialize(&list->body);
  list->count = 0;
}

iree_status_t loom_json_value_list_begin_value(
    loom_json_value_list_t* list, loom_output_stream_t* out_stream) {
  loom_output_stream_for_builder(&list->body, out_stream);
  loom_json_array_writer_t writer = {
      .stream = out_stream,
      .element_count = list->count,
  };
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&writer));
  ++list->count;
  return iree_ok_status();
}

iree_string_view_t loom_json_value_list_body(
    const loom_json_value_list_t* list) {
  return iree_string_builder_view(&list->body);
}

iree_status_t loom_json_write_value_list_array(iree_string_view_t body,
                                               loom_output_stream_t* stream) {
  loom_json_array_writer_t writer;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &writer));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(stream, body));
  return loom_json_array_end(&writer);
}

iree_status_t loom_json_value_list_write_array(
    const loom_json_value_list_t* list, loom_output_stream_t* stream) {
  return loom_json_write_value_list_array(loom_json_value_list_body(list),
                                          stream);
}

iree_status_t loom_json_write_status_object(loom_output_stream_t* stream,
                                            iree_status_code_t code,
                                            iree_string_view_t message) {
  loom_json_object_writer_t writer;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &writer));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &writer, IREE_SV("code"), (uint32_t)code));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &writer, IREE_SV("name"),
      iree_make_cstring_view(iree_status_code_string(code))));
  if (!iree_string_view_is_empty(message)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &writer, IREE_SV("message"), message));
  }
  return loom_json_object_end(&writer);
}
