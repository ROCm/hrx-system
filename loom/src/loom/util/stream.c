// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/stream.h"

#include <stdio.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Batched printf callback for streaming formatted output
//===----------------------------------------------------------------------===//

// Accumulates characters from iree_vfctprintf into a stack buffer, flushing
// to the stream when full. This gives us unbounded output with no heap
// allocation and amortized stream write calls (one per 128 bytes instead of
// one per character).
#define LOOM_STREAM_FORMAT_BUFFER_SIZE 128

typedef struct loom_stream_printf_state_t {
  loom_output_stream_t* stream;
  iree_status_t status;
  size_t position;
  char buffer[LOOM_STREAM_FORMAT_BUFFER_SIZE];
} loom_stream_printf_state_t;

static void loom_stream_printf_callback(char character, void* user_data) {
  loom_stream_printf_state_t* state = (loom_stream_printf_state_t*)user_data;
  if (!iree_status_is_ok(state->status)) return;
  state->buffer[state->position++] = character;
  if (state->position == LOOM_STREAM_FORMAT_BUFFER_SIZE) {
    state->status = loom_output_stream_write(
        state->stream, iree_make_string_view(state->buffer, state->position));
    state->position = 0;
  }
}

iree_status_t loom_output_stream_write_format(loom_output_stream_t* stream,
                                              const char* format, ...) {
  loom_stream_printf_state_t state = {
      .stream = stream,
      .status = iree_ok_status(),
      .position = 0,
  };
  va_list args;
  va_start(args, format);
  int result =
      iree_vfctprintf(loom_stream_printf_callback, &state, format, args);
  va_end(args);
  if (result < 0) {
    return iree_make_status(IREE_STATUS_INTERNAL, "format error");
  }
  if (!iree_status_is_ok(state.status)) return state.status;
  // Flush remaining bytes.
  if (state.position > 0) {
    return loom_output_stream_write(
        stream, iree_make_string_view(state.buffer, state.position));
  }
  return iree_ok_status();
}

static iree_status_t loom_stream_builder_write(void* user_data,
                                               iree_string_view_t text) {
  return iree_string_builder_append_string((iree_string_builder_t*)user_data,
                                           text);
}

void loom_output_stream_for_builder(iree_string_builder_t* builder,
                                    loom_output_stream_t* out_stream) {
  out_stream->write = loom_stream_builder_write;
  out_stream->user_data = builder;
  out_stream->offset = 0;
}

static iree_status_t loom_stream_file_write(void* user_data,
                                            iree_string_view_t text) {
  FILE* file = (FILE*)user_data;
  size_t written = fwrite(text.data, 1, text.size, file);
  if (written != text.size) {
    return iree_make_status(IREE_STATUS_INTERNAL, "fwrite failed");
  }
  return iree_ok_status();
}

void loom_output_stream_for_file(FILE* file, loom_output_stream_t* out_stream) {
  out_stream->write = loom_stream_file_write;
  out_stream->user_data = file;
  out_stream->offset = 0;
}

// Null adapter.
static iree_status_t loom_stream_null_write(void* user_data,
                                            iree_string_view_t text) {
  (void)user_data;
  (void)text;
  return iree_ok_status();
}

void loom_output_stream_null(loom_output_stream_t* out_stream) {
  out_stream->write = loom_stream_null_write;
  out_stream->user_data = NULL;
  out_stream->offset = 0;
}
