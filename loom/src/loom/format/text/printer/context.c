// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/context.h"

#include "loom/ir/module.h"

static bool loom_is_backward_glue(char c) {
  return c == ',' || c == ')' || c == ']' || c == '}';
}

static bool loom_is_forward_glue(char c) {
  return c == '(' || c == '[' || c == '{';
}

// Emits text with automatic spacing. Applies backward/forward glue rules and
// the explicit glue flag.
iree_status_t loom_print_emit(loom_print_context_t* ctx,
                              iree_string_view_t text, bool glue) {
  if (text.size == 0) {
    return iree_ok_status();
  }
  bool suppress_space = glue || ctx->glue_next || !ctx->has_previous_token ||
                        loom_is_backward_glue(text.data[0]) ||
                        loom_is_forward_glue(ctx->last_char);
  ctx->glue_next = false;
  if (!suppress_space) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, text));
  ctx->has_previous_token = true;
  ctx->last_char = text.data[text.size - 1];
  return iree_ok_status();
}

iree_status_t loom_print_emit_cstr(loom_print_context_t* ctx, const char* text,
                                   bool glue) {
  return loom_print_emit(ctx, iree_make_cstring_view(text), glue);
}

void loom_print_set_glue(loom_print_context_t* ctx) { ctx->glue_next = true; }

// Emits a space if the spacing rules require one before the next token.
// Call before writing directly to the stream for content that is not
// backward-glue punctuation (types, attributes, integers; never start
// with ,)]}). After writing, call loom_print_did_write to update state.
iree_status_t loom_print_space_if_needed(loom_print_context_t* ctx) {
  bool suppress = ctx->glue_next || !ctx->has_previous_token ||
                  loom_is_forward_glue(ctx->last_char);
  ctx->glue_next = false;
  if (!suppress) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, ' '));
  }
  return iree_ok_status();
}

// Updates spacing state after writing content directly to the stream.
void loom_print_did_write(loom_print_context_t* ctx) {
  ctx->has_previous_token = true;
  ctx->last_char = ' ';
}

iree_host_size_t loom_print_next_token_start_offset(
    const loom_print_context_t* ctx, bool glue, char first_char) {
  bool suppress_space = glue || ctx->glue_next || !ctx->has_previous_token ||
                        loom_is_backward_glue(first_char) ||
                        loom_is_forward_glue(ctx->last_char);
  return ctx->stream->offset + (suppress_space ? 0 : 1);
}

void loom_print_report_field(loom_print_context_t* ctx,
                             loom_print_field_ref_t field_ref,
                             iree_host_size_t start, iree_host_size_t end) {
  if (!ctx->field_callback.fn) {
    return;
  }
  ctx->field_callback.fn(ctx->field_callback.user_data, field_ref, start, end);
}

// Writes indentation spaces directly to the stream.
static const char SPACES[] = "                                ";

iree_status_t loom_print_indent_at(loom_print_context_t* ctx, uint16_t indent) {
  if (!iree_any_bit_set(ctx->flags, LOOM_TEXT_PRINT_INDENT)) {
    return iree_ok_status();
  }
  iree_host_size_t count = (iree_host_size_t)indent * 2;
  while (count > 0) {
    iree_host_size_t chunk =
        count < sizeof(SPACES) - 1 ? count : sizeof(SPACES) - 1;
    IREE_RETURN_IF_ERROR(loom_output_stream_write(
        ctx->stream, iree_make_string_view(SPACES, chunk)));
    count -= chunk;
  }
  return iree_ok_status();
}

iree_status_t loom_print_indent(loom_print_context_t* ctx) {
  return loom_print_indent_at(ctx, ctx->indent);
}

static iree_status_t loom_print_leading_comments(
    loom_print_context_t* ctx, uint16_t indent,
    const iree_string_view_t* comments, iree_host_size_t comment_count) {
  for (iree_host_size_t i = 0; i < comment_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_print_indent_at(ctx, indent));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(ctx->stream, "//"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write(ctx->stream, comments[i]));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(ctx->stream, '\n'));
  }
  return iree_ok_status();
}

iree_status_t loom_print_op_comments(loom_print_context_t* ctx,
                                     const loom_op_t* op) {
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(ctx->module, op, &comment_count);
  return loom_print_leading_comments(ctx, ctx->indent, comments, comment_count);
}
