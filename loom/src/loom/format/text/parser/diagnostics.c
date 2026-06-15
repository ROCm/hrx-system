// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/diagnostics.h"

#include <string.h>

#include "loom/error/error_catalog.h"

//===----------------------------------------------------------------------===//
// Diagnostic emission
//===----------------------------------------------------------------------===//

static loom_source_range_t loom_parser_token_origin(iree_string_view_t filename,
                                                    iree_string_view_t source,
                                                    loom_token_t token) {
  if (token.kind == LOOM_TOKEN_NONE || token.kind == LOOM_TOKEN_EOF) {
    return (loom_source_range_t){
        .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
        .filename = filename,
        .source = source,
        .start = source.size,
        .end = source.size,
        .start_line = token.line,
        .start_column = token.column,
        .end_line = token.line,
        .end_column = token.end_column,
    };
  }

  return loom_source_range_from_token(filename, source, token.source_text,
                                      token.line, token.column,
                                      token.end_column);
}

static iree_status_t loom_parser_format_token_text(
    loom_parser_t* parser, loom_token_t token,
    iree_string_view_t* out_display_text) {
  if (token.kind == LOOM_TOKEN_NONE || token.kind == LOOM_TOKEN_EOF) {
    *out_display_text = loom_token_kind_name(token.kind);
    return iree_ok_status();
  }

  if (token.kind == LOOM_TOKEN_STRING) {
    iree_host_size_t display_size = 2;
    for (iree_host_size_t i = 0; i < token.text.size; ++i) {
      switch (token.text.data[i]) {
        case '"':
        case '\\':
        case '\n':
        case '\t':
          display_size += 2;
          break;
        default:
          ++display_size;
          break;
      }
    }
    char* display_text = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &parser->parser_arena, display_size, (void**)&display_text));
    iree_host_size_t display_offset = 0;
    display_text[display_offset++] = '"';
    for (iree_host_size_t i = 0; i < token.text.size; ++i) {
      switch (token.text.data[i]) {
        case '"':
          display_text[display_offset++] = '\\';
          display_text[display_offset++] = '"';
          break;
        case '\\':
          display_text[display_offset++] = '\\';
          display_text[display_offset++] = '\\';
          break;
        case '\n':
          display_text[display_offset++] = '\\';
          display_text[display_offset++] = 'n';
          break;
        case '\t':
          display_text[display_offset++] = '\\';
          display_text[display_offset++] = 't';
          break;
        default:
          display_text[display_offset++] = token.text.data[i];
          break;
      }
    }
    display_text[display_offset++] = '"';
    *out_display_text = iree_make_string_view(display_text, display_offset);
    return iree_ok_status();
  }

  char prefix = '\0';
  switch (token.kind) {
    case LOOM_TOKEN_SSA_VALUE:
      prefix = '%';
      break;
    case LOOM_TOKEN_SYMBOL:
      prefix = '@';
      break;
    case LOOM_TOKEN_HASH_ATTR:
      prefix = '#';
      break;
    case LOOM_TOKEN_BLOCK_LABEL:
      prefix = '^';
      break;
    default:
      *out_display_text = token.text;
      return iree_ok_status();
  }

  iree_host_size_t display_size = token.text.size + 1;
  char* display_text = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&parser->parser_arena, display_size,
                                           (void**)&display_text));
  display_text[0] = prefix;
  if (!iree_string_view_is_empty(token.text)) {
    memcpy(display_text + 1, token.text.data, token.text.size);
  }
  *out_display_text = iree_make_string_view(display_text, display_size);
  return iree_ok_status();
}

static iree_status_t loom_parser_emit_diagnostic(
    loom_parser_t* parser, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count,
    loom_token_t token,
    const loom_diagnostic_related_location_t* related_locations,
    iree_host_size_t related_location_count) {
  if (error->severity == LOOM_DIAGNOSTIC_ERROR) {
    ++parser->error_count;
  }
  loom_source_range_t origin =
      loom_parser_token_origin(parser->filename, parser->source, token);
  loom_diagnostic_t diagnostic = {
      .severity = error->severity,
      .error = error,
      .params = params,
      .param_count = param_count,
      .emitter = LOOM_EMITTER_PARSER,
      .origin = origin,
      // The text parser currently reports the parse token itself as the
      // user-facing location. Parsed loc() metadata can diverge from origin in
      // later verifier/reader diagnostics.
      .source_location = origin,
      .related_locations = related_locations,
      .related_location_count = related_location_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_diagnostic_emit(&parser->diagnostic_sink, &diagnostic));

  // When the error limit is reached, emit ERR_PARSE_012 so the sink
  // sees an explicit "too many errors" diagnostic before we stop.
  if (parser->max_errors > 0 && parser->error_count == parser->max_errors) {
    loom_diagnostic_param_t limit_params[] = {
        loom_param_u32(parser->error_count),
    };
    loom_diagnostic_t limit_diagnostic = {
        .severity = LOOM_DIAGNOSTIC_ERROR,
        .error = LOOM_ERR_PARSE_012,
        .params = limit_params,
        .param_count = IREE_ARRAYSIZE(limit_params),
        .emitter = LOOM_EMITTER_PARSER,
        .origin = diagnostic.origin,
        .source_location = diagnostic.source_location,
    };
    IREE_RETURN_IF_ERROR(
        loom_diagnostic_emit(&parser->diagnostic_sink, &limit_diagnostic));
  }

  return iree_ok_status();
}

iree_status_t loom_parser_emit(loom_parser_t* parser,
                               const loom_error_def_t* error,
                               const loom_diagnostic_param_t* params,
                               iree_host_size_t param_count,
                               loom_token_t token) {
  return loom_parser_emit_diagnostic(parser, error, params, param_count, token,
                                     NULL, 0);
}

iree_status_t loom_parser_emit_related(loom_parser_t* parser,
                                       const loom_error_def_t* error,
                                       const loom_diagnostic_param_t* params,
                                       iree_host_size_t param_count,
                                       loom_token_t token,
                                       iree_string_view_t related_label,
                                       loom_token_t related_token) {
  loom_diagnostic_related_location_t related_location = {
      .label = related_label,
      .source_location = loom_parser_token_origin(
          parser->filename, parser->source, related_token),
  };
  return loom_parser_emit_diagnostic(parser, error, params, param_count, token,
                                     &related_location, 1);
}

iree_status_t loom_parser_emit_tokenizer_error(loom_parser_t* parser,
                                               loom_token_t token) {
  return loom_parser_emit(parser, parser->tokenizer.error.error,
                          parser->tokenizer.error.params,
                          parser->tokenizer.error.param_count, token);
}

iree_status_t loom_parser_emit_unexpected_token(loom_parser_t* parser,
                                                loom_token_t token,
                                                iree_string_view_t expected) {
  if (token.kind == LOOM_TOKEN_ERROR) {
    return loom_parser_emit_tokenizer_error(parser, token);
  }
  iree_string_view_t actual_text = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_parser_format_token_text(parser, token, &actual_text));
  loom_diagnostic_param_t params[] = {
      loom_param_string(actual_text),
      loom_param_string(expected),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_003, params,
                          IREE_ARRAYSIZE(params), token);
}

iree_status_t loom_parser_emit_token_text_error(loom_parser_t* parser,
                                                const loom_error_def_t* error,
                                                loom_token_t token) {
  if (token.kind == LOOM_TOKEN_ERROR) {
    return loom_parser_emit_tokenizer_error(parser, token);
  }
  iree_string_view_t token_text = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_parser_format_token_text(parser, token, &token_text));
  loom_diagnostic_param_t params[] = {
      loom_param_string(token_text),
  };
  return loom_parser_emit(parser, error, params, IREE_ARRAYSIZE(params), token);
}

iree_status_t loom_parser_expect(loom_parser_t* parser, loom_token_kind_t kind,
                                 loom_token_t* out_token) {
  // Propagate pending tokenizer infrastructure failures. Malformed user input
  // arrives as LOOM_TOKEN_ERROR and is emitted through the parser sink below.
  if (!iree_status_is_ok(parser->tokenizer.status)) {
    return loom_tokenizer_consume_status(&parser->tokenizer);
  }
  loom_token_t token = loom_tokenizer_next(&parser->tokenizer);
  // A scan inside next() may have produced an infrastructure failure.
  if (!iree_status_is_ok(parser->tokenizer.status)) {
    return loom_tokenizer_consume_status(&parser->tokenizer);
  }
  if (token.kind == LOOM_TOKEN_ERROR) {
    return loom_parser_emit_tokenizer_error(parser, token);
  }
  if (token.kind != kind) {
    return loom_parser_emit_unexpected_token(parser, token,
                                             loom_token_kind_name(kind));
  }
  if (out_token) {
    *out_token = token;
  }
  return iree_ok_status();
}

bool loom_parser_at_error_limit(const loom_parser_t* parser) {
  return parser->max_errors > 0 && parser->error_count >= parser->max_errors;
}
