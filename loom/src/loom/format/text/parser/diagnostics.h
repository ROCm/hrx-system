// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_DIAGNOSTICS_H_
#define LOOM_FORMAT_TEXT_PARSER_DIAGNOSTICS_H_

#include "iree/base/api.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits a structured diagnostic through the parser's sink.
iree_status_t loom_parser_emit(loom_parser_t* parser,
                               const loom_error_def_t* error,
                               const loom_diagnostic_param_t* params,
                               iree_host_size_t param_count,
                               loom_token_t token);

// Emits a structured diagnostic with one related location.
iree_status_t loom_parser_emit_related(loom_parser_t* parser,
                                       const loom_error_def_t* error,
                                       const loom_diagnostic_param_t* params,
                                       iree_host_size_t param_count,
                                       loom_token_t token,
                                       iree_string_view_t related_label,
                                       loom_token_t related_token);

iree_status_t loom_parser_emit_unexpected_token(loom_parser_t* parser,
                                                loom_token_t token,
                                                iree_string_view_t expected);
iree_status_t loom_parser_emit_token_text_error(loom_parser_t* parser,
                                                const loom_error_def_t* error,
                                                loom_token_t token);
iree_status_t loom_parser_emit_tokenizer_error(loom_parser_t* parser,
                                               loom_token_t token);
iree_status_t loom_parser_expect(loom_parser_t* parser, loom_token_kind_t kind,
                                 loom_token_t* out_token);

#define LOOM_PARSE_EXPECT(parser, kind, out_token)                           \
  do {                                                                       \
    uint32_t _expect_errors = (parser)->error_count;                         \
    IREE_RETURN_IF_ERROR(loom_parser_expect((parser), (kind), (out_token))); \
    if ((parser)->error_count > _expect_errors) {                            \
      return iree_ok_status();                                               \
    }                                                                        \
  } while (0)

#define LOOM_PARSE_RESOLVE_VALUE(parser, name_token, out_value_id)          \
  do {                                                                      \
    uint32_t _resolve_errors = (parser)->error_count;                       \
    IREE_RETURN_IF_ERROR(                                                   \
        loom_parser_resolve_value((parser), (name_token), (out_value_id))); \
    if ((parser)->error_count > _resolve_errors) {                          \
      return iree_ok_status();                                              \
    }                                                                       \
  } while (0)

#define LOOM_PARSE_DEFINE_VALUE_NAME(parser, name_token, value_id)          \
  do {                                                                      \
    uint32_t _define_errors = (parser)->error_count;                        \
    IREE_RETURN_IF_ERROR(                                                   \
        loom_parser_define_value_name((parser), (name_token), (value_id))); \
    if ((parser)->error_count > _define_errors) {                           \
      return iree_ok_status();                                              \
    }                                                                       \
  } while (0)

bool loom_parser_at_error_limit(const loom_parser_t* parser);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_DIAGNOSTICS_H_
