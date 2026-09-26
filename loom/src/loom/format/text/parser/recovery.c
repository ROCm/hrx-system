// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/recovery.h"

static bool loom_parser_recovery_is_statement_start(
    loom_token_kind_t kind, loom_region_syntax_t syntax) {
  switch (kind) {
    case LOOM_TOKEN_SSA_VALUE:
    case LOOM_TOKEN_OP_NAME:
    case LOOM_TOKEN_HASH_ATTR:
    case LOOM_TOKEN_BLOCK_LABEL:
    case LOOM_TOKEN_ERROR:
      return true;
    case LOOM_TOKEN_BARE_IDENT:
      return syntax == LOOM_REGION_SYNTAX_LOW_ASM_OPTIONAL ||
             syntax == LOOM_REGION_SYNTAX_PIPELINE;
    default:
      return false;
  }
}

void loom_parser_sync_to_next_op(loom_parser_t* parser,
                                 loom_parser_recovery_point_t point,
                                 loom_region_syntax_t syntax) {
  loom_tokenizer_t* tokenizer = &parser->tokenizer;
  for (;;) {
    loom_token_t token = loom_tokenizer_peek(tokenizer);
    loom_tokenizer_nesting_t nesting = loom_tokenizer_nesting(tokenizer);
    if (token.kind == LOOM_TOKEN_EOF || nesting.braces < point.nesting.braces) {
      return;
    }
    if (nesting.braces == point.nesting.braces) {
      if ((point.nesting.braces > 0 && token.kind == LOOM_TOKEN_RBRACE) ||
          (token.line > point.line &&
           nesting.parentheses <= point.nesting.parentheses &&
           nesting.brackets <= point.nesting.brackets &&
           nesting.angles <= point.nesting.angles &&
           loom_parser_recovery_is_statement_start(token.kind, syntax))) {
        loom_tokenizer_set_nesting(tokenizer, point.nesting);
        return;
      }
    }
    loom_tokenizer_next(tokenizer);
  }
}
