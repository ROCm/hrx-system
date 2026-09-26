// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PARSER_RECOVERY_H_
#define LOOM_FORMAT_TEXT_PARSER_RECOVERY_H_

#include "loom/format/text/parser/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Boundary owned by a statement. Captured before consuming its first token
// so recovery can skip aggregates already opened before an error.
typedef struct loom_parser_recovery_point_t {
  // Delimiters belonging to the enclosing grammar, outside this construct.
  loom_tokenizer_nesting_t nesting;
  // First source line of the construct; sibling recovery requires a later line.
  uint32_t line;
} loom_parser_recovery_point_t;

static inline loom_parser_recovery_point_t loom_parser_recovery_point(
    loom_parser_t* parser) {
  loom_parser_recovery_point_t point = {
      .nesting = loom_tokenizer_nesting(&parser->tokenizer),
      .line = loom_tokenizer_peek(&parser->tokenizer).line,
  };
  return point;
}

// Skips the rest of a failed statement, leaving the next sibling, block label,
// or enclosing region end unconsumed. The syntax determines whether bare names
// can begin statements. Unclosed inner delimiters are abandoned at a region
// end.
void loom_parser_sync_to_next_op(loom_parser_t* parser,
                                 loom_parser_recovery_point_t point,
                                 loom_region_syntax_t syntax);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PARSER_RECOVERY_H_
