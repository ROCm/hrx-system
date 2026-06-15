// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lexical scanner for the loom IR textual format.
//
// The tokenizer operates on an iree_string_view_t source buffer (no NUL
// termination required). Most token payloads are zero-copy slices into that
// buffer. Escaped string literals are decoded into a tokenizer-owned scratch
// buffer that grows to a high-water capacity in |scratch_arena| and is reused
// across scans, while unescaped strings stay zero-copy. The tokenizer is
// stack-allocated and supports one-token lookahead via peek/next.
//
// Character access is bounds-checked: position < source.size. EOF is
// detected by the bounds check, not by a sentinel character.

#ifndef LOOM_FORMAT_TEXT_TOKENIZER_H_
#define LOOM_FORMAT_TEXT_TOKENIZER_H_

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"

typedef struct iree_arena_allocator_t iree_arena_allocator_t;

#ifdef __cplusplus
extern "C" {
#endif

// Token kinds produced by the scanner.
typedef enum loom_token_kind_e {
  // Literals.
  LOOM_TOKEN_INTEGER = 0,
  LOOM_TOKEN_FLOAT = 1,
  LOOM_TOKEN_STRING = 2,

  // References.
  LOOM_TOKEN_SSA_VALUE = 3,
  LOOM_TOKEN_SYMBOL = 4,
  LOOM_TOKEN_HASH_ATTR = 5,
  LOOM_TOKEN_BLOCK_LABEL = 6,

  // Identifiers.
  LOOM_TOKEN_BARE_IDENT = 7,
  LOOM_TOKEN_OP_NAME = 8,

  // Punctuation.
  LOOM_TOKEN_LPAREN = 9,
  LOOM_TOKEN_RPAREN = 10,
  LOOM_TOKEN_LBRACE = 11,
  LOOM_TOKEN_RBRACE = 12,
  LOOM_TOKEN_LBRACKET = 13,
  LOOM_TOKEN_RBRACKET = 14,
  LOOM_TOKEN_LANGLE = 15,
  LOOM_TOKEN_RANGLE = 16,
  LOOM_TOKEN_EQUALS = 17,
  LOOM_TOKEN_COLON = 18,
  LOOM_TOKEN_COMMA = 19,
  LOOM_TOKEN_ARROW = 20,
  LOOM_TOKEN_DIM_X = 21,  // 'x' dimension separator (only when in_dim_list).
  LOOM_TOKEN_PIPE = 22,

  // Special.
  LOOM_TOKEN_EOF = 23,
  LOOM_TOKEN_ERROR = 24,
  LOOM_TOKEN_COUNT_,

  // Sentinel for "no token" (uninitialized lookahead).
  LOOM_TOKEN_NONE = 255,
} loom_token_kind_t;

// A token's semantic payload and exact source spelling.
//
// |text| is the payload the parser should intern or resolve. For prefixed
// tokens (%, @, #, ^), this excludes the prefix. For string literals, this
// excludes the surrounding quotes and decodes JSON-compatible escapes
// (\" \\ \/ \b \f \n \r \t \uXXXX plus surrogate pairs). Unescaped string
// token payloads alias the tokenizer source buffer; escaped string token
// payloads alias tokenizer-owned decode scratch and remain valid only until the
// next scan overwrites that scratch. |source_text| is the exact byte slice from
// the input file, including sigils/quotes, for diagnostics and source ranges.
typedef struct loom_token_t {
  iree_string_view_t text;
  iree_string_view_t source_text;
  uint32_t line;
  uint32_t column;
  uint32_t end_column;  // Column past the last character of this token.
  loom_token_kind_t kind;
} loom_token_t;

#define LOOM_TOKENIZER_MAX_ERROR_PARAMS 2

// Structured lexical error payload for the current LOOM_TOKEN_ERROR lookahead.
//
// |error| points at a stable generated error definition. Any string parameters
// must point at either static storage or the tokenizer's source buffer; they
// remain valid until the next successful scan clears this payload.
typedef struct loom_tokenizer_error_t {
  const loom_error_def_t* error;
  loom_diagnostic_param_t params[LOOM_TOKENIZER_MAX_ERROR_PARAMS];
  iree_host_size_t param_count;
  iree_host_size_t source_start;
  iree_host_size_t source_end;
  uint32_t line;
  uint32_t column;
  uint32_t end_column;
} loom_tokenizer_error_t;

// Returns a sentinel token for parser-synthesized values that have no
// user-authored source token.
static inline loom_token_t loom_token_none(void) {
  loom_token_t token = {
      /*.text=*/iree_string_view_empty(),
      /*.source_text=*/iree_string_view_empty(),
      /*.line=*/0,
      /*.column=*/0,
      /*.end_column=*/0,
      /*.kind=*/LOOM_TOKEN_NONE,
  };
  return token;
}

// Lexical scanner state. Stack-allocated; only escaped string payloads allocate
// from |scratch_arena|. Malformed user input produces a LOOM_TOKEN_ERROR token
// plus a structured payload in |error| so the parser can emit diagnostics and
// recover at sibling-op boundaries. |status| is reserved for infrastructure
// failures such as allocation errors; once set, all subsequent peek/next calls
// return EOF and the caller must consume the status.
typedef struct loom_tokenizer_t {
  iree_string_view_t source;
  iree_host_size_t position;
  uint32_t line;
  uint32_t column;
  loom_token_t peeked;
  loom_tokenizer_error_t error;
  iree_string_view_t filename;
  iree_arena_allocator_t* scratch_arena;
  char* decoded_string_data;
  iree_host_size_t decoded_string_capacity;
  // Pending line comments attached to the next operation or block.
  iree_string_view_t* pending_comments;
  // Number of pending line comments stored in pending_comments.
  iree_host_size_t pending_comment_count;
  // Allocated capacity of pending_comments.
  iree_host_size_t pending_comment_capacity;
  iree_status_t status;

  // Position of the most recently consumed token's end. Updated on
  // every consume (next, try_consume, etc). Allows the parser to
  // compute source ranges spanning multiple tokens.
  uint32_t consumed_end_line;
  uint32_t consumed_end_column;

  // When true, 'x' at identifier-start position produces DIM_X instead
  // of starting an identifier. Set by the shaped type parser during
  // dimension list parsing (e.g., "4x[%M]xf32") and cleared before
  // scanning element types or encoding parameters.
  bool in_dim_list;
} loom_tokenizer_t;

// Initializes a tokenizer over the given source buffer. The source buffer must
// remain valid for the lifetime of the tokenizer. |scratch_arena| backs decoded
// escaped string payloads and must outlive the tokenizer.
void loom_tokenizer_initialize(iree_string_view_t source,
                               iree_string_view_t filename,
                               iree_arena_allocator_t* scratch_arena,
                               loom_tokenizer_t* out_tokenizer);

// Deinitializes the tokenizer, freeing any unconsumed infrastructure status.
void loom_tokenizer_deinitialize(loom_tokenizer_t* tokenizer);

// Returns and transfers ownership of any deferred infrastructure failure.
// Malformed user input is not surfaced here; it appears as LOOM_TOKEN_ERROR
// tokens plus |tokenizer->error| metadata.
iree_status_t loom_tokenizer_consume_status(loom_tokenizer_t* tokenizer);

// Returns pending line comments collected while scanning leading whitespace and
// clears the tokenizer's pending list. The returned array is tokenizer-scratch
// storage; comment payloads are slices into the original source buffer.
void loom_tokenizer_take_pending_comments(
    loom_tokenizer_t* tokenizer, const iree_string_view_t** out_comments,
    iree_host_size_t* out_comment_count);

// Clears pending line comments that were collected before a delimiter and do
// not attach to any operation or block.
void loom_tokenizer_discard_pending_comments(loom_tokenizer_t* tokenizer);

// Returns the next token without consuming it. Repeated calls return
// the same token until loom_tokenizer_next is called.
loom_token_t loom_tokenizer_peek(loom_tokenizer_t* tokenizer);

// Consumes and returns the next token.
loom_token_t loom_tokenizer_next(loom_tokenizer_t* tokenizer);

// Returns a human-readable label for a token kind, suitable for diagnostics.
iree_string_view_t loom_token_kind_name(loom_token_kind_t kind);

// Returns true if the next token matches |kind| without consuming it.
bool loom_tokenizer_at(loom_tokenizer_t* tokenizer, loom_token_kind_t kind);

// Returns true if the next token matches |kind| and |text| without
// consuming it.
bool loom_tokenizer_at_keyword(loom_tokenizer_t* tokenizer,
                               iree_string_view_t text);

// Consumes the next token if it matches |kind|, returns true.
// Otherwise returns false without consuming.
bool loom_tokenizer_try_consume(loom_tokenizer_t* tokenizer,
                                loom_token_kind_t kind);

// Consumes the next token if it is a BARE_IDENT matching |text|,
// returns true. Otherwise returns false without consuming.
bool loom_tokenizer_try_consume_keyword(loom_tokenizer_t* tokenizer,
                                        iree_string_view_t text);

// Scans from the current position (after '<' has been consumed) to
// the matching '>'. Returns the interior text (exclusive of angle
// brackets). Handles nested angle brackets and string literals.
iree_status_t loom_tokenizer_scan_angle_interior(
    loom_tokenizer_t* tokenizer, iree_string_view_t* out_interior);

// Returns a formatted infrastructure error status with filename:line:col
// context. Malformed user input should populate |tokenizer->error| and return
// LOOM_TOKEN_ERROR instead of using this status side channel.
iree_status_t loom_tokenizer_error(const loom_tokenizer_t* tokenizer,
                                   iree_string_view_t message);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_TEXT_TOKENIZER_H_
