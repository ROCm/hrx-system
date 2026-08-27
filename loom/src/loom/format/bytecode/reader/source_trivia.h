// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-trivia wire decoding for validation and IR materialization.

#ifndef LOOM_FORMAT_BYTECODE_READER_SOURCE_TRIVIA_H_
#define LOOM_FORMAT_BYTECODE_READER_SOURCE_TRIVIA_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/reader/decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materialized source comments and authored vertical separation.
//
// Comment views borrow the bytecode storage. The view array is allocated from
// the arena passed to loom_bytecode_source_trivia_materialize.
typedef struct loom_bytecode_source_trivia_t {
  // Borrowed comment payload views in wire order.
  const iree_string_view_t* comments;
  // Number of attached comments.
  uint16_t comment_count;
  // Authored leading vertical separation.
  bool leading_blank_line;
} loom_bytecode_source_trivia_t;

// Validates one source-trivia record and advances |cursor| past it.
//
// This operation performs no allocation and retains no decoded state.
iree_status_t loom_bytecode_source_trivia_validate(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor);

// Materializes one source-trivia record and advances |cursor| past it.
//
// The output is published only after the complete record has been decoded.
iree_status_t loom_bytecode_source_trivia_materialize(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, iree_arena_allocator_t* arena,
    loom_bytecode_source_trivia_t* out_source_trivia);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_SOURCE_TRIVIA_H_
