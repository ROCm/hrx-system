// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Symbol-table validation and immutable fact construction.

#ifndef LOOM_FORMAT_BYTECODE_READER_SYMBOL_VALIDATOR_H_
#define LOOM_FORMAT_BYTECODE_READER_SYMBOL_VALIDATOR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Transactional state for validating one SYMBOLS table.
typedef struct loom_bytecode_symbol_validator_t {
  // Bounded decoder copied from the owning public read.
  loom_bytecode_reader_decoder_t decoder;
  // Finalized dialect and attribute registry context.
  loom_context_t* context;
  // Resettable storage owning validated symbol facts.
  iree_arena_allocator_t* arena;
  // Caller-owned storage for the optional retained index projection.
  iree_arena_allocator_t* metadata_arena;
  // Validated facts preceding SYMBOLS plus the in-progress symbol product.
  loom_bytecode_reader_module_view_t view;
  // Module view receiving the symbol product only after complete validation.
  loom_bytecode_reader_module_view_t* output_view;
} loom_bytecode_symbol_validator_t;

// Initializes a transactional SYMBOLS validator over |module_view|.
void loom_bytecode_symbol_validator_initialize(
    const loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* metadata_arena,
    loom_bytecode_reader_module_view_t* module_view,
    loom_bytecode_symbol_validator_t* out_validator);

// Validates the complete SYMBOLS table and publishes its immutable facts.
iree_status_t loom_bytecode_symbols_validate(
    loom_bytecode_symbol_validator_t* validator,
    const loom_bytecode_reader_section_t* symbols_section,
    const loom_bytecode_reader_section_t* ir_section);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_SYMBOL_VALIDATOR_H_
