// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Full symbol-table and IR-body materialization.

#ifndef LOOM_FORMAT_BYTECODE_READER_SYMBOL_MATERIALIZER_H_
#define LOOM_FORMAT_BYTECODE_READER_SYMBOL_MATERIALIZER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/reader/body.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// State for materializing validated symbol facts and bodies.
typedef struct loom_bytecode_symbol_materializer_t {
  // Bounded decoder copied from the owning public read.
  loom_bytecode_reader_decoder_t decoder;
  // Finalized dialect and attribute registry context.
  loom_context_t* context;
  // Resettable storage for symbol and attribute construction.
  iree_arena_allocator_t* arena;
  // Immutable validated module facts consumed by symbol payloads.
  loom_bytecode_reader_module_view_t view;
  // Module receiving symbol operations and body IR.
  loom_module_t* output_module;
  // Stable-key codec supplied by the embedding compiler.
  loom_low_repr_environment_t low_repr_environment;
  // Nested body decoder sharing this materializer's exact state.
  loom_bytecode_body_materializer_t body_materializer;
} loom_bytecode_symbol_materializer_t;

// Initializes full symbol materialization over an immutable validated view.
void loom_bytecode_symbol_materializer_initialize(
    const loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    iree_arena_allocator_t* arena, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_reader_module_view_t* module_view,
    loom_module_t* output_module,
    const loom_low_repr_environment_t* low_repr_environment,
    loom_bytecode_symbol_materializer_t* out_materializer);

// Predeclares all materializable symbols from validated dense facts.
iree_status_t loom_bytecode_symbols_predeclare(
    loom_bytecode_symbol_materializer_t* materializer);

// Materializes the complete SYMBOLS table and every referenced IR body.
iree_status_t loom_bytecode_symbols_materialize(
    loom_bytecode_symbol_materializer_t* materializer,
    const loom_bytecode_reader_section_t* symbols_section,
    const loom_bytecode_reader_section_t* ir_section);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_SYMBOL_MATERIALIZER_H_
