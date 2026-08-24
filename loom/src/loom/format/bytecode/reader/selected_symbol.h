// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact source-symbol materialization into a compact standalone module.

#ifndef LOOM_FORMAT_BYTECODE_READER_SELECTED_SYMBOL_H_
#define LOOM_FORMAT_BYTECODE_READER_SELECTED_SYMBOL_H_

#include "loom/format/bytecode/function_header.h"
#include "loom/format/bytecode/reader/selected_body.h"

#ifdef __cplusplus
extern "C" {
#endif

// One selected source symbol prepared before output-module allocation.
typedef struct loom_bytecode_selected_symbol_t {
  // Module-local source SYMBOLS ordinal.
  uint32_t source_ordinal;
  // Arena-owned summaries in source region-payload order.
  loom_bytecode_region_summary_t* region_summaries;
  // Number of entries in region_summaries.
  uint8_t region_summary_count;
} loom_bytecode_selected_symbol_t;

// State required to materialize an exact prepared source-symbol sequence.
typedef struct loom_bytecode_selected_symbol_materializer_t {
  // Bounded decoder copied from the owning selected read.
  loom_bytecode_reader_decoder_t decoder;
  // Finalized dialect and attribute registry context.
  loom_context_t* context;
  // Resettable storage for one symbol payload at a time.
  iree_arena_allocator_t* arena;
  // Module receiving compact selected symbols and body IR.
  loom_module_t* output_module;
  // Stable-key codec supplied by the embedding compiler.
  loom_low_repr_environment_t low_repr_environment;
  // Reached-only source-to-output table materializer.
  loom_bytecode_selected_table_materializer_t* tables;
  // Nested selected body decoder sharing the table projection.
  loom_bytecode_selected_body_materializer_t body_materializer;
} loom_bytecode_selected_symbol_materializer_t;

// Initializes selected symbol materialization over a reached-table projection.
void loom_bytecode_selected_symbol_materializer_initialize(
    const loom_bytecode_reader_decoder_t* decoder,
    iree_arena_block_pool_t* block_pool,
    loom_bytecode_selected_table_materializer_t* tables,
    const loom_low_repr_environment_t* low_repr_environment,
    loom_bytecode_selected_symbol_materializer_t* out_materializer);

// Predeclares and materializes |selected_symbols| in source-ordinal order.
//
// The sequence must be strictly increasing and contain only ordinals from the
// validated source metadata. Every symbol is predeclared before any payload is
// decoded so selected symbol references resolve without further reachability.
iree_status_t loom_bytecode_selected_symbols_materialize(
    loom_bytecode_selected_symbol_materializer_t* materializer,
    const loom_bytecode_selected_symbol_t* selected_symbols,
    iree_host_size_t selected_symbol_count);

// Decodes one function-like symbol header without reading root-region bytes.
//
// Shared types, attributes, and external symbol references are projected
// through materializer->tables. The source symbol itself must be resolvable by
// that table's predeclared bindings or external resolver.
iree_status_t loom_bytecode_selected_function_header_materialize(
    loom_bytecode_selected_symbol_materializer_t* materializer,
    uint32_t source_symbol_ordinal,
    loom_bytecode_function_header_t* out_header);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_SELECTED_SYMBOL_H_
