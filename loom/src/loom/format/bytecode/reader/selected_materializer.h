// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact module materialization from an exact validated symbol selection.

#ifndef LOOM_FORMAT_BYTECODE_READER_SELECTED_MATERIALIZER_H_
#define LOOM_FORMAT_BYTECODE_READER_SELECTED_MATERIALIZER_H_

#include "loom/format/bytecode/reader/selected_symbol.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable inputs for one selected module materialization.
typedef struct loom_bytecode_selected_module_materializer_t {
  // Bounded decoder sharing the public diagnostic result state.
  loom_bytecode_reader_decoder_t* decoder;
  // Complete source bytecode containing retained entry and body spans.
  iree_const_byte_span_t bytecode;
  // Finalized dialect and registry context used to build the retained index.
  loom_context_t* context;
  // Resettable storage for preparation and transient materialization state.
  iree_arena_allocator_t* scratch_arena;
  // Block source for the output module and body-local arenas.
  iree_arena_block_pool_t* block_pool;
  // Validated retained metadata for the selected source module.
  const loom_bytecode_module_metadata_t* metadata;
  // Stable-key codec supplied by the embedding compiler.
  loom_low_repr_environment_t low_repr_environment;
  // Host allocator owning the returned module and transient projections.
  iree_allocator_t host_allocator;
} loom_bytecode_selected_module_materializer_t;

// Materializes an ordinary standalone module containing exactly |ordinals|.
//
// |ordinals| must be strictly increasing and in the metadata symbol domain.
// Body summaries are decoded once before output allocation and reused during
// materialization. Rejected symbol entries and bodies are never read.
iree_status_t loom_bytecode_selected_module_materialize(
    const loom_bytecode_selected_module_materializer_t* materializer,
    const iree_host_size_t* ordinals, iree_host_size_t ordinal_count,
    loom_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_SELECTED_MATERIALIZER_H_
