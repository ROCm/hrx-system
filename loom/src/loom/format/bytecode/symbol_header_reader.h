// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lazy function-symbol metadata projection from validated bytecode.

#ifndef LOOM_FORMAT_BYTECODE_SYMBOL_HEADER_READER_H_
#define LOOM_FORMAT_BYTECODE_SYMBOL_HEADER_READER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/function_header.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/low_repr.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_symbol_header_reader_t
    loom_bytecode_symbol_header_reader_t;

// Options controlling lazy symbol-header materialization.
typedef struct loom_bytecode_symbol_header_reader_options_t {
  // Sink for structured malformed-bytecode diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Stable-key codec used by bodyless symbols carrying Low metadata.
  loom_low_repr_environment_t low_repr_environment;
} loom_bytecode_symbol_header_reader_options_t;

// Creates a lazy reader over one validated bytecode module.
//
// The reader borrows |bytecode|, |filename|, |context|, |block_pool|, and
// |metadata| for its lifetime. Its metadata module contains one placeholder
// symbol per source symbol in exact source-ordinal order. Shared tables and
// function signatures are projected only when requested.
iree_status_t loom_bytecode_symbol_header_reader_create(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_module_metadata_t* metadata,
    const loom_bytecode_symbol_header_reader_options_t* options,
    iree_allocator_t allocator,
    loom_bytecode_symbol_header_reader_t** out_reader);

// Releases |reader| and its projected metadata module.
void loom_bytecode_symbol_header_reader_free(
    loom_bytecode_symbol_header_reader_t* reader);

// Returns the reader-owned metadata module.
//
// A source symbol ordinal maps directly to the same local symbol ID in this
// module. The returned module remains owned by |reader|.
loom_module_t* loom_bytecode_symbol_header_reader_module(
    const loom_bytecode_symbol_header_reader_t* reader);

// Decodes one function-like symbol header without reading root-region bytes.
//
// Arrays and IR identities in |out_header| belong to |reader| and remain valid
// until the reader is freed. Callers cache the returned header rather than
// requesting the same source symbol repeatedly.
iree_status_t loom_bytecode_symbol_header_reader_read_function(
    loom_bytecode_symbol_header_reader_t* reader,
    uint32_t source_symbol_ordinal,
    loom_bytecode_function_header_t* out_header);

// Materializes one bodyless source symbol into the metadata module.
//
// Repeated requests for the same symbol are no-ops. Symbols with bodies are
// rejected because this reader never decodes implementation bodies.
iree_status_t loom_bytecode_symbol_header_reader_materialize_bodyless_symbol(
    loom_bytecode_symbol_header_reader_t* reader,
    uint32_t source_symbol_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_SYMBOL_HEADER_READER_H_
