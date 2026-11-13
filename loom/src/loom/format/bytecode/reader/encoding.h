// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Encoding-table validation, retained indexing, and IR materialization.

#ifndef LOOM_FORMAT_BYTECODE_READER_ENCODING_H_
#define LOOM_FORMAT_BYTECODE_READER_ENCODING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/format/bytecode/reader/type.h"
#include "loom/format/bytecode/reader/type_validator.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates ENCODINGS while advancing |types| through each declared prefix.
// Entry ranges are not retained. The caller finishes the remaining TYPES tail.
bool loom_bytecode_encoding_table_validate(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena,
    loom_bytecode_type_validation_t* types,
    const loom_bytecode_reader_section_t* section);

// Validates one complete ENCODINGS section and retains the exact byte range of
// every instance payload, excluding its type-prefix word. Family facts needed
// by later table validation remain scratch-owned in |module_view|. The caller
// finishes |types| after all encoding prefixes have been consumed.
bool loom_bytecode_encoding_table_index(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena,
    loom_bytecode_type_validation_t* types,
    const loom_bytecode_reader_section_t* section,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_encoding_metadata_t** out_entries,
    iree_host_size_t* out_count);

// State required to materialize a validated encoding table into a module.
typedef struct loom_bytecode_encoding_materializer_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Finalized encoding and attribute registry context.
  loom_context_t* context;
  // Validated module tables whose type slots publish completed canonical IDs.
  const loom_bytecode_reader_module_view_t* module_view;
  // Resettable scratch storage for parameter construction.
  iree_arena_allocator_t* scratch_arena;
  // Module receiving canonical encoding-table entries.
  loom_module_t* output_module;
  // Forward type construction, advanced before each encoding is published.
  loom_bytecode_type_materializer_t* types;
} loom_bytecode_encoding_materializer_t;

// Materializes every instance in one validated ENCODINGS section.
bool loom_bytecode_encoding_table_materialize(
    loom_bytecode_encoding_materializer_t* materializer,
    const loom_bytecode_reader_section_t* section);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_ENCODING_H_
