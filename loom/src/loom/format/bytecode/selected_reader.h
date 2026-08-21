// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Selective bytecode IR materialization from an existing validated index.

#ifndef LOOM_FORMAT_BYTECODE_SELECTED_READER_H_
#define LOOM_FORMAT_BYTECODE_SELECTED_READER_H_

#include "loom/format/bytecode/reader.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact validated source-symbol selection for indexed materialization.
typedef struct loom_bytecode_symbol_ordinal_list_t {
  // Number of entries in |ordinals|.
  iree_host_size_t count;
  // Strictly increasing module-local source SYMBOLS ordinals.
  const iree_host_size_t* ordinals;
} loom_bytecode_symbol_ordinal_list_t;

// Materializes exact symbols from a module in an existing validated index.
//
// |metadata|, |bytecode|, and |context| must be the exact inputs retained by a
// successful loom_bytecode_read_index call. This avoids revalidating or
// reparsing the catalog after metadata-only link planning. Only selected
// symbol entries, selected bodies, and transitively reached shared-table facts
// are decoded into the returned ordinary standalone module.
//
// Selection shape is validated as an API precondition. Malformed selected
// payloads follow the same diagnostic, result, verification, and ownership
// contract as loom_bytecode_read_module_ordinal. Rejected symbol entries and
// bodies are never read.
iree_status_t loom_bytecode_materialize_module_symbols(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_file_metadata_t* metadata, uint16_t module_ordinal,
    loom_bytecode_symbol_ordinal_list_t selection,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_SELECTED_READER_H_
