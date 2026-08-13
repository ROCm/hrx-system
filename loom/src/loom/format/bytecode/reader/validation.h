// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bytecode file and module metadata validation.

#ifndef LOOM_FORMAT_BYTECODE_READER_VALIDATION_H_
#define LOOM_FORMAT_BYTECODE_READER_VALIDATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// File-level state retained while validating module directory entries.
typedef struct loom_bytecode_file_reader_t {
  // Full bytecode file bytes.
  iree_const_byte_span_t bytecode;
  // Dialect and encoding registry context.
  loom_context_t* context;
  // Transient metadata arena.
  iree_arena_allocator_t* arena;
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t decoder;
  // Public result accumulator.
  loom_bytecode_read_result_t result;
  // File header bytecode format version.
  uint8_t format_version;
  // File header producer string.
  iree_string_view_t producer;
  // File-level module-name pool.
  iree_string_view_t file_string_pool;
  // Module directory entries.
  loom_bytecode_reader_module_t* modules;
  // Number of module entries.
  iree_host_size_t module_count;
} loom_bytecode_file_reader_t;

// Initializes file validation over caller-owned input and scratch storage.
void loom_bytecode_file_reader_initialize(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_allocator_t* arena,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_bytecode_file_reader_t* out_reader);

// Validates the file header, module directory, and file string pool.
iree_status_t loom_bytecode_file_reader_validate(
    loom_bytecode_file_reader_t* reader);

// Projects the validated file directory into a retained index.
iree_status_t loom_bytecode_file_reader_project_index(
    const loom_bytecode_file_reader_t* reader,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_file_metadata_t* out_metadata);

// Validates one module without producing retained index state.
iree_status_t loom_bytecode_module_validate(
    const loom_bytecode_file_reader_t* file_reader,
    const loom_bytecode_reader_module_t* module,
    loom_bytecode_reader_module_view_t* out_view);

// Validates one module and projects its retained index state.
iree_status_t loom_bytecode_module_index(
    const loom_bytecode_file_reader_t* file_reader,
    const loom_bytecode_reader_module_t* module,
    iree_arena_allocator_t* retained_arena,
    loom_bytecode_module_metadata_t* out_metadata,
    loom_bytecode_reader_module_view_t* out_view);

// Consumes a diagnosed malformed-input unwind marker at the public boundary.
iree_status_t loom_bytecode_reader_normalize_diagnosed_error(
    iree_status_t status, uint32_t error_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_VALIDATION_H_
