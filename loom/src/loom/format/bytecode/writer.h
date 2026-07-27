// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bytecode writer: serializes loom_module_t to .loombc format.
//
// Streaming architecture:
//   1. Intern module metadata (names, sources, symbols) — no output.
//   2. For each function: intern signature, then stream body to IR
//      section through a page-buffered writer. Numbering tables grow
//      as new strings/types/ops are encountered.
//   3. Write SYMBOLS (buffered for offset table patching), then
//      reference sections (STRINGS, TYPES, OPS, etc.) from the
//      now-complete numbering tables.
//   4. Seek back to patch the section directory and module length.
//
// The stream must be writable and seekable. Both iree_io_vec_stream_t
// (in-memory, for tests) and iree_io_stdio_stream_t (files) support
// this. Non-seekable consumers (network) wrap in a vec_stream.
//
// Section data is NOT buffered in memory (except SYMBOLS, which needs
// internal offset table patching and is always small). All other
// sections stream directly through a 4KB page buffer to amortize
// the iree_io_stream_write vtable dispatch cost.

#ifndef LOOM_FORMAT_BYTECODE_WRITER_H_
#define LOOM_FORMAT_BYTECODE_WRITER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "loom/format/bytecode/format.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling writer behavior.
typedef struct loom_bytecode_write_options_t {
  // Producer string embedded in the file header for diagnostics
  // (e.g., "loom-compile 1.0"). If empty, defaults to "loom-c".
  iree_string_view_t producer;
  // Source-location mode to write in the file header. Zero selects
  // LOOM_BYTECODE_LOCATION_MODE_SOURCE_LOCATIONS.
  loom_bytecode_location_mode_t location_mode;
} loom_bytecode_write_options_t;

// Serializes |module| to .loombc format through |stream|.
//
// The stream must have IREE_IO_STREAM_MODE_WRITABLE and
// IREE_IO_STREAM_MODE_SEEKABLE. The module is not mutated.
//
// |block_pool| provides arena blocks for temporary working memory
// (numbering tables, value numbering maps). The writer creates a
// temporary arena from the pool and deinitializes it before returning,
// returning all blocks to the pool. The pool is typically the same one
// used by the module and other compilation artifacts.
//
// |options| may be NULL for defaults (producer="loom-c",
// location_mode=SOURCE_LOCATIONS).
iree_status_t loom_bytecode_write_module(
    const loom_module_t* module, iree_io_stream_t* stream,
    const loom_bytecode_write_options_t* options,
    iree_arena_block_pool_t* block_pool);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_H_
