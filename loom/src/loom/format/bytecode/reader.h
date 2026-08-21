// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bytecode IR materialization.
//
// This is the aggregate bytecode-reading surface. It re-exports metadata-only
// validation and retained indexing from index.h and adds entry points that
// materialize validated bytecode into Loom IR.

#ifndef LOOM_FORMAT_BYTECODE_READER_H_
#define LOOM_FORMAT_BYTECODE_READER_H_

#include "loom/format/bytecode/index.h"
#include "loom/format/low_repr.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options controlling bytecode IR materialization.
typedef struct loom_bytecode_read_options_t {
  // Sink for structured malformed-bytecode and verification diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;

  // Runs the module verifier after successful materialization. Verification
  // diagnostics are emitted to diagnostic_sink and counted in the read result.
  bool verify_module;

  // Maximum verifier errors to emit when verify_module is set. Zero means
  // unlimited, matching loom_verify_options_t.
  uint32_t verify_max_errors;

  // Stable-key codec required when materializing Low function bodies.
  loom_low_repr_environment_t low_repr_environment;
} loom_bytecode_read_options_t;

// Reads a single-module .loombc file and materializes its IR into |out_module|.
//
// Malformed bytecode follows the same diagnostic contract as
// loom_bytecode_read_metadata: diagnostics are emitted and counted in
// |out_result| while the function returns OK unless infrastructure fails. When
// malformed-bytecode diagnostics are emitted, |out_module| is NULL.
//
// |host_allocator| owns the returned module object. All IR storage inside that
// module is arena-owned by the module and released by loom_module_free.
iree_status_t loom_bytecode_read_module(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator);

// Reads and materializes one module from a .loombc file by directory ordinal.
//
// This has the same diagnostic and ownership contract as
// loom_bytecode_read_module, but accepts multi-module files and materializes
// only |module_ordinal|.
iree_status_t loom_bytecode_read_module_ordinal(
    iree_const_byte_span_t bytecode, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    uint16_t module_ordinal, const loom_bytecode_read_options_t* options,
    loom_bytecode_read_result_t* out_result, loom_module_t** out_module,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_H_
