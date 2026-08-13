// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Full module materialization from a validated bytecode module view.

#ifndef LOOM_FORMAT_BYTECODE_READER_MATERIALIZER_H_
#define LOOM_FORMAT_BYTECODE_READER_MATERIALIZER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/format/low_repr.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable inputs for one complete module materialization.
typedef struct loom_bytecode_module_materializer_t {
  // Bounded decoder and shared diagnostic result state.
  const loom_bytecode_reader_decoder_t* decoder;
  // Complete source bytecode containing retained table spans.
  iree_const_byte_span_t bytecode;
  // Finalized dialect and registry context.
  loom_context_t* context;
  // Resettable storage for transient materialization state.
  iree_arena_allocator_t* scratch_arena;
  // Block source for the output module and body-local arenas.
  iree_arena_block_pool_t* block_pool;
  // Validated scratch-lived module facts.
  const loom_bytecode_reader_module_view_t* module_view;
  // Stable-key codec supplied by the embedding compiler.
  loom_low_repr_environment_t low_repr_environment;
  // Host allocator owning the returned module.
  iree_allocator_t host_allocator;
} loom_bytecode_module_materializer_t;

// Materializes a standalone module and publishes it only after all tables and
// bodies have been constructed successfully.
iree_status_t loom_bytecode_module_materialize(
    const loom_bytecode_module_materializer_t* materializer,
    loom_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_MATERIALIZER_H_
