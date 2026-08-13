// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Type-table validation plans and canonical IR materialization.

#ifndef LOOM_FORMAT_BYTECODE_READER_TYPE_H_
#define LOOM_FORMAT_BYTECODE_READER_TYPE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/type_plan.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_reader_module_view_t
    loom_bytecode_reader_module_view_t;

// State required to materialize a validated type plan into a module.
typedef struct loom_bytecode_type_materializer_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Full input bytes containing retained parameterized-type spans.
  iree_const_byte_span_t bytecode;
  // Finalized type and parameterized-type registry context.
  loom_context_t* context;
  // Immutable validated module facts containing the type plan.
  const loom_bytecode_reader_module_view_t* module_view;
  // Resettable scratch storage for composite type construction.
  iree_arena_allocator_t* scratch_arena;
  // Module receiving canonical type-table entries.
  loom_module_t* output_module;
} loom_bytecode_type_materializer_t;

// Validates one TYPES section and builds its immutable topological plan.
iree_status_t loom_bytecode_type_plan_build(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena, iree_const_byte_span_t section_bytes,
    uint64_t section_absolute_offset);

// Materializes every entry in an immutable validated type plan.
iree_status_t loom_bytecode_type_materialize(
    loom_bytecode_type_materializer_t* materializer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_TYPE_H_
