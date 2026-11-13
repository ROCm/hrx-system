// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Type-table validation and immutable topological plan construction.

#ifndef LOOM_FORMAT_BYTECODE_READER_TYPE_VALIDATOR_H_
#define LOOM_FORMAT_BYTECODE_READER_TYPE_VALIDATOR_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/type_plan.h"
#include "loom/ir/context.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_reader_module_view_t
    loom_bytecode_reader_module_view_t;

// Facts retained by the consumer of one forward TYPES cursor.
typedef enum loom_bytecode_type_retention_e {
  LOOM_BYTECODE_TYPE_RETAIN_NONE = 0,
  LOOM_BYTECODE_TYPE_RETAIN_RANGES,
  LOOM_BYTECODE_TYPE_RETAIN_PLAN,
} loom_bytecode_type_retention_t;

// A forward type-table cursor advanced between completed encoding entries.
typedef struct loom_bytecode_type_validation_t {
  // Bounded decoder and public diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Finalized type and attribute descriptor registry.
  loom_context_t* context;
  // Table facts whose encoding prefix bounds each newly decoded type.
  loom_bytecode_reader_module_view_t* module_view;
  // Owner of retained plans/ranges and pending declaration expectations.
  iree_arena_allocator_t* arena;
  // Validated end of the TYPES prefix already consumed from the cursor.
  loom_type_id_t position;
  // Next type entry in the bounded TYPES payload.
  loom_bytecode_reader_cursor_t cursor;
  // Consumer policy chosen once by the module reader.
  loom_bytecode_type_retention_t retention;
  // Retained entry ranges when the consumer requests an index.
  loom_bytecode_table_entry_metadata_t* entries;
  // Last full-reader sparse fact, allowing constant-time append across
  // prefixes.
  loom_bytecode_type_fact_t* last_fact;
} loom_bytecode_type_validation_t;

// Reads the declared type count and allocates only the requested retained
// facts.
bool loom_bytecode_type_validation_begin(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_const_byte_span_t section_bytes, uint64_t section_absolute_offset,
    loom_bytecode_type_retention_t retention, iree_arena_allocator_t* arena,
    loom_bytecode_type_validation_t* out_validation);

// Consumes the newly available prefix. The owning encoding decoder establishes
// monotonic |type_count| within the declared count before calling this method.
bool loom_bytecode_type_validation_advance(
    loom_bytecode_type_validation_t* validation, iree_host_size_t type_count);

// Consumes remaining types after all encodings and verifies exact section use.
bool loom_bytecode_type_validation_finish(
    loom_bytecode_type_validation_t* validation);

// Decodes one retained, already bounded TYPES entry into the same immutable
// fact representation used by the full sequential validator. |type_index| is
// the source table ordinal and therefore bounds all prior-type references.
bool loom_bytecode_type_plan_decode_indexed_entry(
    loom_bytecode_reader_decoder_t* decoder, loom_context_t* context,
    loom_bytecode_reader_module_view_t* module_view,
    iree_arena_allocator_t* scratch_arena, loom_type_id_t type_index,
    iree_const_byte_span_t entry_bytes, uint64_t entry_absolute_offset,
    loom_bytecode_type_plan_entry_t* out_plan_entry,
    loom_bytecode_type_fact_t** out_fact);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_TYPE_VALIDATOR_H_
