// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sparse declaration expectations joining the TYPES and SYMBOLS validators.

#ifndef LOOM_FORMAT_BYTECODE_READER_REFERENCE_BINDING_H_
#define LOOM_FORMAT_BYTECODE_READER_REFERENCE_BINDING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/ir/symbol_map.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_reference_binding_t {
  // First TYPES use, retained for an unresolved or inconsistent binding error.
  uint64_t bytecode_offset;
  // Expected external namespace in the unique source STRINGS table.
  loom_string_id_t namespace_id;
  // Expected payload name in the unique source STRINGS table.
  loom_string_id_t name_id;
} loom_bytecode_reference_binding_t;

typedef struct loom_bytecode_reference_bindings_t {
  // Declaration name to expectation index; matched declarations are erased.
  loom_symbol_map_t pending;
  // Arena-owned expected keys, one per referenced declaration.
  loom_bytecode_reference_binding_t* values;
  // Number of initialized expectations, independent of handle mode and uses.
  iree_host_size_t count;
  // Allocated expectation capacity.
  iree_host_size_t capacity;
} loom_bytecode_reference_bindings_t;

// Records a type's expected declaration key, rejecting conflicting uses.
bool loom_bytecode_reference_binding_expect(
    loom_bytecode_reader_decoder_t* decoder, iree_arena_allocator_t* arena,
    loom_bytecode_reference_bindings_t* bindings,
    loom_string_id_t declaration_name_id, loom_string_id_t namespace_id,
    loom_string_id_t name_id, uint64_t bytecode_offset);

// Matches a decoded symbol against its pending type expectation. A non-type
// symbol supplies INVALID key IDs, which cannot satisfy a type binding.
bool loom_bytecode_reference_binding_match(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reference_bindings_t* bindings,
    loom_string_id_t declaration_name_id, loom_string_id_t namespace_id,
    loom_string_id_t name_id);

// Rejects any type declaration name absent from the complete SYMBOLS table.
bool loom_bytecode_reference_bindings_finish(
    loom_bytecode_reader_decoder_t* decoder,
    const loom_bytecode_reference_bindings_t* bindings);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_REFERENCE_BINDING_H_
