// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Attribute wire grammar for validation and IR materialization.

#ifndef LOOM_FORMAT_BYTECODE_READER_ATTRIBUTE_H_
#define LOOM_FORMAT_BYTECODE_READER_ATTRIBUTE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable state required to validate attribute payloads.
typedef struct loom_bytecode_attribute_validator_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Finalized dialect and parameterized-attribute registry context.
  loom_context_t* context;
  // Scratch-lived module tables referenced by attribute payloads.
  const loom_bytecode_reader_module_view_t* module_view;
} loom_bytecode_attribute_validator_t;

// State required to construct module-owned attribute payloads.
typedef struct loom_bytecode_attribute_materializer_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Finalized dialect and parameterized-attribute registry context.
  loom_context_t* context;
  // Scratch-lived module tables referenced by attribute payloads.
  const loom_bytecode_reader_module_view_t* module_view;
  // Resettable scratch storage for aggregate construction.
  iree_arena_allocator_t* scratch_arena;
  // Module receiving canonical attribute payloads.
  loom_module_t* output_module;
} loom_bytecode_attribute_materializer_t;

// Validation bounds for predicate VALUE arguments encoded as SSA numbers.
typedef struct loom_bytecode_attribute_ssa_validation_scope_t {
  // Symbol owning the SSA namespace, used in malformed-input diagnostics.
  iree_string_view_t symbol_name;
  // Number of scope-local values available at the attribute use.
  uint64_t value_count;
} loom_bytecode_attribute_ssa_validation_scope_t;

// Materialization map for predicate VALUE arguments encoded as SSA numbers.
typedef struct loom_bytecode_attribute_ssa_materialization_scope_t {
  // Symbol owning the SSA namespace, used in malformed-input diagnostics.
  iree_string_view_t symbol_name;
  // Scope-local value numbers mapped to module value IDs.
  const loom_value_id_t* values;
  // Number of entries in |values| available at the attribute use.
  uint64_t value_count;
} loom_bytecode_attribute_ssa_materialization_scope_t;

// One decoded predicate with source-domain VALUE arguments and diagnostics.
typedef struct loom_bytecode_wire_predicate_t {
  // Decoded predicate header, tags, and immediate arguments.
  loom_predicate_t value;
  // Raw wire ordinals for VALUE arguments.
  uint64_t value_numbers[IREE_ARRAYSIZE(((loom_predicate_t*)0)->args)];
  // Absolute wire offsets for argument diagnostics.
  uint64_t argument_offsets[IREE_ARRAYSIZE(((loom_predicate_t*)0)->args)];
} loom_bytecode_wire_predicate_t;

// One decoded canonical signed enum set.
typedef struct loom_bytecode_wire_signed_enum_set_t {
  // Positive words followed by the equally sized negative word span.
  uint64_t words[LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT * 2];
  // Number of words in each signed half of |words|.
  uint8_t word_count;
} loom_bytecode_wire_signed_enum_set_t;

// Reads and validates one attribute wire-kind tag.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline iree_status_t
loom_bytecode_attribute_read_kind(loom_bytecode_reader_decoder_t* decoder,
                                  loom_bytecode_reader_cursor_t* cursor,
                                  loom_bytecode_attr_kind_t* out_kind) {
  const uint64_t kind_offset =
      loom_bytecode_reader_cursor_absolute_position(cursor);
  uint8_t kind = 0;
  IREE_RETURN_IF_ERROR(loom_bytecode_reader_read_u8(decoder, cursor, &kind));
  if (kind >= LOOM_BYTECODE_ATTR_COUNT) {
    return loom_bytecode_reader_emit_enum_value(
        decoder, IREE_SV("attribute_kind"), kind, LOOM_BYTECODE_ATTR_COUNT,
        kind_offset);
  }
  *out_kind = (loom_bytecode_attr_kind_t)kind;
  return iree_ok_status();
}

// Reads one predicate payload without projecting VALUE arguments.
iree_status_t loom_bytecode_attribute_read_predicate(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint64_t predicate_index,
    loom_bytecode_wire_predicate_t* out_predicate);

// Reads and validates a predicate-list count.
iree_status_t loom_bytecode_attribute_read_predicate_count(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor, uint16_t* out_predicate_count);

// Reads and validates one canonical descriptor-backed signed enum set.
iree_status_t loom_bytecode_attribute_read_signed_enum_set(
    loom_bytecode_reader_decoder_t* decoder,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor,
    loom_bytecode_wire_signed_enum_set_t* out_set);

// Finds a parameter by spelling in the descriptor suffix beginning at
// |start_index|. Returns LOOM_ATTR_INDEX_NONE when no parameter matches.
uint8_t loom_bytecode_attribute_find_parameter_index(
    const loom_attr_descriptor_t* parameter_descriptors,
    uint8_t parameter_count, iree_string_view_t parameter_name,
    uint8_t start_index);

// Validates an attribute whose predicate VALUE arguments are STRINGS ordinals.
iree_status_t loom_bytecode_attribute_validate_named(
    loom_bytecode_attribute_validator_t* validator,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    iree_host_size_t available_type_count);

// Validates an attribute whose predicate VALUE arguments are SSA numbers.
iree_status_t loom_bytecode_attribute_validate_ssa(
    loom_bytecode_attribute_validator_t* validator,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    iree_host_size_t available_type_count,
    const loom_bytecode_attribute_ssa_validation_scope_t* ssa_scope);

// Validates a predicate list with SSA-number VALUE arguments and returns its
// encoded predicate count.
iree_status_t loom_bytecode_attribute_validate_predicate_list_ssa(
    loom_bytecode_attribute_validator_t* validator,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_validation_scope_t* ssa_scope,
    uint16_t* out_predicate_count);

// Validates and retains a predicate list with signature-local VALUE ordinals.
//
// The returned predicates are allocated from |retained_arena| and preserve
// wire VALUE arguments as validated signature-local ordinals. Passing NULL for
// |retained_arena| and |out_predicates| performs validation without retention.
iree_status_t loom_bytecode_attribute_retain_predicate_list_ssa(
    loom_bytecode_attribute_validator_t* validator,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_validation_scope_t* ssa_scope,
    iree_arena_allocator_t* retained_arena,
    const loom_predicate_t** out_predicates, uint16_t* out_predicate_count);

// Materializes an attribute whose predicate VALUE arguments are STRINGS
// ordinals.
iree_status_t loom_bytecode_attribute_materialize_named(
    loom_bytecode_attribute_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count);

// Materializes an attribute whose predicate VALUE arguments are SSA numbers.
iree_status_t loom_bytecode_attribute_materialize_ssa(
    loom_bytecode_attribute_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_attr_descriptor_t* descriptor, loom_bytecode_attr_kind_t kind,
    loom_attribute_t* out_attr, iree_host_size_t available_type_count,
    const loom_bytecode_attribute_ssa_materialization_scope_t* ssa_scope);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_ATTRIBUTE_H_
