// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function-local IR body grammar and module materialization.

#ifndef LOOM_FORMAT_BYTECODE_READER_BODY_H_
#define LOOM_FORMAT_BYTECODE_READER_BODY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/module_summary.h"
#include "loom/format/bytecode/reader/attribute.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/format/low_repr.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_bytecode_reader_module_view_t
    loom_bytecode_reader_module_view_t;
typedef struct loom_builder_t loom_builder_t;

// Allocation summary decoded from one bounded root-region payload. Counts
// describe only IR nested inside that root. The value count includes entry
// block arguments prebound to function signature values, but excludes results
// defined by the parent symbol operation.
typedef struct loom_bytecode_region_summary_t {
  // SSA values defined by block arguments and operation results in the root.
  uint32_t value_count;
  // Regions in the root, including nested regions.
  uint32_t region_count;
  // Blocks in the root, including nested regions.
  uint32_t block_count;
  // Live operations in the root, including nested regions.
  uint32_t op_count;
  // Byte offset of the root-region record in the bounded payload span.
  uint8_t payload_offset;
} loom_bytecode_region_summary_t;

enum loom_bytecode_region_materialization_flag_bits_e {
  // Bind the root entry arguments to the supplied signature-value slice,
  // including when that slice is empty.
  LOOM_BYTECODE_REGION_MATERIALIZATION_FLAG_BIND_ENTRY_ARGUMENTS = 1u << 0,
};
typedef uint32_t loom_bytecode_region_materialization_flags_t;

// State required to materialize validated symbol IR root regions.
typedef struct loom_bytecode_body_materializer_t {
  // Shared module and attribute materialization state.
  loom_bytecode_attribute_materializer_t attributes;
  // Block source for root-region-local scratch arenas.
  iree_arena_block_pool_t* block_pool;
  // Stable-key codec supplied by the embedding compiler.
  const loom_low_repr_environment_t* low_repr_environment;
} loom_bytecode_body_materializer_t;

// Scope-local SSA value materialization shared by symbols and IR bodies.
typedef struct loom_bytecode_value_scope_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Immutable validated module tables referenced by value definitions.
  const loom_bytecode_reader_module_view_t* module_view;
  // Module receiving decoded SSA values.
  loom_module_t* output_module;
  // Scratch storage for rebound type payloads.
  iree_arena_allocator_t* arena;
  // Symbol name used in malformed-input diagnostics.
  iree_string_view_t symbol_name;
  // Absolute byte offset of the value-owning payload.
  uint64_t payload_offset;
  // Scope-local value numbers mapped to module value IDs.
  loom_value_id_t* value_map;
  // Maximum number of value definitions in the scope.
  uint64_t value_capacity;
  // Next value definition to decode.
  uint64_t next_value_number;
  // Reserved value prefix available to type bindings.
  uint64_t available_value_count;
  // Next fresh module value ID reserved for this scope.
  loom_value_id_t next_fresh_value_id;
  // Number of reserved fresh module value IDs not yet assigned.
  uint64_t remaining_fresh_value_count;
  // Existing module values substituted into one reserved slice.
  const loom_value_id_t* predefined_values;
  // Scope-local number of the first predefined value.
  uint64_t predefined_value_start;
  // Number of entries in |predefined_values|.
  uint16_t predefined_value_count;
} loom_bytecode_value_scope_t;

// Decodes and validates the allocation summary prefix of one exact root-region
// payload. The returned payload offset lets callers retain the summary and
// later materialize the region without decoding the prefix again.
iree_status_t loom_bytecode_region_summary_read(
    loom_bytecode_reader_decoder_t* decoder, iree_string_view_t symbol_name,
    iree_const_byte_span_t payload_bytes, uint64_t payload_absolute_offset,
    loom_bytecode_region_summary_t* out_summary);

// Initializes a value scope whose entire map consists of fresh module values.
iree_status_t loom_bytecode_value_scope_initialize_fresh(
    loom_bytecode_body_materializer_t* materializer,
    iree_arena_allocator_t* arena, iree_string_view_t symbol_name,
    uint64_t payload_offset, loom_value_id_t* value_map,
    iree_host_size_t value_count, loom_bytecode_value_scope_t* out_scope);

// Materializes the next reserved value definition from |cursor|.
iree_status_t loom_bytecode_value_scope_materialize_definition(
    loom_bytecode_value_scope_t* value_scope,
    loom_bytecode_reader_cursor_t* cursor, loom_value_id_t* out_value_id);

// Materializes one bounded root-region payload into |region_index| on
// |parent_op| using a summary returned by loom_bytecode_region_summary_read.
iree_status_t loom_bytecode_body_materialize_region(
    loom_bytecode_body_materializer_t* materializer,
    iree_string_view_t symbol_name, iree_const_byte_span_t payload_bytes,
    uint64_t payload_absolute_offset,
    const loom_bytecode_region_summary_t* summary, loom_builder_t* builder,
    loom_op_t* parent_op, uint8_t region_index,
    loom_bytecode_region_materialization_flags_t flags,
    const loom_value_id_t* predefined_values, uint16_t predefined_value_count,
    const loom_low_repr_descriptor_set_t* low_descriptor_set);

// Materializes the complete non-symbol module operation forest.
iree_status_t loom_bytecode_body_materialize_module_ops(
    loom_bytecode_body_materializer_t* materializer,
    iree_string_view_t module_name, iree_const_byte_span_t payload_bytes,
    uint64_t payload_absolute_offset,
    const loom_bytecode_module_ops_summary_t* summary);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_BODY_H_
