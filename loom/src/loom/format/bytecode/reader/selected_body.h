// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reached-only function-local IR body materialization.

#ifndef LOOM_FORMAT_BYTECODE_READER_SELECTED_BODY_H_
#define LOOM_FORMAT_BYTECODE_READER_SELECTED_BODY_H_

#include "loom/format/bytecode/reader/body.h"
#include "loom/format/bytecode/reader/selected_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

// State required to materialize selected symbol IR bodies.
typedef struct loom_bytecode_selected_body_materializer_t {
  // Reached-only source-to-output table materializer.
  loom_bytecode_selected_table_materializer_t* tables;
  // Block source for function-local scratch arenas.
  iree_arena_block_pool_t* block_pool;
  // Stable-key codec supplied by the embedding compiler.
  const loom_low_repr_environment_t* low_repr_environment;
} loom_bytecode_selected_body_materializer_t;

// Scope-local SSA value materialization for selected symbols and IR bodies.
typedef struct loom_bytecode_selected_value_scope_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Reached-only source-to-output table materializer.
  loom_bytecode_selected_table_materializer_t* tables;
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
} loom_bytecode_selected_value_scope_t;

// Initializes a selected value scope whose map consists of fresh module values.
iree_status_t loom_bytecode_selected_value_scope_initialize_fresh(
    loom_bytecode_selected_body_materializer_t* materializer,
    iree_arena_allocator_t* arena, iree_string_view_t symbol_name,
    uint64_t payload_offset, loom_value_id_t* value_map,
    iree_host_size_t value_count,
    loom_bytecode_selected_value_scope_t* out_scope);

// Materializes the next selected reserved value definition from |cursor|.
iree_status_t loom_bytecode_selected_value_scope_materialize_definition(
    loom_bytecode_selected_value_scope_t* value_scope,
    loom_bytecode_reader_cursor_t* cursor, loom_value_id_t* out_value_id);

// Materializes one selected bounded symbol IR payload into |parent_op| regions.
iree_status_t loom_bytecode_selected_body_materialize_symbol_regions(
    loom_bytecode_selected_body_materializer_t* materializer,
    iree_string_view_t symbol_name, iree_const_byte_span_t body_bytes,
    uint64_t body_absolute_offset, const loom_bytecode_body_summary_t* summary,
    loom_builder_t* builder, loom_op_t* parent_op, uint8_t first_region_index,
    const loom_bytecode_predefined_region_values_t* predefined_regions,
    uint8_t predefined_region_count,
    const loom_low_repr_descriptor_set_t* low_descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_SELECTED_BODY_H_
