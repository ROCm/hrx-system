// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_VERIFY_VERIFY_STATE_H_
#define LOOM_VERIFY_VERIFY_STATE_H_

#include "iree/base/internal/arena.h"
#include "loom/analysis/consumption.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/special_values.h"
#include "loom/verify/verify.h"

// Maximum region nesting depth tracked by the verifier's scope stack.
#define LOOM_VERIFY_MAX_SCOPE_DEPTH 32

// Reusable scratch tables for validating one op's tied-result metadata.
typedef struct loom_verify_tied_table_t {
  // Bitset tracking whether each result index was claimed.
  uint64_t* result_index_bits;

  // Number of allocated result index bitset words.
  iree_host_size_t result_index_word_capacity;

  // Current source occurrence counter for each result field.
  uint16_t* result_field_occurrences;

  // First source occurrence that claimed each result field.
  uint16_t* first_result_field_occurrences;

  // Number of allocated result occurrence entries.
  iree_host_size_t result_field_occurrence_capacity;

  // Bitset tracking whether each operand index was claimed.
  uint64_t* operand_index_bits;

  // Number of allocated operand index bitset words.
  iree_host_size_t operand_index_word_capacity;

  // Current source occurrence counter for each operand field.
  uint16_t* operand_field_occurrences;

  // First source occurrence that claimed each operand field.
  uint16_t* first_operand_field_occurrences;

  // Number of allocated operand occurrence entries.
  iree_host_size_t operand_field_occurrence_capacity;

  // Sorted scratch copy of valid operand value IDs for the current op.
  loom_value_id_t* operand_value_ids;

  // Number of entries in operand_value_ids.
  iree_host_size_t operand_value_count;

  // Number of allocated operand_value_ids entries.
  iree_host_size_t operand_value_capacity;
} loom_verify_tied_table_t;

// Module-wide canonical type facts gathered before the op walk.
typedef struct loom_verify_type_summary_t {
  // True when every canonical type satisfies representation invariants.
  bool all_well_formed;

  // True when any canonical type may carry an SSA value reference.
  bool may_reference_values;
} loom_verify_type_summary_t;

// State inherited while recursively verifying one region tree.
typedef struct loom_verify_region_scope_t {
  // Region currently being verified.
  const loom_region_t* current;

  // Reusable consumed-value query for current.
  loom_consumption_region_query_t* consumption_query;

  // True when observable effects must be explicit command effects.
  bool command_effects_only;
} loom_verify_region_scope_t;

typedef struct loom_verify_state_t {
  // Module being verified.
  const loom_module_t* module;

  // Final diagnostic sink configured for this verification run.
  loom_diagnostic_sink_t sink;

  // Source resolver used for exact caret ranges.
  loom_source_resolver_t source_resolver;

  // Caller-owned result counters.
  loom_verify_result_t* result;

  // Maximum number of errors to emit before stopping the walk.
  uint32_t max_errors;

  // First non-OK status returned by the diagnostic sink.
  iree_status_t diagnostic_status;

  // Facts derived from the module's canonical type table.
  loom_verify_type_summary_t type_summary;

  // Malformed static encoding diagnostics needed by this verification run.
  struct {
    // Bitset recording encoding records diagnosed at an authored use site.
    uint64_t* diagnosed_bits;

    // Number of allocated words in diagnosed_bits.
    iree_host_size_t word_count;
  } static_encodings;

  // Module symbols named by top-level availability metadata. An unresolved
  // dependency is valid when its symbol is available from an external source.
  struct {
    // Dense bitset indexed by module symbol ID.
    uint64_t* bits;

    // Number of allocated words in bits.
    iree_host_size_t word_count;
  } available_symbols;

  // Scratch arena for all verification-time allocations.
  iree_arena_allocator_t arena;

  // Bitset indexed by value_id; a set bit means the value is visible.
  uint64_t* defined_bits;

  // Number of uint64_t words in defined_bits and consumed_bits.
  iree_host_size_t defined_bits_length;

  // Bitset indexed by value_id; a set bit means the value was consumed.
  uint64_t* consumed_bits;

  // First op that consumed each value_id through an ownership transfer.
  const loom_op_t** consuming_ops;

  // Reusable per-op scratch for tied-result uniqueness checks.
  loom_verify_tied_table_t tied_table;

  // State inherited through the current nested region traversal.
  loom_verify_region_scope_t region_scope;

  // Stack of value IDs defined during the current scoped walk.
  uint32_t* defined_stack;

  // Number of entries in defined_stack.
  iree_host_size_t defined_stack_count;

  // Number of allocated defined_stack entries.
  iree_host_size_t defined_stack_capacity;

  // Defined-stack watermarks at each region scope entry.
  iree_host_size_t scope_watermarks[LOOM_VERIFY_MAX_SCOPE_DEPTH];

  // Number of active region scopes.
  uint32_t scope_depth;
} loom_verify_state_t;

static inline void loom_bitset_set(uint64_t* bits, iree_host_size_t word_count,
                                   uint32_t index) {
  iree_host_size_t word_index = index / 64;
  IREE_ASSERT(word_index < word_count);
  bits[word_index] |= ((uint64_t)1 << (index % 64));
}

static inline void loom_bitset_clear(uint64_t* bits,
                                     iree_host_size_t word_count,
                                     uint32_t index) {
  iree_host_size_t word_index = index / 64;
  IREE_ASSERT(word_index < word_count);
  bits[word_index] &= ~((uint64_t)1 << (index % 64));
}

static inline bool loom_bitset_test(const uint64_t* bits,
                                    iree_host_size_t word_count,
                                    uint32_t index) {
  iree_host_size_t word_index = index / 64;
  IREE_ASSERT(word_index < word_count);
  return (bits[word_index] & ((uint64_t)1 << (index % 64))) != 0;
}

static inline iree_host_size_t loom_bitset_word_count(
    iree_host_size_t bit_count) {
  return bit_count > 0 ? (bit_count + 63) / 64 : 0;
}

void loom_verify_record_diagnostic_status(loom_verify_state_t* state,
                                          iree_status_t status);
iree_status_t loom_verify_take_diagnostic_status(loom_verify_state_t* state);
iree_status_t loom_verify_pending_diagnostic_status(loom_verify_state_t* state);

iree_status_t loom_verify_push_scope(loom_verify_state_t* state);
void loom_verify_pop_scope(loom_verify_state_t* state);
iree_status_t loom_verify_define_value(loom_verify_state_t* state,
                                       loom_value_id_t value_id);
void loom_verify_consume_value(loom_verify_state_t* state,
                               loom_value_id_t value_id,
                               const loom_op_t* consuming_op);

iree_host_size_t loom_verify_source_byte_offset(iree_string_view_t source,
                                                uint32_t line, uint32_t column);

bool loom_verify_at_error_limit(const loom_verify_state_t* state);
const loom_op_vtable_t* loom_verify_lookup_vtable(
    const loom_verify_state_t* state, loom_op_kind_t kind);
iree_string_view_t loom_verify_value_name(const loom_verify_state_t* state,
                                          loom_value_id_t value_id);
iree_string_view_t loom_verify_symbol_name(const loom_verify_state_t* state,
                                           loom_symbol_ref_t ref);
iree_string_view_t loom_verify_symbol_definition_name(
    const loom_symbol_t* symbol);

bool loom_verify_func_args_use_operand_field(const loom_op_vtable_t* vtable);
bool loom_verify_has_func_signature_scope(const loom_op_vtable_t* vtable);

loom_type_t loom_verify_value_type(const loom_verify_state_t* state,
                                   loom_value_id_t value_id);
loom_value_id_t loom_verify_resolve_value_field(const loom_op_t* op,
                                                const loom_op_vtable_t* vtable,
                                                uint8_t field_ref);
bool loom_verify_is_variadic_field(const loom_op_vtable_t* vtable,
                                   uint8_t field_ref);
uint16_t loom_verify_variadic_count(const loom_op_t* op,
                                    const loom_op_vtable_t* vtable,
                                    uint8_t field_ref);
const loom_value_id_t* loom_verify_resolve_variadic_field(
    const loom_op_t* op, const loom_op_vtable_t* vtable, uint8_t field_ref,
    uint16_t* out_count);

iree_string_view_t loom_verify_field_name(const loom_op_vtable_t* vtable,
                                          uint8_t field_ref, char* buffer,
                                          iree_host_size_t buffer_size);
iree_string_view_t loom_verify_indexed_field_name(
    const loom_op_vtable_t* vtable, uint8_t field_ref, uint16_t element_index,
    char* buffer, iree_host_size_t buffer_size);
iree_string_view_t loom_verify_value_field_name(
    const loom_op_vtable_t* vtable, const loom_op_t* op, uint8_t category,
    uint16_t value_index, char* buffer, iree_host_size_t buffer_size);

#endif  // LOOM_VERIFY_VERIFY_STATE_H_
