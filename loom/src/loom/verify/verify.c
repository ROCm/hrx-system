// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/verify/verify_constraints.h"
#include "loom/verify/verify_diagnostics.h"
#include "loom/verify/verify_ownership.h"
#include "loom/verify/verify_state.h"
#include "loom/verify/verify_structure.h"

static iree_status_t loom_verify_op(loom_verify_state_t* state,
                                    const loom_op_t* op);

static iree_status_t loom_verify_region(loom_verify_state_t* state,
                                        loom_region_t* region) {
  if (!region) return iree_ok_status();
  const loom_region_t* saved_region = state->current_region;
  loom_consumption_region_query_t* saved_consumption_query =
      state->current_consumption_query;
  loom_consumption_region_query_t consumption_query;
  state->current_region = region;
  iree_status_t status = loom_consumption_region_query_initialize(
      state->module, region, &state->arena, &consumption_query);
  if (!iree_status_is_ok(status)) {
    state->current_region = saved_region;
    state->current_consumption_query = saved_consumption_query;
    return status;
  }
  state->current_consumption_query = &consumption_query;

  bool scope_pushed = false;
  status = loom_verify_push_scope(state);
  if (iree_status_is_ok(status)) {
    scope_pushed = true;
  }

  for (uint16_t b = 0; iree_status_is_ok(status) && b < region->block_count;
       ++b) {
    loom_block_t* block = loom_region_block(region, b);
    // Define block arguments, then validate any SSA encoding
    // references in their types (encoding values must be visible
    // from the enclosing scope).
    for (uint16_t a = 0; iree_status_is_ok(status) && a < block->arg_count;
         ++a) {
      status = loom_verify_define_value(state, loom_block_arg_id(block, a));
    }
    if (iree_status_is_ok(status)) {
      loom_verify_block_arg_type_well_formedness(state, block);
      status = loom_verify_pending_diagnostic_status(state);
    }
    if (iree_status_is_ok(status)) {
      loom_verify_block_arg_encoding_refs(state, block);
      status = loom_verify_pending_diagnostic_status(state);
    }
    loom_op_t* current = NULL;
    loom_block_for_each_op(block, current) {
      if (!iree_status_is_ok(status) || loom_verify_at_error_limit(state)) {
        break;
      }
      status = loom_verify_op(state, current);
    }
  }

  if (scope_pushed) {
    loom_verify_pop_scope(state);
  }
  state->current_region = saved_region;
  state->current_consumption_query = saved_consumption_query;
  return status;
}

static iree_status_t loom_verify_op(loom_verify_state_t* state,
                                    const loom_op_t* op) {
  const uint32_t initial_error_count = state->result->error_count;
  const loom_op_vtable_t* vtable = loom_verify_lookup_vtable(state, op->kind);
  if (!vtable) {
    // Unknown op kind — emit structured diagnostic.
    char kind_buffer[16];
    iree_snprintf(kind_buffer, sizeof(kind_buffer), "0x%04x", op->kind);
    loom_diagnostic_param_t params[] = {
        loom_param_u32(op->kind),
        loom_param_string(iree_make_cstring_view(kind_buffer)),
    };
    loom_verify_emit_structured(state, op, LOOM_ERR_STRUCTURE_009, params,
                                IREE_ARRAYSIZE(params));
    IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));
    // Still define results so downstream dominance checks don't cascade.
    for (uint16_t i = 0; i < op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_verify_define_value(state, loom_op_const_results(op)[i]));
    }
    return iree_ok_status();
  }

  loom_verify_op_declared_trait_consistency(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));
  loom_verify_op_placement(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  const bool has_signature_scope = loom_verify_has_func_signature_scope(vtable);
  if (has_signature_scope) {
    IREE_RETURN_IF_ERROR(loom_verify_push_scope(state));
    uint16_t arg_count = 0;
    const loom_value_id_t* arg_ids =
        loom_verify_func_signature_arg_ids(op, vtable, &arg_count);
    for (uint16_t i = 0; i < arg_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_verify_define_value(state, arg_ids[i]));
    }
  }

  // Operand references must be valid and in scope. This must run
  // before any type-level checks since those read operand types.
  loom_verify_operand_dominance(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Structural checks: operand/result/attr/region counts match the
  // vtable declaration. Must pass before per-field checks below.
  loom_verify_op_structure(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Successor targets are direct block pointers. Reject malformed edges before
  // region-structure and dominance-sensitive checks observe CFG shape.
  loom_verify_successor_targets(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Per-instance trait callbacks may refine the static vtable declaration but
  // must preserve the same effect and optimization invariants as declared
  // traits. Run this after structural checks so callbacks can safely inspect
  // attributes.
  loom_verify_op_effective_trait_consistency(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Operand and result type payloads must satisfy core representation
  // invariants before table-driven constraints interpret them.
  loom_verify_op_type_well_formedness(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Per-field type constraint checks (operand/result types satisfy
  // the type category declared in the vtable descriptors).
  loom_verify_type_constraints(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // OperandDict format elements are stored as ordinary variadic operands plus
  // a key -> operand-ordinal DICT attribute. Verify that field metadata is
  // canonical and exactly describes the operand segment before later passes
  // depend on keyed lookup.
  loom_verify_operand_dicts(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // SSA encoding references embedded in operand/result types must point to
  // valid LOOM_TYPE_ENCODING values. Ordinary references must be in scope;
  // result co-references and global declaration placeholders have explicit
  // rules in loom_verify_encoding_ref.
  loom_verify_encoding_refs(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Poison may flow through pure SSA computation, but it must not be consumed
  // by an operation that observes values outside ordinary use-def propagation.
  loom_verify_poison_boundaries(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Semantic constraints: table-driven (relation, property) interpreter.
  // Requires structural and type checks to have passed so field refs
  // resolve to valid values with the expected type categories.
  loom_verify_semantic_constraints(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Tied result validation and linear ownership consume marking.
  IREE_RETURN_IF_ERROR(loom_verify_tied_results(state, op, vtable));
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Symbol reference attributes must point to valid symbol table entries.
  IREE_RETURN_IF_ERROR(loom_verify_symbol_definition(state, op, vtable));
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));
  loom_verify_symbol_references(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Region structure: single-block invariants, terminator presence.
  loom_verify_region_structure(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Define result values. Must happen after all checks on this op
  // so that a result cannot appear to dominate its own defining op.
  if (has_signature_scope) {
    loom_verify_pop_scope(state);
  } else {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_verify_define_value(state, loom_op_const_results(op)[i]));
    }
  }

  // Recurse into regions. Region contents see this op's results
  // (defined above) and all enclosing scope values.
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (loom_verify_at_error_limit(state)) break;
    IREE_RETURN_IF_ERROR(loom_verify_region(state, regions[i]));
  }
  loom_verify_func_purity_body_effects(state, op, vtable);
  IREE_RETURN_IF_ERROR(loom_verify_pending_diagnostic_status(state));

  // Op-specific verification callback. Runs last, and only when this op and
  // its nested regions did not emit any prior verifier errors. Callbacks may
  // assume structurally sound, type-correct IR.
  if (vtable->verify && state->result->error_count == initial_error_count) {
    iree_diagnostic_emitter_t emitter = {
        .fn = loom_verify_diagnostic_emitter_fn,
        .user_data = state,
    };
    IREE_RETURN_IF_ERROR(vtable->verify(state->module, op, emitter));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Source resolver: table-based implementation
//===----------------------------------------------------------------------===//

bool loom_source_table_resolve(void* user_data, const loom_module_t* module,
                               loom_location_id_t location,
                               loom_source_range_t* out_range) {
  loom_source_table_resolver_t* table =
      (loom_source_table_resolver_t*)user_data;
  if (!table || table->count == 0) return false;
  if (location == LOOM_LOCATION_UNKNOWN) return false;

  // Look up the location entry from the module's location table.
  if ((iree_host_size_t)location >= module->locations.count) return false;
  const loom_location_entry_t* entry = &module->locations.entries[location];
  if (entry->kind != LOOM_LOCATION_FILE) return false;

  // Find the matching source buffer by source_id.
  const loom_source_entry_t* source_entry = NULL;
  for (iree_host_size_t i = 0; i < table->count; ++i) {
    if (table->entries[i].source_id == entry->file.source_id) {
      source_entry = &table->entries[i];
      break;
    }
  }
  if (!source_entry) return false;

  // Compute byte offsets from line/column into the source buffer.
  iree_host_size_t start_offset = loom_verify_source_byte_offset(
      source_entry->source, entry->file.start_line, entry->file.start_col);
  iree_host_size_t end_offset = loom_verify_source_byte_offset(
      source_entry->source, entry->file.end_line, entry->file.end_col);

  *out_range = (loom_source_range_t){
      .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
      .filename = source_entry->filename,
      .source = source_entry->source,
      .start = start_offset,
      .end = end_offset,
      .start_line = entry->file.start_line,
      .start_column = entry->file.start_col,
      .end_line = entry->file.end_line,
      .end_column = entry->file.end_col,
  };
  return true;
}

//===----------------------------------------------------------------------===//
// Module and function verification entry points
//===----------------------------------------------------------------------===//

static void loom_verify_state_deinitialize(loom_verify_state_t* state);

static iree_status_t loom_verify_state_initialize(
    loom_verify_state_t* state, const loom_module_t* module,
    const loom_verify_options_t* options, loom_verify_result_t* out_result) {
  memset(state, 0, sizeof(*state));
  state->module = module;
  state->result = out_result;
  out_result->error_count = 0;
  out_result->warning_count = 0;

  if (options) {
    state->sink = options->sink;
    state->max_errors = options->max_errors;
    state->source_resolver = options->source_resolver;
  }

  iree_host_size_t value_count = module->values.count;
  // Ensure at least 1 word so we always have valid pointers.
  state->defined_bits_length = value_count > 0 ? (value_count + 63) / 64 : 1;

  // Initialize scratch arena from the module's block pool. All
  // verification-time allocations go here; bulk O(1) free on exit.
  iree_arena_initialize(module->arena.block_pool, &state->arena);

  // Allocate bitsets and defined stack. iree_arena_allocate_array
  // uses checked multiplication (overflow → RESOURCE_EXHAUSTED).
  // On any failure, deinitialize returns all arena blocks to the pool.
  iree_status_t status =
      iree_arena_allocate_array(&state->arena, state->defined_bits_length,
                                sizeof(uint64_t), (void**)&state->defined_bits);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(
        &state->arena, state->defined_bits_length, sizeof(uint64_t),
        (void**)&state->consumed_bits);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t consuming_op_count = value_count > 0 ? value_count : 1;
    status = iree_arena_allocate_array(&state->arena, consuming_op_count,
                                       sizeof(state->consuming_ops[0]),
                                       (void**)&state->consuming_ops);
    if (iree_status_is_ok(status)) {
      memset(state->consuming_ops, 0,
             consuming_op_count * sizeof(state->consuming_ops[0]));
    }
  }
  if (iree_status_is_ok(status)) {
    memset(state->defined_bits, 0,
           state->defined_bits_length * sizeof(uint64_t));
    memset(state->consumed_bits, 0,
           state->defined_bits_length * sizeof(uint64_t));
  }
  if (iree_status_is_ok(status)) {
    // Allocate defined stack. Start with value_count/4 capacity (most
    // values are results that get defined). Grows dynamically via
    // iree_arena_grow_array if the heuristic undersizes.
    state->defined_stack_capacity = value_count > 16 ? value_count / 4 : 16;
    status = iree_arena_allocate_array(
        &state->arena, state->defined_stack_capacity, sizeof(uint32_t),
        (void**)&state->defined_stack);
  }
  if (!iree_status_is_ok(status)) {
    loom_verify_state_deinitialize(state);
    return status;
  }

  return iree_ok_status();
}

static void loom_verify_state_deinitialize(loom_verify_state_t* state) {
  IREE_ASSERT(iree_status_is_ok(state->diagnostic_status));
  iree_arena_deinitialize(&state->arena);
}

iree_status_t loom_verify_module(const loom_module_t* module,
                                 const loom_verify_options_t* options,
                                 loom_verify_result_t* out_result) {
  if (!module || !out_result) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module and out_result must be non-NULL");
  }
  if (!module->context) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module has no context (vtables unavailable)");
  }

  loom_verify_state_t state;
  IREE_RETURN_IF_ERROR(
      loom_verify_state_initialize(&state, module, options, out_result));

  // Walk the module body.
  if (module->body) {
    // Module-level invariant: only symbol-defining ops (func.def,
    // func.decl, etc.) are allowed at the top level. Ops without
    // LOOM_TRAIT_SYMBOL_DEFINE belong inside function bodies.
    if (module->body->block_count > 0) {
      loom_block_t* entry = loom_region_entry_block(module->body);
      const loom_op_t* op = NULL;
      loom_block_for_each_op(entry, op) {
        const loom_op_vtable_t* vtable =
            loom_verify_lookup_vtable(&state, op->kind);
        if (vtable &&
            !iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
          iree_string_view_t op_name = loom_op_vtable_name(vtable);
          loom_diagnostic_param_t params[] = {
              loom_param_string(op_name),
          };
          loom_verify_emit_structured(&state, op, LOOM_ERR_STRUCTURE_011,
                                      params, IREE_ARRAYSIZE(params));
          iree_status_t diagnostic_status =
              loom_verify_pending_diagnostic_status(&state);
          if (!iree_status_is_ok(diagnostic_status)) {
            loom_verify_state_deinitialize(&state);
            return diagnostic_status;
          }
        }
      }
    }

    iree_status_t walk_status = loom_verify_region(&state, module->body);
    if (!iree_status_is_ok(walk_status)) {
      loom_verify_state_deinitialize(&state);
      return walk_status;
    }
  }

  loom_verify_state_deinitialize(&state);
  return iree_ok_status();
}

iree_status_t loom_verify_function(const loom_module_t* module,
                                   loom_func_like_t function,
                                   const loom_verify_options_t* options,
                                   loom_verify_result_t* out_result) {
  if (!module || !loom_func_like_isa(function) || !out_result) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, function, and out_result must be non-NULL");
  }
  if (!module->context) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "module has no context (vtables unavailable)");
  }

  loom_verify_state_t state;
  IREE_RETURN_IF_ERROR(
      loom_verify_state_initialize(&state, module, options, out_result));

  iree_status_t verify_status = loom_verify_op(&state, function.op);
  if (!iree_status_is_ok(verify_status)) {
    loom_verify_state_deinitialize(&state);
    return verify_status;
  }

  loom_verify_state_deinitialize(&state);
  return iree_ok_status();
}
