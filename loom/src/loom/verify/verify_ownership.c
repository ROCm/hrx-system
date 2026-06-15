// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify_ownership.h"

#include <stdlib.h>
#include <string.h>

#include "loom/analysis/consumption.h"
#include "loom/error/error_catalog.h"
#include "loom/verify/verify_diagnostics.h"

static int loom_compare_value_ids(const void* lhs, const void* rhs) {
  loom_value_id_t lhs_id = *(const loom_value_id_t*)lhs;
  loom_value_id_t rhs_id = *(const loom_value_id_t*)rhs;
  return (lhs_id > rhs_id) - (lhs_id < rhs_id);
}

// Prepares the reusable tied-result scratch tables for an op with
// |result_count| results and |operand_count| operands. Grows the
// scratch storage from the verifier arena on demand, then clears only the
// active index-bitset words needed by the current op. Operand occurrences are
// counted per logical operand field, not across all operands. For a body op:
//   test.invoke @callee(%arg) ... -> (%arg as f32, %arg as f32)
// operand 0 has occurrence 0 in the ordinary operand list and occurrences 1/2
// in the tied-result clauses, so duplicate-tie tracking starts at 1. Signature
// ties use the function signature arg domain and the tied-result clause is the
// first operand-field spelling we target here, so they start at 0. If a future
// format emits tied-result clauses before ordinary operand spellings, or emits
// extra same-field spellings first, teach parser field spans about that role
// instead of layering more source-order heuristics into the verifier.
static iree_status_t loom_verify_tied_table_reset(
    loom_verify_tied_table_t* tied_table, iree_arena_allocator_t* arena,
    iree_host_size_t result_count, iree_host_size_t operand_count,
    uint16_t operand_occurrence_base) {
  iree_host_size_t result_word_count = loom_bitset_word_count(result_count);
  if (result_word_count > tied_table->result_index_word_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, 0, result_word_count, sizeof(uint64_t),
                              &tied_table->result_index_word_capacity,
                              (void**)&tied_table->result_index_bits));
  }
  if (result_word_count > 0) {
    memset(tied_table->result_index_bits, 0,
           result_word_count * sizeof(tied_table->result_index_bits[0]));
  }
  if (result_count > tied_table->result_field_occurrence_capacity) {
    iree_host_size_t result_occurrence_capacity =
        tied_table->result_field_occurrence_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, 0, result_count, sizeof(uint16_t), &result_occurrence_capacity,
        (void**)&tied_table->result_field_occurrences));
    iree_host_size_t first_occurrence_capacity =
        tied_table->result_field_occurrence_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, 0, result_count, sizeof(uint16_t), &first_occurrence_capacity,
        (void**)&tied_table->first_result_field_occurrences));
    IREE_ASSERT(first_occurrence_capacity == result_occurrence_capacity);
    tied_table->result_field_occurrence_capacity = result_occurrence_capacity;
  }
  if (result_count > 0) {
    memset(tied_table->result_field_occurrences, 0,
           result_count * sizeof(tied_table->result_field_occurrences[0]));
  }

  iree_host_size_t operand_word_count = loom_bitset_word_count(operand_count);
  if (operand_word_count > tied_table->operand_index_word_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, 0, operand_word_count, sizeof(uint64_t),
                              &tied_table->operand_index_word_capacity,
                              (void**)&tied_table->operand_index_bits));
  }
  if (operand_word_count > 0) {
    memset(tied_table->operand_index_bits, 0,
           operand_word_count * sizeof(tied_table->operand_index_bits[0]));
  }
  if (operand_count > tied_table->operand_field_occurrence_capacity) {
    iree_host_size_t operand_occurrence_capacity =
        tied_table->operand_field_occurrence_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, 0, operand_count, sizeof(uint16_t), &operand_occurrence_capacity,
        (void**)&tied_table->operand_field_occurrences));
    iree_host_size_t first_occurrence_capacity =
        tied_table->operand_field_occurrence_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, 0, operand_count, sizeof(uint16_t), &first_occurrence_capacity,
        (void**)&tied_table->first_operand_field_occurrences));
    IREE_ASSERT(first_occurrence_capacity == operand_occurrence_capacity);
    tied_table->operand_field_occurrence_capacity = operand_occurrence_capacity;
  }
  for (iree_host_size_t i = 0; i < operand_count; ++i) {
    tied_table->operand_field_occurrences[i] = operand_occurrence_base;
  }

  if (operand_count > tied_table->operand_value_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, 0, operand_count, sizeof(loom_value_id_t),
                              &tied_table->operand_value_capacity,
                              (void**)&tied_table->operand_value_ids));
  }
  tied_table->operand_value_count = 0;

  return iree_ok_status();
}

// Records one tied-result claim for |result_index| and returns the field
// occurrence metadata for the current claim. If this result index was claimed
// previously, |out_first_occurrence| receives the first claimant occurrence and
// the function returns true.
static bool loom_verify_tied_table_claim_result(
    loom_verify_tied_table_t* tied_table, uint16_t result_index,
    uint16_t* out_current_occurrence, uint16_t* out_first_occurrence) {
  *out_current_occurrence =
      tied_table->result_field_occurrences[result_index]++;
  if (loom_bitset_test(tied_table->result_index_bits,
                       tied_table->result_index_word_capacity, result_index)) {
    *out_first_occurrence =
        tied_table->first_result_field_occurrences[result_index];
    return true;
  }
  loom_bitset_set(tied_table->result_index_bits,
                  tied_table->result_index_word_capacity, result_index);
  tied_table->first_result_field_occurrences[result_index] =
      *out_current_occurrence;
  return false;
}

// Records one tied-result claim for |operand_index| and returns the field
// occurrence metadata for the current claim. If this operand index was claimed
// previously, |out_first_occurrence| receives the first claimant occurrence and
// the function returns true.
static bool loom_verify_tied_table_claim_operand(
    loom_verify_tied_table_t* tied_table, uint16_t operand_index,
    uint16_t* out_current_occurrence, uint16_t* out_first_occurrence) {
  *out_current_occurrence =
      tied_table->operand_field_occurrences[operand_index]++;
  if (loom_bitset_test(tied_table->operand_index_bits,
                       tied_table->operand_index_word_capacity,
                       operand_index)) {
    *out_first_occurrence =
        tied_table->first_operand_field_occurrences[operand_index];
    return true;
  }
  loom_bitset_set(tied_table->operand_index_bits,
                  tied_table->operand_index_word_capacity, operand_index);
  tied_table->first_operand_field_occurrences[operand_index] =
      *out_current_occurrence;
  return false;
}

// Returns true if |value_id| appears in more than one operand slot in the
// current op's sorted operand-value scratch array.
static bool loom_verify_tied_table_has_duplicate_operand_value(
    const loom_verify_tied_table_t* tied_table, loom_value_id_t value_id) {
  iree_host_size_t low = 0;
  iree_host_size_t high = tied_table->operand_value_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    if (tied_table->operand_value_ids[mid] < value_id) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low < tied_table->operand_value_count &&
         tied_table->operand_value_ids[low] == value_id &&
         (low + 1 < tied_table->operand_value_count &&
          tied_table->operand_value_ids[low + 1] == value_id);
}

static void loom_verify_emit_consumed_value_use(loom_verify_state_t* state,
                                                const loom_op_t* use_op,
                                                uint16_t operand_index,
                                                loom_value_id_t value_id,
                                                const loom_op_t* consuming_op) {
  const loom_op_vtable_t* consuming_vtable =
      consuming_op ? loom_verify_lookup_vtable(state, consuming_op->kind)
                   : NULL;
  iree_string_view_t consuming_op_name =
      consuming_vtable ? loom_op_vtable_name(consuming_vtable)
                       : IREE_SV("<unknown op>");
  iree_string_view_t value_name = loom_verify_value_name(state, value_id);
  loom_diagnostic_field_ref_t operand_ref =
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, operand_index);
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_string(value_name), operand_ref),
      loom_param_string(consuming_op_name),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("consumed here"),
      .op = consuming_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = use_op,
      .error = LOOM_ERR_DOMINANCE_002,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  loom_verify_emit_diagnostic(state, &emission);
}

void loom_verify_operand_dominance(loom_verify_state_t* state,
                                   const loom_op_t* op,
                                   const loom_op_vtable_t* vtable) {
  if (loom_verify_func_args_are_operands(vtable) &&
      iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    return;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_value_id_t value_id = operands[i];
    if (value_id == LOOM_VALUE_ID_INVALID ||
        value_id >= state->module->values.count) {
      loom_diagnostic_param_t params[] = {
          loom_param_u32(value_id),
          loom_param_u32((uint32_t)state->module->values.count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_DOMINANCE_003, params,
                                  IREE_ARRAYSIZE(params));
      continue;
    }
    if (!loom_bitset_test(state->defined_bits, state->defined_bits_length,
                          value_id)) {
      iree_string_view_t value_name = loom_verify_value_name(state, value_id);
      loom_diagnostic_field_ref_t operand_ref =
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i);
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(loom_param_string(value_name), operand_ref),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_DOMINANCE_001, params,
                                  IREE_ARRAYSIZE(params));
    }
    if (loom_bitset_test(state->consumed_bits, state->defined_bits_length,
                         value_id)) {
      loom_verify_emit_consumed_value_use(state, op, i, value_id,
                                          state->consuming_ops[value_id]);
    }
  }
}

static bool loom_verify_op_observes_poison(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t* out_boundary_kind) {
  loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (iree_any_bit_set(traits, LOOM_TRAIT_POISON_BOUNDARY)) {
    *out_boundary_kind = IREE_SV("observation boundary");
    return true;
  }
  const loom_trait_flags_t boundary_traits =
      LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_WRITES_MEMORY |
      LOOM_TRAIT_NON_DETERMINISTIC | LOOM_TRAIT_UNKNOWN_EFFECTS;
  if (iree_any_bit_set(traits, boundary_traits)) {
    *out_boundary_kind = IREE_SV("effectful operation");
    return true;
  }
  *out_boundary_kind = IREE_SV("");
  return false;
}

void loom_verify_poison_boundaries(loom_verify_state_t* state,
                                   const loom_op_t* op,
                                   const loom_op_vtable_t* vtable) {
  iree_string_view_t boundary_kind = IREE_SV("");
  if (!loom_verify_op_observes_poison(state->module, op, &boundary_kind)) {
    return;
  }

  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (loom_verify_at_error_limit(state)) return;
    loom_value_id_t value_id = operands[i];
    if (value_id == LOOM_VALUE_ID_INVALID) continue;
    if (value_id >= state->module->values.count) continue;
    if (!loom_bitset_test(state->defined_bits, state->defined_bits_length,
                          value_id)) {
      continue;
    }
    if (!loom_value_is_poison(state->module, value_id)) continue;

    const loom_value_t* value = loom_module_value(state->module, value_id);
    const loom_op_t* poison_op = loom_value_def_op(value);
    iree_string_view_t value_name = loom_verify_value_name(state, value_id);
    iree_string_view_t op_name = loom_op_vtable_name(vtable);
    loom_diagnostic_field_ref_t operand_ref =
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i);
    loom_diagnostic_param_t params[] = {
        loom_param_with_field_ref(loom_param_string(value_name), operand_ref),
        loom_param_string(boundary_kind),
        loom_param_string(op_name),
    };
    loom_diagnostic_related_op_t related_ops[] = {{
        .label = IREE_SV("poison produced here"),
        .op = poison_op,
    }};
    loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_TYPE_012,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
        .related_ops = related_ops,
        .related_op_count = IREE_ARRAYSIZE(related_ops),
    };
    loom_verify_emit_diagnostic(state, &emission);
  }
}

static void loom_verify_emit_duplicate_tied_result(loom_verify_state_t* state,
                                                   const loom_op_t* op,
                                                   iree_string_view_t op_name,
                                                   uint16_t result_index,
                                                   uint16_t current_occurrence,
                                                   uint16_t first_occurrence) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_u32(result_index),
          loom_diagnostic_field_ref_with_occurrence(
              LOOM_DIAGNOSTIC_FIELD_RESULT, result_index, current_occurrence)),
      loom_param_string(op_name),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("previously tied here"),
      .op = op,
      .field_ref = loom_diagnostic_field_ref_with_occurrence(
          LOOM_DIAGNOSTIC_FIELD_RESULT, result_index, first_occurrence),
  }};
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_DOMINANCE_006,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  loom_verify_emit_diagnostic(state, &emission);
}

static void loom_verify_emit_duplicate_tied_operand(loom_verify_state_t* state,
                                                    const loom_op_t* op,
                                                    iree_string_view_t op_name,
                                                    uint16_t operand_index,
                                                    uint16_t current_occurrence,
                                                    uint16_t first_occurrence) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_u32(operand_index),
                                loom_diagnostic_field_ref_with_occurrence(
                                    LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                    operand_index, current_occurrence)),
      loom_param_string(op_name),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("previously tied here"),
      .op = op,
      .field_ref = loom_diagnostic_field_ref_with_occurrence(
          LOOM_DIAGNOSTIC_FIELD_OPERAND, operand_index, first_occurrence),
  }};
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_DOMINANCE_007,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  loom_verify_emit_diagnostic(state, &emission);
}

iree_status_t loom_verify_tied_results(loom_verify_state_t* state,
                                       const loom_op_t* op,
                                       const loom_op_vtable_t* vtable) {
  if (op->tied_result_count == 0) return iree_ok_status();

  const bool has_signature_ties = loom_verify_has_func_signature_scope(vtable);
  uint16_t tied_operand_count = 0;
  const loom_value_id_t* tied_operands = NULL;
  if (has_signature_ties) {
    tied_operands =
        loom_verify_func_signature_arg_ids(op, vtable, &tied_operand_count);
  } else {
    tied_operand_count = op->operand_count;
    tied_operands = loom_op_const_operands(op);
  }

  IREE_RETURN_IF_ERROR(loom_verify_tied_table_reset(
      &state->tied_table, &state->arena, op->result_count, tied_operand_count,
      has_signature_ties ? 0 : 1));

  iree_string_view_t op_name = loom_op_vtable_name(vtable);
  const loom_tied_result_t* tied = loom_op_tied_results(op);

  // Copy and sort this op's valid operand values so ties to repeated operand
  // values can be diagnosed as ambiguous.
  for (uint16_t i = 0; i < tied_operand_count; ++i) {
    loom_value_id_t value_id = tied_operands[i];
    if (value_id == LOOM_VALUE_ID_INVALID ||
        value_id >= state->module->values.count) {
      continue;
    }
    state->tied_table
        .operand_value_ids[state->tied_table.operand_value_count++] = value_id;
  }
  if (state->tied_table.operand_value_count > 1) {
    qsort(state->tied_table.operand_value_ids,
          state->tied_table.operand_value_count, sizeof(loom_value_id_t),
          loom_compare_value_ids);
  }

  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (tied[i].result_index >= op->result_count) {
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(
              loom_param_u32(tied[i].result_index),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT,
                                        tied[i].result_index)),
          loom_param_string(op_name),
          loom_param_u32(op->result_count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_DOMINANCE_004, params,
                                  IREE_ARRAYSIZE(params));
    } else {
      uint16_t current_occurrence = 0;
      uint16_t first_occurrence = 0;
      if (loom_verify_tied_table_claim_result(
              &state->tied_table, tied[i].result_index, &current_occurrence,
              &first_occurrence)) {
        loom_verify_emit_duplicate_tied_result(
            state, op, op_name, tied[i].result_index, current_occurrence,
            first_occurrence);
      }
    }
    if (tied[i].operand_index >= tied_operand_count || !tied_operands) {
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(
              loom_param_u32(tied[i].operand_index),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                        tied[i].operand_index)),
          loom_param_string(op_name),
          loom_param_u32(tied_operand_count),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_DOMINANCE_005, params,
                                  IREE_ARRAYSIZE(params));
      continue;
    }
    uint16_t current_operand_occurrence = 0;
    uint16_t first_operand_occurrence = 0;
    if (loom_verify_tied_table_claim_operand(
            &state->tied_table, tied[i].operand_index,
            &current_operand_occurrence, &first_operand_occurrence)) {
      loom_verify_emit_duplicate_tied_operand(
          state, op, op_name, tied[i].operand_index, current_operand_occurrence,
          first_operand_occurrence);
    }

    loom_value_id_t consumed_id = tied_operands[tied[i].operand_index];
    // DOMINANCE/008 stays single-site for now because the duplicate-value check
    // runs over a sorted operand-value multiset, which intentionally discards
    // the source slot of the earlier equal-value operand.
    if (consumed_id != LOOM_VALUE_ID_INVALID &&
        consumed_id < state->module->values.count &&
        loom_verify_tied_table_has_duplicate_operand_value(&state->tied_table,
                                                           consumed_id)) {
      loom_diagnostic_field_ref_t operand_ref = loom_diagnostic_field_ref(
          LOOM_DIAGNOSTIC_FIELD_OPERAND, tied[i].operand_index);
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(loom_param_u32(tied[i].operand_index),
                                    operand_ref),
          loom_param_string(op_name),
          loom_param_with_field_ref(
              loom_param_string(loom_verify_value_name(state, consumed_id)),
              operand_ref),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_DOMINANCE_008, params,
                                  IREE_ARRAYSIZE(params));
    }

    // Ties on regular body ops consume the operand's storage. Func-like symbol
    // signatures are caller-side ownership contracts, so their entry args are
    // validated as tie targets but not marked consumed at function entry.
    const loom_region_t* parent_region =
        op->parent_block ? op->parent_block->parent_region : NULL;
    if (!has_signature_ties && parent_region &&
        iree_any_bit_set(parent_region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
      if (!state->current_consumption_query) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "verifier CFG tied-result checks require a consumption query");
      }
      loom_consumption_use_t use = {0};
      bool found_use = false;
      IREE_RETURN_IF_ERROR(loom_consumption_find_use_after(
          state->current_consumption_query, op, consumed_id, &use, &found_use));
      if (found_use) {
        loom_verify_emit_consumed_value_use(state, use.op, use.operand_index,
                                            consumed_id, op);
      }
    } else if (!has_signature_ties) {
      loom_verify_consume_value(state, consumed_id, op);
    }
  }

  return iree_ok_status();
}
