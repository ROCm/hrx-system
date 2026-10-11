// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-memory matching and address materialization for generated lowering
// rules.

#ifndef LOOM_CODEGEN_LOW_LOWER_RULE_SOURCE_MEMORY_H_
#define LOOM_CODEGEN_LOW_LOWER_RULE_SOURCE_MEMORY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/rules.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_rule_match_context_t
    loom_low_lower_rule_match_context_t;

// Selects the producer-retained complete byte offset when this recipe accepts
// it and it replaces a multi-term reconstruction. Single canonical terms keep
// their native scale/displacement opportunities. Canonical analysis and memory
// effects are independent of this materialization choice.
static inline bool loom_low_lower_rule_source_memory_reuses_view_base(
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access) {
  return iree_any_bit_set(source_memory->flags,
                          LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_REUSE_VIEW_BASE) &&
         access->dynamic_term_count > 1 &&
         access->dynamic_term_count == access->dynamic_view_base_term_count &&
         access->dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID;
}

// Returns the displacement consumed by the selected recipe. The canonical
// static offset remains unchanged for address facts and memory effects.
static inline int64_t loom_low_lower_rule_source_memory_static_byte_offset(
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* access) {
  return access->static_byte_offset -
         (loom_low_lower_rule_source_memory_reuses_view_base(source_memory,
                                                             access)
              ? access->dynamic_view_base_value_static_byte_offset
              : 0);
}

typedef struct loom_low_lower_rule_source_memory_state_t {
  // Source op whose canonical access is retained in this state.
  const loom_op_t* source_op;
  // True after canonical source-memory planning has been attempted.
  bool plan_attempted;
  // True when |access_plan| contains a canonical source-memory plan.
  bool plan_available;
  // Caller-owned storage receiving the canonical source-memory plan.
  loom_low_source_memory_access_plan_t* access_plan;
  // Shared preselection's immutable source plan, or NULL in read-only legality.
  const loom_low_source_memory_access_plan_t* retained_access;
  // Exact source-access rejection facts when planning did not succeed.
  loom_low_source_memory_access_diagnostic_t diagnostic;
} loom_low_lower_rule_source_memory_state_t;

// Initializes source-memory match state for one source op.
void loom_low_lower_rule_source_memory_state_initialize(
    const loom_op_t* source_op,
    loom_low_source_memory_access_plan_t* access_plan_storage,
    loom_low_lower_rule_source_memory_state_t* out_state);

typedef struct loom_low_lower_rule_source_memory_match_t {
  // True when at least one emit row carries a source-memory contract.
  bool has_source_memory;
  // True when every source-memory shape constraint matched, independent of
  // complete-address materialization constraints.
  bool constraints_compatible;
  // True when every source-memory and complete-address constraint matched.
  bool all_emits_match;
  // First diagnostic row selected by a rejected source-memory emit.
  uint16_t diagnostic_index;
} loom_low_lower_rule_source_memory_match_t;

// Returns the source-memory-plan alignment used by generated diagnostics.
uint32_t loom_low_lower_rule_source_memory_minimum_alignment(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op);

// Returns true when |memory_space| is present in the generated mask.
bool loom_low_lower_rule_memory_space_matches(
    loom_low_lower_memory_space_mask_t memory_space_mask,
    loom_value_fact_memory_space_t memory_space);

// Matches one generated source-memory row's shape and address layout against
// a canonical access plan, independently of address-width materialization.
bool loom_low_lower_rule_source_memory_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_lower_source_memory_shape_t* source_memory_shape,
    const loom_low_lower_source_memory_diagnostics_t* diagnostics,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_low_source_memory_access_rejection_flags_t access_rejection_bits,
    uint16_t* out_diagnostic_index);

// Matches every source-memory emit attached to |rule|.
loom_low_lower_rule_source_memory_match_t
loom_low_lower_rule_source_memory_emits_match(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule);

// Resolves one canonical dynamic address term in the descriptor-declared byte
// arithmetic carrier. Fixed-width source integers use the selected memory
// contract's numeric conversion before the term is consumed directly by a
// target descriptor.
iree_status_t loom_low_lower_rule_materialize_source_memory_dynamic_term(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    uint8_t term_ordinal, loom_value_id_t* out_value_id);

// Materializes the recipe's dynamic byte offset in its declared carrier. When
// reusing a complete view base this includes its embedded static contribution;
// static_byte_offset above returns the matching residual displacement. Wider
// source terms are projected before arithmetic; matching owns the range proof.
iree_status_t loom_low_lower_rule_materialize_source_memory_byte_offset(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t* out_value_id);

// Materializes the complete byte offset, including the static bias selected by
// the source-memory plan, in the descriptor-declared carrier.
iree_status_t
loom_low_lower_rule_materialize_source_memory_complete_byte_offset(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t* out_value_id);

// Materializes the complete target address using its planned semantic
// coordinate carrier. Descriptor register widths alone do not encode typed
// coordinates.
iree_status_t loom_low_lower_rule_materialize_source_memory_address(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_type_t coordinate_type, loom_value_id_t* out_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_RULE_SOURCE_MEMORY_H_
