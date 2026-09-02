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

typedef struct loom_low_lower_rule_source_memory_state_t {
  // Source op whose canonical access is retained in this state.
  const loom_op_t* source_op;
  // True after canonical source-memory planning has been attempted.
  bool plan_attempted;
  // True when |access_plan| contains a canonical source-memory plan.
  bool plan_available;
  // Caller-owned storage receiving the canonical source-memory plan.
  loom_low_source_memory_access_plan_t* access_plan;
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

// Matches one generated source-memory row against a canonical access plan.
bool loom_low_lower_rule_source_memory_matches(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    uint16_t* out_diagnostic_index);

// Matches every source-memory emit attached to |rule|.
loom_low_lower_rule_source_memory_match_t
loom_low_lower_rule_source_memory_emits_match(
    const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_rule_t* rule);

// Materializes the canonical dynamic byte offset selected by a source-memory
// plan.
iree_status_t loom_low_lower_rule_materialize_source_memory_byte_offset(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t* out_value_id);

// Materializes the complete target address selected by a source-memory plan.
iree_status_t loom_low_lower_rule_materialize_source_memory_address(
    loom_low_lower_context_t* context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    const loom_low_lower_source_memory_t* source_memory,
    const loom_low_source_memory_access_plan_t* source_memory_access,
    loom_value_id_t* out_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_RULE_SOURCE_MEMORY_H_
