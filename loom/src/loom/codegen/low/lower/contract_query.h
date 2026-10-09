// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Read-only target contract queries over source-to-target-low rule tables.
//
// This layer owns the shared contract matching path used before mutation. It
// interprets rule-table guards against a selected target bundle and descriptor
// set, returning compact target contract query results without emitting
// diagnostics or lowering IR. Callers provide bridges for source-value register
// metadata and optional materialization predicates.

#ifndef LOOM_CODEGEN_LOW_LOWER_CONTRACT_QUERY_H_
#define LOOM_CODEGEN_LOW_LOWER_CONTRACT_QUERY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/rule_match.h"
#include "loom/ir/ir.h"
#include "loom/target/contract.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t (*loom_low_lower_contract_accept_rule_fn_t)(
    void* user_data, const loom_low_lower_rule_selection_t* selection,
    bool* out_accepted);

typedef struct loom_low_lower_contract_accept_rule_callback_t {
  // Optional callback deciding whether a matched source graph can be claimed.
  loom_low_lower_contract_accept_rule_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_low_lower_contract_accept_rule_callback_t;

typedef struct loom_low_lower_contract_query_options_t {
  // Optional composed contract index used for direct op-to-case lookup.
  const loom_target_contract_index_t* contract_index;
  // Rule sets to query in priority order.
  loom_low_lower_rule_set_list_t rule_sets;
  // Source-value to target-low register metadata mapper.
  loom_low_lower_rule_match_map_value_callback_t map_value;
  // Optional source value materializer predicate bridge.
  loom_low_lower_rule_match_can_materialize_value_callback_t can_materialize;
  // Optional rule-local descriptor-ref resolver.
  loom_low_lower_rule_match_descriptor_ref_callback_t descriptor_ref;
  // Optional target-owned descriptor-matrix projection.
  loom_low_lower_descriptor_matrix_t descriptor_matrix;
  // Optional source-graph ownership filter applied after a rule matches.
  loom_low_lower_contract_accept_rule_callback_t accept_rule;
} loom_low_lower_contract_query_options_t;

// Returns true and assigns the generated lower-rule row referenced by a
// composed target-contract case. The row may be an emission program or a
// contract-only guard recipe. Target-owned systems without lower-rule rows,
// including explicit unsupported-family markers, return false.
static inline bool loom_low_lower_contract_case_lower_rule_index(
    const loom_target_contract_index_t* index,
    const loom_target_contract_case_t* contract_case,
    uint16_t* out_rule_index) {
  const loom_target_contract_binding_t* binding =
      &index->bindings[contract_case->binding_index];
  switch (contract_case->system) {
    case LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE: {
      const loom_target_contract_descriptor_rule_t* descriptor_rule =
          &binding->fragment->descriptor_rules[contract_case->row_index];
      *out_rule_index = descriptor_rule->rule_index;
      return true;
    }
    case LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ALIAS:
    case LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ELIDE:
    case LOOM_TARGET_CONTRACT_SYSTEM_RECIPE_RULE:
      *out_rule_index = contract_case->row_index;
      return *out_rule_index != LOOM_TARGET_CONTRACT_ROW_NONE;
    case LOOM_TARGET_CONTRACT_SYSTEM_UNSUPPORTED:
    case LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX:
    default:
      return false;
  }
}

// Queries one generated descriptor-matrix case through the shared matrix
// source adapter and target-owned descriptor projection. When provided,
// |out_request| receives the adapted semantic request submitted to the target
// query so planning can retain a legal selection without repeating source
// analysis.
iree_status_t loom_low_lower_query_descriptor_matrix_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_descriptor_matrix_t* descriptor_matrix,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, loom_contract_request_t* out_request,
    loom_target_contract_query_result_t* out_result);

// Queries source-to-target-low rule tables for one source op.
//
// A LEGAL result means one opt-in rule matched the selected target contract. An
// UNSUPPORTED result means an explicit unsupported-family marker was present
// or at least one opt-in rule set covered the source op kind but all candidates
// rejected it. UNHANDLED means no opt-in rule set had an opinion. Non-OK status
// is reserved for malformed tables or allocation failures while preparing rare
// rejection payloads.
iree_status_t loom_low_lower_query_target_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

// Queries one source op and retains the matched source graph when a generated
// lower rule is selected. Descriptor-matrix and target-owned legal results
// leave |out_selection| empty.
iree_status_t loom_low_lower_query_target_contract_with_selection(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op, loom_target_contract_query_result_t* out_result,
    loom_low_lower_rule_selection_t* out_selection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_CONTRACT_QUERY_H_
