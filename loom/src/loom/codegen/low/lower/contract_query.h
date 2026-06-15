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
#include "loom/codegen/low/lower/lower_rules.h"
#include "loom/ir/ir.h"
#include "loom/target/contract.h"

#ifdef __cplusplus
extern "C" {
#endif

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
} loom_low_lower_contract_query_options_t;

// Returns true and assigns the generated lower-rule row referenced by a
// composed target-contract case. Cases without source-to-low lowering payloads
// return false so callers can leave target-owned cases unhandled.
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
      *out_rule_index = contract_case->row_index;
      return *out_rule_index != LOOM_TARGET_CONTRACT_ROW_NONE;
    case LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX:
    default:
      return false;
  }
}

// Queries one generated descriptor-matrix case through the shared matrix
// source adapter and target-owned descriptor projection.
iree_status_t loom_low_lower_query_descriptor_matrix_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_descriptor_matrix_t* descriptor_matrix,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

// Queries source-to-target-low rule tables for one source op.
//
// A LEGAL result means one opt-in rule matched the selected target contract. An
// UNSUPPORTED result means at least one opt-in rule set covered the source op
// kind but all candidates rejected it. UNHANDLED means no opt-in rule set had
// an opinion. Non-OK status is reserved for malformed tables or allocation
// failures while preparing rare rejection payloads.
iree_status_t loom_low_lower_query_target_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_CONTRACT_QUERY_H_
