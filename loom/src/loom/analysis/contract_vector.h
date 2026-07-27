// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector op adapters for generic contract requests.

#ifndef LOOM_ANALYSIS_CONTRACT_VECTOR_H_
#define LOOM_ANALYSIS_CONTRACT_VECTOR_H_

#include "loom/analysis/contract.h"
#include "loom/ir/module.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maps vector dot ops to generic packed-vector contract requests.
//
// This helper preserves only target-independent structure: vector lane
// grouping, payload numeric facts, accumulator/result types, and lowering
// policy. Target feature bits and descriptor names remain target-owned
// projection outputs.
bool loom_contract_request_from_vector_dot_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_lowering_policy_t policy, loom_contract_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic);

typedef struct loom_contract_vector_mma_options_t {
  // Number of K payload elements reduced into each accumulator contribution.
  uint16_t k_group_size;

  // Fragment ownership facts required by the target projection.
  loom_contract_fragment_t fragment;

  // Requested target primitive capability class.
  loom_contract_capability_class_t capability_class;

  // Fallback and target primitive selection policy.
  loom_lowering_policy_t policy;
} loom_contract_vector_mma_options_t;

// Maps vector.mma plus fragment value facts to a generic matrix contract.
//
// The op itself carries only physical lhs/rhs/init values. This adapter queries
// fragment facts for roles, logical M/N/K shape, schemas, and auxiliary data,
// then emits the target-independent request consumed by target projection
// tables. Missing or incompatible facts are reported through |out_diagnostic|.
bool loom_contract_request_from_vector_mma_op(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* op, const loom_contract_vector_mma_options_t* options,
    loom_contract_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_CONTRACT_VECTOR_H_
