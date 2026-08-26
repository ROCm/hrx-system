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
#include "loom/ops/vector/fragment.h"
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

typedef enum loom_contract_vector_mma_fragment_projection_e {
  // Use the target-supplied fragment and K-group facts verbatim.
  LOOM_CONTRACT_VECTOR_MMA_FRAGMENT_PROJECTION_EXPLICIT = 0,
  // Derive one packed-vector instruction from physical operand payloads.
  LOOM_CONTRACT_VECTOR_MMA_FRAGMENT_PROJECTION_PACKED_VECTOR = 1,
} loom_contract_vector_mma_fragment_projection_t;

typedef struct loom_contract_vector_mma_options_t {
  // How target-independent physical fragment facts are constructed.
  loom_contract_vector_mma_fragment_projection_t fragment_projection;

  // Number of K payload elements reduced into each accumulator contribution.
  // Ignored when fragment_projection derives this value from packed vectors.
  uint16_t k_group_size;

  // Fragment ownership facts required by the target projection.
  // Ignored when fragment_projection derives these facts from packed vectors.
  loom_contract_fragment_t fragment;

  // Requested target primitive capability class.
  loom_contract_capability_class_t capability_class;

  // Fallback and target primitive selection policy.
  loom_lowering_policy_t policy;
} loom_contract_vector_mma_options_t;

// Projects one matrix fragment into its target-independent operand contract.
// The caller supplies the already-indexed fragment fact and the semantic role
// of the value at its use site. This is the canonical numeric interpretation
// shared by whole-matrix contract construction and target value typing.
bool loom_contract_vector_operand_from_fragment(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_vector_fragment_fact_t fragment, loom_contract_operand_role_t role,
    loom_contract_operand_t* out_operand,
    loom_contract_rejection_bits_t* out_rejection_bits);

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
