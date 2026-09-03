// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P native matrix contraction and accumulator movement lowering.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_MATRIX_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_MATRIX_H_

#include "loom/analysis/contract_vector.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Supplies AIE2P options for the shared vector.mma contract adapter.
iree_status_t loom_aie2p_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options);

// Selects the native AIE2P matrix packet for a generic matrix contract.
iree_status_t loom_aie2p_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* request,
    loom_target_contract_query_result_t* out_result);

// Returns true when |plan| is owned by native matrix lowering.
bool loom_aie2p_matrix_plan_isa(loom_low_lower_plan_t plan);

// Selects a native matrix or accumulator-movement plan.
iree_status_t loom_aie2p_select_matrix_plan(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan);

// Marks source values required to emit a selected matrix plan.
void loom_aie2p_mark_matrix_plan_demands(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan);

// Describes a selected matrix plan for compile reports.
void loom_aie2p_describe_matrix_plan(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     loom_low_lower_plan_report_t* out_report);

// Emits a selected matrix plan into target Low.
iree_status_t loom_aie2p_emit_matrix_plan(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_MATRIX_H_
