// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P bounded-scalar vector-load lowering.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_VECTOR_LOAD_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_VECTOR_LOAD_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries wide underaligned vector loads retained for bounded target lowering.
iree_status_t loom_aie2p_query_vector_load_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

// Returns true when |plan| is owned by bounded vector-load lowering.
bool loom_aie2p_vector_load_plan_isa(loom_low_lower_plan_t plan);

// Prepares a bounded scalar-to-packet plan from retained source-memory facts.
iree_status_t loom_aie2p_prepare_vector_load_plan(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t* out_plan);

// Marks the retained canonical address inputs needed by |plan|.
void loom_aie2p_mark_vector_load_plan_demands(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan);

// Describes a selected bounded vector-load plan for compile reports.
void loom_aie2p_describe_vector_load_plan(
    loom_low_lower_plan_report_t* out_report);

// Emits one advancing scalar-load cursor per vector packet and binds the
// resulting native carrier.
iree_status_t loom_aie2p_emit_vector_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_VECTOR_LOAD_H_
