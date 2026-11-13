// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P construction of vector carriers from scalar elements.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_VECTOR_CONSTRUCT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_VECTOR_CONSTRUCT_H_

#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries constructors implemented by native scalar pushes and packet packing.
iree_status_t loom_aie2p_query_vector_construct_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

// Selects one compact plan for the verified logical element segment and result
// carrier. Other operations leave |out_plan| empty.
iree_status_t loom_aie2p_select_vector_construct_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Describes the native push realization for compiler reporting.
void loom_aie2p_describe_vector_construct_plan(
    loom_low_lower_plan_report_t* out_report);

// Emits scalar pushes in logical lane order and binds the complete carrier.
iree_status_t loom_aie2p_emit_vector_construct_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_VECTOR_CONSTRUCT_H_
