// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P static vector transpose lowering.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_TRANSPOSE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_TRANSPOSE_H_

#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries the complete family of verified static transposes with AIE2P
// carriers.
iree_status_t loom_aie2p_query_transpose_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

// Selects one retained native or bounded permutation-network plan. Other
// operations leave |out_plan| empty.
iree_status_t loom_aie2p_select_transpose_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Publishes the actual source carrier for an identity transpose after its
// producer has finalized its binding.
iree_status_t loom_aie2p_finalize_transpose_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Retains the complete source carrier consumed by the selected plan.
void loom_aie2p_mark_transpose_plan_demands(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t plan);

// Describes the selected alias, VSHUFFLE, composed, or network realization.
void loom_aie2p_describe_transpose_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report);

// Emits the selected static transpose realization.
iree_status_t loom_aie2p_emit_transpose_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_TRANSPOSE_H_
