// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P patterned even/odd vector permutation lowering.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_INTERLEAVE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_INTERLEAVE_H_

#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries interleave and deinterleave forms implemented by native AIE2P packet
// plans.
iree_status_t loom_aie2p_query_interleave_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

// Selects a native packet plan for the complete admitted interleave and
// deinterleave carrier family. Other operations leave |out_plan| empty.
iree_status_t loom_aie2p_select_interleave_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Retains every source carrier consumed by the selected plan.
void loom_aie2p_mark_interleave_plan_demands(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t plan);

// Describes the selected VSHUFFLE or block-routing realization.
void loom_aie2p_describe_interleave_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report);

// Emits the selected native interleave or deinterleave realization.
iree_status_t loom_aie2p_emit_interleave_plan(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_INTERLEAVE_H_
