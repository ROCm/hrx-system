// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P source load-FIFO state import lowering.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_FIFO_STATE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_FIFO_STATE_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reports the source load-FIFO entry-state query as natively legal.
iree_status_t loom_aie2p_query_fifo_state_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

// Returns true when |plan| imports the architectural load-FIFO states.
bool loom_aie2p_fifo_state_plan_isa(loom_low_lower_plan_t plan);

// Retains one canonical source query and plans both results as load-FIFO
// registers. Other source operations leave |out_plan| empty.
iree_status_t loom_aie2p_select_fifo_state_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Emits both canonical FIFO states in the Low function preamble.
iree_status_t loom_aie2p_emit_fifo_state_preamble(
    void* user_data, loom_low_lower_context_t* context);

// Describes the entry-state import in compiler reports.
void loom_aie2p_describe_fifo_state_plan(
    loom_low_lower_plan_report_t* out_report);

// Binds a source query to the canonical preamble values.
iree_status_t loom_aie2p_emit_fifo_state_plan(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_FIFO_STATE_H_
