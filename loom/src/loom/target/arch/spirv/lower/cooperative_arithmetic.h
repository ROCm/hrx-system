// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_COOPERATIVE_ARITHMETIC_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_COOPERATIVE_ARITHMETIC_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native cooperative values remain indivisible during source packetization.
// The query consumes retained fragment facts rather than source carrier width.
iree_status_t loom_spirv_query_cooperative_arithmetic(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

// Selects one immutable native instruction and its exact result carrier.
iree_status_t loom_spirv_select_cooperative_arithmetic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

bool loom_spirv_cooperative_arithmetic_plan_isa(loom_low_lower_plan_t plan);

// Emits the selected instruction without consulting source analyses.
iree_status_t loom_spirv_emit_cooperative_arithmetic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_COOPERATIVE_ARITHMETIC_H_
