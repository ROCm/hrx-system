// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P immutable gather lowering.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_GATHER_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_GATHER_H_

#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects an immutable gather plan for one source operation.
iree_status_t loom_aie2p_select_gather_plan(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan);

// Marks source values required to emit a selected immutable gather plan.
void loom_aie2p_mark_gather_plan_demands(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan);

// Describes a selected immutable gather plan for compile reports.
void loom_aie2p_describe_gather_plan(loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     loom_low_lower_plan_report_t* out_report);

// Emits a selected immutable gather plan into target Low.
iree_status_t loom_aie2p_emit_gather_plan(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan);

// Materializes module-owned payloads reserved by immutable gather plans.
iree_status_t loom_aie2p_finalize_gather_module(
    loom_module_t* module, loom_low_lower_module_state_t* module_state,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_GATHER_H_
