// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P source function-storage lowering.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_STORAGE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_STORAGE_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |plan| is owned by function-storage lowering.
bool loom_aie2p_storage_plan_isa(loom_low_lower_plan_t plan);

// Selects a function-storage plan for one source operation.
iree_status_t loom_aie2p_select_storage_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan);

// Marks source values required to emit a selected function-storage plan.
void loom_aie2p_mark_storage_plan_demands(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan);

// Describes a selected function-storage plan for compile reports.
void loom_aie2p_describe_storage_plan(loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_low_lower_plan_t plan,
                                      loom_low_lower_plan_report_t* out_report);

// Emits a selected function-storage plan into target Low.
iree_status_t loom_aie2p_emit_storage_plan(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_STORAGE_H_
