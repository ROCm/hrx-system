// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V source-to-low lowering for Workgroup storage.

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves the allocation-wide register class for a source Workgroup view.
// |out_is_workgroup| distinguishes ordinary views from Workgroup views that
// cannot be represented with one coherent scalar carrier.
iree_status_t loom_spirv_resolve_workgroup_view_reg_class(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_is_workgroup, uint16_t* out_reg_class_id);

// Resolves the same allocation-wide Workgroup view carrier during read-only
// target-contract queries.
iree_status_t loom_spirv_resolve_workgroup_contract_view_reg_class(
    const loom_target_contract_query_environment_t* environment,
    loom_value_id_t source_value_id, bool* out_is_workgroup,
    uint16_t* out_reg_class_id);

// Selects Workgroup storage plans for buffer allocation and typed views.
iree_status_t loom_spirv_select_workgroup_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Marks the exact source SSA storage consumed by a selected Workgroup plan.
void loom_spirv_mark_workgroup_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Emits a previously selected Workgroup storage plan.
iree_status_t loom_spirv_lower_workgroup_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_WORKGROUP_H_
