// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for kernel preamble query operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_PREAMBLE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_PREAMBLE_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a plan for kernel preamble source ops.
iree_status_t loom_amdgpu_select_preamble_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Emits structural preamble live-ins for the current low function.
iree_status_t loom_amdgpu_emit_preamble(void* user_data,
                                        loom_low_lower_context_t* context);

// Emits entry-block setup packets that depend on structural ABI imports.
iree_status_t loom_amdgpu_emit_entry_setup(void* user_data,
                                           loom_low_lower_context_t* context);

// Lowers a kernel preamble source op using its pre-bound live-in value.
iree_status_t loom_amdgpu_lower_preamble_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op);

// Looks up the current dispatch packet pointer live-in.
iree_status_t loom_amdgpu_lookup_current_dispatch_ptr(
    loom_low_lower_context_t* context, loom_value_id_t* out_low_value_id);

// Looks up the current dispatch packet ID live-in.
iree_status_t loom_amdgpu_lookup_current_dispatch_id(
    loom_low_lower_context_t* context, loom_value_id_t* out_low_value_id);

// Looks up the current global workgroup coordinate for |dimension|.
iree_status_t loom_amdgpu_lookup_current_workgroup_id(
    loom_low_lower_context_t* context, loom_kernel_dimension_t dimension,
    loom_value_id_t* out_low_value_id);

// Looks up or extracts the current workitem id for |dimension|.
iree_status_t loom_amdgpu_lookup_current_workitem_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension, loom_value_id_t* out_low_value_id);

// Emits the dynamic workgroup count for |dimension| from the dispatch packet.
iree_status_t loom_amdgpu_emit_current_workgroup_count(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension, loom_type_t result_type,
    loom_value_id_t* out_low_value_id);

// Emits the current workgroup id flattened with dynamic launch dimensions.
iree_status_t loom_amdgpu_emit_current_workgroup_linear_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_linear_id);

// Emits the current workitem id flattened within the static workgroup.
iree_status_t loom_amdgpu_emit_current_workitem_linear_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t result_type, loom_value_id_t* out_linear_id);

// Verifies AMDGPU low legality for launch preamble query source ops.
iree_status_t loom_amdgpu_low_legality_verify_kernel_preamble(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_PREAMBLE_H_
