// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU planning and legality for vector fragment memory operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PLAN_H_

#include "loom/target/arch/amdgpu/lower/matrix_fragment_state.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the AMDGPU matrix-fragment store plan for a source op without
// requiring a low-lowering context. |target_facts| supplies the processor
// matrix capabilities without resolving target IR.
bool loom_amdgpu_analyze_vector_fragment_store_plan(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_target_facts_t* target_facts,
    loom_func_like_t source_function, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan);

// Selects an AMDGPU matrix-fragment load plan.
iree_status_t loom_amdgpu_select_vector_fragment_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU matrix-fragment store plan.
iree_status_t loom_amdgpu_select_vector_fragment_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected);

// Verifies source vector.fragment.load/store legality for AMDGPU target-low
// selection.
iree_status_t loom_amdgpu_low_legality_verify_vector_fragment_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PLAN_H_
