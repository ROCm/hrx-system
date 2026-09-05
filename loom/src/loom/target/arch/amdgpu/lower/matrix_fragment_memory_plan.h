// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU planning and legality for vector fragment memory operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_PLAN_H_

#include "loom/codegen/low/representation_plan.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_state.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Enumerates exact target result representations with costed publication
// choices for an accumulator RESULT store. The caller validates the operation
// and supplies storage for the complete target representation catalog.
iree_status_t loom_amdgpu_query_accumulator_fragment_store_representations(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_representation_candidate_t* out_candidates,
    iree_host_size_t* out_candidate_count);

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
