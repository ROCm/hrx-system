// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Joint AMDGPU matrix instruction-layout and result-publication planning.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_PUBLICATION_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_PUBLICATION_PLAN_H_

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the lowest-cost exact physical layout in |contract_layout|'s
// canonical layout family for the result publications in |source_function|.
// Instruction queries and fragment stores use this shared function so operand
// ordering and result coordinates cannot diverge.
const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_select_matrix_fragment_publication_layout(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_target_facts_t* target_facts,
    loom_func_like_t source_function,
    const loom_amdgpu_matrix_fragment_layout_t* contract_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_PUBLICATION_PLAN_H_
