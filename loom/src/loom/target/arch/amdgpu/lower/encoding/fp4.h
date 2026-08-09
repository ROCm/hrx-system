// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU packed FP4 payload lowering.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP4_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP4_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when vector.decode has an exact packed E2M1 schema: unscaled or
// E4M3FN-scaled F16, or group-32 E8M0-scaled BF16. The active descriptor set
// must provide a native or portable packet recipe for the matched schema.
bool loom_amdgpu_vector_decode_can_lower_as_fp4_conversion(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op);

// Selects a native or portable packed E2M1 vector.decode plan.
iree_status_t loom_amdgpu_select_fp4_decode_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan,
    bool* out_selected);

// Lowers one selected packed E2M1 vector.decode plan.
iree_status_t loom_amdgpu_lower_vector_fp4_decode(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan);

// Returns the stable compile-report key for a selected packed FP4 plan.
iree_string_view_t loom_amdgpu_fp4_decode_plan_key(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP4_H_
