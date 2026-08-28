// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dense AMDGPU FP8/BF8 vector conversion route orchestration.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP8_VECTOR_CONVERSION_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP8_VECTOR_CONVERSION_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the exact FP8/BF8 decode action producing each physical result
// register and canonicalizes an identity F32 scale to an unscaled plan.
void loom_amdgpu_select_vector_fp8_decode_plan(
    loom_low_lower_context_t* context,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* plan);

// Returns the stable compile-report strategy key for an FP8/BF8 vector
// conversion plan.
iree_string_view_t loom_amdgpu_vector_fp8_conversion_plan_key(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan);

// Lowers one selected FP8/BF8 vector conversion through native or software
// packet recipes.
iree_status_t loom_amdgpu_lower_vector_fp8_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP8_VECTOR_CONVERSION_H_
