// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU vector conversion lowering for F16, BF16, FP8, and BF8 formats.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_ENCODING_VECTOR_CONVERSION_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_ENCODING_VECTOR_CONVERSION_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects an AMDGPU vector narrow-float conversion plan.
iree_status_t loom_amdgpu_select_vector_16bit_float_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan,
    bool* out_selected);

// Lowers an AMDGPU vector narrow-float conversion plan.
iree_status_t loom_amdgpu_lower_vector_16bit_float_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan);

// Returns the stable compile-report strategy key for a selected narrow-float
// vector conversion plan.
iree_string_view_t loom_amdgpu_vector_16bit_float_conversion_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan);

// Returns true when an explicit vector.decode is a dense FP8 conversion that
// AMDGPU source-low lowering can handle.
bool loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op);

// Resolves the single scale auxiliary accepted by direct vector.decode
// lowering and verifies the physical type implied by |scale_format|.
bool loom_amdgpu_vector_decode_scale_source(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_value_fact_numeric_format_flags_t scale_format,
    loom_value_id_t* out_scale_source);

// Looks up and materializes a selected vector conversion scale as one VGPR.
iree_status_t loom_amdgpu_lookup_vector_scale_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t* out_low_scale);

// Verifies AMDGPU low legality for direct FP8 vector.decode operations.
iree_status_t loom_amdgpu_low_legality_verify_vector_decode(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for direct FP8 vector.encode operations.
iree_status_t loom_amdgpu_low_legality_verify_vector_encode(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_ENCODING_VECTOR_CONVERSION_H_
