// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for scalar and vector value-construction source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUES_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUES_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects an AMDGPU constant materialization plan for index.constant.
iree_status_t loom_amdgpu_select_index_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU constant materialization plan for scalar.constant.
iree_status_t loom_amdgpu_select_scalar_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU constant materialization plan for vector.constant.
iree_status_t loom_amdgpu_select_vector_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU constant materialization plan.
iree_status_t loom_amdgpu_lower_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_constant_plan_t* plan);

// Selects an AMDGPU index.cast plan.
iree_status_t loom_amdgpu_select_index_cast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_index_cast_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU index.cast plan.
iree_status_t loom_amdgpu_lower_index_cast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_index_cast_plan_t* plan);

// Selects an AMDGPU full-width address-domain arithmetic plan.
iree_status_t loom_amdgpu_select_address_i64_alu_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_address_i64_alu_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU full-width address-domain arithmetic plan.
iree_status_t loom_amdgpu_lower_address_i64_alu(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_address_i64_alu_plan_t* plan);

// Selects an AMDGPU full-width address-domain comparison plan.
iree_status_t loom_amdgpu_select_index_cmp_i64_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_i64_compare_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU scalar i64 comparison plan.
iree_status_t loom_amdgpu_select_scalar_cmpi_i64_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_i64_compare_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU i64 comparison plan.
iree_status_t loom_amdgpu_lower_i64_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_i64_compare_plan_t* plan);

// Selects an AMDGPU scalar i64 ALU plan.
iree_status_t loom_amdgpu_select_scalar_i64_alu_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_i64_alu_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU scalar i64 ALU plan.
iree_status_t loom_amdgpu_lower_scalar_i64_alu(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_i64_alu_plan_t* plan);

// Selects an AMDGPU scalar conversion plan.
iree_status_t loom_amdgpu_select_scalar_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_conversion_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU scalar conversion plan.
iree_status_t loom_amdgpu_lower_scalar_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan);

// Selects an AMDGPU vector conversion plan.
iree_status_t loom_amdgpu_select_vector_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_conversion_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU vector conversion plan.
iree_status_t loom_amdgpu_lower_vector_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_conversion_plan_t* plan);

// Selects an AMDGPU vector.extract plan.
iree_status_t loom_amdgpu_select_vector_extract_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_extract_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU vector.extract plan.
iree_status_t loom_amdgpu_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan);

// Selects an AMDGPU vector 16-bit-float conversion plan.
iree_status_t loom_amdgpu_select_vector_16bit_float_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan,
    bool* out_selected);

// Returns true when an explicit vector.decode op is a dense FP8 conversion that
// AMDGPU source-low lowering can handle with native packets or software decode.
bool loom_amdgpu_vector_decode_can_lower_as_fp8_conversion(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op);

// Lowers an AMDGPU vector 16-bit-float conversion plan.
iree_status_t loom_amdgpu_lower_vector_16bit_float_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan);

// Returns the stable compile-report strategy key for a selected vector
// 16-bit/narrow-float conversion plan.
iree_string_view_t loom_amdgpu_vector_16bit_float_conversion_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan);

// Selects a plan for structural value-construction source ops.
iree_status_t loom_amdgpu_select_structural_value_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Preselects target-owned value-construction combines that must claim their
// producer before generated source-lowering contracts see it.
iree_status_t loom_amdgpu_preselect_structural_value_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Lowers a structural value-construction source op using its selected plan.
iree_status_t loom_amdgpu_lower_structural_value_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Marks the exact source values consumed by a selected structural value plan.
void loom_amdgpu_mark_structural_value_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Verifies AMDGPU low legality for vector coordinate construction source ops.
iree_status_t loom_amdgpu_low_legality_verify_vector_iota(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for vector aggregate construction source ops.
iree_status_t loom_amdgpu_low_legality_verify_vector_from_elements(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for full-width address arithmetic source ops.
iree_status_t loom_amdgpu_low_legality_verify_address_i64_alu(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for full-width address compare source ops.
iree_status_t loom_amdgpu_low_legality_verify_address_compare(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for i64 scalar compares owned by value lowering.
iree_status_t loom_amdgpu_low_legality_verify_scalar_cmpi_i64(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for scalar i64 ALU ops owned by value lowering.
iree_status_t loom_amdgpu_low_legality_verify_scalar_i64_alu(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for unsupported scalar i64 signed remainder.
iree_status_t loom_amdgpu_low_legality_verify_scalar_remsi_i64(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for scalar conversions owned by value lowering.
iree_status_t loom_amdgpu_low_legality_verify_scalar_conversion(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for vector conversions owned by value lowering.
iree_status_t loom_amdgpu_low_legality_verify_vector_conversion(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for direct FP8 vector.decode ops owned by value
// lowering.
iree_status_t loom_amdgpu_low_legality_verify_vector_decode(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUES_H_
