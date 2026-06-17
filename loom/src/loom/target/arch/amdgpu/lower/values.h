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

// Selects an AMDGPU full-width offset addition plan.
iree_status_t loom_amdgpu_select_offset_add_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_offset_add_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU full-width offset addition plan.
iree_status_t loom_amdgpu_lower_offset_add(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_offset_add_plan_t* plan);

// Selects an AMDGPU full-width offset comparison plan.
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

// Selects an AMDGPU vector bf16 conversion plan.
iree_status_t loom_amdgpu_select_vector_bf16_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_bf16_conversion_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU vector bf16 conversion plan.
iree_status_t loom_amdgpu_lower_vector_bf16_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bf16_conversion_plan_t* plan);

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

// Verifies AMDGPU low legality for full-width offset arithmetic source ops.
iree_status_t loom_amdgpu_low_legality_verify_offset_add(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for full-width offset compare source ops.
iree_status_t loom_amdgpu_low_legality_verify_offset_compare(
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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUES_H_
