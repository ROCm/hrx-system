// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for arithmetic source operations that need target-owned
// planning beyond generated descriptor contracts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_ARITHMETIC_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_ARITHMETIC_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a mixed-source f32-result FMA plan for scalar.fmaf operands widened
// from f16 sources.
iree_status_t loom_amdgpu_select_scalar_fmaf_mix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fma_mix_plan_t* out_plan, bool* out_selected);

// Classifies a scalar f32 value as a legal mixed-FMA packet source.
bool loom_amdgpu_select_fma_mix_source(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_id_t* out_source,
    loom_amdgpu_fma_mix_source_kind_t* out_source_kind,
    uint32_t* out_source_register_offset);

// Selects a mixed-source FMA/MAD descriptor that rounds into one f16 result
// lane.
bool loom_amdgpu_select_fma_mix_half_result_descriptor(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fma_mix_source_kind_t* source_kinds, bool high_result,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref);

// Selects a packed f16 FMA plan for vector.fmaf over even lane-pair vectors.
iree_status_t loom_amdgpu_select_vector_packed_fmaf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_packed_ternary_plan_t* out_plan, bool* out_selected);

// Selects a packed i16 MAD plan for vector.fmai over even lane-pair vectors.
iree_status_t loom_amdgpu_select_vector_packed_fmai_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_packed_ternary_plan_t* out_plan, bool* out_selected);

// Selects a mixed-source f32-result multiply plan for scalar.mulf operands
// widened from f16 sources.
iree_status_t loom_amdgpu_select_scalar_mulf_mix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_mulf_mix_plan_t* out_plan, bool* out_selected);

// Selects a mixed-source f32-result multiply plan for vector.mulf by a splatted
// scalar widened from an f16 source.
iree_status_t loom_amdgpu_select_vector_mulf_mix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_mulf_mix_plan_t* out_plan, bool* out_selected);

// Lowers scalar.fmaf to one mixed-source AMDGPU FMA/MAD descriptor packet.
iree_status_t loom_amdgpu_lower_scalar_fmaf_mix(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan);

// Emits one mixed-source AMDGPU FMA/MAD descriptor packet.
iree_status_t loom_amdgpu_emit_fma_mix_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan, loom_type_t result_type,
    loom_value_id_t* out_result);

// Emits one mixed-source AMDGPU FMA/MAD descriptor packet whose partial result
// is tied to an existing accumulator register.
iree_status_t loom_amdgpu_emit_tied_fma_mix_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan, loom_value_id_t accumulator,
    loom_type_t result_type, loom_value_id_t* out_result);

// Lowers a packed vector ternary op to one AMDGPU packet per lane pair.
iree_status_t loom_amdgpu_lower_vector_packed_ternary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_packed_ternary_plan_t* plan);

// Lowers scalar or vector mulf to mixed-source AMDGPU FMA/MAD packets with a
// zero addend.
iree_status_t loom_amdgpu_lower_mulf_mix(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_mulf_mix_plan_t* plan);

// Emits a source-to-low diagnostic explaining FMAAK/FMAMK literal selection.
iree_status_t loom_amdgpu_emit_fmaf_literal_operand_form_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op);

// Marks the exact source values consumed by a selected mixed FMA plan.
void loom_amdgpu_mark_fma_mix_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan);

// Marks the exact source values consumed by a selected packed ternary plan.
void loom_amdgpu_mark_packed_ternary_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_packed_ternary_plan_t* plan);

// Marks the exact source values consumed by a selected mixed multiply plan.
void loom_amdgpu_mark_mulf_mix_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_mulf_mix_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_ARITHMETIC_H_
