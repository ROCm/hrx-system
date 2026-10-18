// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for full-width index, address, and scalar integer values.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_INTEGER64_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_INTEGER64_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the numeric transport from source facts. |minimum_unit_count| is one
// for consumers accepting a compact unsigned word or two for pair consumers.
loom_amdgpu_integer_operand_form_t loom_amdgpu_select_integer_operand_form(
    loom_value_facts_t facts, uint8_t minimum_unit_count);

// Materializes the selected integer operand in |register_class_id| (SGPR or
// VGPR). Pair forms preserve a complete existing pair or extend a single word;
// an unsigned-word form extracts the proven u32 payload. Emission consumes no
// source facts. Scalar materialization requires selected uniformity.
iree_status_t loom_amdgpu_lookup_or_materialize_integer_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, uint32_t register_class_id,
    loom_amdgpu_integer_operand_form_t form, loom_value_id_t* out_low_value);

// Sign-extends a one-unit SGPR or VGPR value into a two-unit carrier in the
// same register class. The low word is retained and the high word replicates
// its sign bit.
iree_status_t loom_amdgpu_emit_i64_from_i32(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_value_id_t low_source,
                                            loom_value_id_t* out_low_result);

// Emits a two-word scalar carry or borrow chain. The low descriptor produces
// the low result and SCC; the high descriptor consumes that SCC and produces
// the high result and SCC. Both inputs and the result are SGPR pairs.
iree_status_t loom_amdgpu_emit_sgpr64_binary_carry(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t low_descriptor_ref,
    loom_amdgpu_descriptor_ref_t high_descriptor_ref, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, loom_value_id_t* out_low_result);

// Adds two SGPR pairs and returns both the low 64-bit sum and a normalized
// one-word overflow value. |out_overflow| is zero when the mathematical sum is
// representable in 64 bits and one otherwise.
iree_status_t loom_amdgpu_emit_sgpr64_add_checked(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs,
    loom_value_id_t* out_low_sum, loom_value_id_t* out_overflow);

// Multiplies two SGPR pairs and returns both the low 64-bit product and a
// one-word overflow indicator. |out_overflow| is zero when the mathematical
// product is representable in 64 bits and nonzero otherwise.
iree_status_t loom_amdgpu_emit_sgpr64_mul_checked(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs,
    loom_value_id_t* out_low_product, loom_value_id_t* out_overflow);

// Compares two unsigned SGPR pairs and returns SCC true when |low_lhs| is less
// than or equal to |low_rhs|.
iree_status_t loom_amdgpu_emit_sgpr64_ule_scc(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_value_id_t low_lhs,
                                              loom_value_id_t low_rhs,
                                              loom_value_id_t* out_low_scc);

// Emits the low 64 bits of a product from one- or two-unit operands in the same
// SGPR or VGPR class. One-unit inputs are unsigned words with an implicit zero
// high word. The result always has two units; absent cross products are
// omitted.
iree_status_t loom_amdgpu_emit_i64_mul_lo(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t low_lhs,
                                          loom_value_id_t low_rhs,
                                          loom_value_id_t* out_low_product);

// Selects an AMDGPU index.cast plan.
iree_status_t loom_amdgpu_select_index_cast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_index_cast_plan_t* out_plan, bool* out_selected);

// Publishes the cast's selected width in its actual source or conversion bank.
iree_status_t loom_amdgpu_finalize_index_cast_plan(
    loom_low_lower_context_t* context,
    const loom_amdgpu_index_cast_plan_t* plan);

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

// Selects an AMDGPU scalar i64 population-count plan.
iree_status_t loom_amdgpu_select_scalar_i64_ctpop_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_i64_ctpop_plan_t* out_plan, bool* out_selected);

// Lowers an AMDGPU scalar i64 population-count plan.
iree_status_t loom_amdgpu_lower_scalar_i64_ctpop(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_i64_ctpop_plan_t* plan);

// Extracts the low 32 bits of an already-lowered source value into a VGPR.
iree_status_t loom_amdgpu_extract_low_32_bits_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_value_id_t* out_low_source);

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

// Verifies AMDGPU low legality for i64 scalar compares.
iree_status_t loom_amdgpu_low_legality_verify_scalar_cmpi_i64(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for scalar i64 ALU ops.
iree_status_t loom_amdgpu_low_legality_verify_scalar_i64_alu(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

// Verifies AMDGPU low legality for scalar i64 population count.
iree_status_t loom_amdgpu_low_legality_verify_scalar_i64_ctpop(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_VALUE_INTEGER64_H_
