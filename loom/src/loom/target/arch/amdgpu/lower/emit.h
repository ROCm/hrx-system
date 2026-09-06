// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared AMDGPU descriptor-backed low emission helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_EMIT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_EMIT_H_

#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Interns a low lowering helper string in the active module.
iree_status_t loom_amdgpu_intern(loom_low_lower_context_t* context,
                                 iree_string_view_t string,
                                 loom_string_id_t* out_string_id);

// Appends one signed integer attribute to a descriptor-backed low op.
iree_status_t loom_amdgpu_append_i64_attr(loom_low_lower_context_t* context,
                                          iree_string_view_t name,
                                          int64_t value,
                                          loom_named_attr_t* attrs,
                                          iree_host_size_t attr_capacity,
                                          iree_host_size_t* inout_attr_count);

// Emits an SGPR byte offset from an optional source dynamic index plus a static
// byte offset.
iree_status_t loom_amdgpu_emit_sgpr_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dynamic_index, int64_t dynamic_index_byte_stride,
    uint32_t dynamic_index_byte_shift, uint32_t static_byte_offset,
    loom_value_id_t* out_low_offset);

// Emits an SGPR byte offset from one already-selected source memory dynamic
// term.
iree_status_t loom_amdgpu_emit_sgpr_byte_offset_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_dynamic_term_t* term,
    loom_value_id_t* out_low_offset);

// Emits an SGPR byte offset from the scalar-address terms selected for a source
// memory access plus a static byte offset.
iree_status_t loom_amdgpu_emit_sgpr_byte_offset_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_dynamic_index_kind_t* dynamic_term_kinds,
    uint32_t static_byte_offset, loom_value_id_t* out_low_offset);

// Maps a source result to the low register type already selected by the active
// lowering policy and verifies that it is a register payload.
iree_status_t loom_amdgpu_low_result_type(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_result,
                                          loom_type_t* out_low_type);

// Builds a low register range value from one or more already-emitted register
// units.
iree_status_t loom_amdgpu_build_low_register_range(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* low_registers, uint32_t register_count,
    loom_type_t result_type, loom_value_id_t* out_low_result);

// Extracts one 32-bit unit from an already-emitted low register range.
iree_status_t loom_amdgpu_extract_low_register_unit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t register_count,
    uint32_t register_offset, loom_type_t unit_type,
    loom_value_id_t* out_register_unit);

// Binds a source result to one or more already-emitted low register units.
iree_status_t loom_amdgpu_bind_low_register_range(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, const loom_value_id_t* low_registers,
    uint32_t register_count);

// Resolves an optional AMDGPU descriptor ref against the active descriptor set.
iree_status_t loom_amdgpu_resolve_descriptor_ref_if_present(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor, bool* out_present);

typedef struct loom_amdgpu_descriptor_resolution_t {
  // Descriptor ref to resolve against the active descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Destination receiving the resolved descriptor when every row is present.
  loom_low_lower_resolved_descriptor_t* out_descriptor;
} loom_amdgpu_descriptor_resolution_t;

// Resolves an ordered descriptor ref set, reporting |out_present| false when
// the first missing descriptor ref is encountered.
iree_status_t loom_amdgpu_resolve_descriptor_refs_if_present(
    loom_low_lower_context_t* context,
    const loom_amdgpu_descriptor_resolution_t* resolutions,
    iree_host_size_t resolution_count, bool* out_present);

// Resolves selected v_cndmask_b32 descriptor forms against the active
// descriptor set. Required flags must be present for |out_present| to become
// true; optional flags are resolved when available and otherwise left empty.
iree_status_t loom_amdgpu_resolve_cndmask_b32_descriptors(
    loom_low_lower_context_t* context,
    loom_amdgpu_cndmask_b32_descriptor_flags_t required_flags,
    loom_amdgpu_cndmask_b32_descriptor_flags_t optional_flags,
    loom_amdgpu_cndmask_b32_descriptors_t* out_descriptors, bool* out_present);

// Resolves a required AMDGPU descriptor ref against the active descriptor set.
iree_status_t loom_amdgpu_resolve_descriptor_ref(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor);

// Selects the exact descriptor used to emit |descriptor_ref| with |immediate|,
// including target inline forms. Returns NONE when no form is available.
loom_amdgpu_descriptor_ref_t
loom_amdgpu_select_vgpr_binary_immediate_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t immediate);

// Returns true when |descriptor_set| can emit |descriptor_ref| with |immediate|
// through the normal VGPR immediate helper, including target inline forms.
bool loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t immediate);

// Returns true when a VGPR compare against |immediate| can use either the
// supplied RHS-inline form or a materialized VGPR immediate.
bool loom_amdgpu_descriptor_set_can_emit_vgpr_compare_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_descriptor_ref_t src1_inline_descriptor_ref,
    uint32_t immediate);

// Resolves one optional explicit packet descriptor and its immediate names.
iree_status_t loom_amdgpu_resolve_explicit_packet_plan(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_explicit_packet_immediate_template_t* immediates,
    iree_host_size_t immediate_count,
    loom_amdgpu_explicit_packet_plan_t* out_plan, bool* out_present);

// Resolves one explicit packet descriptor row and its immediate names.
iree_status_t loom_amdgpu_resolve_explicit_packet_row_plan(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_explicit_packet_immediate_template_t* immediates,
    iree_host_size_t immediate_count,
    loom_amdgpu_explicit_packet_plan_t* out_plan);

// Emits one descriptor-backed low.op with source provenance.
iree_status_t loom_amdgpu_emit_low_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_op_t** out_low_op);

// Emits one explicit descriptor packet selected during source-to-low planning.
iree_status_t loom_amdgpu_emit_explicit_packet_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_explicit_packet_plan_t* plan);

// Emits one descriptor-backed low.const with an imm32 attribute.
iree_status_t loom_amdgpu_emit_const_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_value_id_t* out_value_id);

// Emits one resolved descriptor-backed low.const with an imm32 attribute.
iree_status_t loom_amdgpu_emit_resolved_const_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, uint32_t value,
    loom_type_t result_type, loom_value_id_t* out_value_id);

// Emits one resolved VGPR descriptor op with two VGPR operands and one imm32
// immediate attribute.
iree_status_t loom_amdgpu_emit_resolved_vgpr_binary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, uint32_t immediate, loom_type_t lane_type,
    loom_value_id_t* out_value);

// Emits one resolved VGPR descriptor op with one register operand.
iree_status_t loom_amdgpu_emit_resolved_vgpr_unary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t value, loom_type_t lane_type, loom_value_id_t* out_value);

// Emits one resolved VGPR descriptor op with two VGPR operands.
iree_status_t loom_amdgpu_emit_resolved_vgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_value);

// Emits one resolved VGPR descriptor op with three register operands.
iree_status_t loom_amdgpu_emit_resolved_vgpr_ternary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t third, loom_type_t lane_type,
    loom_value_id_t* out_value);

// Materializes distinct SGPR sources beyond the active target's VOP3 scalar
// bus limit. Repeated source IDs share the same legalized value.
iree_status_t loom_amdgpu_legalize_vop3_scalar_sources(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t sources[3]);

// Emits one resolved VGPR descriptor op with one VGPR operand and one imm32
// immediate attribute.
iree_status_t loom_amdgpu_emit_resolved_vgpr_unary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t value, uint32_t immediate, loom_type_t lane_type,
    loom_value_id_t* out_value);

// Emits a fresh VGPR carrying the same 32-bit bit payload as |low_source|.
iree_status_t loom_amdgpu_emit_vgpr_b32_copy(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_value_id_t low_source,
                                             loom_value_id_t* out_value);

// Returns true when |low_value| is a VGPR value whose defining descriptor
// writes only the low 16-bit register part.
bool loom_amdgpu_low_value_defines_vgpr_low16(loom_low_lower_context_t* context,
                                              loom_value_id_t low_value);

// Returns |low_value| when it is already a one-unit VGPR, otherwise emits a
// fresh VGPR carrying the same 32-bit bit payload.
iree_status_t loom_amdgpu_materialize_low_vgpr_b32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value);

// Returns a one-unit VGPR carrying a full 32-bit payload. If |low_value| is
// defined by a descriptor that writes only the low 16-bit register part, emits
// a full-width extraction before returning.
iree_status_t loom_amdgpu_materialize_full_low_vgpr_b32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value);

// Returns |low_value| when it is already a VGPR register range, otherwise
// emits fresh VGPRs carrying the same 32-bit bit payloads as each SGPR unit.
iree_status_t loom_amdgpu_materialize_low_vgpr_b32_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value);

// Materializes target-owned storage contracts before low structural ops consume
// operands that otherwise have the correct low type.
iree_status_t loom_amdgpu_materialize_structural_operand(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, iree_host_size_t operand_index,
    loom_value_id_t source_value_id, loom_value_id_t low_value_id,
    loom_type_t required_low_type, loom_value_id_t* out_low_value_id);

// Emits one binary SGPR descriptor op.
iree_status_t loom_amdgpu_emit_sgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_value);

// Emits one SGPR descriptor op with one SGPR operand and one imm32 immediate.
iree_status_t loom_amdgpu_emit_sgpr_binary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    uint32_t immediate, loom_type_t lane_type, loom_value_id_t* out_value);

// Emits |value| scaled by an unsigned 32-bit constant into a one-unit SGPR.
iree_status_t loom_amdgpu_emit_sgpr_scale_u32(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_value_id_t value,
                                              uint32_t scale,
                                              loom_type_t lane_type,
                                              loom_value_id_t* out_value);

// Emits |value * scale + addend| into a one-unit SGPR, using a fused scalar
// shift-add descriptor when the target provides the exact power-of-two form.
iree_status_t loom_amdgpu_emit_sgpr_scaled_add_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t scale, loom_value_id_t addend,
    loom_type_t lane_type, loom_value_id_t* out_value);

// Emits an SGPR x2 value zero-extending the supplied one-unit SGPR.
iree_status_t loom_amdgpu_emit_sgpr64_from_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_wide_value);

// Emits an SGPR x2 unsigned integer constant.
iree_status_t loom_amdgpu_emit_sgpr64_constant_u64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t value, loom_value_id_t* out_low_wide_value);

// Emits SCC true when any active lane in an EXEC-width SGPRx2 lane mask is set.
//
// Wave64 compares the full SGPR pair. Wave32 VOPC producers define the low
// half and may leave the high half unspecified, so this compares only the
// defined low half.
iree_status_t loom_amdgpu_emit_lane_mask_nonzero_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_mask, uint32_t wavefront_size,
    loom_value_id_t* out_low_scc);

// Emits SCC true when two EXEC-width SGPRx2 lane masks are equal.
//
// Wave64 compares the full SGPR pairs. Wave32 compares the defined low halves
// and ignores unspecified high halves.
iree_status_t loom_amdgpu_emit_lane_mask_equal_scc(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs, uint32_t wavefront_size,
    loom_value_id_t* out_low_scc);

// Emits an SGPR x2 add using the target carry-chain instructions.
iree_status_t loom_amdgpu_emit_sgpr64_add(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t low_lhs,
                                          loom_value_id_t low_rhs,
                                          loom_value_id_t* out_low_sum);

// Emits an SGPR x2 base plus one-unit SGPR unsigned byte offset.
iree_status_t loom_amdgpu_emit_sgpr64_add_u32_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_base, loom_value_id_t low_offset,
    loom_value_id_t* out_low_sum);

// Emits one binary VGPR descriptor op.
iree_status_t loom_amdgpu_emit_vgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t lane_type, loom_value_id_t* out_value);

// Emits one unary VGPR descriptor op.
iree_status_t loom_amdgpu_emit_vgpr_unary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    loom_type_t lane_type, loom_value_id_t* out_value);

// Emits a VGPR compare against an immediate using an already-resolved inline
// form when available and otherwise materializing the immediate in a VGPR.
iree_status_t loom_amdgpu_emit_resolved_vgpr_compare_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* src1_inline_descriptor,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    uint32_t immediate, loom_type_t vgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_mask);

// Emits a VGPR compare against an immediate, resolving the supplied RHS-inline
// descriptor form when the immediate is encodable.
iree_status_t loom_amdgpu_emit_vgpr_compare_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_descriptor_ref_t src1_inline_descriptor_ref,
    loom_value_id_t value, uint32_t immediate, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_mask);

// Emits a VGPR select using a lane-mask condition.
iree_status_t loom_amdgpu_emit_vgpr_select(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t false_value,
                                           loom_value_id_t true_value,
                                           loom_value_id_t condition,
                                           loom_type_t vgpr_type,
                                           loom_value_id_t* out_value);

// Emits one VGPR descriptor op with one VGPR operand and one imm32 immediate.
iree_status_t loom_amdgpu_emit_vgpr_binary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t value,
    uint32_t immediate, loom_type_t lane_type, loom_value_id_t* out_value);

// Emits one VGPR immediate-shift descriptor op. If |shift| is zero, returns
// |value| unchanged.
iree_status_t loom_amdgpu_emit_vgpr_shift(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t shift,
    loom_value_id_t value, loom_type_t lane_type, loom_value_id_t* out_value);

// Emits |value << shift| + |addend| using a resolved V_LSHL_ADD_U32
// immediate-shift descriptor.
iree_status_t loom_amdgpu_emit_resolved_vgpr_lshl_add_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t value, loom_value_id_t addend, uint32_t shift,
    loom_type_t lane_type, loom_value_id_t* out_value);

// Tries to emit |value << shift| + |addend| using the V_LSHL_ADD_U32
// immediate-shift form. If the active descriptor set lacks the packet or the
// shift is not encodable, returns with |out_selected| false and emits nothing.
iree_status_t loom_amdgpu_try_emit_vgpr_lshl_add_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, loom_value_id_t addend, uint32_t shift,
    loom_type_t lane_type, loom_value_id_t* out_value, bool* out_selected);

// Emits a VGPR x2 value zero-extending the supplied one-unit low register.
iree_status_t loom_amdgpu_emit_vgpr64_from_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_wide_value);

// Emits a VGPR x2 add using the target carry-chain instructions.
iree_status_t loom_amdgpu_emit_vgpr64_add(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t low_lhs,
                                          loom_value_id_t low_rhs,
                                          loom_value_id_t* out_low_sum);

// Emits a VGPR x2 subtract using the target borrow-chain instructions.
iree_status_t loom_amdgpu_emit_vgpr64_sub(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t low_lhs,
                                          loom_value_id_t low_rhs,
                                          loom_value_id_t* out_low_difference);

// Emits the low 64 bits of a VGPR x2 multiply.
iree_status_t loom_amdgpu_emit_vgpr64_mul_lo(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_value_id_t low_lhs,
                                             loom_value_id_t low_rhs,
                                             loom_value_id_t* out_low_product);

// Emits a VGPR x2 left shift by a one-unit VGPR shift amount.
iree_status_t loom_amdgpu_emit_vgpr64_shl(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t low_value,
                                          loom_value_id_t low_shift,
                                          loom_value_id_t* out_low_shifted);

typedef enum loom_amdgpu_vgpr_sdwa_extract_flag_bits_e {
  // No additional source selection modifiers are applied.
  LOOM_AMDGPU_VGPR_SDWA_EXTRACT_FLAG_NONE = 0u,
  // Sign-extends the selected byte or word to the destination dword.
  LOOM_AMDGPU_VGPR_SDWA_EXTRACT_FLAG_SIGN_EXTEND = 1u << 0,
} loom_amdgpu_vgpr_sdwa_extract_flag_bits_t;
typedef uint32_t loom_amdgpu_vgpr_sdwa_extract_flags_t;

// Tries to emit a CDNA SDWA byte/word extract. If the active descriptor set has
// no SDWA form, or the selected bit range is not representable by SDWA, returns
// with |out_selected| false and emits nothing.
iree_status_t loom_amdgpu_try_emit_vgpr_b32_sdwa_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t bit_offset, uint32_t bit_count,
    loom_amdgpu_vgpr_sdwa_extract_flags_t flags, loom_type_t lane_type,
    loom_value_id_t* out_value, bool* out_selected);

typedef enum loom_amdgpu_vgpr_bfe_extract_flag_bits_e {
  // Selects the unsigned bitfield extraction form.
  LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_NONE = 0u,
  // Selects the signed bitfield extraction form.
  LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_SIGN_EXTEND = 1u << 0,
} loom_amdgpu_vgpr_bfe_extract_flag_bits_t;
typedef uint32_t loom_amdgpu_vgpr_bfe_extract_flags_t;

// Tries to emit a V_BFE offset/width-inline bitfield extract. If the active
// descriptor set has no BFE form, or the selected bit range is not
// representable by the descriptor, returns with |out_selected| false and emits
// nothing.
iree_status_t loom_amdgpu_try_emit_vgpr_b32_bfe_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t bit_offset, uint32_t bit_count,
    loom_amdgpu_vgpr_bfe_extract_flags_t flags, loom_type_t lane_type,
    loom_value_id_t* out_value, bool* out_selected);

typedef enum loom_amdgpu_vgpr_scale_u32_flag_bits_e {
  // No additional facts about the input value are known.
  LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE = 0u,
  // The input value is known to be in the unsigned 24-bit range.
  LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24 = 1u << 0,
} loom_amdgpu_vgpr_scale_u32_flag_bits_t;
typedef uint32_t loom_amdgpu_vgpr_scale_u32_flags_t;

// Emits |value| scaled by an unsigned 32-bit constant into a one-unit VGPR.
iree_status_t loom_amdgpu_emit_vgpr_scale_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t scale,
    loom_amdgpu_vgpr_scale_u32_flags_t flags, loom_type_t lane_type,
    loom_value_id_t* out_value);

// Emits a low.slice from a register range.
iree_status_t loom_amdgpu_emit_low_slice(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_value_id_t source,
                                         uint32_t lane_offset,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_value);

// Emits a value into M0 for special-register packet operands.
iree_status_t loom_amdgpu_emit_m0_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* consumer_descriptor,
    uint32_t value, loom_value_id_t* out_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_EMIT_H_
