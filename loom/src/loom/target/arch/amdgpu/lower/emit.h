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

// Resolves a required AMDGPU descriptor ref against the active descriptor set.
iree_status_t loom_amdgpu_resolve_descriptor_ref(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor);

// Returns true when |descriptor_set| contains the target-generated descriptor
// reference.
bool loom_amdgpu_descriptor_set_has_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref);

// Returns true when |descriptor_set| contains a descriptor with |key|.
bool loom_amdgpu_descriptor_set_has_key(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key);

// Returns true when a descriptor row has an implicit resource operand.
bool loom_amdgpu_descriptor_has_implicit_resource_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

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

// Emits a fresh VGPR carrying the same 32-bit bit payload as |low_source|.
iree_status_t loom_amdgpu_emit_vgpr_b32_copy(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_value_id_t low_source,
                                             loom_value_id_t* out_value);

// Returns |low_value| when it is already a one-unit VGPR, otherwise emits a
// fresh VGPR carrying the same 32-bit bit payload.
iree_status_t loom_amdgpu_materialize_low_vgpr_b32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value);

// Returns |low_value| when it is already a VGPR register range, otherwise
// emits fresh VGPRs carrying the same 32-bit bit payloads as each SGPR unit.
iree_status_t loom_amdgpu_materialize_low_vgpr_b32_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value);

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

// Emits an SGPR x2 value zero-extending the supplied one-unit SGPR.
iree_status_t loom_amdgpu_emit_sgpr64_from_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_wide_value);

// Emits an SGPR x2 unsigned integer constant.
iree_status_t loom_amdgpu_emit_sgpr64_constant_u64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t value, loom_value_id_t* out_low_wide_value);

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

// Emits round-to-nearest-even conversion from one f32 lane to one BF16 lane.
// The result is held in the low 16 bits of a one-unit VGPR.
iree_status_t loom_amdgpu_emit_f32_to_bf16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane);

// Emits round-to-nearest-even conversion from two f32 lanes to one packed BF16
// register. The low source becomes the low 16 bits of the result.
iree_status_t loom_amdgpu_emit_f32_pair_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed);

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
