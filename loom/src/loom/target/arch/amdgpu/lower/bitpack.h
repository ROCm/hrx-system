// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for packed bitfield and bitstream source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_BITPACK_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_BITPACK_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects an AMDGPU vector.bitpack plan for a dispatch-row-owned bitpack op.
iree_status_t loom_amdgpu_select_vector_bitpack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitpack_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.bitpack op to AMDGPU descriptor-backed low packets.
iree_status_t loom_amdgpu_lower_vector_bitpack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitpack_plan_t* plan);

// Packs an already-masked low-bit payload into a 32-bit packed VGPR register.
iree_status_t loom_amdgpu_pack_bits_into_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_bits, uint32_t bit_offset, loom_type_t lane_type,
    loom_value_id_t* inout_packed);

// Masks and packs one 32-bit source lane into a 32-bit packed VGPR register.
iree_status_t loom_amdgpu_pack_lane_bits_into_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, uint32_t width, uint32_t bit_offset,
    loom_type_t lane_type, loom_value_id_t* inout_packed);

// Selects the best available full-register i8 byte permutation plan.
void loom_amdgpu_select_i8_pack_permute_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_i8_pack_permute_plan_t* out_plan);

// Packs complete groups of four 32-bit source lanes to i8 bytes using
// the selected byte permutation plan.
iree_status_t loom_amdgpu_pack_i8_lanes_with_permute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_i8_pack_permute_plan_t* plan,
    const loom_value_id_t* source_lanes, uint32_t source_lane_count,
    loom_type_t lane_type, loom_value_id_t* out_packed_registers);

typedef enum loom_amdgpu_bitfield_extract_mode_e {
  // Leaves high bits after the selected field unchanged by shifting only.
  LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_RAW_SHIFTED = 0,
  // Clears high bits above the selected field.
  LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_ZERO_EXTEND,
  // Sign-extends the selected field to a 32-bit value.
  LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND,
} loom_amdgpu_bitfield_extract_mode_t;

// Extracts a bitfield from a VGPR word using the best available target packet
// sequence for the requested semantics.
iree_status_t loom_amdgpu_extract_vgpr_bitfield(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t bit_offset, uint32_t bit_count,
    loom_amdgpu_bitfield_extract_mode_t mode, loom_type_t lane_type,
    loom_value_id_t* out_value);

// Selects an AMDGPU vector.bitunpack plan for a dispatch-row-owned bitunpack
// op.
iree_status_t loom_amdgpu_select_vector_bitunpack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitunpack_plan_t* out_plan, bool* out_selected);

// Lowers a source vector.bitunpack op to AMDGPU descriptor-backed low packets.
iree_status_t loom_amdgpu_lower_vector_bitunpack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_BITPACK_H_
