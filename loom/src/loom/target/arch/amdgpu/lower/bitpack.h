// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for vector bitstream packing source operations.

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
