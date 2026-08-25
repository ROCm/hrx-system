// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU narrow-float payload emission for fragment memory plans.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_NARROW_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_NARROW_H_

#include "loom/target/arch/amdgpu/lower/encoding/float16.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_access.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packs selected result-fragment lanes into a 16-bit memory packet.
iree_status_t loom_amdgpu_emit_fragment_memory_packed_16bit_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_payload,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    uint16_t register_index, uint16_t result_register_count,
    uint16_t packet_register_count, loom_value_id_t low_scale,
    loom_type_t vgpr_type, loom_value_id_t* out_packet);

// Widens selected low-subword F16 lanes into an F32 memory packet.
iree_status_t loom_amdgpu_emit_fragment_memory_f16_to_f32_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_payload,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_type_t vgpr_type, loom_value_id_t* out_packet);

// Loads and converts one selected FP8 packet into packed 16-bit registers.
iree_status_t loom_amdgpu_emit_fragment_memory_fp8_to_packed_16bit_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_fragment_memory_address_state_t* address_state,
    loom_value_id_t low_packet_resource, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t low_soffset,
    loom_value_id_t* out_low_packet);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_NARROW_H_
