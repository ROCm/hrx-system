// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU matrix-fragment physical memory layout planning.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_FRAGMENT_MEMORY_LAYOUT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_FRAGMENT_MEMORY_LAYOUT_H_

#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiles participant and register coordinates into physical byte-address
// coefficients. Returns false and records the first stable constraint key when
// the coefficients cannot be represented by the target address plan.
bool loom_amdgpu_fragment_memory_compile_address_layout(
    loom_contract_operand_role_t role,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_matrix_result_representation_flags_t representation_flags,
    uint8_t view_rank,
    const loom_low_source_memory_axis_byte_stride_t* axis_byte_strides,
    loom_amdgpu_fragment_memory_address_layout_t* out_address_layout,
    loom_amdgpu_fragment_memory_runtime_axis_t* out_runtime_axes,
    iree_string_view_t* out_constraint_key);

// Selects the physical packet interpretation for a compiled address layout.
bool loom_amdgpu_fragment_memory_select_packetization(
    loom_contract_operand_role_t role,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_payload_form_t payload_form,
    uint16_t element_byte_count,
    const loom_amdgpu_fragment_memory_address_layout_t* address_layout,
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axes,
    uint8_t view_rank,
    loom_amdgpu_fragment_memory_packetization_t* out_packetization,
    iree_string_view_t* out_constraint_key);

// Rewrites |address_layout| for a transposing 16-bit load in which each lane
// reads eight contiguous elements and every group of eight lanes exchanges them
// as an 8x8 transpose: lane 8g+i receives element i of the reads issued by
// lanes 8g..8g+7. The fragment must hold eight elements per lane at a constant
// byte stride, and eight adjacent lanes must hold adjacent elements. On success
// |address_layout| describes the 16 contiguous bytes each lane reads.
bool loom_amdgpu_fragment_memory_transpose_b16x8_address_layout(
    uint16_t element_byte_count, uint16_t register_count,
    loom_amdgpu_fragment_memory_address_layout_t* address_layout);

// Verifies that indexed-address terms outside the scalar base fit U32 VADDR.
bool loom_amdgpu_fragment_memory_source_plan_supports_addressing(
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_fragment_memory_scalar_base_t* scalar_base,
    iree_string_view_t* out_constraint_key);

// Verifies that source offsets outside the scalar base, participant offsets,
// and register offsets together fit unsigned 32-bit vector addressing.
bool loom_amdgpu_fragment_memory_address_range_fits_u32(
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_fragment_memory_scalar_base_t* scalar_base,
    const loom_amdgpu_fragment_memory_address_layout_t* address_layout,
    const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axes,
    uint8_t view_rank, uint16_t wave_size, uint16_t register_count,
    iree_string_view_t* out_constraint_key);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_FRAGMENT_MEMORY_LAYOUT_H_
