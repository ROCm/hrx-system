// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU matrix-fragment memory address planning and emission.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_ADDRESS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_ADDRESS_H_

#include "loom/target/arch/amdgpu/lower/matrix_fragment_state.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_fragment_memory_address_t {
  // Low packet address operand after static offset immediates are split out.
  loom_value_id_t low_vaddr;
  // Encoded descriptor offset immediate value.
  int64_t immediate_offset;
} loom_amdgpu_fragment_memory_address_t;

typedef struct loom_amdgpu_fragment_memory_runtime_axis_address_state_t {
  // Materialized runtime byte stride for this physical view axis.
  loom_amdgpu_fragment_memory_address_accumulator_t byte_stride;
  // Coordinate magnitude represented by scaled_byte_stride, or zero if empty.
  uint32_t scaled_coordinate;
  // Cached byte stride multiplied by scaled_coordinate.
  loom_amdgpu_fragment_memory_address_accumulator_t scaled_byte_stride;
} loom_amdgpu_fragment_memory_runtime_axis_address_state_t;

typedef struct loom_amdgpu_fragment_memory_address_state_t {
  // Shared source and lane base advanced across packet coordinates.
  loom_amdgpu_fragment_memory_address_accumulator_t cursor;
  // Runtime coordinate currently accumulated into cursor for each axis.
  uint32_t current_coordinates[LOOM_MATRIX_FRAGMENT_AXIS_COUNT];
  // Materialized stride and packet-step state for each runtime axis.
  loom_amdgpu_fragment_memory_runtime_axis_address_state_t
      runtime_axes[LOOM_MATRIX_FRAGMENT_AXIS_COUNT];
} loom_amdgpu_fragment_memory_address_state_t;

// Returns the exact byte offset contributed by one subgroup lane ID.
uint64_t loom_amdgpu_fragment_memory_relative_lane_byte_offset(
    const loom_amdgpu_fragment_memory_address_layout_t* address_layout,
    uint8_t lane);

// Returns true when a plan uses its original dynamic view-base value.
bool loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint8_t term_index);

// Returns the static byte-address term for one fragment register.
bool loom_amdgpu_fragment_memory_register_terms(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint64_t* out_static_byte_offset);

// Returns true when a physical register group maps to contiguous bytes.
bool loom_amdgpu_fragment_memory_register_group_is_contiguous(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t register_count, uint32_t register_byte_count);

// Returns one fragment element's static byte offset as a signed value.
bool loom_amdgpu_fragment_memory_static_offset_i64(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, int64_t* out_static_byte_offset);

// Returns one fragment element's VGPR-address byte offset when it fits u32.
bool loom_amdgpu_fragment_memory_vaddr_static_offset_u32(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, uint64_t* out_static_byte_offset);

// Returns whether runtime fragment terms contribute only a subgroup-uniform
// common offset to one packet. Source-origin terms are classified separately.
bool loom_amdgpu_fragment_memory_runtime_packet_offset_is_subgroup_uniform(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index);

// Emits the dynamic and lane-coordinate base shared by a fragment's packets.
iree_status_t loom_amdgpu_initialize_fragment_memory_address_state(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_matrix_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_state_t* out_state);

// Emits one packet's VGPR address and descriptor immediate offset.
iree_status_t loom_amdgpu_emit_fragment_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_fragment_memory_address_state_t* address_state,
    loom_type_t vgpr_type, loom_amdgpu_fragment_memory_address_t* out_address);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_FRAGMENT_MEMORY_ADDRESS_H_
