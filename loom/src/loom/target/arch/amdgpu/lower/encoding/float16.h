// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU narrow floating-point payload lowering for F16 and BF16 storage.
//
// This shard owns packet materialization and packed-register extraction for
// 16-bit floating-point lanes. FP8/BF8 decode, matrix-fragment stores, vector
// construction, and vector conversion share these physical payload helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FLOAT16_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FLOAT16_H_

#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_float16_pack_descriptor_flag_bits_e {
  // No optional packet helpers are available.
  LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_NONE = 0u,
  // Native F32-pair-to-BF16-pair conversion descriptor is available.
  LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE_BF16 = 1u << 0,
  // Integer low-16-bit pair packing descriptor is available.
  LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16 = 1u << 1,
  // Integer three-input add descriptor with a source-2 literal is available.
  LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_ADD3_SRC2_LITERAL = 1u << 2,
  // Native F32-to-F16 conversion descriptor is available.
  LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_F16_CONVERT = 1u << 3,
  // Native F16-pair-to-packed-B32 descriptor is available.
  LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE_F16_PACK = 1u << 4,
} loom_amdgpu_float16_pack_descriptor_flag_bits_t;
typedef uint32_t loom_amdgpu_float16_pack_descriptor_flags_t;

typedef struct loom_amdgpu_float16_pack_descriptors_t {
  // Availability bits for optional descriptor fields in this plan.
  loom_amdgpu_float16_pack_descriptor_flags_t flags;
  // Native F32-pair-to-BF16-pair conversion descriptor.
  loom_low_lower_resolved_descriptor_t native_bf16_descriptor;
  // Native F32-to-F16 conversion descriptor.
  loom_low_lower_resolved_descriptor_t f16_convert_descriptor;
  // Native F16-pair-to-packed-B32 descriptor.
  loom_low_lower_resolved_descriptor_t native_f16_pack_descriptor;
  // Integer low-16-bit pair packing descriptor.
  loom_low_lower_resolved_descriptor_t pack_u16_descriptor;
  // Integer three-input add descriptor with a source-2 literal.
  loom_low_lower_resolved_descriptor_t add3_src2_literal_descriptor;
} loom_amdgpu_float16_pack_descriptors_t;

// Returns optional F16 and BF16 pack packet helpers for the active descriptor
// set. The returned descriptors are function-local target lowering state and
// remain valid until the current loom_low_lower_function call returns.
iree_status_t loom_amdgpu_get_float16_pack_descriptors(
    loom_low_lower_context_t* context,
    const loom_amdgpu_float16_pack_descriptors_t** out_descriptors);

// Returns true when |descriptor_set| can emit one f32-to-BF16 lane conversion.
bool loom_amdgpu_bf16_descriptor_set_can_emit_f32_to_bf16_lane(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns true when |descriptor_set| can pack two low-aligned 16-bit lanes.
bool loom_amdgpu_descriptor_set_can_emit_packed_u16_lane_pair(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns true when |descriptor_set| can emit a packed BF16 pair from two f32
// lanes through either native conversion or integer rounding and packing.
bool loom_amdgpu_bf16_descriptor_set_can_emit_f32_pair_to_packed_bf16(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns true when |descriptor_set| can emit one f32-to-F16 lane conversion.
bool loom_amdgpu_f16_descriptor_set_can_emit_f32_to_f16_lane(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns true when |descriptor_set| can convert and pack two f32 lanes into
// one packed F16 register with round-to-nearest-even semantics.
bool loom_amdgpu_f16_descriptor_set_can_emit_f32_pair_to_packed_f16(
    const loom_low_descriptor_set_t* descriptor_set);

// Emits round-to-nearest-even conversion from one f32 lane to one BF16 lane.
// The result is held in the low 16 bits of a one-unit VGPR.
iree_status_t loom_amdgpu_emit_f32_to_bf16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane);

// Emits round-to-nearest-even conversion from one f32 lane to one BF16 lane
// using already-resolved optional integer pack descriptors.
iree_status_t loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_float16_pack_descriptors_t* descriptors,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane);

// Emits round-to-nearest-even conversion from two f32 lanes to one packed BF16
// register. The low source becomes the low 16 bits of the result.
iree_status_t loom_amdgpu_emit_f32_pair_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed);

// Emits round-to-nearest-even conversion from two f32 lanes to one packed BF16
// register using already-resolved optional native and integer pack descriptors.
iree_status_t loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_float16_pack_descriptors_t* descriptors,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed);

// Emits round-to-nearest-even conversion from one f32 lane to one F16 lane.
// The result is held in the low 16 bits of a one-unit VGPR.
iree_status_t loom_amdgpu_emit_f32_to_f16_lane_with_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_float16_pack_descriptors_t* descriptors,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane);

// Converts two f32 lanes to F16 with round-to-nearest-even semantics and packs
// them into one register. The low source becomes the low 16-bit lane.
iree_status_t loom_amdgpu_emit_f32_pair_to_packed_f16_with_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_float16_pack_descriptors_t* descriptors,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed);

// Emits one packed VGPR containing two low-aligned 16-bit payloads. The low
// source becomes the low 16 bits of the result. |pack_u16_descriptor| may be
// NULL, in which case the helper falls back to shift/or packing.
iree_status_t loom_amdgpu_emit_packed_u16_lane_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor,
    loom_value_id_t low_lane, loom_value_id_t high_lane, loom_type_t lane_type,
    loom_value_id_t* out_packed);

// Extracts one packed BF16 register lane and aligns its bits as an F32 payload.
iree_status_t loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_register, uint32_t register_lane,
    loom_type_t result_lane_type, loom_value_id_t* out_lane);

// Extracts one lane from a packed BF16 register range and aligns its bits as an
// F32 payload.
iree_status_t loom_amdgpu_extract_bf16_range_lane_as_f32_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t lane_index, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_value_id_t* out_lane);

// Extracts one lane from a packed F16 register range into the low 16 bits of a
// one-unit VGPR.
iree_status_t loom_amdgpu_extract_f16_lane_as_low_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t lane_index, loom_type_t lane_type, loom_value_id_t* out_lane);

// Converts one F32 lane to F16 and packs it into one half of |inout_packed|.
iree_status_t loom_amdgpu_pack_f32_lane_to_f16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, uint32_t register_lane, loom_type_t lane_type,
    loom_value_id_t* inout_packed);

// Converts one F32 lane to F16 and duplicates it into both packed lanes.
iree_status_t loom_amdgpu_splat_f32_lane_to_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_packed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FLOAT16_H_
