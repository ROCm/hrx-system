// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU narrow floating-point payload lowering for BF16 storage.
//
// This shard owns packet materialization for BF16 lanes and packed BF16
// registers. FP8/BF8 decode, matrix-fragment stores, and vector fptrunc all
// share this path once they need to form BF16 payloads from F32 lanes.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_BF16_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_BF16_H_

#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_bf16_pack_descriptor_flag_bits_e {
  // No optional packet helpers are available.
  LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_NONE = 0u,
  // Native F32-pair-to-BF16-pair conversion descriptor is available.
  LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE = 1u << 0,
  // Integer low-16-bit pair packing descriptor is available.
  LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16 = 1u << 1,
  // Integer three-input add descriptor with a source-2 literal is available.
  LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_ADD3_SRC2_LITERAL = 1u << 2,
} loom_amdgpu_bf16_pack_descriptor_flag_bits_t;
typedef uint32_t loom_amdgpu_bf16_pack_descriptor_flags_t;

typedef struct loom_amdgpu_bf16_pack_descriptors_t {
  // Availability bits for optional descriptor fields in this plan.
  loom_amdgpu_bf16_pack_descriptor_flags_t flags;
  // Native F32-pair-to-BF16-pair conversion descriptor.
  loom_low_lower_resolved_descriptor_t native_descriptor;
  // Integer low-16-bit pair packing descriptor.
  loom_low_lower_resolved_descriptor_t pack_u16_descriptor;
  // Integer three-input add descriptor with a source-2 literal.
  loom_low_lower_resolved_descriptor_t add3_src2_literal_descriptor;
} loom_amdgpu_bf16_pack_descriptors_t;

// Returns optional BF16 pack packet helpers for the active descriptor set. The
// returned descriptor set is function-local target lowering state and remains
// valid until the current loom_low_lower_function call returns.
iree_status_t loom_amdgpu_get_bf16_pack_descriptors(
    loom_low_lower_context_t* context,
    const loom_amdgpu_bf16_pack_descriptors_t** out_descriptors);

// Returns true when |descriptor_set| can emit one f32-to-BF16 lane conversion.
bool loom_amdgpu_bf16_descriptor_set_can_emit_f32_to_bf16_lane(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns true when |descriptor_set| can pack two already-rounded BF16 lanes.
bool loom_amdgpu_bf16_descriptor_set_can_emit_packed_lane_pair(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns true when |descriptor_set| can emit a packed BF16 pair from two f32
// lanes through either native conversion or integer rounding and packing.
bool loom_amdgpu_bf16_descriptor_set_can_emit_f32_pair_to_packed_bf16(
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
    const loom_amdgpu_bf16_pack_descriptors_t* descriptors,
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
    const loom_amdgpu_bf16_pack_descriptors_t* descriptors,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed);

// Emits one packed VGPR containing two already-rounded BF16 bit payloads. The
// low source becomes the low 16 bits of the result. |pack_u16_descriptor| may
// be NULL, in which case the helper falls back to shift/or packing.
iree_status_t loom_amdgpu_emit_packed_bf16_lane_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor,
    loom_value_id_t low_lane, loom_value_id_t high_lane, loom_type_t lane_type,
    loom_value_id_t* out_packed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_BF16_H_
