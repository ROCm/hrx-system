// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU exact packed FP8 encode selection and emission.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP8_ENCODE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP8_ENCODE_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_fp8_encode_emission_state_t {
  // VGPR type used by source and result packet values.
  loom_type_t lane_type;
  // Lane-mask type produced by encoding comparisons.
  loom_type_t mask_type;
  // Positive E4M3 maximum materialized for saturating encodes.
  loom_value_id_t positive_maximum;
  // Negative E4M3 maximum materialized for saturating encodes.
  loom_value_id_t negative_maximum;
  // Maximum finite or overflow-midpoint magnitude used by software encoding.
  loom_value_id_t maximum_magnitude;
  // IEEE F32 infinity magnitude used to identify NaNs.
  loom_value_id_t infinity_magnitude;
  // Smallest normal result magnitude used to select subnormal encoding.
  loom_value_id_t minimum_normal_magnitude;
  // Canonical signed-NaN payload magnitude in the encoded result byte.
  loom_value_id_t nan_encoding;
  // Positive F32 magnitude yielding the canonical FNUZ bridge NaN encoding.
  loom_value_id_t fnuz_nan_bridge_magnitude;
  // V_PERM selector gathering low-pair F32 sign bytes into packed FP8 lanes.
  loom_value_id_t low_sign_permute_selector;
  // V_PERM selector gathering high-pair F32 sign bytes into packed FP8 lanes.
  loom_value_id_t high_sign_permute_selector;
  // Bit mask selecting the low packed FP8 pair's sign bits.
  loom_value_id_t low_sign_mask;
  // Bit mask selecting the high packed FP8 pair's sign bits.
  loom_value_id_t high_sign_mask;
  // F32 mantissa bits discarded by the encoded format.
  uint32_t mantissa_shift;
  // RNE bias below the retained mantissa LSB.
  uint32_t rounding_bias;
  // Two's-complement exponent-bias adjustment after the normal-path shift.
  uint32_t normal_adjustment;
  // F32 bit pattern added to round subnormal results at the FP8 quantum.
  uint32_t subnormal_magic;
  // Two's-complement removal of subnormal_magic from the rounded bit pattern.
  uint32_t subnormal_adjustment;
} loom_amdgpu_fp8_encode_emission_state_t;

// Selects the strongest exact FP8 encode strategy present in |descriptor_set|.
bool loom_amdgpu_select_fp8_encode_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_amdgpu_fp8_encode_plan_t* out_plan);

// Returns true when |plan| encodes independent lanes in software.
bool loom_amdgpu_fp8_encode_plan_is_software(
    const loom_amdgpu_fp8_encode_plan_t* plan);

// Returns true when |plan| bridges exact OCP semantics through native FNUZ.
bool loom_amdgpu_fp8_encode_plan_is_fnuz_bridge(
    const loom_amdgpu_fp8_encode_plan_t* plan);

// Initializes function-local values shared by every packet emitted for |plan|.
// |encoded_lane_count| bounds pair-specific constants needed by the operation.
iree_status_t loom_amdgpu_initialize_fp8_encode_emission(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan, uint32_t encoded_lane_count,
    loom_type_t lane_type, loom_amdgpu_fp8_encode_emission_state_t* out_state);

// Duplicates the low F16 lane into both halves of one packed source register.
iree_status_t loom_amdgpu_emit_fp8_encode_duplicate_f16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_type_t lane_type, loom_value_id_t* out_packed);

// Encodes one F32 bit payload to an exact OCP FP8 byte in a full VGPR.
iree_status_t loom_amdgpu_emit_fp8_encode_software_f32_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source, loom_value_id_t* out_encoded);

// Encodes one low F16 payload to an exact OCP E5M2 byte in a full VGPR.
iree_status_t loom_amdgpu_emit_fp8_encode_software_f16_e5m2_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source, loom_value_id_t* out_encoded);

// Encodes one to four F32 lanes through native FNUZ pair conversions. The last
// logical lane fills unused physical bytes in the complete result register.
iree_status_t loom_amdgpu_emit_fp8_encode_fnuz_f32_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    const loom_value_id_t* source_lanes, uint32_t source_lane_count,
    loom_value_id_t* out_packed);

// Emits the low encoded pair. F32 plans consume |low_source| and |high_source|;
// F16 plans consume one packed pair in |low_source|.
iree_status_t loom_amdgpu_emit_fp8_encode_low_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t low_source, loom_value_id_t high_source,
    loom_value_id_t* out_packed);

// Continues |packed| with the high encoded pair. The result remains tied to
// |packed| so register allocation preserves the partial-register definition.
iree_status_t loom_amdgpu_emit_fp8_encode_high_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t packed, loom_value_id_t low_source,
    loom_value_id_t high_source, loom_value_id_t* out_packed);

// Returns the stable compile-report strategy key for |plan|.
iree_string_view_t loom_amdgpu_fp8_encode_plan_key(
    const loom_amdgpu_fp8_encode_plan_t* plan, loom_scalar_type_t source_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP8_ENCODE_H_
