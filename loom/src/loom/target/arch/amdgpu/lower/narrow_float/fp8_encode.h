// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native packed FP8 encode selection and emission.

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
  // Lane-mask type produced by ordered saturation comparisons.
  loom_type_t mask_type;
  // Positive E4M3 maximum materialized for saturating encodes.
  loom_value_id_t positive_maximum;
  // Negative E4M3 maximum materialized for saturating encodes.
  loom_value_id_t negative_maximum;
} loom_amdgpu_fp8_encode_emission_state_t;

// Selects an exact native FP8 encode strategy for a scalar element pair.
bool loom_amdgpu_select_fp8_encode_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_amdgpu_fp8_encode_plan_t* out_plan);

// Initializes function-local values shared by every packet emitted for |plan|.
iree_status_t loom_amdgpu_initialize_fp8_encode_emission(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan, loom_type_t lane_type,
    loom_amdgpu_fp8_encode_emission_state_t* out_state);

// Duplicates the low F16 lane into both halves of one packed source register.
iree_status_t loom_amdgpu_emit_fp8_encode_duplicate_f16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_type_t lane_type, loom_value_id_t* out_packed);

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
