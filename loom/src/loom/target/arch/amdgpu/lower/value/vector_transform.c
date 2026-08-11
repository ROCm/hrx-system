// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/value/vector_transform.h"

#include "iree/base/internal/math.h"
#include "loom/ops/encoding/hadamard.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"

static bool loom_amdgpu_hadamard_normalization_bits(
    loom_encoding_transform_normalization_t normalization,
    uint32_t slice_extent, uint32_t* out_scale_bits) {
  *out_scale_bits = 0;
  if (normalization == LOOM_ENCODING_TRANSFORM_NORMALIZATION_NONE) {
    return true;
  }
  if (normalization != LOOM_ENCODING_TRANSFORM_NORMALIZATION_ORTHONORMAL) {
    return false;
  }

  // Exact f32 encodings of 1/sqrt(N) for the supported power-of-two slices.
  static const uint32_t kOrthonormalScaleBits[] = {
      UINT32_C(0x3F800000),  // 1/sqrt(1)
      UINT32_C(0x3F3504F3),  // 1/sqrt(2)
      UINT32_C(0x3F000000),  // 1/sqrt(4)
      UINT32_C(0x3EB504F3),  // 1/sqrt(8)
      UINT32_C(0x3E800000),  // 1/sqrt(16)
      UINT32_C(0x3E3504F3),  // 1/sqrt(32)
  };
  if (!iree_math_is_power_of_two_i64((int64_t)slice_extent) ||
      slice_extent > 32u) {
    return false;
  }
  const uint32_t scale_ordinal =
      (uint32_t)iree_math_count_trailing_zeros_u32(slice_extent);
  *out_scale_bits = kOrthonormalScaleBits[scale_ordinal];
  return true;
}

static bool loom_amdgpu_vector_transform_describe(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    uint32_t* out_lane_count, uint32_t* out_slice_extent,
    uint32_t* out_normalization_scale_bits) {
  *out_lane_count = 0;
  *out_slice_extent = 0;
  *out_normalization_scale_bits = 0;

  const loom_value_id_t source = loom_vector_transform_source(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  if (loom_type_element_type(source_type) != LOOM_SCALAR_TYPE_F32) {
    return false;
  }

  loom_encoding_hadamard_descriptor_t descriptor;
  if (!loom_encoding_hadamard_try_read_verified_descriptor(
          module, loom_vector_transform_transform(source_op), &descriptor)) {
    return false;
  }

  iree_host_size_t lane_count = 0;
  if (!loom_type_static_element_count(source_type, &lane_count) ||
      lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return false;
  }
  const uint8_t final_axis = (uint8_t)(loom_type_rank(source_type) - 1u);
  if (loom_type_dim_is_dynamic_at(source_type, final_axis)) {
    return false;
  }
  const int64_t slice_extent_i64 =
      loom_type_dim_static_size_at(source_type, final_axis);

  uint32_t normalization_scale_bits = 0;
  if (!loom_amdgpu_hadamard_normalization_bits(descriptor.normalization,
                                               (uint32_t)slice_extent_i64,
                                               &normalization_scale_bits)) {
    return false;
  }

  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32) ||
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_F32) ||
      (normalization_scale_bits != 0 &&
       normalization_scale_bits != UINT32_C(0x3F800000) &&
       !loom_amdgpu_descriptor_set_has_ref(
           descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32_LIT))) {
    return false;
  }

  *out_lane_count = (uint32_t)lane_count;
  *out_slice_extent = (uint32_t)slice_extent_i64;
  *out_normalization_scale_bits = normalization_scale_bits;
  return true;
}

bool loom_amdgpu_vector_transform_can_lower(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op) {
  uint32_t lane_count = 0;
  uint32_t slice_extent = 0;
  uint32_t normalization_scale_bits = 0;
  return loom_amdgpu_vector_transform_describe(
      module, descriptor_set, source_op, &lane_count, &slice_extent,
      &normalization_scale_bits);
}

iree_status_t loom_amdgpu_select_vector_transform_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_transform_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_transform_plan_t){0};
  *out_selected = false;
  uint32_t lane_count = 0;
  uint32_t slice_extent = 0;
  uint32_t normalization_scale_bits = 0;
  if (!loom_amdgpu_vector_transform_describe(
          loom_low_lower_context_module(context),
          loom_low_lower_context_descriptor_set(context), source_op,
          &lane_count, &slice_extent, &normalization_scale_bits)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_vector_transform_plan_t){
      .source = loom_vector_transform_source(source_op),
      .result = loom_vector_transform_result(source_op),
      .lane_count = lane_count,
      .slice_extent = slice_extent,
      .normalization_scale_bits = normalization_scale_bits,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_transform(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_transform_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t result_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &result_lane_type));

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->lane_count, i, source_lane_type,
        &lanes[i]));
  }

  for (uint32_t slice_offset = 0; slice_offset < plan->lane_count;
       slice_offset += plan->slice_extent) {
    for (uint32_t half_span = 1; half_span < plan->slice_extent;
         half_span <<= 1) {
      const uint32_t span = half_span << 1;
      for (uint32_t base = 0; base < plan->slice_extent; base += span) {
        for (uint32_t lane = 0; lane < half_span; ++lane) {
          const uint32_t lhs_index = slice_offset + base + lane;
          const uint32_t rhs_index = lhs_index + half_span;
          const loom_value_id_t lhs = lanes[lhs_index];
          const loom_value_id_t rhs = lanes[rhs_index];
          loom_value_id_t sum = LOOM_VALUE_ID_INVALID;
          loom_value_id_t difference = LOOM_VALUE_ID_INVALID;
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
              context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32, lhs,
              rhs, result_lane_type, &sum));
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
              context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_F32, lhs,
              rhs, result_lane_type, &difference));
          lanes[lhs_index] = sum;
          lanes[rhs_index] = difference;
        }
      }
    }
  }

  if (plan->normalization_scale_bits != 0 &&
      plan->normalization_scale_bits != UINT32_C(0x3F800000)) {
    for (uint32_t i = 0; i < plan->lane_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32_LIT,
          lanes[i], plan->normalization_scale_bits, result_lane_type,
          &lanes[i]));
    }
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lanes, plan->lane_count);
}

iree_string_view_t loom_amdgpu_vector_transform_plan_key(
    const loom_amdgpu_vector_transform_plan_t* plan) {
  (void)plan;
  return IREE_SV("amdgpu.vector_transform.strategy.f32_register_hadamard");
}
