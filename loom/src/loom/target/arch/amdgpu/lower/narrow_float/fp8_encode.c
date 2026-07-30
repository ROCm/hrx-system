// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/narrow_float/fp8_encode.h"

#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#define LOOM_AMDGPU_FP8_E4M3_POSITIVE_MAXIMUM_BITS UINT32_C(0x43E00000)
#define LOOM_AMDGPU_FP8_E4M3_NEGATIVE_MAXIMUM_BITS UINT32_C(0xC3E00000)

static bool loom_amdgpu_fp8_encode_has_refs(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_descriptor_ref_t* descriptor_refs,
    iree_host_size_t descriptor_ref_count) {
  return loom_amdgpu_descriptor_set_has_all_refs(
      descriptor_set, descriptor_refs, descriptor_ref_count);
}

bool loom_amdgpu_select_fp8_encode_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_amdgpu_fp8_encode_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_fp8_encode_plan_t){0};
  if (source_type != LOOM_SCALAR_TYPE_F16 &&
      source_type != LOOM_SCALAR_TYPE_BF16 &&
      source_type != LOOM_SCALAR_TYPE_F32) {
    return false;
  }

  if (source_type == LOOM_SCALAR_TYPE_F16 &&
      result_type == LOOM_SCALAR_TYPE_F8E5M2) {
    const loom_amdgpu_descriptor_ref_t direct_refs[] = {
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF8_F16_OCP_LOW,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF8_F16_OCP_HIGH,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
    };
    if (loom_amdgpu_fp8_encode_has_refs(descriptor_set, direct_refs,
                                        IREE_ARRAYSIZE(direct_refs))) {
      *out_plan = (loom_amdgpu_fp8_encode_plan_t){
          .kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F16_PAIR,
          .low_descriptor_ref = direct_refs[0],
          .high_descriptor_ref = direct_refs[1],
      };
      return true;
    }
  }

  loom_amdgpu_descriptor_ref_t low_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_descriptor_ref_t high_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_fp8_encode_kind_t kind = LOOM_AMDGPU_FP8_ENCODE_KIND_NONE;
  switch (result_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3;
      low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_FP8_F32_OCP_LOW;
      high_descriptor_ref =
          LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_FP8_F32_OCP_HIGH;
      break;
    case LOOM_SCALAR_TYPE_F8E5M2:
      kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR;
      low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF8_F32_OCP_LOW;
      high_descriptor_ref =
          LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF8_F32_OCP_HIGH;
      break;
    default:
      return false;
  }

  const loom_amdgpu_descriptor_ref_t base_refs[] = {
      low_descriptor_ref,
      high_descriptor_ref,
      source_type == LOOM_SCALAR_TYPE_F16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      source_type == LOOM_SCALAR_TYPE_BF16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      source_type == LOOM_SCALAR_TYPE_BF16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  if (!loom_amdgpu_fp8_encode_has_refs(descriptor_set, base_refs,
                                       IREE_ARRAYSIZE(base_refs))) {
    return false;
  }
  if (kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3) {
    const loom_amdgpu_descriptor_ref_t saturation_refs[] = {
        LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_OGT_F32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_OLT_F32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
    };
    if (!loom_amdgpu_fp8_encode_has_refs(descriptor_set, saturation_refs,
                                         IREE_ARRAYSIZE(saturation_refs))) {
      return false;
    }
  }

  *out_plan = (loom_amdgpu_fp8_encode_plan_t){
      .kind = kind,
      .low_descriptor_ref = low_descriptor_ref,
      .high_descriptor_ref = high_descriptor_ref,
  };
  return true;
}

iree_status_t loom_amdgpu_initialize_fp8_encode_emission(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan, loom_type_t lane_type,
    loom_amdgpu_fp8_encode_emission_state_t* out_state) {
  *out_state = (loom_amdgpu_fp8_encode_emission_state_t){
      .lane_type = lane_type,
      .positive_maximum = LOOM_VALUE_ID_INVALID,
      .negative_maximum = LOOM_VALUE_ID_INVALID,
  };
  if (plan->kind != LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &out_state->mask_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_FP8_E4M3_POSITIVE_MAXIMUM_BITS, lane_type,
      &out_state->positive_maximum));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                                    LOOM_AMDGPU_FP8_E4M3_NEGATIVE_MAXIMUM_BITS,
                                    lane_type, &out_state->negative_maximum);
}

iree_status_t loom_amdgpu_emit_fp8_encode_duplicate_f16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_type_t lane_type,
    loom_value_id_t* out_packed) {
  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, source,
      UINT32_C(0xFFFF), lane_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      low_lane, lane_type, &high_lane));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_lane,
      high_lane, lane_type, out_packed);
}

static iree_status_t loom_amdgpu_emit_fp8_e4m3_saturate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source, loom_value_id_t* out_value) {
  loom_value_id_t positive_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_OGT_F32, source,
      state->positive_maximum, state->mask_type, &positive_overflow));
  loom_value_id_t positive_saturated = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
      context, source_op, source, state->positive_maximum, positive_overflow,
      state->lane_type, &positive_saturated));

  loom_value_id_t negative_overflow = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_OLT_F32,
      positive_saturated, state->negative_maximum, state->mask_type,
      &negative_overflow));
  return loom_amdgpu_emit_vgpr_select(
      context, source_op, positive_saturated, state->negative_maximum,
      negative_overflow, state->lane_type, out_value);
}

static iree_status_t loom_amdgpu_prepare_fp8_encode_f32_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t low_source, loom_value_id_t high_source,
    loom_value_id_t* out_low_source, loom_value_id_t* out_high_source) {
  *out_low_source = low_source;
  *out_high_source = high_source;
  if (plan->kind != LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_e4m3_saturate(
      context, source_op, state, low_source, out_low_source));
  return loom_amdgpu_emit_fp8_e4m3_saturate(context, source_op, state,
                                            high_source, out_high_source);
}

static iree_status_t loom_amdgpu_emit_fp8_encode_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t packed,
    loom_value_id_t low_source, loom_value_id_t high_source,
    loom_value_id_t* out_packed) {
  loom_value_id_t operands[3] = {0};
  iree_host_size_t operand_count = 0;
  if (packed != LOOM_VALUE_ID_INVALID) {
    operands[operand_count++] = packed;
  }
  if (plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F16_PAIR) {
    operands[operand_count++] = low_source;
  } else {
    loom_value_id_t prepared_low = LOOM_VALUE_ID_INVALID;
    loom_value_id_t prepared_high = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_fp8_encode_f32_pair(
        context, source_op, plan, state, low_source, high_source, &prepared_low,
        &prepared_high));
    operands[operand_count++] = prepared_low;
    operands[operand_count++] = prepared_high;
  }

  const loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  const loom_tied_result_t* tied_results =
      packed == LOOM_VALUE_ID_INVALID ? NULL : &tied_result;
  const iree_host_size_t tied_result_count =
      packed == LOOM_VALUE_ID_INVALID ? 0 : 1;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  loom_op_t* encode_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_named_attr_slice_empty(), &state->lane_type, 1, tied_results,
      tied_result_count, source_op->location, &encode_op));
  *out_packed = loom_value_slice_get(loom_low_op_results(encode_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fp8_encode_low_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t low_source, loom_value_id_t high_source,
    loom_value_id_t* out_packed) {
  return loom_amdgpu_emit_fp8_encode_pair(
      context, source_op, plan, state, plan->low_descriptor_ref,
      LOOM_VALUE_ID_INVALID, low_source, high_source, out_packed);
}

iree_status_t loom_amdgpu_emit_fp8_encode_high_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t packed, loom_value_id_t low_source,
    loom_value_id_t high_source, loom_value_id_t* out_packed) {
  return loom_amdgpu_emit_fp8_encode_pair(context, source_op, plan, state,
                                          plan->high_descriptor_ref, packed,
                                          low_source, high_source, out_packed);
}

iree_string_view_t loom_amdgpu_fp8_encode_plan_key(
    const loom_amdgpu_fp8_encode_plan_t* plan, loom_scalar_type_t source_type) {
  switch (plan->kind) {
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR:
      switch (source_type) {
        case LOOM_SCALAR_TYPE_F32:
          return IREE_SV("amdgpu.fp8_encode.strategy.f32_pair_native");
        case LOOM_SCALAR_TYPE_F16:
          return IREE_SV("amdgpu.fp8_encode.strategy.f16_via_f32_pair_native");
        case LOOM_SCALAR_TYPE_BF16:
          return IREE_SV("amdgpu.fp8_encode.strategy.bf16_via_f32_pair_native");
        default:
          return iree_string_view_empty();
      }
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3:
      switch (source_type) {
        case LOOM_SCALAR_TYPE_F32:
          return IREE_SV(
              "amdgpu.fp8_encode.strategy.f32_pair_saturating_native");
        case LOOM_SCALAR_TYPE_F16:
          return IREE_SV(
              "amdgpu.fp8_encode.strategy."
              "f16_via_f32_pair_saturating_native");
        case LOOM_SCALAR_TYPE_BF16:
          return IREE_SV(
              "amdgpu.fp8_encode.strategy."
              "bf16_via_f32_pair_saturating_native");
        default:
          return iree_string_view_empty();
      }
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F16_PAIR:
      return IREE_SV("amdgpu.fp8_encode.strategy.f16_pair_native");
    default:
      return iree_string_view_empty();
  }
}
