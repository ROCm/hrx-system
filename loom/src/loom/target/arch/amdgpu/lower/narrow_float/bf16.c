// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/narrow_float/bf16.h"

#include "iree/base/bitfield.h"
#include "loom/ir/attribute.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static iree_status_t loom_amdgpu_emit_resolved_vgpr_ternary_immediate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, uint32_t immediate, loom_type_t lane_type,
    loom_value_id_t* out_value) {
  IREE_ASSERT(descriptor->descriptor != NULL);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("imm32"), immediate, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, attr_count), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bf16_pack_descriptors_t* descriptors,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, source_lane, &source_lane));

  loom_value_id_t upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
      source_lane, lane_type, &upper));
  loom_value_id_t lsb = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, upper, 1,
      lane_type, &lsb));
  loom_value_id_t rounded = LOOM_VALUE_ID_INVALID;
  if (descriptors != NULL &&
      iree_any_bit_set(
          descriptors->flags,
          LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_ADD3_SRC2_LITERAL)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary_immediate(
        context, source_op, &descriptors->add3_src2_literal_descriptor,
        source_lane, lsb, UINT32_C(0x7FFF), lane_type, &rounded));
  } else {
    loom_value_id_t bias = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, lsb,
        UINT32_C(0x7FFF), lane_type, &bias));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, source_lane,
        bias, lane_type, &rounded));
  }
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
      rounded, lane_type, out_lane);
}

iree_status_t loom_amdgpu_resolve_bf16_pack_descriptors(
    loom_low_lower_context_t* context,
    loom_amdgpu_bf16_pack_descriptors_t* out_descriptors) {
  *out_descriptors = (loom_amdgpu_bf16_pack_descriptors_t){0};

  bool has_native_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32,
      &out_descriptors->native_descriptor, &has_native_descriptor));
  if (has_native_descriptor) {
    out_descriptors->flags |= LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE;
  }

  bool has_pack_u16_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32,
      &out_descriptors->pack_u16_descriptor, &has_pack_u16_descriptor));
  if (has_pack_u16_descriptor) {
    out_descriptors->flags |=
        LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16;
  }

  bool has_add3_src2_literal_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD3_U32_SRC2_LIT,
      &out_descriptors->add3_src2_literal_descriptor,
      &has_add3_src2_literal_descriptor));
  if (has_add3_src2_literal_descriptor) {
    out_descriptors->flags |=
        LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_ADD3_SRC2_LITERAL;
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_f32_to_bf16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  loom_amdgpu_bf16_pack_descriptors_t descriptors = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_bf16_pack_descriptors(context, &descriptors));
  return loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
      context, source_op, &descriptors, source_lane, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bf16_pack_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_lane, loom_value_id_t high_lane, loom_type_t lane_type,
    loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {low_lane, high_lane};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_packed = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_native_f32_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_source_lane, &low_source_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, high_source_lane, &high_source_lane));
  return loom_amdgpu_emit_bf16_pack_descriptor(
      context, source_op, descriptor, low_source_lane, high_source_lane,
      lane_type, out_packed);
}

iree_status_t loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bf16_pack_descriptors_t* descriptors,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  if (iree_any_bit_set(descriptors->flags,
                       LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE)) {
    return loom_amdgpu_emit_native_f32_to_packed_bf16(
        context, source_op, &descriptors->native_descriptor, low_source_lane,
        high_source_lane, lane_type, out_packed);
  }

  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
      context, source_op, descriptors, low_source_lane, lane_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
      context, source_op, descriptors, high_source_lane, lane_type,
      &high_lane));
  if (iree_any_bit_set(descriptors->flags,
                       LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16)) {
    return loom_amdgpu_emit_bf16_pack_descriptor(
        context, source_op, &descriptors->pack_u16_descriptor, low_lane,
        high_lane, lane_type, out_packed);
  }

  loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      high_lane, lane_type, &high_bits));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_lane,
      high_bits, lane_type, out_packed);
}

iree_status_t loom_amdgpu_emit_f32_pair_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  loom_amdgpu_bf16_pack_descriptors_t descriptors = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_bf16_pack_descriptors(context, &descriptors));
  return loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
      context, source_op, &descriptors, low_source_lane, high_source_lane,
      lane_type, out_packed);
}
