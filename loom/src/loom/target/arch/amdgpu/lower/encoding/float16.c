// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/encoding/float16.h"

#include "iree/base/bitfield.h"
#include "loom/ir/attribute.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

iree_status_t loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_float16_pack_descriptors_t* descriptors,
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
          LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_ADD3_SRC2_LITERAL)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary_immediate(
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

static iree_status_t loom_amdgpu_initialize_float16_pack_descriptors(
    loom_low_lower_context_t* context,
    loom_amdgpu_float16_pack_descriptors_t* descriptors) {
  *descriptors = (loom_amdgpu_float16_pack_descriptors_t){0};

  bool has_native_bf16_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32,
      &descriptors->native_bf16_descriptor, &has_native_bf16_descriptor));
  if (has_native_bf16_descriptor) {
    descriptors->flags |=
        LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE_BF16;
  }

  bool has_f16_convert_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F16_F32,
      &descriptors->f16_convert_descriptor, &has_f16_convert_descriptor));
  if (has_f16_convert_descriptor) {
    descriptors->flags |=
        LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_F16_CONVERT;
  }

  bool has_native_f16_pack_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PACK_B32_F16,
      &descriptors->native_f16_pack_descriptor,
      &has_native_f16_pack_descriptor));
  if (has_native_f16_pack_descriptor) {
    descriptors->flags |=
        LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE_F16_PACK;
  }

  bool has_pack_u16_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32,
      &descriptors->pack_u16_descriptor, &has_pack_u16_descriptor));
  if (has_pack_u16_descriptor) {
    descriptors->flags |= LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16;
  }

  bool has_add3_src2_literal_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD3_U32_SRC2_LIT,
      &descriptors->add3_src2_literal_descriptor,
      &has_add3_src2_literal_descriptor));
  if (has_add3_src2_literal_descriptor) {
    descriptors->flags |=
        LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_ADD3_SRC2_LITERAL;
  }

  return iree_ok_status();
}

typedef struct loom_amdgpu_float16_pack_descriptor_cache_t {
  // Non-zero once descriptors has been initialized for this lowering context.
  uint32_t initialized;
  // Function-local F16 and BF16 pack descriptor set.
  loom_amdgpu_float16_pack_descriptors_t descriptors;
} loom_amdgpu_float16_pack_descriptor_cache_t;

static int loom_amdgpu_float16_pack_descriptor_cache_state_key;

iree_status_t loom_amdgpu_get_float16_pack_descriptors(
    loom_low_lower_context_t* context,
    const loom_amdgpu_float16_pack_descriptors_t** out_descriptors) {
  *out_descriptors = NULL;
  loom_amdgpu_float16_pack_descriptor_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_float16_pack_descriptor_cache_state_key,
      sizeof(*cache), (void**)&cache));
  if (cache->initialized == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_float16_pack_descriptors(
        context, &cache->descriptors));
    cache->initialized = 1;
  }
  *out_descriptors = &cache->descriptors;
  return iree_ok_status();
}

bool loom_amdgpu_bf16_descriptor_set_can_emit_f32_to_bf16_lane(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (!loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16) ||
      !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, 1)) {
    return false;
  }
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD3_U32_SRC2_LIT)) {
    return true;
  }
  return loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
             UINT32_C(0x7FFF)) &&
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32);
}

bool loom_amdgpu_descriptor_set_can_emit_packed_u16_lane_pair(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_U16_U32)) {
    return true;
  }
  return loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
             UINT16_MAX) &&
         loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
             16) &&
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32);
}

bool loom_amdgpu_bf16_descriptor_set_can_emit_f32_pair_to_packed_bf16(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32)) {
    return true;
  }
  return loom_amdgpu_bf16_descriptor_set_can_emit_f32_to_bf16_lane(
             descriptor_set) &&
         loom_amdgpu_descriptor_set_can_emit_packed_u16_lane_pair(
             descriptor_set);
}

bool loom_amdgpu_f16_descriptor_set_can_emit_f32_to_f16_lane(
    const loom_low_descriptor_set_t* descriptor_set) {
  return loom_amdgpu_descriptor_set_has_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F16_F32);
}

bool loom_amdgpu_f16_descriptor_set_can_emit_f32_pair_to_packed_f16(
    const loom_low_descriptor_set_t* descriptor_set) {
  return loom_amdgpu_f16_descriptor_set_can_emit_f32_to_f16_lane(
             descriptor_set) &&
         (loom_amdgpu_descriptor_set_has_ref(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PACK_B32_F16) ||
          loom_amdgpu_descriptor_set_can_emit_packed_u16_lane_pair(
              descriptor_set));
}

iree_status_t loom_amdgpu_emit_f32_to_bf16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  const loom_amdgpu_float16_pack_descriptors_t* descriptors = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_get_float16_pack_descriptors(context, &descriptors));
  return loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
      context, source_op, descriptors, source_lane, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_float16_pack_descriptor(
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
  return loom_amdgpu_emit_float16_pack_descriptor(
      context, source_op, descriptor, low_source_lane, high_source_lane,
      lane_type, out_packed);
}

iree_status_t loom_amdgpu_emit_f32_to_f16_lane_with_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_float16_pack_descriptors_t* descriptors,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_TRUE(iree_any_bit_set(
      descriptors->flags,
      LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_F16_CONVERT));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, source_lane, &source_lane));
  return loom_amdgpu_emit_resolved_vgpr_unary(
      context, source_op, &descriptors->f16_convert_descriptor, source_lane,
      lane_type, out_lane);
}

iree_status_t loom_amdgpu_emit_f32_pair_to_packed_f16_with_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_float16_pack_descriptors_t* descriptors,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_f16_lane_with_descriptors(
      context, source_op, descriptors, low_source_lane, lane_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_f16_lane_with_descriptors(
      context, source_op, descriptors, high_source_lane, lane_type,
      &high_lane));
  if (iree_any_bit_set(
          descriptors->flags,
          LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE_F16_PACK)) {
    return loom_amdgpu_emit_float16_pack_descriptor(
        context, source_op, &descriptors->native_f16_pack_descriptor, low_lane,
        high_lane, lane_type, out_packed);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, low_lane, &low_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, high_lane, &high_lane));
  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(descriptors->flags,
                       LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16)
          ? &descriptors->pack_u16_descriptor
          : NULL;
  return loom_amdgpu_emit_packed_u16_lane_pair(
      context, source_op, pack_u16_descriptor, low_lane, high_lane, lane_type,
      out_packed);
}

iree_status_t loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_float16_pack_descriptors_t* descriptors,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  if (iree_any_bit_set(
          descriptors->flags,
          LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE_BF16)) {
    return loom_amdgpu_emit_native_f32_to_packed_bf16(
        context, source_op, &descriptors->native_bf16_descriptor,
        low_source_lane, high_source_lane, lane_type, out_packed);
  }

  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
      context, source_op, descriptors, low_source_lane, lane_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
      context, source_op, descriptors, high_source_lane, lane_type,
      &high_lane));
  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(descriptors->flags,
                       LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16)
          ? &descriptors->pack_u16_descriptor
          : NULL;
  return loom_amdgpu_emit_packed_u16_lane_pair(
      context, source_op, pack_u16_descriptor, low_lane, high_lane, lane_type,
      out_packed);
}

iree_status_t loom_amdgpu_emit_packed_u16_lane_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor,
    loom_value_id_t low_lane, loom_value_id_t high_lane, loom_type_t lane_type,
    loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  if (pack_u16_descriptor != NULL) {
    return loom_amdgpu_emit_float16_pack_descriptor(
        context, source_op, pack_u16_descriptor, low_lane, high_lane, lane_type,
        out_packed);
  }
  loom_value_id_t low_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_lane,
      UINT16_MAX, lane_type, &low_bits));
  loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      high_lane, lane_type, &high_bits));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_bits,
      high_bits, lane_type, out_packed);
}

iree_status_t loom_amdgpu_emit_f32_pair_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_lane, loom_value_id_t high_source_lane,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  const loom_amdgpu_float16_pack_descriptors_t* descriptors = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_get_float16_pack_descriptors(context, &descriptors));
  return loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
      context, source_op, descriptors, low_source_lane, high_source_lane,
      lane_type, out_packed);
}

iree_status_t loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_register, uint32_t register_lane,
    loom_type_t result_lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (register_lane == 0) {
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
        source_register, result_lane_type, out_lane);
  }
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      source_register, UINT32_C(0xFFFF0000), result_lane_type, out_lane);
}

iree_status_t loom_amdgpu_extract_bf16_range_lane_as_f32_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t lane_index, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_value_id_t* out_lane) {
  const uint32_t register_index = lane_index / 2u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, source_register_count, register_index,
      source_lane_type, &source_register));
  return loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
      context, source_op, source_register, lane_index % 2u, result_lane_type,
      out_lane);
}

iree_status_t loom_amdgpu_extract_f16_lane_as_low_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_register_count,
    uint32_t lane_index, loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t register_index = lane_index / 2u;
  const uint32_t register_bit_offset = (lane_index % 2u) * 16u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, source_register_count, register_index,
      lane_type, &source_register));
  if (register_bit_offset == 0) {
    *out_lane = source_register;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      register_bit_offset, source_register, lane_type, out_lane);
}

iree_status_t loom_amdgpu_pack_f32_lane_to_f16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, uint32_t register_lane, loom_type_t lane_type,
    loom_value_id_t* inout_packed) {
  loom_value_id_t half_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F16_F32, source_lane,
      lane_type, &half_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, half_lane, &half_lane));
  return loom_amdgpu_pack_bits_into_register(context, source_op, half_lane,
                                             register_lane * 16u, lane_type,
                                             inout_packed);
}

iree_status_t loom_amdgpu_splat_f32_lane_to_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor,
    loom_value_id_t source_lane, loom_type_t lane_type,
    loom_value_id_t* out_packed) {
  loom_value_id_t half_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F16_F32, source_lane,
      lane_type, &half_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, half_lane, &half_lane));

  return loom_amdgpu_emit_packed_u16_lane_pair(
      context, source_op, pack_u16_descriptor, half_lane, half_lane, lane_type,
      out_packed);
}
