// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_narrow.h"

#include <stdint.h>

#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/types.h"

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_amdgpu_build_fragment_memory_packet(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         const loom_value_id_t* low_registers,
                                         uint16_t register_count,
                                         loom_type_t vgpr_type,
                                         loom_value_id_t* out_packet) {
  const loom_type_t packet_type =
      loom_low_register_carrier_type_with_unit_count(vgpr_type, register_count);
  return loom_amdgpu_build_low_register_range(context, source_op, low_registers,
                                              register_count, packet_type,
                                              out_packet);
}

IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD static iree_status_t
loom_amdgpu_emit_fragment_memory_uniform_narrowed_16bit_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t bit_pattern, uint16_t result_register_count,
    uint16_t packet_register_count, loom_type_t vgpr_type,
    loom_value_id_t* out_packet) {
  uint32_t packed_bit_pattern = bit_pattern;
  if (result_register_count > 1) {
    packed_bit_pattern |= packed_bit_pattern << 16;
  }
  loom_value_id_t low_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      packed_bit_pattern, vgpr_type, &low_register));
  loom_value_id_t
      low_registers[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS];
  for (uint16_t i = 0; i < packet_register_count; ++i) {
    low_registers[i] = low_register;
  }
  return loom_amdgpu_build_fragment_memory_packet(
      context, source_op, low_registers, packet_register_count, vgpr_type,
      out_packet);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_16bit_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_payload, uint16_t payload_register_count,
    uint16_t lane_index, loom_type_t vgpr_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint16_t source_register_index = lane_index / 2u;
  if (source_register_index >= payload_register_count) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment packed b16 lane");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t source_register = low_payload;
  if (payload_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_payload, source_register_index, vgpr_type,
        &source_register));
  }
  if ((lane_index & 1u) == 0) {
    *out_lane = source_register;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
      source_register, vgpr_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_f32_to_16bit_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_payload_form_t payload_form,
    const loom_amdgpu_float16_pack_descriptors_t* float16_pack_descriptors,
    loom_value_id_t low_source, uint16_t source_register_count,
    uint16_t register_index, loom_value_id_t low_scale, loom_type_t vgpr_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_register = low_source;
  if (source_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source, register_index, vgpr_type,
        &source_register));
  }
  if (low_scale != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
        source_register, low_scale, vgpr_type, &source_register));
  }
  switch (payload_form) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16:
      return loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
          context, source_op, float16_pack_descriptors, source_register,
          vgpr_type, out_lane);
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16:
      return loom_amdgpu_emit_f32_to_f16_lane_with_descriptors(
          context, source_op, float16_pack_descriptors, source_register,
          vgpr_type, out_lane);
    default:
      IREE_ASSERT_UNREACHABLE("selected AMDGPU narrowed store payload form");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_emit_fragment_memory_f16_to_f32_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t source_register_count,
    uint16_t register_index, loom_type_t vgpr_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_register = low_source;
  if (source_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source, register_index, vgpr_type,
        &source_register));
  }
  return loom_amdgpu_emit_vgpr_unary(context, source_op,
                                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16,
                                     source_register, vgpr_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_f32_pair_to_packed_16bit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_payload_form_t payload_form,
    const loom_amdgpu_float16_pack_descriptors_t* float16_pack_descriptors,
    loom_value_id_t low_source, uint16_t register_index, loom_type_t vgpr_type,
    loom_value_id_t low_scale, loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_low_slice(context, source_op, low_source, register_index,
                                 vgpr_type, &low_source_register));
  loom_value_id_t high_source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source, register_index + 1u, vgpr_type,
      &high_source_register));
  if (low_scale != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
        low_source_register, low_scale, vgpr_type, &low_source_register));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
        high_source_register, low_scale, vgpr_type, &high_source_register));
  }
  switch (payload_form) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16:
      return loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
          context, source_op, float16_pack_descriptors, low_source_register,
          high_source_register, vgpr_type, out_packed);
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16:
      return loom_amdgpu_emit_f32_pair_to_packed_f16_with_descriptors(
          context, source_op, float16_pack_descriptors, low_source_register,
          high_source_register, vgpr_type, out_packed);
    default:
      IREE_ASSERT_UNREACHABLE("selected AMDGPU narrowed store payload form");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_amdgpu_emit_fragment_memory_packed_16bit_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_payload,
    const loom_amdgpu_float16_pack_descriptors_t* float16_pack_descriptors,
    uint16_t register_index, uint16_t result_register_count,
    uint16_t packet_register_count, loom_value_id_t low_scale,
    loom_type_t vgpr_type, loom_value_id_t* out_packet) {
  *out_packet = LOOM_VALUE_ID_INVALID;
  if (plan->narrowed_result.uniform_bit_pattern !=
      LOOM_AMDGPU_FRAGMENT_MEMORY_UNIFORM_BIT_PATTERN_INVALID) {
    return loom_amdgpu_emit_fragment_memory_uniform_narrowed_16bit_packet(
        context, source_op, (uint16_t)plan->narrowed_result.uniform_bit_pattern,
        result_register_count, packet_register_count, vgpr_type, out_packet);
  }
  if (result_register_count == 1) {
    if (plan->narrowed_result.round_source != LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_emit_fragment_memory_f32_to_16bit_lane(
          context, source_op, plan->payload_form, float16_pack_descriptors,
          low_payload, plan->register_count, register_index, low_scale,
          vgpr_type, out_packet);
    }
    return loom_amdgpu_emit_fragment_memory_packed_16bit_lane(
        context, source_op, low_payload, plan->payload_register_count,
        register_index, vgpr_type, out_packet);
  }

  if (plan->narrowed_result.round_source == LOOM_VALUE_ID_INVALID) {
    loom_type_t packet_type = vgpr_type;
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
        context, packet_register_count, vgpr_type, &packet_type));
    if (register_index == 0 &&
        packet_register_count == plan->payload_register_count) {
      *out_packet = low_payload;
      return iree_ok_status();
    }
    return loom_amdgpu_emit_low_slice(context, source_op, low_payload,
                                      register_index / 2u, packet_type,
                                      out_packet);
  }

  loom_value_id_t
      packed_registers[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS] = {0};
  for (uint16_t i = 0; i < packet_register_count; ++i) {
    const uint16_t source_register_index = register_index + i * 2u;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_f32_pair_to_packed_16bit(
            context, source_op, plan->payload_form, float16_pack_descriptors,
            low_payload, source_register_index, vgpr_type, low_scale,
            &packed_registers[i]));
  }
  return loom_amdgpu_build_fragment_memory_packet(
      context, source_op, packed_registers, packet_register_count, vgpr_type,
      out_packet);
}

iree_status_t loom_amdgpu_emit_fragment_memory_f16_to_f32_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_payload,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_type_t vgpr_type, loom_value_id_t* out_packet) {
  *out_packet = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_EQ(packet->result_register_count, packet->packet_register_count);
  loom_value_id_t
      converted_registers[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS] = {
          0};
  for (uint16_t i = 0; i < packet->packet_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_f16_to_f32_lane(
        context, source_op, low_payload, plan->register_count,
        packet->register_index + i, vgpr_type, &converted_registers[i]));
  }
  return loom_amdgpu_build_fragment_memory_packet(
      context, source_op, converted_registers, packet->packet_register_count,
      vgpr_type, out_packet);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_fp8_source_byte(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_value_id_t low_source_packet, uint16_t packet_register_count,
    uint16_t byte_index, loom_type_t vgpr_type, loom_value_id_t* out_low_byte) {
  *out_low_byte = LOOM_VALUE_ID_INVALID;
  const uint16_t source_register_index = byte_index / 4u;
  if (source_register_index >= packet_register_count) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 fragment packet");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t low_source_register = low_source_packet;
  if (packet_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source_packet, source_register_index, vgpr_type,
        &low_source_register));
  }

  const uint16_t bit_offset = (byte_index & 3u) * 8u;
  if (bit_offset != 0 &&
      iree_any_bit_set(decode_plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32)) {
    loom_named_attr_t attrs[2] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("offset"), bit_offset,
                                    attrs, IREE_ARRAYSIZE(attrs), &attr_count));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("width"), 8, attrs,
                                    IREE_ARRAYSIZE(attrs), &attr_count));

    const loom_value_id_t operands[] = {low_source_register};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &decode_plan->bfe_u32_descriptor, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &vgpr_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &low_op));
    *out_low_byte = loom_value_slice_get(loom_low_op_results(low_op), 0);
    return iree_ok_status();
  }

  loom_value_id_t shifted_byte = low_source_register;
  if (bit_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        bit_offset, low_source_register, vgpr_type, &shifted_byte));
  }
  *out_low_byte = shifted_byte;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_fp8_source_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t source_register_index, loom_type_t vgpr_type,
    loom_value_id_t* out_source_register) {
  *out_source_register = LOOM_VALUE_ID_INVALID;
  if (source_register_index >= packet_register_count) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 fragment packet");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t source_register = low_source;
  if (packet_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source, source_register_index, vgpr_type,
        &source_register));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, source_register, &source_register));
  *out_source_register = source_register;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_shifted_fp8_pair_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index, loom_type_t vgpr_type,
    loom_value_id_t* out_source_register) {
  *out_source_register = LOOM_VALUE_ID_INVALID;
  const uint16_t byte_index =
      result_register_index * LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
  const uint16_t source_register_index = byte_index / 4u;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_register(
      context, source_op, low_source, packet_register_count,
      source_register_index, vgpr_type, out_source_register));

  const uint16_t bit_offset = (byte_index & 3u) * 8u;
  if (bit_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        bit_offset, *out_source_register, vgpr_type, out_source_register));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_scalef32_fp8_to_packed_bf16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_low_lower_resolved_descriptor_t* scalef32_bf16_descriptor,
    loom_value_id_t low_scale, loom_type_t vgpr_type,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_shifted_fp8_pair_register(
          context, source_op, low_source, packet_register_count,
          result_register_index, vgpr_type, &source_register));

  const loom_value_id_t operands[] = {source_register, low_scale};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, scalef32_bf16_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &vgpr_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &convert_op));
  *out_low_packet = loom_value_slice_get(loom_low_op_results(convert_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_scale_f32_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_scale, loom_type_t vgpr_type,
    loom_value_id_t* inout_low_lane, loom_value_id_t* inout_high_lane) {
  if (low_scale == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, *inout_low_lane,
      low_scale, vgpr_type, inout_low_lane));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
      *inout_high_lane, low_scale, vgpr_type, inout_high_lane);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_native_fp8_to_packed_bf16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_low_lower_resolved_descriptor_t* native_f32_pair_descriptor,
    loom_type_t native_f32_pair_type, loom_value_id_t low_scale,
    const loom_amdgpu_float16_pack_descriptors_t* bf16_pack_descriptors,
    loom_type_t vgpr_type, loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_shifted_fp8_pair_register(
          context, source_op, low_source, packet_register_count,
          result_register_index, vgpr_type, &source_register));

  const loom_value_id_t operands[] = {source_register};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, native_f32_pair_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &native_f32_pair_type, 1,
      /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &convert_op));
  const loom_value_id_t converted_pair =
      loom_value_slice_get(loom_low_op_results(convert_op), 0);
  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, converted_pair, 0, vgpr_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, converted_pair, 1, vgpr_type, &high_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_scale_f32_pair(
      context, source_op, low_scale, vgpr_type, &low_lane, &high_lane));
  return loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
      context, source_op, bf16_pack_descriptors, low_lane, high_lane, vgpr_type,
      out_low_packet);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_scale_packed_bf16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_packed, loom_value_id_t low_scale,
    const loom_amdgpu_float16_pack_descriptors_t* bf16_pack_descriptors,
    loom_type_t vgpr_type, loom_value_id_t* out_low_packed) {
  *out_low_packed = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
      context, source_op, low_packed, 0, vgpr_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
      context, source_op, low_packed, 1, vgpr_type, &high_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_scale_f32_pair(
      context, source_op, low_scale, vgpr_type, &low_lane, &high_lane));
  return loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
      context, source_op, bf16_pack_descriptors, low_lane, high_lane, vgpr_type,
      out_low_packed);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_scalef32_fp8_to_packed_f16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_low_lower_resolved_descriptor_t* scalef32_f16_descriptor,
    loom_value_id_t low_scale, loom_type_t vgpr_type,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_shifted_fp8_pair_register(
          context, source_op, low_source, packet_register_count,
          result_register_index, vgpr_type, &source_register));

  const loom_value_id_t operands[] = {source_register, low_scale};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, scalef32_f16_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &vgpr_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &convert_op));
  *out_low_packet = loom_value_slice_get(loom_low_op_results(convert_op), 0);
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_identity_e8m0_pk8_fp8_to_16bit_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_packet, uint16_t packet_register_count,
    uint16_t result_register_count,
    const loom_low_lower_resolved_descriptor_t* e8m0_pk8_descriptor,
    loom_value_id_t low_identity_scale, loom_type_t vgpr_type,
    loom_value_id_t* out_low_result_registers) {
  if (result_register_count == 0 || result_register_count % 4u != 0 ||
      packet_register_count < result_register_count / 2u) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 E8M0 pk8 fragment packet");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t source_pair_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &source_pair_type));
  loom_type_t result_quad_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 4, &result_quad_type));

  for (uint16_t result_register_index = 0;
       result_register_index < result_register_count;
       result_register_index += 4u) {
    const uint16_t source_register_index = result_register_index / 2u;
    loom_value_id_t source_registers[2] = {LOOM_VALUE_ID_INVALID,
                                           LOOM_VALUE_ID_INVALID};
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_register(
        context, source_op, low_source_packet, packet_register_count,
        source_register_index, vgpr_type, &source_registers[0]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_register(
        context, source_op, low_source_packet, packet_register_count,
        source_register_index + 1u, vgpr_type, &source_registers[1]));

    loom_op_t* source_pair_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        loom_low_lower_context_builder(context), source_registers,
        IREE_ARRAYSIZE(source_registers), source_pair_type, source_op->location,
        &source_pair_op));
    const loom_value_id_t source_pair =
        loom_value_slice_get(loom_low_op_results(source_pair_op), 0);

    const loom_value_id_t operands[] = {source_pair, low_identity_scale};
    loom_op_t* convert_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, e8m0_pk8_descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_named_attr_slice_empty(), &result_quad_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &convert_op));
    const loom_value_id_t converted_quad =
        loom_value_slice_get(loom_low_op_results(convert_op), 0);
    for (uint16_t quad_index = 0; quad_index < 4u; ++quad_index) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, converted_quad, quad_index, vgpr_type,
          &out_low_result_registers[result_register_index + quad_index]));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_native_fp8_to_packed_f16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_low_lower_resolved_descriptor_t* native_f16_pair_descriptor,
    loom_type_t vgpr_type, loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_shifted_fp8_pair_register(
          context, source_op, low_source, packet_register_count,
          result_register_index, vgpr_type, &source_register));

  const loom_value_id_t operands[] = {source_register};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, native_f16_pair_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &vgpr_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &convert_op));
  *out_low_packet = loom_value_slice_get(loom_low_op_results(convert_op), 0);
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_full_fp8_to_packed_bf16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_decode_value_flags_t decode_value_flags,
    loom_type_t vgpr_type, loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_elements[LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT] =
      {0};
  for (uint16_t element_index = 0;
       element_index < LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
       ++element_index) {
    const uint16_t byte_index =
        result_register_index * LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT +
        element_index;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_byte(
        context, source_op, decode_plan, low_source, packet_register_count,
        byte_index, vgpr_type, &low_elements[element_index]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_bf16_lane(
        context, source_op, decode_plan, low_elements[element_index],
        decode_value_flags, vgpr_type, sgpr_type, mask_type,
        &low_elements[element_index]));
  }

  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(decode_plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16)
          ? &decode_plan->pack_u16_descriptor
          : NULL;
  return loom_amdgpu_emit_packed_u16_lane_pair(
      context, source_op, pack_u16_descriptor, low_elements[0], low_elements[1],
      vgpr_type, out_low_packet);
}

static iree_status_t loom_amdgpu_fragment_memory_prepare_fp8_pair_sources(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_packet, uint16_t packet_register_count,
    uint16_t result_register_count, loom_type_t vgpr_type,
    loom_amdgpu_fp8_packed_u16_pair_source_t* out_pair_sources) {
  loom_value_id_t
      source_registers[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS] = {0};
  for (uint16_t source_register_index = 0;
       source_register_index < packet_register_count; ++source_register_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_register(
        context, source_op, low_source_packet, packet_register_count,
        source_register_index, vgpr_type,
        &source_registers[source_register_index]));
  }

  for (uint16_t result_register_index = 0;
       result_register_index < result_register_count; ++result_register_index) {
    const uint16_t byte_index =
        result_register_index * LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
    const uint16_t source_register_index = byte_index / 4u;
    if (source_register_index >= packet_register_count) {
      IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 fragment packet");
      IREE_BUILTIN_UNREACHABLE();
    }
    out_pair_sources[result_register_index] =
        (loom_amdgpu_fp8_packed_u16_pair_source_t){
            .source_register = source_registers[source_register_index],
            .byte_offset = byte_index & 3u,
            .live_lane_count = 2u,
        };
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_fp8_to_packed_bf16_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_packet, uint16_t packet_register_count,
    uint16_t result_register_count,
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_packed_bf16_strategy_t strategy,
    loom_amdgpu_fp8_packed_u16_repairs_t repairs, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_result_registers) {
  loom_amdgpu_fp8_packed_u16_pair_source_t
      pair_sources[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_prepare_fp8_pair_sources(
      context, source_op, low_source_packet, packet_register_count,
      result_register_count, vgpr_type, pair_sources));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
      context, source_op, decode_plan, pair_sources, result_register_count,
      strategy, repairs, vgpr_type, sgpr_type, mask_type,
      out_low_result_registers));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_fp8_to_packed_f16_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_packet, uint16_t packet_register_count,
    uint16_t result_register_count,
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_packed_f16_strategy_t strategy,
    loom_amdgpu_fp8_packed_u16_repairs_t repairs, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_result_registers) {
  loom_amdgpu_fp8_packed_u16_pair_source_t
      pair_sources[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_prepare_fp8_pair_sources(
      context, source_op, low_source_packet, packet_register_count,
      result_register_count, vgpr_type, pair_sources));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
      context, source_op, decode_plan, pair_sources, result_register_count,
      strategy, repairs, vgpr_type, sgpr_type, mask_type,
      out_low_result_registers));
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fragment_memory_fp8_to_packed_16bit_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_fragment_memory_address_state_t* address_state,
    loom_value_id_t low_packet_resource, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t low_soffset,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_fragment_memory_address_t address;
  const loom_amdgpu_fp8_decode_action_kind_t decode_kind =
      plan->fp8_load_decode.kind;
  const loom_amdgpu_fp8_decode_value_flags_t decode_value_flags =
      loom_amdgpu_fp8_decode_action_value_flags(&plan->fp8_load_decode, 0);
  const loom_scalar_type_t result_element_type =
      loom_amdgpu_fragment_memory_load_fp8_result_element_type(
          plan->payload_form);
  const bool use_identity_e8m0_pk8_descriptor =
      decode_kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_BF16 ||
      decode_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F16;
  const bool use_scalef32_descriptor =
      decode_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR ||
      decode_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F16_PAIR;
  const bool use_native_f16_pair =
      decode_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F16_PAIR;
  const bool use_native_f32_pair =
      decode_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR ||
      decode_kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR_BF16_PACK;
  const bool use_packed_bf16 =
      decode_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_NORMAL ||
      decode_kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_REPAIR ||
      decode_kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_VIA_F16;
  const bool use_packed_f16 =
      decode_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_NORMAL ||
      decode_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_EXACT_REPAIR;
  const bool use_full_bf16 =
      decode_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16;

  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
  if (use_native_f32_pair || use_packed_bf16 || use_packed_f16 ||
      use_full_bf16) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_decode_plan(
        context, plan->view_element_format, plan->descriptor_source_format,
        &decode_plan));
  }
  const loom_low_lower_resolved_descriptor_t* scalef32_descriptor = NULL;
  if (use_scalef32_descriptor) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
        context, plan->descriptor_source_format, result_element_type,
        &scalef32_descriptor));
    IREE_ASSERT(scalef32_descriptor != NULL);
  }
  const loom_low_lower_resolved_descriptor_t* e8m0_pk8_descriptor = NULL;
  if (use_identity_e8m0_pk8_descriptor) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_e8m0_pk8_descriptor(
        context, plan->descriptor_source_format, result_element_type,
        &e8m0_pk8_descriptor));
    IREE_ASSERT(e8m0_pk8_descriptor != NULL);
  }
  const bool has_fp8_load_scale =
      plan->fp8_load_scale_source != LOOM_VALUE_ID_INVALID;
  const loom_amdgpu_fp8_native_descriptors_t* native_f16_descriptors = NULL;
  if (use_native_f16_pair) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
        context, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F16,
        &native_f16_descriptors));
    IREE_ASSERT(native_f16_descriptors != NULL);
    IREE_ASSERT(
        iree_any_bit_set(native_f16_descriptors->flags,
                         LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR));
  }
  const loom_amdgpu_fp8_native_descriptors_t* native_f32_descriptors = NULL;
  if (use_native_f32_pair) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
        context, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F32,
        &native_f32_descriptors));
    IREE_ASSERT(native_f32_descriptors != NULL);
    IREE_ASSERT(
        iree_any_bit_set(native_f32_descriptors->flags,
                         LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR));
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
      context, source_op, plan, packet->register_index,
      /*element_index=*/0, packet->descriptor_ref, address_state, vgpr_type,
      &address));

  loom_type_t packet_type = vgpr_type;
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
      context, packet->packet_register_count, vgpr_type, &packet_type));
  loom_value_id_t low_source_packet = LOOM_VALUE_ID_INVALID;
  const uint32_t vector_lane_count =
      (uint32_t)packet->result_register_count *
      LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
      context, source_op, layout, plan, packet, /*element_index=*/0,
      vector_lane_count, packet_type, &address, low_packet_resource,
      low_soffset, &low_source_packet));
  if (packet->result_register_count == 1) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_low_subword_load_packet(
            context, source_op, low_source_packet, vgpr_type,
            &low_source_packet));
  }

  loom_type_t sgpr_type = loom_type_none();
  loom_type_t native_f32_pair_type = loom_type_none();
  loom_amdgpu_float16_pack_descriptors_t bf16_pack_descriptors = {0};
  if (use_packed_bf16 || use_packed_f16 || use_full_bf16) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  }
  if (use_native_f32_pair) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &native_f32_pair_type));
  }
  const bool scale_after_decode =
      has_fp8_load_scale && !use_scalef32_descriptor && !use_native_f32_pair;
  if (use_native_f32_pair || scale_after_decode) {
    bf16_pack_descriptors = (loom_amdgpu_float16_pack_descriptors_t){
        .flags =
            (decode_kind ==
                     LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR_BF16_PACK
                 ? LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE_BF16
                 : LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_NONE) |
            (iree_any_bit_set(decode_plan->flags,
                              LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16)
                 ? LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16
                 : LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_NONE) |
            (iree_any_bit_set(
                 decode_plan->flags,
                 LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_ADD3_SRC2_LITERAL)
                 ? LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_HAS_ADD3_SRC2_LITERAL
                 : LOOM_AMDGPU_FLOAT16_PACK_DESCRIPTOR_FLAG_NONE),
        .native_bf16_descriptor = decode_plan->native_bf16_pack_descriptor,
        .pack_u16_descriptor = decode_plan->pack_u16_descriptor,
        .add3_src2_literal_descriptor =
            decode_plan->add3_src2_literal_descriptor,
    };
  }
  loom_value_id_t low_conversion_scale = LOOM_VALUE_ID_INVALID;
  if (has_fp8_load_scale) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, plan->fp8_load_scale_source, &low_conversion_scale));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
        context, source_op, low_conversion_scale, &low_conversion_scale));
  } else if (use_scalef32_descriptor) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS, vgpr_type,
        &low_conversion_scale));
  }
  loom_value_id_t low_identity_e8m0_scale = LOOM_VALUE_ID_INVALID;
  if (use_identity_e8m0_pk8_descriptor) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        LOOM_AMDGPU_FP8_E8M0FNU_PACKED_IDENTITY_SCALE_BITS, vgpr_type,
        &low_identity_e8m0_scale));
  }

  loom_value_id_t
      low_result_registers[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(low_result_registers); ++i) {
    low_result_registers[i] = LOOM_VALUE_ID_INVALID;
  }
  switch (decode_kind) {
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_BF16:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F16: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_identity_e8m0_pk8_fp8_to_16bit_packet(
              context, source_op, low_source_packet,
              packet->packet_register_count, packet->result_register_count,
              e8m0_pk8_descriptor, low_identity_e8m0_scale, vgpr_type,
              low_result_registers));
      break;
    }
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F16_PAIR:
      for (uint16_t result_register_index = 0;
           result_register_index < packet->result_register_count;
           ++result_register_index) {
        if (decode_kind ==
            LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR) {
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_emit_fragment_memory_scalef32_fp8_to_packed_bf16_register(
                  context, source_op, low_source_packet,
                  packet->packet_register_count, result_register_index,
                  scalef32_descriptor, low_conversion_scale, vgpr_type,
                  &low_result_registers[result_register_index]));
        } else {
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_emit_fragment_memory_scalef32_fp8_to_packed_f16_register(
                  context, source_op, low_source_packet,
                  packet->packet_register_count, result_register_index,
                  scalef32_descriptor, low_conversion_scale, vgpr_type,
                  &low_result_registers[result_register_index]));
        }
      }
      break;
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F16_PAIR:
      for (uint16_t result_register_index = 0;
           result_register_index < packet->result_register_count;
           ++result_register_index) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_memory_native_fp8_to_packed_f16_register(
                context, source_op, low_source_packet,
                packet->packet_register_count, result_register_index,
                &native_f16_descriptors->pair_descriptor, vgpr_type,
                &low_result_registers[result_register_index]));
      }
      break;
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR_BF16_PACK:
      for (uint16_t result_register_index = 0;
           result_register_index < packet->result_register_count;
           ++result_register_index) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_memory_native_fp8_to_packed_bf16_register(
                context, source_op, low_source_packet,
                packet->packet_register_count, result_register_index,
                &native_f32_descriptors->pair_descriptor, native_f32_pair_type,
                low_conversion_scale, &bf16_pack_descriptors, vgpr_type,
                &low_result_registers[result_register_index]));
      }
      break;
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_NORMAL:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_REPAIR:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_VIA_F16: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_fp8_to_packed_bf16_packet(
              context, source_op, low_source_packet,
              packet->packet_register_count, packet->result_register_count,
              decode_plan,
              loom_amdgpu_fp8_decode_action_packed_bf16_strategy(
                  &plan->fp8_load_decode),
              (loom_amdgpu_fp8_packed_u16_repairs_t)
                  plan->fp8_load_decode.detail_flags,
              vgpr_type, sgpr_type, mask_type, low_result_registers));
      break;
    }
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_NORMAL:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_EXACT_REPAIR: {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_fp8_to_packed_f16_packet(
              context, source_op, low_source_packet,
              packet->packet_register_count, packet->result_register_count,
              decode_plan,
              loom_amdgpu_fp8_decode_action_packed_f16_strategy(
                  &plan->fp8_load_decode),
              (loom_amdgpu_fp8_packed_u16_repairs_t)
                  plan->fp8_load_decode.detail_flags,
              vgpr_type, sgpr_type, mask_type, low_result_registers));
      break;
    }
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16:
      for (uint16_t result_register_index = 0;
           result_register_index < packet->result_register_count;
           ++result_register_index) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_memory_full_fp8_to_packed_bf16_register(
                context, source_op, low_source_packet,
                packet->packet_register_count, result_register_index,
                decode_plan, decode_value_flags, vgpr_type, sgpr_type,
                mask_type, &low_result_registers[result_register_index]));
      }
      break;
    default:
      IREE_ASSERT_UNREACHABLE("selected FP8 fragment decode kind");
      IREE_BUILTIN_UNREACHABLE();
  }
  if (scale_after_decode) {
    for (uint16_t result_register_index = 0;
         result_register_index < packet->result_register_count;
         ++result_register_index) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_scale_packed_bf16_register(
              context, source_op, low_result_registers[result_register_index],
              low_conversion_scale, &bf16_pack_descriptors, vgpr_type,
              &low_result_registers[result_register_index]));
    }
  }
  return loom_amdgpu_build_fragment_memory_packet(
      context, source_op, low_result_registers, packet->result_register_count,
      vgpr_type, out_low_packet);
}
