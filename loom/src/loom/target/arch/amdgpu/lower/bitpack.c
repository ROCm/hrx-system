// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/bitpack.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/storage.h"
#include "loom/target/arch/amdgpu/contracts/arithmetic_lower_rules.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static void loom_amdgpu_bitpack_plan_from_accepted_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_bitpack_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_bitpack_plan_t){0};
  IREE_ASSERT(loom_vector_bitpack_isa(source_op));
  const int64_t width = loom_vector_bitpack_width(source_op);
  const loom_value_id_t source = loom_vector_bitpack_source(source_op);
  const loom_value_id_t result = loom_vector_bitpack_result(source_op);
  loom_vector_packed_integer_payload_from_lanes_match_t match = {0};
  const bool matched = loom_vector_packed_integer_payload_from_lanes_match(
      loom_module_value_type(module, source),
      loom_module_value_type(module, result), (uint32_t)width,
      /*storage_unit_bit_count=*/32, LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS,
      &match);
  IREE_ASSERT(matched);
  IREE_ASSERT_EQ(match.source_shape.element_type, LOOM_SCALAR_TYPE_I32);
  IREE_ASSERT_EQ(match.result_shape.element_type, LOOM_SCALAR_TYPE_I8);
  IREE_ASSERT_EQ(match.result_shape.payload_bit_count % 32u, 0);
  IREE_ASSERT_NE(match.result_shape.storage_unit_count, 0);

  out_plan->source = source;
  out_plan->result = result;
  out_plan->width = (uint32_t)width;
  out_plan->lane_count = match.source_shape.lane_count;
  out_plan->result_register_count = match.result_shape.storage_unit_count;
}

static void loom_amdgpu_bitunpack_plan_from_accepted_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_bitunpack_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_bitunpack_plan_t){0};

  int64_t width = 0;
  if (loom_vector_bitunpacku_isa(source_op)) {
    width = loom_vector_bitunpacku_width(source_op);
    out_plan->source = loom_vector_bitunpacku_source(source_op);
    out_plan->result = loom_vector_bitunpacku_result(source_op);
    out_plan->is_signed = false;
  } else if (loom_vector_bitunpacks_isa(source_op)) {
    width = loom_vector_bitunpacks_width(source_op);
    out_plan->source = loom_vector_bitunpacks_source(source_op);
    out_plan->result = loom_vector_bitunpacks_result(source_op);
    out_plan->is_signed = true;
  } else {
    IREE_ASSERT_UNREACHABLE(
        "bitstream plan selector called for unsupported source op");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_vector_packed_integer_lanes_from_payload_match_t match = {0};
  const bool matched = loom_vector_packed_integer_lanes_from_payload_match(
      loom_module_value_type(module, out_plan->source),
      loom_module_value_type(module, out_plan->result), (uint32_t)width,
      /*storage_unit_bit_count=*/32, LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS,
      LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES, &match);
  IREE_ASSERT(matched);

  if (match.result_shape.element_type == LOOM_SCALAR_TYPE_I32) {
    out_plan->result_kind = LOOM_AMDGPU_BITUNPACK_RESULT_KIND_I32_LANES;
    out_plan->result_register_count = match.lane_count;
  } else {
    IREE_ASSERT_LE(width, 8);
    IREE_ASSERT_EQ(match.result_shape.element_type, LOOM_SCALAR_TYPE_I8);
    IREE_ASSERT_EQ(match.result_shape.payload_bit_count, match.lane_count * 8u);
    out_plan->result_kind = LOOM_AMDGPU_BITUNPACK_RESULT_KIND_PACKED_I8;
    out_plan->result_register_count = match.result_shape.storage_unit_count;
  }
  IREE_ASSERT_NE(out_plan->result_kind, LOOM_AMDGPU_BITUNPACK_RESULT_KIND_NONE);

  out_plan->width = (uint32_t)width;
  out_plan->source_register_count = match.source_shape.storage_unit_count;
  out_plan->lane_count = match.lane_count;
}

static iree_status_t loom_amdgpu_select_vector_bitstream_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_selected) {
  *out_selected = false;
  loom_low_lower_rule_selection_t selection = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_select_contract(
      context, &loom_amdgpu_arithmetic_lower_rule_set, source_op, &selection));
  *out_selected = selection.rule != NULL;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_bitpack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitpack_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_bitpack_plan_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_bitstream_contract(
      context, source_op, out_selected));
  if (*out_selected) {
    loom_amdgpu_bitpack_plan_from_accepted_op(
        loom_low_lower_context_module(context), source_op, out_plan);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_bitunpack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitunpack_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_bitunpack_plan_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_bitstream_contract(
      context, source_op, out_selected));
  if (*out_selected) {
    loom_amdgpu_bitunpack_plan_from_accepted_op(
        loom_low_lower_context_module(context), source_op, out_plan);
  }
  return iree_ok_status();
}

static uint32_t loom_amdgpu_low_mask(uint32_t width) {
  return width == 32 ? UINT32_MAX : (UINT32_C(1) << width) - 1u;
}

iree_status_t loom_amdgpu_pack_bits_into_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_bits, uint32_t bit_offset, loom_type_t lane_type,
    loom_value_id_t* inout_packed) {
  if (*inout_packed == LOOM_VALUE_ID_INVALID) {
    loom_value_id_t shifted = low_bits;
    if (bit_offset != 0) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          bit_offset, low_bits, lane_type, &shifted));
    }
    *inout_packed = shifted;
    return iree_ok_status();
  }

  if (bit_offset != 0) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    bool selected_lshl_add = false;
    // Packed lanes occupy disjoint bit ranges, so add and OR are equivalent
    // after shifting the incoming bits into their target lane.
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_lshl_add_u32(
        context, source_op, low_bits, *inout_packed, bit_offset, lane_type,
        &packed, &selected_lshl_add));
    if (selected_lshl_add) {
      *inout_packed = packed;
      return iree_ok_status();
    }
  }

  loom_value_id_t shifted = low_bits;
  if (bit_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        bit_offset, low_bits, lane_type, &shifted));
  }
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, *inout_packed,
      shifted, lane_type, inout_packed);
}

iree_status_t loom_amdgpu_pack_lane_bits_into_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, uint32_t width, uint32_t bit_offset,
    loom_type_t lane_type, loom_value_id_t* inout_packed) {
  loom_value_id_t low_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, source_lane,
      loom_amdgpu_low_mask(width), lane_type, &low_bits));
  return loom_amdgpu_pack_bits_into_register(
      context, source_op, low_bits, bit_offset, lane_type, inout_packed);
}

iree_status_t loom_amdgpu_lower_vector_bitpack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitpack_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  const uint32_t lanes_per_register = 32u / plan->width;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * lanes_per_register;
    for (uint32_t register_lane = 0; register_lane < lanes_per_register;
         ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      if (lane_index >= plan->lane_count) {
        break;
      }
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, lane_index, lane_type, &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_pack_lane_bits_into_register(
          context, source_op, source_lane, plan->width,
          register_lane * plan->width, lane_type, &packed));
    }
    packed_registers[register_index] = packed;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static bool loom_amdgpu_signed_bitunpack_lane_prefers_bfe(
    const loom_amdgpu_bitunpack_plan_t* plan, uint32_t source_bit_offset) {
  // The high byte of a 32-bit word already sign-extends with one ASHR.
  return plan->width == 8 && source_bit_offset < 24;
}

static bool loom_amdgpu_unsigned_bitunpack_lane_prefers_bfe(
    uint32_t source_bit_offset) {
  // Offset zero already extracts with a single mask packet.
  return source_bit_offset != 0;
}

static iree_status_t loom_amdgpu_emit_bitunpacku_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (source_bit_offset == 0 && plan->width == 32) {
    *out_lane = low_source;
    return iree_ok_status();
  }

  bool selected_sdwa = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_sdwa_extract(
      context, source_op, low_source, source_bit_offset, plan->width,
      LOOM_AMDGPU_VGPR_SDWA_EXTRACT_FLAG_NONE, lane_type, out_lane,
      &selected_sdwa));
  if (selected_sdwa) {
    return iree_ok_status();
  }

  if (loom_amdgpu_unsigned_bitunpack_lane_prefers_bfe(source_bit_offset)) {
    bool selected_bfe = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_bfe_extract(
        context, source_op, low_source, source_bit_offset, plan->width,
        LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_NONE, lane_type, out_lane,
        &selected_bfe));
    if (selected_bfe) {
      return iree_ok_status();
    }
  }

  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      source_bit_offset, low_source, lane_type, &shifted));

  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, shifted,
      loom_amdgpu_low_mask(plan->width), lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bitunpacks_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (source_bit_offset == 0 && plan->width == 32) {
    *out_lane = low_source;
    return iree_ok_status();
  }

  bool selected_sdwa = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_sdwa_extract(
      context, source_op, low_source, source_bit_offset, plan->width,
      LOOM_AMDGPU_VGPR_SDWA_EXTRACT_FLAG_SIGN_EXTEND, lane_type, out_lane,
      &selected_sdwa));
  if (selected_sdwa) {
    return iree_ok_status();
  }

  if (loom_amdgpu_signed_bitunpack_lane_prefers_bfe(plan, source_bit_offset)) {
    bool selected_bfe = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_b32_bfe_extract(
        context, source_op, low_source, source_bit_offset, plan->width,
        LOOM_AMDGPU_VGPR_BFE_EXTRACT_FLAG_SIGN_EXTEND, lane_type, out_lane,
        &selected_bfe));
    if (selected_bfe) {
      return iree_ok_status();
    }
  }

  loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      32u - source_bit_offset - plan->width, low_source, lane_type,
      &shifted_left));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
      32u - plan->width, shifted_left, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bitunpack_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  if (plan->is_signed) {
    return loom_amdgpu_emit_bitunpacks_lane(context, source_op, plan,
                                            low_source, source_bit_offset,
                                            lane_type, out_lane);
  }
  return loom_amdgpu_emit_bitunpacku_lane(context, source_op, plan, low_source,
                                          source_bit_offset, lane_type,
                                          out_lane);
}

static iree_status_t loom_amdgpu_source_bitunpack_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    uint32_t source_bit_offset, loom_type_t lane_type,
    loom_value_id_t* out_source_register,
    uint32_t* out_source_register_bit_offset) {
  const uint32_t source_register_index = source_bit_offset / 32u;
  *out_source_register_bit_offset = source_bit_offset & 31u;
  *out_source_register = low_source;
  if (plan->source_register_count == 1) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source,
                                    source_register_index, lane_type,
                                    out_source_register);
}

static iree_status_t loom_amdgpu_lower_vector_bitunpack_i32_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    loom_type_t lane_type) {
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    const uint32_t source_bit_offset = i * plan->width;
    loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
    uint32_t source_register_bit_offset = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_source_bitunpack_register(
        context, source_op, plan, low_source, source_bit_offset, lane_type,
        &source_register, &source_register_bit_offset));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bitunpack_lane(
        context, source_op, plan, source_register, source_register_bit_offset,
        lane_type, &lane_results[i]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lane_results, plan->lane_count);
}

static iree_status_t loom_amdgpu_lower_vector_bitunpack_packed_i8(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan, loom_value_id_t low_source,
    loom_type_t lane_type) {
  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * 4u;
    for (uint32_t register_lane = 0; register_lane < 4; ++register_lane) {
      const uint32_t lane_ordinal = lane_base + register_lane;
      if (lane_ordinal >= plan->lane_count) {
        break;
      }
      const uint32_t source_bit_offset = lane_ordinal * plan->width;
      loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
      uint32_t source_register_bit_offset = 0;
      IREE_RETURN_IF_ERROR(loom_amdgpu_source_bitunpack_register(
          context, source_op, plan, low_source, source_bit_offset, lane_type,
          &source_register, &source_register_bit_offset));
      loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bitunpack_lane(
          context, source_op, plan, source_register, source_register_bit_offset,
          lane_type, &lane));
      if (plan->is_signed) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_pack_lane_bits_into_register(
            context, source_op, lane, 8, register_lane * 8u, lane_type,
            &packed));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_pack_bits_into_register(
            context, source_op, lane, register_lane * 8u, lane_type, &packed));
      }
    }
    result_registers[register_index] = packed;
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_lower_vector_bitunpack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitunpack_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  switch (plan->result_kind) {
    case LOOM_AMDGPU_BITUNPACK_RESULT_KIND_I32_LANES:
      return loom_amdgpu_lower_vector_bitunpack_i32_lanes(
          context, source_op, plan, low_source, lane_type);
    case LOOM_AMDGPU_BITUNPACK_RESULT_KIND_PACKED_I8:
      return loom_amdgpu_lower_vector_bitunpack_packed_i8(
          context, source_op, plan, low_source, lane_type);
    case LOOM_AMDGPU_BITUNPACK_RESULT_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unsupported bitunpack result kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}
