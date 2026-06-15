// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/bitpack.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef enum loom_amdgpu_bitpack_rejection_e {
  LOOM_AMDGPU_BITPACK_REJECTION_NONE = 0,
  LOOM_AMDGPU_BITPACK_REJECTION_OP = 1,
  LOOM_AMDGPU_BITPACK_REJECTION_WIDTH = 2,
  LOOM_AMDGPU_BITPACK_REJECTION_SOURCE_TYPE = 3,
  LOOM_AMDGPU_BITPACK_REJECTION_RESULT_TYPE = 4,
  LOOM_AMDGPU_BITPACK_REJECTION_LANE_COUNT = 5,
  LOOM_AMDGPU_BITPACK_REJECTION_LANE_GROUP = 6,
} loom_amdgpu_bitpack_rejection_t;

typedef enum loom_amdgpu_bitunpack_rejection_e {
  LOOM_AMDGPU_BITUNPACK_REJECTION_NONE = 0,
  LOOM_AMDGPU_BITUNPACK_REJECTION_OP = 1,
  LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_RANGE = 2,
  LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_DWORD_DIVISOR = 3,
  LOOM_AMDGPU_BITUNPACK_REJECTION_SOURCE_STORAGE = 4,
  LOOM_AMDGPU_BITUNPACK_REJECTION_PAYLOAD_DIVISIBILITY = 5,
  LOOM_AMDGPU_BITUNPACK_REJECTION_LANE_COUNT = 6,
  LOOM_AMDGPU_BITUNPACK_REJECTION_RESULT_TYPE = 7,
} loom_amdgpu_bitunpack_rejection_t;

typedef struct loom_amdgpu_bitpack_rejection_key_t {
  // Rejection kind matched by this row.
  loom_amdgpu_bitpack_rejection_t rejection;
  // Stable diagnostic constraint key returned for the rejection kind.
  iree_string_view_t constraint_key;
} loom_amdgpu_bitpack_rejection_key_t;

static const loom_amdgpu_bitpack_rejection_key_t kAmdgpuBitpackRejectionKeys[] =
    {
        {
            .rejection = LOOM_AMDGPU_BITPACK_REJECTION_WIDTH,
            .constraint_key = IREE_SVL("bitpack.width"),
        },
        {
            .rejection = LOOM_AMDGPU_BITPACK_REJECTION_SOURCE_TYPE,
            .constraint_key = IREE_SVL("bitpack.source_type"),
        },
        {
            .rejection = LOOM_AMDGPU_BITPACK_REJECTION_RESULT_TYPE,
            .constraint_key = IREE_SVL("bitpack.result_type"),
        },
        {
            .rejection = LOOM_AMDGPU_BITPACK_REJECTION_LANE_COUNT,
            .constraint_key = IREE_SVL("bitpack.lane_count"),
        },
        {
            .rejection = LOOM_AMDGPU_BITPACK_REJECTION_LANE_GROUP,
            .constraint_key = IREE_SVL("bitpack.lane_group"),
        },
};

typedef struct loom_amdgpu_bitunpack_rejection_key_t {
  // Rejection kind matched by this row.
  loom_amdgpu_bitunpack_rejection_t rejection;
  // Stable diagnostic constraint key returned for the rejection kind.
  iree_string_view_t constraint_key;
} loom_amdgpu_bitunpack_rejection_key_t;

static const loom_amdgpu_bitunpack_rejection_key_t
    kAmdgpuBitunpackRejectionKeys[] = {
        {
            .rejection = LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_RANGE,
            .constraint_key = IREE_SVL("bitunpack.width_range"),
        },
        {
            .rejection = LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_DWORD_DIVISOR,
            .constraint_key = IREE_SVL("bitunpack.width_dword_divisor"),
        },
        {
            .rejection = LOOM_AMDGPU_BITUNPACK_REJECTION_SOURCE_STORAGE,
            .constraint_key = IREE_SVL("bitunpack.source_storage"),
        },
        {
            .rejection = LOOM_AMDGPU_BITUNPACK_REJECTION_PAYLOAD_DIVISIBILITY,
            .constraint_key = IREE_SVL("bitunpack.payload_divisibility"),
        },
        {
            .rejection = LOOM_AMDGPU_BITUNPACK_REJECTION_LANE_COUNT,
            .constraint_key = IREE_SVL("bitunpack.lane_count"),
        },
        {
            .rejection = LOOM_AMDGPU_BITUNPACK_REJECTION_RESULT_TYPE,
            .constraint_key = IREE_SVL("bitunpack.result_type"),
        },
};

static bool loom_amdgpu_bitpack_reject(
    loom_amdgpu_bitpack_rejection_t rejection,
    loom_amdgpu_bitpack_rejection_t* out_rejection) {
  if (out_rejection) {
    *out_rejection = rejection;
  }
  return false;
}

static bool loom_amdgpu_bitunpack_reject(
    loom_amdgpu_bitunpack_rejection_t rejection,
    loom_amdgpu_bitunpack_rejection_t* out_rejection) {
  if (out_rejection) {
    *out_rejection = rejection;
  }
  return false;
}

static bool loom_amdgpu_bitpack_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_bitpack_plan_t* out_plan,
    loom_amdgpu_bitpack_rejection_t* out_rejection) {
  *out_plan = (loom_amdgpu_bitpack_plan_t){0};
  if (out_rejection) {
    *out_rejection = LOOM_AMDGPU_BITPACK_REJECTION_NONE;
  }
  if (!loom_vector_bitpack_isa(source_op)) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_OP,
                                      out_rejection);
  }
  if (loom_vector_bitpack_width(source_op) != 8) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_WIDTH,
                                      out_rejection);
  }

  const loom_value_id_t source = loom_vector_bitpack_source(source_op);
  const loom_value_id_t result = loom_vector_bitpack_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const uint32_t source_lane_count =
      loom_amdgpu_vector_i32_lane_count(source_type);
  const uint32_t result_lane_count =
      loom_amdgpu_vector_i8_lane_count(result_type);
  if (source_lane_count == 0) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_SOURCE_TYPE,
                                      out_rejection);
  }
  if (result_lane_count == 0) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_RESULT_TYPE,
                                      out_rejection);
  }
  if (result_lane_count != source_lane_count) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_LANE_COUNT,
                                      out_rejection);
  }
  if ((result_lane_count % 4) != 0) {
    return loom_amdgpu_bitpack_reject(LOOM_AMDGPU_BITPACK_REJECTION_LANE_GROUP,
                                      out_rejection);
  }

  out_plan->source = source;
  out_plan->result = result;
  out_plan->result_register_count = result_lane_count / 4u;
  return true;
}

static bool loom_amdgpu_bitunpack_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_bitunpack_plan_t* out_plan,
    loom_amdgpu_bitunpack_rejection_t* out_rejection) {
  *out_plan = (loom_amdgpu_bitunpack_plan_t){0};
  if (out_rejection) {
    *out_rejection = LOOM_AMDGPU_BITUNPACK_REJECTION_NONE;
  }

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
    return loom_amdgpu_bitunpack_reject(LOOM_AMDGPU_BITUNPACK_REJECTION_OP,
                                        out_rejection);
  }

  if (width < 1 || width > 32) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_RANGE, out_rejection);
  }
  if ((32 % width) != 0) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_WIDTH_DWORD_DIVISOR, out_rejection);
  }

  const loom_type_t source_type =
      loom_module_value_type(module, out_plan->source);
  uint32_t source_payload_bit_count = 0;
  uint32_t source_register_count = 0;
  if (!loom_amdgpu_type_packed_integer_storage(
          source_type, &source_payload_bit_count, &source_register_count)) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_SOURCE_STORAGE, out_rejection);
  }
  if ((source_payload_bit_count % (uint32_t)width) != 0) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_PAYLOAD_DIVISIBILITY, out_rejection);
  }

  const uint32_t lane_count = source_payload_bit_count / (uint32_t)width;
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_LANE_COUNT, out_rejection);
  }

  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);
  const uint32_t i32_lane_count =
      loom_amdgpu_vector_i32_lane_count(result_type);
  if (i32_lane_count == lane_count) {
    out_plan->result_kind = LOOM_AMDGPU_BITUNPACK_RESULT_KIND_I32_LANES;
    out_plan->result_register_count = lane_count;
  } else {
    uint32_t result_payload_bit_count = 0;
    uint32_t result_register_count = 0;
    if (width > 8 ||
        loom_amdgpu_vector_i8_lane_count(result_type) != lane_count ||
        !loom_amdgpu_type_packed_integer_storage(
            result_type, &result_payload_bit_count, &result_register_count) ||
        result_payload_bit_count != lane_count * 8u) {
      return loom_amdgpu_bitunpack_reject(
          LOOM_AMDGPU_BITUNPACK_REJECTION_RESULT_TYPE, out_rejection);
    }
    out_plan->result_kind = LOOM_AMDGPU_BITUNPACK_RESULT_KIND_PACKED_I8;
    out_plan->result_register_count = result_register_count;
  }
  if (out_plan->result_kind == LOOM_AMDGPU_BITUNPACK_RESULT_KIND_NONE) {
    return loom_amdgpu_bitunpack_reject(
        LOOM_AMDGPU_BITUNPACK_REJECTION_RESULT_TYPE, out_rejection);
  }

  out_plan->width = (uint32_t)width;
  out_plan->source_register_count = source_register_count;
  out_plan->lane_count = lane_count;
  return true;
}

iree_status_t loom_amdgpu_select_vector_bitpack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitpack_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_bitpack_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan, NULL);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_bitunpack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_bitunpack_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_bitunpack_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan, NULL);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_pack_i8_bits_into_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_bits, uint32_t register_lane, loom_type_t lane_type,
    loom_value_id_t* inout_packed) {
  if (*inout_packed == LOOM_VALUE_ID_INVALID) {
    loom_value_id_t shifted = low_bits;
    if (register_lane != 0) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          register_lane * 8u, low_bits, lane_type, &shifted));
    }
    *inout_packed = shifted;
    return iree_ok_status();
  }

  if (register_lane != 0) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    bool selected_lshl_add = false;
    // Packed bytes occupy disjoint bit ranges, so add and OR are equivalent
    // after shifting the incoming low byte into its target lane.
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vgpr_lshl_add_u32(
        context, source_op, low_bits, *inout_packed, register_lane * 8u,
        lane_type, &packed, &selected_lshl_add));
    if (selected_lshl_add) {
      *inout_packed = packed;
      return iree_ok_status();
    }
  }

  loom_value_id_t shifted = low_bits;
  if (register_lane != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        register_lane * 8u, low_bits, lane_type, &shifted));
  }
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, *inout_packed,
      shifted, lane_type, inout_packed);
}

static iree_status_t loom_amdgpu_pack_i8_lane_into_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_lane, uint32_t register_lane, loom_type_t lane_type,
    loom_value_id_t* inout_packed) {
  loom_value_id_t low_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, source_lane,
      UINT32_C(255), lane_type, &low_bits));
  return loom_amdgpu_pack_i8_bits_into_register(
      context, source_op, low_bits, register_lane, lane_type, inout_packed);
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
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    const uint32_t lane_base = register_index * 4u;
    for (uint32_t lane_index = 0; lane_index < 4; ++lane_index) {
      loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, lane_base + lane_index, lane_type,
          &source_lane));
      IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_lane_into_register(
          context, source_op, source_lane, lane_index, lane_type, &packed));
    }
    packed_registers[register_index] = packed;
  }

  if (plan->result_register_count == 1) {
    return loom_low_lower_bind_value(context, plan->result,
                                     packed_registers[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context),
                            packed_registers, plan->result_register_count,
                            result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static uint32_t loom_amdgpu_low_mask(uint32_t width) {
  return width == 32 ? UINT32_MAX : (UINT32_C(1) << width) - 1u;
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
  if (plan->lane_count == 1) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bitunpack_lane(
        context, source_op, plan, low_source, 0, lane_type, &low_result));
    return loom_low_lower_bind_value(context, plan->result, low_result);
  }

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

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, plan->lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
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
        IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_lane_into_register(
            context, source_op, lane, register_lane, lane_type, &packed));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_bits_into_register(
            context, source_op, lane, register_lane, lane_type, &packed));
      }
    }
    result_registers[register_index] = packed;
  }

  if (plan->result_register_count == 1) {
    return loom_low_lower_bind_value(context, plan->result,
                                     result_registers[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context),
                            result_registers, plan->result_register_count,
                            result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
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

static iree_string_view_t loom_amdgpu_bitpack_rejection_key(
    loom_amdgpu_bitpack_rejection_t rejection) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kAmdgpuBitpackRejectionKeys);
       ++i) {
    const loom_amdgpu_bitpack_rejection_key_t* row =
        &kAmdgpuBitpackRejectionKeys[i];
    if (row->rejection == rejection) {
      return row->constraint_key;
    }
  }
  return IREE_SV("bitpack.shape");
}

static iree_string_view_t loom_amdgpu_bitunpack_rejection_key(
    loom_amdgpu_bitunpack_rejection_t rejection) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuBitunpackRejectionKeys); ++i) {
    const loom_amdgpu_bitunpack_rejection_key_t* row =
        &kAmdgpuBitunpackRejectionKeys[i];
    if (row->rejection == rejection) {
      return row->constraint_key;
    }
  }
  return IREE_SV("bitunpack.shape");
}

iree_status_t loom_amdgpu_low_legality_verify_vector_bitstream(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  switch (op->kind) {
    case LOOM_OP_VECTOR_BITPACK: {
      loom_amdgpu_bitpack_plan_t unused_plan = {0};
      loom_amdgpu_bitpack_rejection_t rejection =
          LOOM_AMDGPU_BITPACK_REJECTION_NONE;
      if (loom_amdgpu_bitpack_plan_from_op(module, op, &unused_plan,
                                           &rejection)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, loom_amdgpu_bitpack_rejection_key(rejection));
    }
    case LOOM_OP_VECTOR_BITUNPACKS:
    case LOOM_OP_VECTOR_BITUNPACKU: {
      loom_amdgpu_bitunpack_plan_t unused_plan = {0};
      loom_amdgpu_bitunpack_rejection_t rejection =
          LOOM_AMDGPU_BITUNPACK_REJECTION_NONE;
      if (loom_amdgpu_bitunpack_plan_from_op(module, op, &unused_plan,
                                             &rejection)) {
        return iree_ok_status();
      }
      return loom_amdgpu_low_legality_reject(
          context, op, loom_amdgpu_bitunpack_rejection_key(rejection));
    }
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}
