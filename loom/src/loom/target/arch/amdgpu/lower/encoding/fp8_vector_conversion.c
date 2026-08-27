// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/encoding/fp8_vector_conversion.h"

#include <stdint.h>

#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/storage.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/float16.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/encoding/vector_conversion.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static loom_amdgpu_vector_extract_plan_t loom_amdgpu_vector_fp8_extract_plan(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return (loom_amdgpu_vector_extract_plan_t){
      .source = plan->storage_source,
      .result = plan->result,
      .lane_count = plan->lane_count,
      .register_count = plan->storage_register_count,
      .result_register_count = 1,
      .element_register_count = 1,
      .lane_bit_count = 8,
      .sign_extend_packed_lane = false,
  };
}

static iree_status_t loom_amdgpu_extract_vector_fp8_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_extract_plan_t* extract_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    uint32_t lane_index, loom_value_id_t* out_low_byte) {
  const uint64_t storage_lane =
      (uint64_t)plan->storage_lane_offset +
      (uint64_t)lane_index * (uint64_t)plan->storage_lane_stride;
  IREE_ASSERT_LE(storage_lane, UINT32_MAX);
  const uint32_t lanes_per_register = 32u / extract_plan->lane_bit_count;
  const uint32_t register_offset = (uint32_t)storage_lane / lanes_per_register;
  const uint32_t register_bit_offset =
      ((uint32_t)storage_lane % lanes_per_register) *
      extract_plan->lane_bit_count;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, extract_plan->register_count,
      register_offset, source_lane_type, &source_register));
  return loom_amdgpu_extract_vgpr_bitfield(
      context, source_op, source_register, register_bit_offset,
      extract_plan->lane_bit_count,
      LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_RAW_SHIFTED, source_lane_type,
      out_low_byte);
}

typedef struct loom_amdgpu_vector_fp8_decode_value_flag_cache_t {
  // Per-result-lane FP8 decode simplification flags.
  loom_amdgpu_fp8_decode_value_flags_t
      lane_flags[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
} loom_amdgpu_vector_fp8_decode_value_flag_cache_t;

static void loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_fp8_decode_value_flag_cache_t* out_cache) {
  *out_cache = (loom_amdgpu_vector_fp8_decode_value_flag_cache_t){0};
  IREE_ASSERT_LE(plan->lane_count, IREE_ARRAYSIZE(out_cache->lane_flags));
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) {
    return;
  }

  const loom_value_facts_t content_facts =
      loom_value_fact_table_lookup(fact_table, plan->content_fact_source);
  loom_value_facts_t all_equal_facts = {0};
  const bool has_all_equal_facts = loom_value_facts_query_all_equal_element(
      &fact_table->context, content_facts, &all_equal_facts);

  loom_value_fact_small_static_lanes_t small_lanes = {0};
  const bool has_small_lanes = loom_value_facts_query_small_static_lanes(
      &fact_table->context, content_facts, &small_lanes);
  for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
    const uint64_t storage_lane =
        (uint64_t)plan->storage_lane_offset +
        (uint64_t)lane_index * (uint64_t)plan->storage_lane_stride;
    loom_value_facts_t lane_facts = {0};
    if (has_small_lanes && storage_lane < small_lanes.count) {
      lane_facts = small_lanes.lanes[storage_lane];
    } else if (has_all_equal_facts) {
      lane_facts = all_equal_facts;
    } else {
      continue;
    }
    out_cache->lane_flags[lane_index] =
        loom_amdgpu_fp8_decode_value_flags_from_facts(lane_facts);
  }
}

static loom_amdgpu_fp8_decode_value_flags_t
loom_amdgpu_vector_fp8_decode_value_flags(
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* cache,
    uint32_t lane_index) {
  IREE_ASSERT_LT(lane_index, IREE_ARRAYSIZE(cache->lane_flags));
  return cache->lane_flags[lane_index];
}

static iree_status_t loom_amdgpu_ensure_fp8_software_decode(
    loom_low_lower_context_t* context,
    loom_value_fact_numeric_format_flags_t source_format,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_type_t* sgpr_type, loom_type_t* mask_type,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan) {
  if (loom_type_kind(*sgpr_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, sgpr_type));
  }
  if (loom_type_kind(*mask_type) != LOOM_TYPE_NONE && *decode_plan != NULL) {
    return iree_ok_status();
  }
  if (loom_type_kind(*mask_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, mask_type));
  }
  if (*decode_plan == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_decode_plan(
        context, source_format, descriptor_source_format, decode_plan));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_fp8_to_f32_software_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_extract_plan_t* extract_plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t lane_index,
    loom_value_id_t* out_low_lane) {
  *out_low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
      context, plan->source_format, plan->descriptor_source_format, sgpr_type,
      mask_type, decode_plan));

  loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_lane(
      context, source_op, plan, extract_plan, low_source, source_lane_type,
      lane_index, &low_byte));
  const loom_amdgpu_fp8_decode_value_flags_t value_flags =
      loom_amdgpu_vector_fp8_decode_value_flags(value_flag_cache, lane_index);
  return loom_amdgpu_emit_fp8_to_f32_lane(
      context, source_op, *decode_plan, low_byte, value_flags, result_lane_type,
      *sgpr_type, *mask_type, out_low_lane);
}

static iree_status_t loom_amdgpu_emit_fp8_lane_to_f32_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_byte, loom_type_t result_lane_type,
    loom_value_id_t* out_low_lane) {
  *out_low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, low_byte, &low_byte));
  return loom_amdgpu_emit_resolved_vgpr_unary(
      context, source_op, descriptor, low_byte, result_lane_type, out_low_lane);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_to_f32_native_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_extract_plan_t* extract_plan,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, uint32_t lane_index,
    loom_value_id_t* out_low_lane) {
  loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_lane(
      context, source_op, plan, extract_plan, low_source, source_lane_type,
      lane_index, &low_byte));
  return loom_amdgpu_emit_fp8_lane_to_f32_native(
      context, source_op, descriptor, low_byte, result_lane_type, out_low_lane);
}

typedef struct loom_amdgpu_vector_fp8_pair_storage_t {
  // Source register containing the selected adjacent FP8 byte pair.
  uint32_t source_register_index;
  // First FP8 byte offset within the source register.
  uint32_t byte_offset;
  // Number of live logical lanes consumed from this pair.
  uint32_t live_lane_count;
} loom_amdgpu_vector_fp8_pair_storage_t;

static bool loom_amdgpu_vector_fp8_query_storage_pair(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    uint32_t lane_index,
    loom_amdgpu_vector_fp8_pair_storage_t* out_pair_storage) {
  *out_pair_storage = (loom_amdgpu_vector_fp8_pair_storage_t){0};
  if (lane_index >= plan->lane_count) {
    return false;
  }

  const uint64_t storage_lane =
      (uint64_t)plan->storage_lane_offset +
      (uint64_t)lane_index * (uint64_t)plan->storage_lane_stride;
  const uint64_t next_storage_lane =
      (uint64_t)plan->storage_lane_offset +
      (uint64_t)(lane_index + 1u) * (uint64_t)plan->storage_lane_stride;
  if (next_storage_lane != storage_lane + 1u) {
    return false;
  }
  if (next_storage_lane >= plan->storage_lane_count) {
    return false;
  }

  const uint32_t register_index = (uint32_t)(storage_lane / 4u);
  const uint32_t byte_offset = (uint32_t)(storage_lane % 4u);
  if (byte_offset >= 3u || next_storage_lane / 4u != register_index) {
    return false;
  }
  IREE_ASSERT_LT(register_index, plan->storage_register_count);
  const uint32_t remaining_lane_count = plan->lane_count - lane_index;
  const uint32_t live_lane_count =
      remaining_lane_count < 2u ? remaining_lane_count : 2u;
  *out_pair_storage = (loom_amdgpu_vector_fp8_pair_storage_t){
      .source_register_index = register_index,
      .byte_offset = byte_offset,
      .live_lane_count = live_lane_count,
  };
  return true;
}

typedef struct loom_amdgpu_vector_fp8_octet_storage_t {
  // First source register containing the selected adjacent FP8 byte octet.
  uint32_t source_register_index;
} loom_amdgpu_vector_fp8_octet_storage_t;

static loom_amdgpu_vector_fp8_octet_storage_t
loom_amdgpu_vector_fp8_storage_octet(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    uint32_t lane_index) {
  IREE_ASSERT_LT(lane_index + 7u, plan->lane_count);
  IREE_ASSERT_EQ(plan->storage_lane_stride, 1u);
  const uint64_t storage_lane =
      (uint64_t)plan->storage_lane_offset + (uint64_t)lane_index;
  IREE_ASSERT_EQ(storage_lane & 3u, 0u);
  IREE_ASSERT_LT(storage_lane + 7u, plan->storage_lane_count);
  const uint64_t source_register_index = storage_lane / 4u;
  IREE_ASSERT_LT(source_register_index + 1u, plan->storage_register_count);
  IREE_ASSERT_LE(source_register_index, UINT32_MAX);
  return (loom_amdgpu_vector_fp8_octet_storage_t){
      .source_register_index = (uint32_t)source_register_index,
  };
}

static bool loom_amdgpu_vector_fp8_query_storage_octet(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    uint32_t lane_index,
    loom_amdgpu_vector_fp8_octet_storage_t* out_octet_storage) {
  *out_octet_storage = (loom_amdgpu_vector_fp8_octet_storage_t){0};
  if (lane_index + 7u >= plan->lane_count || plan->storage_lane_stride != 1u) {
    return false;
  }

  const uint64_t storage_lane =
      (uint64_t)plan->storage_lane_offset + (uint64_t)lane_index;
  if ((storage_lane & 3u) != 0 ||
      storage_lane + 7u >= plan->storage_lane_count) {
    return false;
  }

  const uint64_t source_register_index = storage_lane / 4u;
  if (source_register_index + 1u >= plan->storage_register_count ||
      source_register_index > UINT32_MAX) {
    return false;
  }
  *out_octet_storage = loom_amdgpu_vector_fp8_storage_octet(plan, lane_index);
  return true;
}

static iree_status_t loom_amdgpu_extract_vector_fp8_pair_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_pair_storage_t* pair_storage,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_value_id_t* out_source_register) {
  *out_source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, plan->storage_register_count,
      pair_storage->source_register_index, source_lane_type,
      out_source_register));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, *out_source_register, out_source_register));
  if (pair_storage->byte_offset == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      pair_storage->byte_offset * 8u, *out_source_register, source_lane_type,
      out_source_register);
}

static iree_status_t loom_amdgpu_extract_vector_fp8_octet_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_octet_storage_t* octet_storage,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t source_pair_type, loom_value_id_t* out_source_pair) {
  loom_value_id_t source_registers[2] = {LOOM_VALUE_ID_INVALID,
                                         LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(source_registers); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->storage_register_count,
        octet_storage->source_register_index + i, source_lane_type,
        &source_registers[i]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
        context, source_op, source_registers[i], &source_registers[i]));
  }

  loom_op_t* source_pair_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), source_registers,
      IREE_ARRAYSIZE(source_registers), source_pair_type, source_op->location,
      &source_pair_op));
  *out_source_pair =
      loom_value_slice_get(loom_low_op_results(source_pair_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vector_fp8_pair_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_pair_storage_t* pair_storage,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    const loom_value_id_t* extra_operands, uint32_t extra_operand_count,
    loom_type_t result_type, loom_value_id_t* out_converted_pair) {
  IREE_ASSERT_LE(extra_operand_count, 1u);
  *out_converted_pair = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_pair_register(
      context, source_op, plan, pair_storage, low_source, source_lane_type,
      &source_register));

  loom_value_id_t operands[2] = {source_register, LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < extra_operand_count; ++i) {
    operands[i + 1u] = extra_operands[i];
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, 1u + extra_operand_count,
      loom_named_attr_slice_empty(), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  *out_converted_pair = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_emit_vector_fp8_pair_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    const loom_value_id_t* extra_operands, uint32_t extra_operand_count,
    loom_type_t result_type, uint32_t lane_index,
    loom_amdgpu_vector_fp8_pair_storage_t* out_pair_storage,
    loom_value_id_t* out_converted_pair) {
  *out_converted_pair = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                 out_pair_storage)) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vector_fp8_pair_descriptor(
      context, source_op, plan, out_pair_storage, descriptor, low_source,
      source_lane_type, extra_operands, extra_operand_count, result_type,
      out_converted_pair);
}

static iree_status_t
loom_amdgpu_try_emit_vector_fp8_tail_byte_select_descriptor(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_fp8_native_descriptor_refs_t* native_refs,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_value_id_t* out_converted_lane) {
  *out_converted_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t lane_index = plan->lane_count - 1u;
  const uint64_t storage_lane =
      (uint64_t)plan->storage_lane_offset +
      (uint64_t)lane_index * (uint64_t)plan->storage_lane_stride;
  const uint32_t source_register_index =
      (uint32_t)(storage_lane / LOOM_AMDGPU_FP8_REGISTER_BYTE_COUNT);
  const uint32_t byte_selector =
      (uint32_t)(storage_lane % LOOM_AMDGPU_FP8_REGISTER_BYTE_COUNT);

  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      native_refs->byte_select[byte_selector];
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, plan->storage_register_count,
      source_register_index, source_lane_type, &source_register));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, source_register, &source_register));
  return loom_amdgpu_emit_resolved_vgpr_unary(context, source_op, &descriptor,
                                              source_register, result_lane_type,
                                              out_converted_lane);
}

static iree_status_t loom_amdgpu_extract_vector_fp8_native_pair_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t converted_pair, loom_type_t result_lane_type,
    uint32_t low_lane_count, loom_value_id_t* out_low_lanes) {
  for (uint32_t i = 0; i < low_lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, converted_pair, /*register_count=*/2,
        /*register_offset=*/i, result_lane_type, &out_low_lanes[i]));
  }
  return iree_ok_status();
}

static loom_amdgpu_fp8_decode_value_flags_t
loom_amdgpu_vector_fp8_pair_decode_value_flags(
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    uint32_t lane_index, uint32_t live_lane_count) {
  IREE_ASSERT_GE(live_lane_count, 1u);
  IREE_ASSERT_LE(live_lane_count, 2u);
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      loom_amdgpu_vector_fp8_decode_value_flags(value_flag_cache, lane_index);
  if (live_lane_count == 2u) {
    value_flags &= loom_amdgpu_vector_fp8_decode_value_flags(value_flag_cache,
                                                             lane_index + 1u);
  }
  return value_flags;
}

static bool loom_amdgpu_vector_fp8_query_storage_pair_set(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    uint32_t pair_count, loom_amdgpu_vector_fp8_pair_storage_t* pair_storage,
    loom_amdgpu_fp8_decode_value_flags_t* out_value_flags) {
  const uint32_t required_pair_count = (plan->lane_count + 1u) / 2u;
  if (required_pair_count == 0 ||
      required_pair_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS ||
      pair_count != required_pair_count ||
      plan->storage_register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }

  *out_value_flags = LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  bool have_value_flags = false;
  for (uint32_t register_index = 0; register_index < pair_count;
       ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    if (!loom_amdgpu_vector_fp8_query_storage_pair(
            plan, lane_base, &pair_storage[register_index])) {
      return false;
    }
    const loom_amdgpu_fp8_decode_value_flags_t pair_value_flags =
        loom_amdgpu_vector_fp8_pair_decode_value_flags(
            value_flag_cache, lane_base,
            pair_storage[register_index].live_lane_count);
    *out_value_flags = have_value_flags ? *out_value_flags & pair_value_flags
                                        : pair_value_flags;
    have_value_flags = true;
  }
  return true;
}

static iree_status_t loom_amdgpu_materialize_vector_fp8_pair_sources(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_pair_storage_t* pair_storage,
    uint32_t pair_count, loom_value_id_t low_source,
    loom_type_t source_lane_type,
    loom_amdgpu_fp8_packed_u16_pair_source_t* out_pair_sources) {
  loom_value_id_t source_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(source_registers); ++i) {
    source_registers[i] = LOOM_VALUE_ID_INVALID;
  }
  for (uint32_t register_index = 0; register_index < pair_count;
       ++register_index) {
    const uint32_t source_register_index =
        pair_storage[register_index].source_register_index;
    if (source_registers[source_register_index] == LOOM_VALUE_ID_INVALID) {
      loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, low_source, plan->storage_register_count,
          source_register_index, source_lane_type, &source_register));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
          context, source_op, source_register, &source_register));
      source_registers[source_register_index] = source_register;
    }
  }

  for (uint32_t register_index = 0; register_index < pair_count;
       ++register_index) {
    out_pair_sources[register_index] =
        (loom_amdgpu_fp8_packed_u16_pair_source_t){
            .source_register = source_registers[pair_storage[register_index]
                                                    .source_register_index],
            .byte_offset = pair_storage[register_index].byte_offset,
            .live_lane_count = pair_storage[register_index].live_lane_count,
        };
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_f32_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source, const loom_value_id_t* extra_operands,
    uint32_t extra_operand_count, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t result_pair_type,
    uint32_t lane_index, loom_value_id_t* out_low_lanes,
    uint32_t* out_low_lane_count) {
  out_low_lanes[0] = LOOM_VALUE_ID_INVALID;
  out_low_lanes[1] = LOOM_VALUE_ID_INVALID;
  *out_low_lane_count = 0;

  loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
  loom_value_id_t converted_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vector_fp8_pair_descriptor(
      context, source_op, plan, descriptor, low_source, source_lane_type,
      extra_operands, extra_operand_count, result_pair_type, lane_index,
      &pair_storage, &converted_pair));
  if (converted_pair == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  const uint32_t low_lane_count = pair_storage.live_lane_count;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_native_pair_lanes(
      context, source_op, converted_pair, result_lane_type, low_lane_count,
      out_low_lanes));
  *out_low_lane_count = low_lane_count;
  return iree_ok_status();
}

static bool loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return plan->scale_source != LOOM_VALUE_ID_INVALID;
}

static bool loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(plan) &&
         plan->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F32;
}

static bool loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(plan) &&
         plan->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0;
}

static bool loom_amdgpu_vector_fp8_scalef32_is_identity(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  if (!loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan)) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  uint32_t scale_bit_pattern = 0;
  const loom_type_t scale_type =
      loom_module_value_type(module, plan->scale_source);
  if (loom_amdgpu_type_is_f32(scale_type)) {
    return loom_amdgpu_value_as_f32_bit_pattern(
               module, fact_table, plan->scale_source, &scale_bit_pattern) &&
           scale_bit_pattern == LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS;
  }
  return loom_vector_static_rank1_lane_count(scale_type, LOOM_SCALAR_TYPE_F32,
                                             1) == 1 &&
         loom_amdgpu_source_lane_as_u32_bits(
             fact_table, module, plan->scale_source, 0, &scale_bit_pattern) &&
         scale_bit_pattern == LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS;
}

static void loom_amdgpu_vector_fp8_unscaled_plan(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan) {
  *out_plan = *plan;
  out_plan->scale_source = LOOM_VALUE_ID_INVALID;
  out_plan->scale_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
  out_plan->scale_group_element_count = 0;
}

static uint32_t loom_amdgpu_vector_fp8_e8m0_pk8_result_register_count(
    loom_scalar_type_t result_element_type) {
  return result_element_type == LOOM_SCALAR_TYPE_F32 ? 8u : 4u;
}

static uint32_t loom_amdgpu_vector_fp8_e8m0_pk8_scale_sel(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    uint32_t lane_index) {
  if (!loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan) ||
      plan->scale_group_element_count == 0 ||
      plan->scale_group_element_count >= plan->lane_count) {
    return 0;
  }
  const uint32_t scale_selector = lane_index / plan->scale_group_element_count;
  IREE_ASSERT_LE(scale_selector, 15u);
  return scale_selector;
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_e8m0_pk8_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_value_id_t low_scale,
    loom_type_t source_lane_type, loom_type_t result_lane_type,
    loom_value_id_t* out_low_registers, uint32_t* out_selected_count) {
  *out_selected_count = 0;
  IREE_ASSERT_TRUE(
      !loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(plan) ||
      loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan));
  const bool requires_full_selection =
      loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan);
  const uint32_t result_registers_per_octet =
      loom_amdgpu_vector_fp8_e8m0_pk8_result_register_count(
          plan->result_element_type);
  const uint32_t result_register_count =
      plan->result_element_type == LOOM_SCALAR_TYPE_F32
          ? plan->lane_count
          : plan->result_register_count;
  for (uint32_t i = 0; i < result_register_count; ++i) {
    out_low_registers[i] = LOOM_VALUE_ID_INVALID;
  }

  const loom_low_lower_resolved_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_e8m0_pk8_descriptor(
      context, plan->descriptor_source_format, plan->result_element_type,
      &descriptor));
  if (descriptor == NULL) {
    IREE_ASSERT_FALSE(requires_full_selection);
    return iree_ok_status();
  }
  if (plan->lane_count == 0 || (plan->lane_count & 7u) != 0) {
    IREE_ASSERT_FALSE(requires_full_selection);
    return iree_ok_status();
  }
  if (!requires_full_selection) {
    for (uint32_t lane_index = 0; lane_index < plan->lane_count;
         lane_index += 8u) {
      loom_amdgpu_vector_fp8_octet_storage_t octet_storage = {0};
      if (!loom_amdgpu_vector_fp8_query_storage_octet(plan, lane_index,
                                                      &octet_storage)) {
        return iree_ok_status();
      }
    }
  }

  loom_type_t source_pair_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &source_pair_type));
  loom_type_t result_octet_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, result_registers_per_octet, &result_octet_type));

  loom_value_id_t low_e8m0_scale = low_scale;
  loom_string_id_t scale_sel_name_id = LOOM_STRING_ID_INVALID;
  for (uint32_t lane_index = 0; lane_index < plan->lane_count;
       lane_index += 8u) {
    const loom_amdgpu_vector_fp8_octet_storage_t octet_storage =
        loom_amdgpu_vector_fp8_storage_octet(plan, lane_index);
    if (low_e8m0_scale == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          LOOM_AMDGPU_FP8_E8M0FNU_PACKED_IDENTITY_SCALE_BITS, result_lane_type,
          &low_e8m0_scale));
    }

    loom_value_id_t source_pair = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_octet_registers(
        context, source_op, plan, &octet_storage, low_source, source_lane_type,
        source_pair_type, &source_pair));

    const loom_value_id_t operands[] = {source_pair, low_e8m0_scale};
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    const uint32_t scale_sel =
        loom_amdgpu_vector_fp8_e8m0_pk8_scale_sel(plan, lane_index);
    if (scale_sel != 0) {
      if (scale_sel_name_id == LOOM_STRING_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, IREE_SV("scale_sel"),
                                                &scale_sel_name_id));
      }
      attrs[attr_count++] = (loom_named_attr_t){
          .name_id = scale_sel_name_id,
          .value = loom_attr_i64(scale_sel),
      };
    }
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &result_octet_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &low_op));
    const loom_value_id_t converted_octet =
        loom_value_slice_get(loom_low_op_results(low_op), 0);
    const uint32_t result_register_base =
        plan->result_element_type == LOOM_SCALAR_TYPE_F32 ? lane_index
                                                          : lane_index / 2u;
    for (uint32_t i = 0; i < result_registers_per_octet; ++i) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, converted_octet, result_registers_per_octet, i,
          result_lane_type, &out_low_registers[result_register_base + i]));
      ++(*out_selected_count);
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_try_lower_vector_fp8_identity_scalef32_to_packed_16bit_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_value_id_t* out_low_packets,
    uint32_t* out_selected_count) {
  *out_selected_count = 0;
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    out_low_packets[i] = LOOM_VALUE_ID_INVALID;
  }

  const loom_low_lower_resolved_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
      context, plan->descriptor_source_format, plan->result_element_type,
      &descriptor));
  if (descriptor == NULL) {
    return iree_ok_status();
  }

  loom_value_id_t low_scale_one = LOOM_VALUE_ID_INVALID;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    const uint32_t lane_index = register_index * 2u;
    loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
    if (!loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                   &pair_storage)) {
      continue;
    }
    if (low_scale_one == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS, result_lane_type,
          &low_scale_one));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_pair_descriptor(
        context, source_op, plan, &pair_storage, descriptor, low_source,
        source_lane_type, &low_scale_one, /*extra_operand_count=*/1,
        result_lane_type, &out_low_packets[register_index]));
    if (out_low_packets[register_index] == LOOM_VALUE_ID_INVALID) {
      IREE_ASSERT_UNREACHABLE("selected pair storage was already queried");
      IREE_BUILTIN_UNREACHABLE();
    }
    ++(*out_selected_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pair_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t lane_index,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
  if (!loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                 &pair_storage)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
      context, plan->source_format, plan->descriptor_source_format, sgpr_type,
      mask_type, decode_plan));

  const loom_amdgpu_fp8_decode_value_flags_t value_flags =
      loom_amdgpu_vector_fp8_pair_decode_value_flags(
          value_flag_cache, lane_index, pair_storage.live_lane_count);
  if (!loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(*decode_plan,
                                                    value_flags)) {
    return iree_ok_status();
  }

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_pair_register(
      context, source_op, plan, &pair_storage, low_source, source_lane_type,
      &source_register));

  const loom_amdgpu_fp8_packed_u16_pair_source_t pair_source = {
      .source_register = source_register,
      .byte_offset = pair_storage.byte_offset,
      .live_lane_count = pair_storage.live_lane_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
      context, source_op, *decode_plan, &pair_source, 1, value_flags,
      result_lane_type, *sgpr_type, *mask_type, out_low_packet));
  return iree_ok_status();
}

typedef enum loom_amdgpu_vector_fp8_packed_bf16_selection_flag_bits_e {
  LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE = 0u,
  LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED = 1u << 0,
} loom_amdgpu_vector_fp8_packed_bf16_selection_flag_bits_t;
typedef uint32_t loom_amdgpu_vector_fp8_packed_bf16_selection_flags_t;

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t pair_count,
    loom_amdgpu_vector_fp8_packed_bf16_selection_flags_t selection_flags,
    loom_value_id_t* out_low_packets) {
  for (uint32_t i = 0; i < pair_count; ++i) {
    out_low_packets[i] = LOOM_VALUE_ID_INVALID;
  }
  loom_amdgpu_vector_fp8_pair_storage_t
      pair_storage[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_query_storage_pair_set(
          plan, value_flag_cache, pair_count, pair_storage, &value_flags)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
      context, plan->source_format, plan->descriptor_source_format, sgpr_type,
      mask_type, decode_plan));
  if (!loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(*decode_plan,
                                                    value_flags)) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(
          selection_flags,
          LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED) &&
      !loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(*decode_plan,
                                                       value_flags)) {
    return iree_ok_status();
  }

  loom_amdgpu_fp8_packed_u16_pair_source_t
      pair_sources[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_vector_fp8_pair_sources(
      context, source_op, plan, pair_storage, pair_count, low_source,
      source_lane_type, pair_sources));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
      context, source_op, *decode_plan, pair_sources, pair_count, value_flags,
      result_lane_type, *sgpr_type, *mask_type, out_low_packets));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t pair_count,
    loom_value_id_t* out_low_packets) {
  for (uint32_t i = 0; i < pair_count; ++i) {
    out_low_packets[i] = LOOM_VALUE_ID_INVALID;
  }
  loom_amdgpu_vector_fp8_pair_storage_t
      pair_storage[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_query_storage_pair_set(
          plan, value_flag_cache, pair_count, pair_storage, &value_flags)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
      context, plan->source_format, plan->descriptor_source_format, sgpr_type,
      mask_type, decode_plan));
  if (!loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(*decode_plan,
                                                          value_flags)) {
    return iree_ok_status();
  }

  loom_amdgpu_fp8_packed_u16_pair_source_t
      pair_sources[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_vector_fp8_pair_sources(
      context, source_op, plan, pair_storage, pair_count, low_source,
      source_lane_type, pair_sources));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
      context, source_op, *decode_plan, pair_sources, pair_count, value_flags,
      result_lane_type, *sgpr_type, *mask_type, out_low_packets));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pair_to_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t lane_index,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
  if (!loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                 &pair_storage)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
      context, plan->source_format, plan->descriptor_source_format, sgpr_type,
      mask_type, decode_plan));
  const loom_amdgpu_fp8_decode_value_flags_t value_flags =
      loom_amdgpu_vector_fp8_pair_decode_value_flags(
          value_flag_cache, lane_index, pair_storage.live_lane_count);
  if (!loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(*decode_plan,
                                                          value_flags)) {
    return iree_ok_status();
  }

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_pair_register(
      context, source_op, plan, &pair_storage, low_source, source_lane_type,
      &source_register));
  const loom_amdgpu_fp8_packed_u16_pair_source_t pair_source = {
      .source_register = source_register,
      .byte_offset = pair_storage.byte_offset,
      .live_lane_count = pair_storage.live_lane_count,
  };
  return loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
      context, source_op, *decode_plan, &pair_source, 1, value_flags,
      result_lane_type, *sgpr_type, *mask_type, out_low_packet);
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pairs_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, loom_value_id_t* out_low_lanes) {
  out_low_lanes[0] = LOOM_VALUE_ID_INVALID;
  if (plan->lane_count == 0 ||
      plan->lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES ||
      (plan->lane_count & 1u) != 0) {
    return iree_ok_status();
  }

  const uint32_t pair_count = plan->lane_count / 2u;
  loom_value_id_t packed_bf16[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(packed_bf16); ++i) {
    packed_bf16[i] = LOOM_VALUE_ID_INVALID;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_bf16(
      context, source_op, plan, value_flag_cache, decode_plan, low_source,
      source_lane_type, result_lane_type, sgpr_type, mask_type, pair_count,
      LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE, packed_bf16));
  if (packed_bf16[0] == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  for (uint32_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
        context, source_op, packed_bf16[pair_index], 0, result_lane_type,
        &out_low_lanes[pair_index * 2u]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
        context, source_op, packed_bf16[pair_index], 1, result_lane_type,
        &out_low_lanes[pair_index * 2u + 1u]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_lower_vector_fp8_pair_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t* sgpr_type,
    loom_type_t* mask_type, uint32_t lane_index,
    loom_value_id_t* out_low_lanes) {
  out_low_lanes[0] = LOOM_VALUE_ID_INVALID;
  out_low_lanes[1] = LOOM_VALUE_ID_INVALID;

  if (lane_index + 1u >= plan->lane_count) {
    return iree_ok_status();
  }

  loom_value_id_t packed_bf16 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pair_to_packed_bf16(
      context, source_op, plan, value_flag_cache, decode_plan, low_source,
      source_lane_type, result_lane_type, sgpr_type, mask_type, lane_index,
      &packed_bf16));
  if (packed_bf16 == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
      context, source_op, packed_bf16, 0, result_lane_type, &out_low_lanes[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
      context, source_op, packed_bf16, 1, result_lane_type, &out_low_lanes[1]));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vector_fp8_to_f32_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_value_id_t* lanes) {
  IREE_ASSERT_TRUE(
      !loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(plan) ||
      loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan));

  for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
    lanes[lane_index] = LOOM_VALUE_ID_INVALID;
  }
  loom_value_id_t low_e8m0_scale = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
        context, source_op, plan, &low_e8m0_scale));
  }
  uint32_t e8m0_pk8_lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_e8m0_pk8_native(
      context, source_op, plan, low_source, low_e8m0_scale, source_lane_type,
      result_lane_type, lanes, &e8m0_pk8_lane_count));
  if (e8m0_pk8_lane_count == plan->lane_count) {
    return iree_ok_status();
  }
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_ASSERT_EQ(e8m0_pk8_lane_count, plan->lane_count);
    IREE_ASSERT_UNREACHABLE(
        "accepted AMDGPU E8M0 pk8 FP8 vector decode did not cover all lanes");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t mask_type = loom_type_none();
  loom_type_t sgpr_type = loom_type_none();
  const loom_amdgpu_vector_extract_plan_t extract_plan =
      loom_amdgpu_vector_fp8_extract_plan(plan);
  const loom_amdgpu_fp8_native_descriptors_t* native_descriptors = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
      context, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F32,
      &native_descriptors));
  const bool native_pair_present =
      native_descriptors != NULL &&
      iree_any_bit_set(native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR);
  const bool native_lane_present =
      native_descriptors != NULL &&
      iree_any_bit_set(native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_LANE);

  loom_type_t result_pair_type = loom_type_none();
  if (native_pair_present) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &result_pair_type));
  }

  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(context, plan,
                                                            &value_flag_cache);

  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
  if (!native_pair_present && !native_lane_present) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pairs_to_f32(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type, lanes));
    if (lanes[0] != LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }
  }

  for (uint32_t i = 0; i < plan->lane_count;) {
    if (lanes[i] != LOOM_VALUE_ID_INVALID) {
      ++i;
      continue;
    }
    if (native_pair_present) {
      loom_value_id_t native_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                         LOOM_VALUE_ID_INVALID};
      uint32_t native_lane_count = 0;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_f32_native(
              context, source_op, plan, &native_descriptors->pair_descriptor,
              low_source, /*extra_operands=*/NULL, /*extra_operand_count=*/0,
              source_lane_type, result_lane_type, result_pair_type, i,
              native_lanes, &native_lane_count));
      if (native_lanes[0] != LOOM_VALUE_ID_INVALID) {
        for (uint32_t lane_offset = 0; lane_offset < native_lane_count;
             ++lane_offset) {
          lanes[i + lane_offset] = native_lanes[lane_offset];
        }
        i += native_lane_count;
        continue;
      }
    }

    if (native_lane_present) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_fp8_to_f32_native_lane(
          context, source_op, plan, &extract_plan,
          &native_descriptors->lane_descriptor, low_source, source_lane_type,
          result_lane_type, i, &lanes[i]));
      ++i;
      continue;
    }

    loom_value_id_t packed_decode_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                              LOOM_VALUE_ID_INVALID};
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pair_to_f32(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type, i,
        packed_decode_lanes));
    if (packed_decode_lanes[0] != LOOM_VALUE_ID_INVALID) {
      lanes[i] = packed_decode_lanes[0];
      lanes[i + 1u] = packed_decode_lanes[1];
      i += 2u;
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_fp8_to_f32_software_lane(
        context, source_op, plan, &extract_plan, &value_flag_cache,
        &decode_plan, low_source, source_lane_type, result_lane_type,
        &sgpr_type, &mask_type, i, &lanes[i]));
    ++i;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vector_fp8_scalef32_to_f32_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_value_id_t low_scale,
    loom_type_t source_lane_type, loom_type_t result_lane_type,
    loom_value_id_t* lanes) {
  loom_amdgpu_vector_16bit_float_conversion_plan_t unscaled_plan;
  loom_amdgpu_vector_fp8_unscaled_plan(plan, &unscaled_plan);
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_to_f32_lanes(
      context, source_op, &unscaled_plan, low_source, source_lane_type,
      result_lane_type, lanes));
  for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
        lanes[lane_index], low_scale, result_lane_type, &lanes[lane_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_fp8_scalef32_to_f32_software_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_vector_extract_plan_t* extract_plan,
    const loom_amdgpu_vector_fp8_decode_value_flag_cache_t* value_flag_cache,
    const loom_amdgpu_fp8_decode_plan_t** decode_plan,
    loom_value_id_t low_source, loom_value_id_t low_scale,
    loom_type_t source_lane_type, loom_type_t result_lane_type,
    loom_type_t* sgpr_type, loom_type_t* mask_type, uint32_t lane_index,
    loom_value_id_t* out_low_lane) {
  loom_value_id_t unscaled_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_fp8_to_f32_software_lane(
      context, source_op, plan, extract_plan, value_flag_cache, decode_plan,
      low_source, source_lane_type, result_lane_type, sgpr_type, mask_type,
      lane_index, &unscaled_lane));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, unscaled_lane,
      low_scale, result_lane_type, out_low_lane);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_scalef32_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_TRUE(
      loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
      context, source_op, plan, &low_scale));

  const loom_low_lower_resolved_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
      context, plan->descriptor_source_format, plan->result_element_type,
      &descriptor));

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint32_t selected_lane_count = 0;
  for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
    lanes[lane_index] = LOOM_VALUE_ID_INVALID;
  }

  if (descriptor != NULL) {
    loom_type_t result_pair_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &result_pair_type));

    if (plan->lane_count == 2u) {
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      loom_value_id_t converted_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vector_fp8_pair_descriptor(
          context, source_op, plan, descriptor, low_source, source_lane_type,
          &low_scale, /*extra_operand_count=*/1, result_pair_type,
          /*lane_index=*/0, &pair_storage, &converted_pair));
      if (converted_pair != LOOM_VALUE_ID_INVALID) {
        return loom_low_lower_bind_value(context, plan->result, converted_pair);
      }
    }

    for (uint32_t lane_index = 0; lane_index < plan->lane_count;
         lane_index += 2u) {
      loom_value_id_t native_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                         LOOM_VALUE_ID_INVALID};
      uint32_t native_lane_count = 0;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_f32_native(
              context, source_op, plan, descriptor, low_source, &low_scale,
              /*extra_operand_count=*/1, source_lane_type, result_lane_type,
              result_pair_type, lane_index, native_lanes, &native_lane_count));
      if (native_lanes[0] == LOOM_VALUE_ID_INVALID) {
        continue;
      }
      for (uint32_t lane_offset = 0; lane_offset < native_lane_count;
           ++lane_offset) {
        lanes[lane_index + lane_offset] = native_lanes[lane_offset];
      }
      selected_lane_count += native_lane_count;
    }
    if (selected_lane_count == plan->lane_count) {
      return loom_amdgpu_bind_low_register_range(
          context, source_op, plan->result, lanes, plan->lane_count);
    }
  }

  if (selected_lane_count != 0) {
    loom_type_t mask_type = loom_type_none();
    loom_type_t sgpr_type = loom_type_none();
    const loom_amdgpu_vector_extract_plan_t extract_plan =
        loom_amdgpu_vector_fp8_extract_plan(plan);
    loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
    loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(
        context, plan, &value_flag_cache);
    const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
    for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
      if (lanes[lane_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lower_vector_fp8_scalef32_to_f32_software_lane(
              context, source_op, plan, &extract_plan, &value_flag_cache,
              &decode_plan, low_source, low_scale, source_lane_type,
              result_lane_type, &sgpr_type, &mask_type, lane_index,
              &lanes[lane_index]));
    }
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               lanes, plan->lane_count);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_scalef32_to_f32_lanes(
      context, source_op, plan, low_source, low_scale, source_lane_type,
      result_lane_type, lanes));
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lanes, plan->lane_count);
}

static iree_status_t loom_amdgpu_pack_f32_lanes_to_16bit_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_scalar_type_t result_element_type,
    const loom_amdgpu_float16_pack_descriptors_t* bf16_pack_descriptors,
    const loom_value_id_t* f32_lanes, uint32_t f32_lane_count,
    loom_type_t lane_type, loom_value_id_t* out_packed) {
  IREE_ASSERT_GE(f32_lane_count, 1u);
  IREE_ASSERT_LE(f32_lane_count, 2u);

  *out_packed = LOOM_VALUE_ID_INVALID;
  if (result_element_type == LOOM_SCALAR_TYPE_BF16) {
    loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
    if (f32_lane_count == 2u) {
      high_lane = f32_lanes[1];
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
          lane_type, &high_lane));
    }
    return loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
        context, source_op, bf16_pack_descriptors, f32_lanes[0], high_lane,
        lane_type, out_packed);
  }

  IREE_ASSERT_EQ(result_element_type, LOOM_SCALAR_TYPE_F16);
  for (uint32_t register_lane = 0; register_lane < f32_lane_count;
       ++register_lane) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lane_to_f16_register(
        context, source_op, f32_lanes[register_lane], register_lane, lane_type,
        out_packed));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_packed_16bit_from_f32_native(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t low_source, const loom_value_id_t* extra_operands,
    uint32_t extra_operand_count, loom_type_t source_lane_type,
    loom_type_t result_lane_type, loom_type_t result_pair_type,
    const loom_amdgpu_float16_pack_descriptors_t* bf16_pack_descriptors,
    uint32_t lane_index, loom_value_id_t* out_low_pair) {
  *out_low_pair = LOOM_VALUE_ID_INVALID;

  loom_value_id_t f32_lanes[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  uint32_t f32_lane_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_f32_native(
          context, source_op, plan, descriptor, low_source, extra_operands,
          extra_operand_count, source_lane_type, result_lane_type,
          result_pair_type, lane_index, f32_lanes, &f32_lane_count));
  if (f32_lanes[0] == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_amdgpu_pack_f32_lanes_to_16bit_register(
      context, source_op, plan->result_element_type, bf16_pack_descriptors,
      f32_lanes, f32_lane_count, result_lane_type, out_low_pair);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_scalef32_to_packed_16bit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_TRUE(
      loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
      context, source_op, plan, &low_scale));

  const loom_low_lower_resolved_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
      context, plan->descriptor_source_format, plan->result_element_type,
      &descriptor));

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  uint32_t missing_register_count = plan->result_register_count;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    packed_registers[register_index] = LOOM_VALUE_ID_INVALID;
  }

  if (descriptor != NULL) {
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      const uint32_t lane_index = register_index * 2u;
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vector_fp8_pair_descriptor(
          context, source_op, plan, descriptor, low_source, source_lane_type,
          &low_scale, /*extra_operand_count=*/1, result_lane_type, lane_index,
          &pair_storage, &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        --missing_register_count;
      }
    }
  }

  const loom_low_lower_resolved_descriptor_t* f32_descriptor = NULL;
  if (missing_register_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
        context, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F32,
        &f32_descriptor));
  }
  if (f32_descriptor != NULL) {
    loom_type_t result_pair_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &result_pair_type));
    const loom_amdgpu_float16_pack_descriptors_t* bf16_pack_descriptors = NULL;
    if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_get_float16_pack_descriptors(
          context, &bf16_pack_descriptors));
    }
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_packed_16bit_from_f32_native(
              context, source_op, plan, f32_descriptor, low_source, &low_scale,
              /*extra_operand_count=*/1, source_lane_type, result_lane_type,
              result_pair_type, bf16_pack_descriptors, register_index * 2u,
              &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        --missing_register_count;
      }
    }
  }

  if (missing_register_count == 0) {
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }

  if (missing_register_count == plan->result_register_count) {
    loom_value_id_t f32_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_scalef32_to_f32_lanes(
        context, source_op, plan, low_source, low_scale, source_lane_type,
        result_lane_type, f32_lanes));

    const loom_amdgpu_float16_pack_descriptors_t* bf16_pack_descriptors = NULL;
    if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_get_float16_pack_descriptors(
          context, &bf16_pack_descriptors));
    }
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      const uint32_t lane_base = register_index * 2u;
      const uint32_t remaining_lane_count = plan->lane_count - lane_base;
      const uint32_t register_lane_count =
          remaining_lane_count < 2u ? remaining_lane_count : 2u;
      IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lanes_to_16bit_register(
          context, source_op, plan->result_element_type, bf16_pack_descriptors,
          &f32_lanes[lane_base], register_lane_count, result_lane_type,
          &packed_registers[register_index]));
    }

    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }

  loom_type_t mask_type = loom_type_none();
  loom_type_t sgpr_type = loom_type_none();
  const loom_amdgpu_vector_extract_plan_t extract_plan =
      loom_amdgpu_vector_fp8_extract_plan(plan);
  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(context, plan,
                                                            &value_flag_cache);
  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;

  const loom_amdgpu_float16_pack_descriptors_t* bf16_pack_descriptors = NULL;
  if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_float16_pack_descriptors(
        context, &bf16_pack_descriptors));
  } else {
    IREE_ASSERT_EQ(plan->result_element_type, LOOM_SCALAR_TYPE_F16);
  }
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const uint32_t lane_base = register_index * 2u;
    loom_value_id_t f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                    LOOM_VALUE_ID_INVALID};
    const uint32_t remaining_lane_count = plan->lane_count - lane_base;
    const uint32_t register_lane_count =
        remaining_lane_count < 2u ? remaining_lane_count : 2u;
    for (uint32_t register_lane = 0; register_lane < register_lane_count;
         ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lower_vector_fp8_scalef32_to_f32_software_lane(
              context, source_op, plan, &extract_plan, &value_flag_cache,
              &decode_plan, low_source, low_scale, source_lane_type,
              result_lane_type, &sgpr_type, &mask_type, lane_index,
              &f32_lanes[register_lane]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lanes_to_16bit_register(
        context, source_op, plan->result_element_type, bf16_pack_descriptors,
        f32_lanes, register_lane_count, result_lane_type,
        &packed_registers[register_index]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan)) {
    return loom_amdgpu_lower_vector_fp8_scalef32_to_f32(
        context, source_op, plan, low_source, source_lane_type,
        result_lane_type);
  }

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_to_f32_lanes(
      context, source_op, plan, low_source, source_lane_type, result_lane_type,
      lanes));
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             lanes, plan->lane_count);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  loom_type_t mask_type = loom_type_none();
  loom_type_t sgpr_type = loom_type_none();

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  loom_value_id_t low_e8m0_scale = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
        context, source_op, plan, &low_e8m0_scale));
  }
  uint32_t e8m0_pk8_register_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_e8m0_pk8_native(
      context, source_op, plan, low_source, low_e8m0_scale, source_lane_type,
      result_lane_type, packed_registers, &e8m0_pk8_register_count));
  if (e8m0_pk8_register_count == plan->result_register_count) {
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_ASSERT_EQ(e8m0_pk8_register_count, plan->result_register_count);
    IREE_ASSERT_UNREACHABLE(
        "accepted AMDGPU E8M0 pk8 FP8 vector decode did not cover all lanes");
    IREE_BUILTIN_UNREACHABLE();
  }

  uint32_t identity_scalef32_register_count = 0;
  if (e8m0_pk8_register_count == 0) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_try_lower_vector_fp8_identity_scalef32_to_packed_16bit_native(
            context, source_op, plan, low_source, source_lane_type,
            result_lane_type, packed_registers,
            &identity_scalef32_register_count));
    if (identity_scalef32_register_count == plan->result_register_count) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  const loom_amdgpu_vector_extract_plan_t extract_plan =
      loom_amdgpu_vector_fp8_extract_plan(plan);
  const loom_amdgpu_fp8_native_descriptors_t* native_descriptors = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
      context, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F32,
      &native_descriptors));
  const bool native_pair_present =
      native_descriptors != NULL &&
      iree_any_bit_set(native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR);
  const bool native_lane_present =
      native_descriptors != NULL &&
      iree_any_bit_set(native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_LANE);

  loom_type_t result_pair_type = loom_type_none();
  if (native_pair_present) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &result_pair_type));
  }

  const loom_amdgpu_float16_pack_descriptors_t* bf16_pack_descriptors = NULL;
  if (native_pair_present || native_lane_present) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_float16_pack_descriptors(
        context, &bf16_pack_descriptors));
  }

  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(context, plan,
                                                            &value_flag_cache);

  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
  if (!native_pair_present && !native_lane_present &&
      identity_scalef32_register_count == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_bf16(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type,
        plan->result_register_count,
        LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE,
        packed_registers));
    if (packed_registers[0] != LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  } else if (identity_scalef32_register_count == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_bf16(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type,
        plan->result_register_count,
        LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED,
        packed_registers));
    if (packed_registers[0] != LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const uint32_t lane_base = register_index * 2u;
    if (native_pair_present) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_packed_16bit_from_f32_native(
              context, source_op, plan, &native_descriptors->pair_descriptor,
              low_source, /*extra_operands=*/NULL, /*extra_operand_count=*/0,
              source_lane_type, result_lane_type, result_pair_type,
              bf16_pack_descriptors, lane_base,
              &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
    }

    if (native_lane_present) {
      loom_value_id_t f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
      for (uint32_t register_lane = 0; register_lane < 2u; ++register_lane) {
        const uint32_t lane_index = lane_base + register_lane;
        if (lane_index >= plan->lane_count) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
              context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
              result_lane_type, &f32_lanes[register_lane]));
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_fp8_to_f32_native_lane(
            context, source_op, plan, &extract_plan,
            &native_descriptors->lane_descriptor, low_source, source_lane_type,
            result_lane_type, lane_index, &f32_lanes[register_lane]));
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, bf16_pack_descriptors, f32_lanes[0],
              f32_lanes[1], result_lane_type,
              &packed_registers[register_index]));
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pair_to_packed_bf16(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type, lane_base,
        &packed_registers[register_index]));
    if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
      continue;
    }

    loom_value_id_t lanes[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
    for (uint32_t register_lane = 0; register_lane < 2u; ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      if (lane_index >= plan->lane_count) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
            result_lane_type, &lanes[register_lane]));
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fp8_software_decode(
          context, plan->source_format, plan->descriptor_source_format,
          &sgpr_type, &mask_type, &decode_plan));
      loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_lane(
          context, source_op, plan, &extract_plan, low_source, source_lane_type,
          lane_index, &low_byte));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_bf16_lane(
          context, source_op, decode_plan, low_byte,
          loom_amdgpu_vector_fp8_decode_value_flags(&value_flag_cache,
                                                    lane_index),
          result_lane_type, sgpr_type, mask_type, &lanes[register_lane]));
    }
    const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
        iree_any_bit_set(decode_plan->flags,
                         LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16)
            ? &decode_plan->pack_u16_descriptor
            : NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_u16_lane_pair(
        context, source_op, pack_u16_descriptor, lanes[0], lanes[1],
        result_lane_type, &packed_registers[register_index]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

static iree_status_t loom_amdgpu_lower_vector_fp8_to_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES];
  uint32_t missing_register_count = plan->result_register_count;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    packed_registers[register_index] = LOOM_VALUE_ID_INVALID;
  }

  loom_value_id_t low_e8m0_scale = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
        context, source_op, plan, &low_e8m0_scale));
  }
  uint32_t e8m0_pk8_register_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_e8m0_pk8_native(
      context, source_op, plan, low_source, low_e8m0_scale, source_lane_type,
      result_lane_type, packed_registers, &e8m0_pk8_register_count));
  missing_register_count -= e8m0_pk8_register_count;
  if (missing_register_count == 0) {
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    IREE_ASSERT_EQ(e8m0_pk8_register_count, plan->result_register_count);
    IREE_ASSERT_UNREACHABLE(
        "accepted AMDGPU E8M0 pk8 FP8 vector decode did not cover all lanes");
    IREE_BUILTIN_UNREACHABLE();
  }

  const loom_amdgpu_fp8_native_descriptors_t* native_descriptors = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
      context, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F16,
      &native_descriptors));
  const bool native_pair_present =
      native_descriptors != NULL &&
      iree_any_bit_set(native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR);
  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  const bool native_byte_select_present =
      (plan->lane_count & 1u) != 0 && native_descriptors != NULL &&
      iree_any_bit_set(
          native_descriptors->flags,
          LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_BYTE_SELECT_FAMILY);
  if (native_byte_select_present) {
    const bool native_byte_select_row_present =
        loom_amdgpu_fp8_native_descriptor_refs(
            plan->descriptor_source_format, LOOM_SCALAR_TYPE_F16, &native_refs);
    IREE_ASSERT_TRUE(native_byte_select_row_present);
  }

  if (native_pair_present) {
    const uint32_t pair_register_count =
        plan->result_register_count - (native_byte_select_present ? 1u : 0u);
    for (uint32_t register_index = 0; register_index < pair_register_count;
         ++register_index) {
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_emit_vector_fp8_pair_descriptor(
          context, source_op, plan, &native_descriptors->pair_descriptor,
          low_source, source_lane_type, /*extra_operands=*/NULL,
          /*extra_operand_count=*/0, result_lane_type, register_index * 2u,
          &pair_storage, &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        --missing_register_count;
      }
    }
  }

  if (native_byte_select_present) {
    const uint32_t tail_register_index = plan->result_register_count - 1u;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_try_emit_vector_fp8_tail_byte_select_descriptor(
            context, source_op, plan, &native_refs, low_source,
            source_lane_type, result_lane_type,
            &packed_registers[tail_register_index]));
    --missing_register_count;
  }

  if (missing_register_count == 0) {
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }

  if (missing_register_count == plan->result_register_count) {
    uint32_t identity_scalef32_register_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_try_lower_vector_fp8_identity_scalef32_to_packed_16bit_native(
            context, source_op, plan, low_source, source_lane_type,
            result_lane_type, packed_registers,
            &identity_scalef32_register_count));
    missing_register_count -= identity_scalef32_register_count;
    if (missing_register_count == 0) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  const loom_amdgpu_fp8_native_descriptors_t* f32_native_descriptors = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
      context, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F32,
      &f32_native_descriptors));
  const bool f32_native_pair_present =
      f32_native_descriptors != NULL &&
      iree_any_bit_set(f32_native_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR);
  if (f32_native_pair_present && missing_register_count != 0) {
    loom_type_t result_pair_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_range_type(context, 2, &result_pair_type));
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_try_lower_vector_fp8_storage_pair_descriptor_to_packed_16bit_from_f32_native(
              context, source_op, plan,
              &f32_native_descriptors->pair_descriptor, low_source,
              /*extra_operands=*/NULL, /*extra_operand_count=*/0,
              source_lane_type, result_lane_type, result_pair_type,
              /*bf16_pack_descriptors=*/NULL, register_index * 2u,
              &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        --missing_register_count;
      }
    }
    if (missing_register_count == 0) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  loom_type_t mask_type = loom_type_none();
  loom_type_t sgpr_type = loom_type_none();
  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(context, plan,
                                                            &value_flag_cache);
  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
  if (missing_register_count == plan->result_register_count) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pairs_to_packed_f16(
        context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
        source_lane_type, result_lane_type, &sgpr_type, &mask_type,
        plan->result_register_count, packed_registers));
    if (packed_registers[0] != LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  if (missing_register_count != 0) {
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_try_lower_vector_fp8_pair_to_packed_f16(
          context, source_op, plan, &value_flag_cache, &decode_plan, low_source,
          source_lane_type, result_lane_type, &sgpr_type, &mask_type,
          register_index * 2u, &packed_registers[register_index]));
      if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
        --missing_register_count;
      }
    }
    if (missing_register_count == 0) {
      return loom_amdgpu_bind_low_register_range(context, source_op,
                                                 plan->result, packed_registers,
                                                 plan->result_register_count);
    }
  }

  if (missing_register_count == plan->result_register_count) {
    loom_value_id_t f32_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_to_f32_lanes(
        context, source_op, plan, low_source, source_lane_type,
        result_lane_type, f32_lanes));
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
      const uint32_t lane_base = register_index * 2u;
      for (uint32_t register_lane = 0; register_lane < 2u; ++register_lane) {
        const uint32_t lane_index = lane_base + register_lane;
        if (lane_index >= plan->lane_count) {
          break;
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lane_to_f16_register(
            context, source_op, f32_lanes[lane_index], register_lane,
            result_lane_type, &packed));
      }
      packed_registers[register_index] = packed;
    }
    return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                               packed_registers,
                                               plan->result_register_count);
  }

  const loom_amdgpu_vector_extract_plan_t extract_plan =
      loom_amdgpu_vector_fp8_extract_plan(plan);
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (packed_registers[register_index] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const uint32_t lane_base = register_index * 2u;
    const uint32_t remaining_lane_count = plan->lane_count - lane_base;
    const uint32_t register_lane_count =
        remaining_lane_count < 2u ? remaining_lane_count : 2u;
    loom_value_id_t f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                    LOOM_VALUE_ID_INVALID};
    for (uint32_t register_lane = 0; register_lane < 2u; ++register_lane) {
      const uint32_t lane_index = lane_base + register_lane;
      if (register_lane >= register_lane_count) {
        break;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_fp8_to_f32_software_lane(
          context, source_op, plan, &extract_plan, &value_flag_cache,
          &decode_plan, low_source, source_lane_type, result_lane_type,
          &sgpr_type, &mask_type, lane_index, &f32_lanes[register_lane]));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lanes_to_16bit_register(
        context, source_op, plan->result_element_type,
        /*bf16_pack_descriptors=*/NULL, f32_lanes, register_lane_count,
        result_lane_type, &packed_registers[register_index]));
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             packed_registers,
                                             plan->result_register_count);
}

typedef enum loom_amdgpu_vector_fp8_conversion_capability_bits_e {
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NONE = 0u,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_E8M0_PK8_RESULT = 1u << 0,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_RESULT_PAIR = 1u << 1,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_F32_PAIR = 1u << 2,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR = 1u << 3,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE = 1u << 4,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PAIR = 1u << 5,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F16_PAIR = 1u << 6,
  LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PACK = 1u << 7,
} loom_amdgpu_vector_fp8_conversion_capability_bits_t;
typedef uint32_t loom_amdgpu_vector_fp8_conversion_capabilities_t;

static bool loom_amdgpu_vector_fp8_plan_has_pair_storage(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  if (plan->lane_count == 0 || plan->storage_lane_stride != 1u) {
    return false;
  }
  const uint32_t pair_count = (plan->lane_count + 1u) / 2u;
  const uint32_t first_byte_offset = plan->storage_lane_offset & 3u;
  if (first_byte_offset == 3u ||
      (pair_count > 1u && (first_byte_offset & 1u) != 0)) {
    return false;
  }
  const uint64_t required_storage_lane_count =
      ((uint64_t)plan->lane_count + 1u) & ~UINT64_C(1);
  const uint64_t last_storage_lane =
      (uint64_t)plan->storage_lane_offset + required_storage_lane_count - 1u;
  return last_storage_lane < (uint64_t)plan->storage_lane_count &&
         last_storage_lane / 4u < (uint64_t)plan->storage_register_count;
}

static bool loom_amdgpu_vector_fp8_plan_has_octet_storage(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  if (plan->lane_count == 0 || (plan->lane_count & 7u) != 0 ||
      plan->storage_lane_stride != 1u ||
      (plan->storage_lane_offset & 3u) != 0) {
    return false;
  }
  const uint64_t last_storage_lane =
      (uint64_t)plan->storage_lane_offset + (uint64_t)plan->lane_count - 1u;
  return last_storage_lane < (uint64_t)plan->storage_lane_count &&
         last_storage_lane / 4u < (uint64_t)plan->storage_register_count;
}

static bool loom_amdgpu_vector_fp8_descriptor_set_has_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  return descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
         loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref);
}

static bool loom_amdgpu_vector_fp8_has_scalef32_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_numeric_format_flags_t source_format,
    loom_scalar_type_t result_element_type) {
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  return loom_amdgpu_fp8_scalef32_descriptor_ref(
             source_format, result_element_type, &descriptor_ref) &&
         loom_amdgpu_vector_fp8_descriptor_set_has_ref(descriptor_set,
                                                       descriptor_ref);
}

static bool loom_amdgpu_vector_fp8_has_e8m0_pk8_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_numeric_format_flags_t source_format,
    loom_scalar_type_t result_element_type) {
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  return loom_amdgpu_fp8_e8m0_pk8_descriptor_ref(
             source_format, result_element_type, &descriptor_ref) &&
         loom_amdgpu_vector_fp8_descriptor_set_has_ref(descriptor_set,
                                                       descriptor_ref);
}

static bool loom_amdgpu_vector_fp8_native_descriptor_set_refs(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_numeric_format_flags_t source_format,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_fp8_native_descriptor_refs_t* out_refs) {
  *out_refs = (loom_amdgpu_fp8_native_descriptor_refs_t){0};
  if (!loom_amdgpu_fp8_native_descriptor_refs(source_format,
                                              result_element_type, out_refs)) {
    return false;
  }
  out_refs->pair = loom_amdgpu_vector_fp8_descriptor_set_has_ref(descriptor_set,
                                                                 out_refs->pair)
                       ? out_refs->pair
                       : LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  out_refs->lane = loom_amdgpu_vector_fp8_descriptor_set_has_ref(descriptor_set,
                                                                 out_refs->lane)
                       ? out_refs->lane
                       : LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  return out_refs->pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE ||
         out_refs->lane != LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
}

static loom_amdgpu_vector_fp8_conversion_capabilities_t
loom_amdgpu_vector_fp8_conversion_capabilities(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities =
      LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NONE;
  const bool has_pair_storage =
      loom_amdgpu_vector_fp8_plan_has_pair_storage(plan);
  if (has_pair_storage && loom_amdgpu_vector_fp8_has_scalef32_descriptor(
                              descriptor_set, plan->descriptor_source_format,
                              plan->result_element_type)) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_RESULT_PAIR;
  }
  if (has_pair_storage && loom_amdgpu_vector_fp8_has_scalef32_descriptor(
                              descriptor_set, plan->descriptor_source_format,
                              LOOM_SCALAR_TYPE_F32)) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_F32_PAIR;
  }
  if (loom_amdgpu_vector_fp8_plan_has_octet_storage(plan) &&
      loom_amdgpu_vector_fp8_has_e8m0_pk8_descriptor(
          descriptor_set, plan->descriptor_source_format,
          plan->result_element_type)) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_E8M0_PK8_RESULT;
  }
  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  if (loom_amdgpu_vector_fp8_native_descriptor_set_refs(
          descriptor_set, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F32,
          &native_refs)) {
    if (has_pair_storage &&
        native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      capabilities |=
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR;
    }
    if (native_refs.lane != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      capabilities |=
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE;
    }
  }
  if (has_pair_storage &&
      loom_amdgpu_vector_fp8_native_descriptor_set_refs(
          descriptor_set, plan->descriptor_source_format, LOOM_SCALAR_TYPE_BF16,
          &native_refs) &&
      native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PAIR;
  }
  if (has_pair_storage &&
      loom_amdgpu_vector_fp8_native_descriptor_set_refs(
          descriptor_set, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F16,
          &native_refs) &&
      native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F16_PAIR;
  }
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32)) {
    capabilities |=
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PACK;
  }
  return capabilities;
}

static bool loom_amdgpu_vector_fp8_plan_packed_u16_decode_plan(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_fp8_decode_plan_t* out_decode_plan,
    loom_amdgpu_fp8_decode_value_flags_t* out_value_flags) {
  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(context, plan,
                                                            &value_flag_cache);
  loom_amdgpu_vector_fp8_pair_storage_t
      pair_storage[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_query_storage_pair_set(
          plan, &value_flag_cache, plan->result_register_count, pair_storage,
          &value_flags)) {
    return false;
  }
  loom_amdgpu_initialize_fp8_decode_plan_from_descriptor_set(
      loom_low_lower_context_descriptor_set(context), plan->source_format,
      plan->descriptor_source_format, out_decode_plan);
  *out_value_flags = value_flags;
  return true;
}

static bool loom_amdgpu_vector_fp8_try_packed_bf16_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_fp8_packed_bf16_selection_flags_t selection_flags,
    iree_string_view_t* out_plan_key) {
  *out_plan_key = iree_string_view_empty();
  loom_amdgpu_fp8_decode_plan_t decode_plan;
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_plan_packed_u16_decode_plan(
          context, plan, &decode_plan, &value_flags) ||
      !loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(&decode_plan,
                                                    value_flags)) {
    return false;
  }
  if (iree_any_bit_set(
          selection_flags,
          LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED) &&
      !loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(&decode_plan,
                                                       value_flags)) {
    return false;
  }
  const loom_amdgpu_fp8_packed_u16_repairs_t repairs =
      loom_amdgpu_fp8_pair_to_packed_bf16_repairs(&decode_plan, value_flags);
  *out_plan_key = loom_amdgpu_fp8_packed_bf16_strategy_key(
      loom_amdgpu_fp8_selects_exact_bf16_via_f16(&decode_plan, value_flags),
      repairs);
  return true;
}

static iree_string_view_t loom_amdgpu_vector_fp8_packed_bf16_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  iree_string_view_t plan_key = iree_string_view_empty();
  if (!loom_amdgpu_vector_fp8_try_packed_bf16_plan_key(
          context, plan, LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE,
          &plan_key)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp8_software_packed_bf16_decode");
  }
  return plan_key;
}

static iree_string_view_t loom_amdgpu_vector_fp8_packed_f16_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  loom_amdgpu_fp8_decode_plan_t decode_plan;
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_plan_packed_u16_decode_plan(
          context, plan, &decode_plan, &value_flags) ||
      !loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(&decode_plan,
                                                          value_flags)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp8_software_packed_f16_decode");
  }
  const loom_amdgpu_fp8_packed_u16_repairs_t repairs =
      loom_amdgpu_fp8_pair_to_packed_f16_repairs(&decode_plan, value_flags);
  return loom_amdgpu_fp8_packed_f16_repair_reason_key(repairs);
}

static iree_string_view_t loom_amdgpu_vector_fp8_e8m0_pk8_conversion_plan_key(
    loom_scalar_type_t result_element_type) {
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy.fp8_e8m0_pk8_f32");
    case LOOM_SCALAR_TYPE_BF16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy.fp8_e8m0_pk8_bf16");
    case LOOM_SCALAR_TYPE_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy.fp8_e8m0_pk8_f16");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t
loom_amdgpu_vector_fp8_scalef32_native_conversion_plan_key(
    loom_scalar_type_t result_element_type) {
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_pair");
    case LOOM_SCALAR_TYPE_BF16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_bf16_pair");
    case LOOM_SCALAR_TYPE_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f16_pair");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t
loom_amdgpu_vector_fp8_scalef32_native_f32_lane_conversion_plan_key(
    loom_scalar_type_t result_element_type) {
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_lane");
    case LOOM_SCALAR_TYPE_BF16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_lane_manual_bf16_pack");
    case LOOM_SCALAR_TYPE_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_lane_manual_f16_pack");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t
loom_amdgpu_vector_fp8_scalef32_native_f32_pair_to_16bit_conversion_plan_key(
    loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities,
    loom_scalar_type_t result_element_type) {
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_BF16:
      if (iree_any_bit_set(
              capabilities,
              LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PACK)) {
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "fp8_scalef32_native_f32_pair_native_bf16_pack");
      }
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_pair_manual_bf16_pack");
    case LOOM_SCALAR_TYPE_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_pair_manual_f16_pack");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t
loom_amdgpu_vector_fp8_scalef32_software_conversion_plan_key(
    loom_scalar_type_t result_element_type) {
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_software_f32_decode");
    case LOOM_SCALAR_TYPE_BF16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_software_packed_bf16_decode");
    case LOOM_SCALAR_TYPE_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_software_packed_f16_decode");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t
loom_amdgpu_vector_fp8_unscaled_f32_conversion_plan_key(
    loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities) {
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy.fp8_native_f32_pair");
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy.fp8_native_f32_lane");
  }
  return IREE_SV(
      "amdgpu.vector_16bit_float_conversion.strategy.fp8_software_f32_decode");
}

static iree_string_view_t
loom_amdgpu_vector_fp8_unscaled_bf16_conversion_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities) {
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PAIR)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy.fp8_native_bf16_pair");
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR |
              LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE)) {
    iree_string_view_t preferred_plan_key = iree_string_view_empty();
    if (loom_amdgpu_vector_fp8_try_packed_bf16_plan_key(
            context, plan,
            LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED,
            &preferred_plan_key)) {
      return preferred_plan_key;
    }
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR)) {
    if (iree_any_bit_set(
            capabilities,
            LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_BF16_PACK)) {
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_native_f32_pair_native_bf16_pack");
    }
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp8_native_f32_pair_manual_bf16_pack");
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp8_native_f32_lane_manual_bf16_pack");
  }
  return loom_amdgpu_vector_fp8_packed_bf16_plan_key(context, plan);
}

static iree_string_view_t
loom_amdgpu_vector_fp8_unscaled_f16_conversion_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities) {
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F16_PAIR)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy.fp8_native_f16_pair");
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR)) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp8_native_f32_pair_manual_f16_pack");
  }
  return loom_amdgpu_vector_fp8_packed_f16_plan_key(context, plan);
}

iree_string_view_t loom_amdgpu_vector_fp8_conversion_plan_key(
    loom_low_lower_context_t* context,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  const loom_amdgpu_vector_16bit_float_conversion_plan_t* fp8_plan = plan;
  loom_amdgpu_vector_16bit_float_conversion_plan_t unscaled_plan;
  if (loom_amdgpu_vector_fp8_scalef32_is_identity(context, plan)) {
    loom_amdgpu_vector_fp8_unscaled_plan(plan, &unscaled_plan);
    fp8_plan = &unscaled_plan;
  }
  const loom_amdgpu_vector_fp8_conversion_capabilities_t capabilities =
      loom_amdgpu_vector_fp8_conversion_capabilities(context, fp8_plan);
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(fp8_plan)) {
    if (iree_any_bit_set(
            capabilities,
            LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_E8M0_PK8_RESULT)) {
      return loom_amdgpu_vector_fp8_e8m0_pk8_conversion_plan_key(
          fp8_plan->result_element_type);
    }
    IREE_ASSERT_UNREACHABLE(
        "accepted AMDGPU E8M0 pk8 FP8 vector decode did not cover all lanes");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(fp8_plan)) {
    if (iree_any_bit_set(
            capabilities,
            LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_RESULT_PAIR)) {
      return loom_amdgpu_vector_fp8_scalef32_native_conversion_plan_key(
          fp8_plan->result_element_type);
    }
    const bool has_native_f32_pair = iree_any_bit_set(
        capabilities,
        LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_PAIR);
    const bool has_native_f32_lane =
        !has_native_f32_pair &&
        iree_any_bit_set(
            capabilities,
            LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_NATIVE_F32_LANE);
    if (fp8_plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
      if (has_native_f32_pair) {
        return loom_amdgpu_vector_fp8_scalef32_native_conversion_plan_key(
            fp8_plan->result_element_type);
      }
      if (has_native_f32_lane) {
        return loom_amdgpu_vector_fp8_scalef32_native_f32_lane_conversion_plan_key(
            fp8_plan->result_element_type);
      }
    }
    if (loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_16BIT_FLOAT,
                                      fp8_plan->result_element_type) &&
        (iree_any_bit_set(
             capabilities,
             LOOM_AMDGPU_VECTOR_FP8_CONVERSION_CAPABILITY_SCALEF32_F32_PAIR) ||
         has_native_f32_pair)) {
      return loom_amdgpu_vector_fp8_scalef32_native_f32_pair_to_16bit_conversion_plan_key(
          capabilities, fp8_plan->result_element_type);
    }
    if (loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_16BIT_FLOAT,
                                      fp8_plan->result_element_type) &&
        has_native_f32_lane) {
      return loom_amdgpu_vector_fp8_scalef32_native_f32_lane_conversion_plan_key(
          fp8_plan->result_element_type);
    }
    return loom_amdgpu_vector_fp8_scalef32_software_conversion_plan_key(
        fp8_plan->result_element_type);
  }
  switch (fp8_plan->result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return loom_amdgpu_vector_fp8_unscaled_f32_conversion_plan_key(
          capabilities);
    case LOOM_SCALAR_TYPE_BF16:
      return loom_amdgpu_vector_fp8_unscaled_bf16_conversion_plan_key(
          context, fp8_plan, capabilities);
    case LOOM_SCALAR_TYPE_F16:
      return loom_amdgpu_vector_fp8_unscaled_f16_conversion_plan_key(
          context, fp8_plan, capabilities);
    default:
      return iree_string_view_empty();
  }
}

iree_status_t loom_amdgpu_lower_vector_fp8_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  const loom_amdgpu_vector_16bit_float_conversion_plan_t* fp8_plan = plan;
  loom_amdgpu_vector_16bit_float_conversion_plan_t unscaled_plan;
  if (loom_amdgpu_vector_fp8_scalef32_is_identity(context, plan)) {
    loom_amdgpu_vector_fp8_unscaled_plan(plan, &unscaled_plan);
    fp8_plan = &unscaled_plan;
  }
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(fp8_plan)) {
    if (fp8_plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
      return loom_amdgpu_lower_vector_fp8_scalef32_to_f32(
          context, source_op, fp8_plan, low_source, source_lane_type,
          result_lane_type);
    }
    return loom_amdgpu_lower_vector_fp8_scalef32_to_packed_16bit(
        context, source_op, fp8_plan, low_source, source_lane_type,
        result_lane_type);
  }
  if (fp8_plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
    return loom_amdgpu_lower_vector_fp8_to_packed_bf16(
        context, source_op, fp8_plan, low_source, source_lane_type,
        result_lane_type);
  }
  if (fp8_plan->result_element_type == LOOM_SCALAR_TYPE_F16) {
    return loom_amdgpu_lower_vector_fp8_to_packed_f16(
        context, source_op, fp8_plan, low_source, source_lane_type,
        result_lane_type);
  }
  return loom_amdgpu_lower_vector_fp8_to_f32(context, source_op, fp8_plan,
                                             low_source, source_lane_type,
                                             result_lane_type);
}
