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

typedef struct loom_amdgpu_vector_fp8_selection_state_t {
  // Active low-lowering context.
  loom_low_lower_context_t* context;
  // Vector conversion plan being selected.
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan;
  // Physical result actions not yet selected.
  uint32_t missing_action_count;
  // Whether software decode resources have been initialized.
  bool software_resources_initialized;
  // Per-result-lane FP8 decode simplification flags.
  loom_amdgpu_vector_fp8_decode_value_flag_cache_t value_flag_cache;
  // Target-supported software FP8 decode operations.
  loom_amdgpu_fp8_decode_plan_t decode_plan;
} loom_amdgpu_vector_fp8_selection_state_t;

static void loom_amdgpu_vector_fp8_selection_require_software_resources(
    loom_amdgpu_vector_fp8_selection_state_t* state) {
  if (state->software_resources_initialized) {
    return;
  }
  loom_amdgpu_vector_fp8_decode_value_flag_cache_initialize(
      state->context, state->plan, &state->value_flag_cache);
  loom_amdgpu_initialize_fp8_decode_plan_from_descriptor_set(
      loom_low_lower_context_descriptor_set(state->context),
      state->plan->source_format, state->plan->descriptor_source_format,
      &state->decode_plan);
  state->software_resources_initialized = true;
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, plan->storage_register_count,
      pair_storage->source_register_index, source_lane_type, &source_register));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, source_register, &source_register));
  if (pair_storage->byte_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        pair_storage->byte_offset * 8u, source_register, source_lane_type,
        &source_register));
  }

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

static iree_status_t loom_amdgpu_emit_vector_fp8_tail_byte_select_descriptor(
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

static bool loom_amdgpu_vector_fp8_query_uniform_packed_pair_value_flags(
    loom_amdgpu_vector_fp8_selection_state_t* state,
    loom_amdgpu_fp8_decode_value_flags_t* out_value_flags) {
  const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  const uint32_t pair_count = plan->result_register_count;
  const uint32_t required_pair_count = (plan->lane_count + 1u) / 2u;
  if (required_pair_count == 0 ||
      required_pair_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS ||
      pair_count != required_pair_count ||
      plan->storage_register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }

  loom_amdgpu_vector_fp8_pair_storage_t
      pair_storage[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  for (uint32_t register_index = 0; register_index < pair_count;
       ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    if (!loom_amdgpu_vector_fp8_query_storage_pair(
            plan, lane_base, &pair_storage[register_index])) {
      return false;
    }
  }

  loom_amdgpu_vector_fp8_selection_require_software_resources(state);
  for (uint32_t register_index = 0; register_index < pair_count;
       ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    const loom_amdgpu_fp8_decode_value_flags_t pair_value_flags =
        loom_amdgpu_vector_fp8_pair_decode_value_flags(
            &state->value_flag_cache, lane_base,
            pair_storage[register_index].live_lane_count);
    *out_value_flags = register_index == 0
                           ? pair_value_flags
                           : *out_value_flags & pair_value_flags;
  }
  return true;
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

static iree_status_t loom_amdgpu_emit_vector_fp8_e8m0_pk8(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_value_id_t low_scale,
    loom_type_t source_lane_type, loom_type_t result_lane_type,
    loom_value_id_t* out_low_registers) {
  IREE_ASSERT_TRUE(
      !loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(plan) ||
      loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan));
  IREE_ASSERT_GT(plan->lane_count, 0u);
  IREE_ASSERT_EQ(plan->lane_count & 7u, 0u);
  const uint32_t result_registers_per_octet =
      plan->result_element_type == LOOM_SCALAR_TYPE_F32 ? 8u : 4u;
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
  IREE_ASSERT(descriptor != NULL);

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
    }
  }
  return iree_ok_status();
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

typedef enum loom_amdgpu_vector_fp8_packed_bf16_selection_flag_bits_e {
  LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE = 0u,
  LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED = 1u << 0,
} loom_amdgpu_vector_fp8_packed_bf16_selection_flag_bits_t;
typedef uint32_t loom_amdgpu_vector_fp8_packed_bf16_selection_flags_t;

static uint32_t loom_amdgpu_vector_fp8_decode_action_count(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  return plan->result_element_type == LOOM_SCALAR_TYPE_F32
             ? plan->lane_count
             : plan->result_register_count;
}

static void loom_amdgpu_vector_fp8_set_decode_action(
    loom_amdgpu_vector_fp8_selection_state_t* state, uint32_t action_index,
    uint32_t action_span, loom_amdgpu_fp8_decode_action_t action) {
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  const uint32_t action_count =
      loom_amdgpu_vector_fp8_decode_action_count(plan);
  IREE_ASSERT_GE(action_span, 1u);
  IREE_ASSERT_LE(action_index + action_span, action_count);
  IREE_ASSERT_GE(state->missing_action_count, action_span);
  IREE_ASSERT_EQ(plan->strategy.fp8_decode.actions[action_index].kind,
                 LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE);
  plan->strategy.fp8_decode.actions[action_index] = action;
  for (uint32_t i = 1; i < action_span; ++i) {
    IREE_ASSERT_EQ(plan->strategy.fp8_decode.actions[action_index + i].kind,
                   LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE);
    plan->strategy.fp8_decode.actions[action_index + i].kind =
        LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_CONTINUATION;
  }
  state->missing_action_count -= action_span;
}

static loom_amdgpu_fp8_decode_action_t
loom_amdgpu_vector_fp8_full_decode_action(
    loom_amdgpu_vector_fp8_selection_state_t* state,
    loom_amdgpu_fp8_decode_action_kind_t kind, uint32_t lane_base,
    uint32_t lane_count) {
  IREE_ASSERT_GE(lane_count, 1u);
  IREE_ASSERT_LE(lane_count, 2u);
  loom_amdgpu_vector_fp8_selection_require_software_resources(state);
  loom_amdgpu_fp8_decode_action_t action = {
      .kind = kind,
      .value_flags = loom_amdgpu_vector_fp8_decode_value_flags(
          &state->value_flag_cache, lane_base),
  };
  if (lane_count == 2u) {
    const loom_amdgpu_fp8_decode_value_flags_t high_value_flags =
        loom_amdgpu_vector_fp8_decode_value_flags(&state->value_flag_cache,
                                                  lane_base + 1u);
    action.value_flags |=
        (loom_amdgpu_fp8_decode_value_flags_t)(high_value_flags << 4u);
  }
  return action;
}

static bool loom_amdgpu_vector_fp8_select_packed_pair_action(
    loom_amdgpu_vector_fp8_selection_state_t* state,
    loom_scalar_type_t result_element_type, uint32_t lane_index,
    loom_amdgpu_fp8_decode_action_t* out_action) {
  const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
  if (!loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                 &pair_storage)) {
    return false;
  }
  loom_amdgpu_vector_fp8_selection_require_software_resources(state);
  const loom_amdgpu_fp8_decode_value_flags_t value_flags =
      loom_amdgpu_vector_fp8_pair_decode_value_flags(
          &state->value_flag_cache, lane_index, pair_storage.live_lane_count);
  switch (result_element_type) {
    case LOOM_SCALAR_TYPE_BF16:
      if (!loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(&state->decode_plan,
                                                        value_flags)) {
        return false;
      }
      *out_action = loom_amdgpu_select_fp8_packed_bf16_decode_action(
          &state->decode_plan, value_flags);
      return true;
    case LOOM_SCALAR_TYPE_F16:
      if (!loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(
              &state->decode_plan, value_flags)) {
        return false;
      }
      *out_action = loom_amdgpu_select_fp8_packed_f16_decode_action(
          &state->decode_plan, value_flags);
      return true;
    default:
      IREE_ASSERT_UNREACHABLE("packed FP8 decode result element type");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_amdgpu_vector_fp8_select_all_packed_bf16_actions(
    loom_amdgpu_vector_fp8_selection_state_t* state,
    loom_amdgpu_vector_fp8_packed_bf16_selection_flags_t selection_flags) {
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  if (plan->result_element_type == LOOM_SCALAR_TYPE_F32 &&
      (plan->lane_count & 1u) != 0) {
    return false;
  }
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_query_uniform_packed_pair_value_flags(
          state, &value_flags) ||
      !loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(&state->decode_plan,
                                                    value_flags)) {
    return false;
  }
  if (iree_any_bit_set(
          selection_flags,
          LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED) &&
      !loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(&state->decode_plan,
                                                       value_flags)) {
    return false;
  }
  const loom_amdgpu_fp8_decode_action_t action =
      loom_amdgpu_select_fp8_packed_bf16_decode_action(&state->decode_plan,
                                                       value_flags);
  const uint32_t action_count =
      loom_amdgpu_vector_fp8_decode_action_count(plan);
  const uint32_t action_span =
      plan->result_element_type == LOOM_SCALAR_TYPE_F32 ? 2u : 1u;
  for (uint32_t i = 0; i < action_count; i += action_span) {
    loom_amdgpu_vector_fp8_set_decode_action(state, i, action_span, action);
  }
  return true;
}

static bool loom_amdgpu_vector_fp8_select_all_packed_f16_actions(
    loom_amdgpu_vector_fp8_selection_state_t* state) {
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  loom_amdgpu_fp8_decode_value_flags_t value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (!loom_amdgpu_vector_fp8_query_uniform_packed_pair_value_flags(
          state, &value_flags) ||
      !loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(&state->decode_plan,
                                                          value_flags)) {
    return false;
  }
  const loom_amdgpu_fp8_decode_action_t action =
      loom_amdgpu_select_fp8_packed_f16_decode_action(&state->decode_plan,
                                                      value_flags);
  for (uint32_t i = 0; i < plan->result_register_count; ++i) {
    loom_amdgpu_vector_fp8_set_decode_action(state, i, 1u, action);
  }
  return true;
}

static bool loom_amdgpu_vector_fp8_has_native_bf16_pack(
    const loom_low_descriptor_set_t* descriptor_set) {
  return loom_amdgpu_descriptor_set_has_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF16_F32);
}

static bool loom_amdgpu_vector_fp8_select_e8m0_pk8_actions(
    loom_amdgpu_vector_fp8_selection_state_t* state,
    loom_scalar_type_t descriptor_result_element_type) {
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(state->context);
  if (!loom_amdgpu_vector_fp8_plan_has_octet_storage(plan) ||
      !loom_amdgpu_vector_fp8_has_e8m0_pk8_descriptor(
          descriptor_set, plan->descriptor_source_format,
          descriptor_result_element_type)) {
    return false;
  }
  loom_amdgpu_fp8_decode_action_kind_t kind =
      LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE;
  switch (descriptor_result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      if (plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
        kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32;
      } else if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16 &&
                 loom_amdgpu_vector_fp8_has_native_bf16_pack(descriptor_set)) {
        kind =
            LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32_BF16_PACK;
      } else {
        kind =
            LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32_LANES_PACK;
      }
      break;
    case LOOM_SCALAR_TYPE_BF16:
      kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_BF16;
      break;
    case LOOM_SCALAR_TYPE_F16:
      kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F16;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("supported FP8 pk8 result element type");
      IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t action_span =
      plan->result_element_type == LOOM_SCALAR_TYPE_F32 ? 8u : 4u;
  const uint32_t action_count =
      loom_amdgpu_vector_fp8_decode_action_count(plan);
  for (uint32_t i = 0; i < action_count; i += action_span) {
    loom_amdgpu_vector_fp8_set_decode_action(
        state, i, action_span, (loom_amdgpu_fp8_decode_action_t){.kind = kind});
  }
  return true;
}

static loom_amdgpu_fp8_decode_action_kind_t
loom_amdgpu_vector_fp8_native_f32_pair_action_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t result_element_type) {
  return result_element_type == LOOM_SCALAR_TYPE_BF16 &&
                 loom_amdgpu_vector_fp8_has_native_bf16_pack(descriptor_set)
             ? LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR_BF16_PACK
             : LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR;
}

static loom_amdgpu_fp8_decode_action_kind_t
loom_amdgpu_vector_fp8_native_f32_lanes_action_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t result_element_type) {
  return result_element_type == LOOM_SCALAR_TYPE_BF16 &&
                 loom_amdgpu_vector_fp8_has_native_bf16_pack(descriptor_set)
             ? LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_LANES_BF16_PACK
             : LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_LANES_PACK;
}

static void loom_amdgpu_vector_fp8_select_unscaled_f32_result_actions(
    loom_amdgpu_vector_fp8_selection_state_t* state) {
  loom_low_lower_context_t* context = state->context;
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (loom_amdgpu_vector_fp8_select_e8m0_pk8_actions(state,
                                                     LOOM_SCALAR_TYPE_F32)) {
    return;
  }

  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  loom_amdgpu_vector_fp8_native_descriptor_set_refs(
      descriptor_set, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F32,
      &native_refs);
  const bool has_native_pair =
      native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const bool has_native_lane =
      native_refs.lane != LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if ((plan->lane_count & 1u) == 0 && !has_native_pair && !has_native_lane &&
      loom_amdgpu_vector_fp8_select_all_packed_bf16_actions(
          state, LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE)) {
    return;
  }

  for (uint32_t lane_index = 0; lane_index < plan->lane_count;) {
    loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
    if (has_native_pair && loom_amdgpu_vector_fp8_query_storage_pair(
                               plan, lane_index, &pair_storage)) {
      loom_amdgpu_vector_fp8_set_decode_action(
          state, lane_index, pair_storage.live_lane_count,
          (loom_amdgpu_fp8_decode_action_t){
              .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR,
          });
      lane_index += pair_storage.live_lane_count;
      continue;
    }
    if (has_native_lane) {
      loom_amdgpu_vector_fp8_set_decode_action(
          state, lane_index, 1u,
          (loom_amdgpu_fp8_decode_action_t){
              .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_LANE,
          });
      ++lane_index;
      continue;
    }
    loom_amdgpu_fp8_decode_action_t packed_action = {0};
    if (lane_index + 1u < plan->lane_count &&
        loom_amdgpu_vector_fp8_select_packed_pair_action(
            state, LOOM_SCALAR_TYPE_BF16, lane_index, &packed_action)) {
      loom_amdgpu_vector_fp8_set_decode_action(state, lane_index, 2u,
                                               packed_action);
      lane_index += 2u;
      continue;
    }
    loom_amdgpu_vector_fp8_set_decode_action(
        state, lane_index, 1u,
        loom_amdgpu_vector_fp8_full_decode_action(
            state, LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_F32, lane_index,
            1u));
    ++lane_index;
  }
}

static void loom_amdgpu_vector_fp8_select_f32_result_actions(
    loom_amdgpu_vector_fp8_selection_state_t* state) {
  loom_low_lower_context_t* context = state->context;
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (!loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan)) {
    loom_amdgpu_vector_fp8_select_unscaled_f32_result_actions(state);
    return;
  }

  if (loom_amdgpu_vector_fp8_has_scalef32_descriptor(
          descriptor_set, plan->descriptor_source_format,
          LOOM_SCALAR_TYPE_F32)) {
    for (uint32_t lane_index = 0; lane_index < plan->lane_count;
         lane_index += 2u) {
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      if (loom_amdgpu_vector_fp8_query_storage_pair(plan, lane_index,
                                                    &pair_storage)) {
        loom_amdgpu_vector_fp8_set_decode_action(
            state, lane_index, pair_storage.live_lane_count,
            (loom_amdgpu_fp8_decode_action_t){
                .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F32_PAIR,
            });
      }
    }
  }
  if (state->missing_action_count == plan->lane_count) {
    loom_amdgpu_vector_fp8_select_unscaled_f32_result_actions(state);
    return;
  }
  for (uint32_t lane_index = 0; lane_index < plan->lane_count; ++lane_index) {
    if (plan->strategy.fp8_decode.actions[lane_index].kind !=
        LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE) {
      continue;
    }
    loom_amdgpu_vector_fp8_set_decode_action(
        state, lane_index, 1u,
        loom_amdgpu_vector_fp8_full_decode_action(
            state, LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_F32, lane_index,
            1u));
  }
}

static void loom_amdgpu_vector_fp8_select_f32_fallback_packed_actions(
    loom_amdgpu_vector_fp8_selection_state_t* state) {
  loom_low_lower_context_t* context = state->context;
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (loom_amdgpu_vector_fp8_select_e8m0_pk8_actions(state,
                                                     LOOM_SCALAR_TYPE_F32)) {
    return;
  }

  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  loom_amdgpu_vector_fp8_native_descriptor_set_refs(
      descriptor_set, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F32,
      &native_refs);
  const bool has_native_pair =
      native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const bool has_native_lane =
      native_refs.lane != LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if ((plan->lane_count & 1u) == 0 && !has_native_pair && !has_native_lane &&
      loom_amdgpu_vector_fp8_select_all_packed_bf16_actions(
          state, LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE)) {
    return;
  }

  const loom_amdgpu_fp8_decode_action_kind_t full_kind =
      plan->result_element_type == LOOM_SCALAR_TYPE_BF16
          ? LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16
          : LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_F16;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    const uint32_t lane_count = lane_base + 1u < plan->lane_count ? 2u : 1u;
    loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
    if (has_native_pair && loom_amdgpu_vector_fp8_query_storage_pair(
                               plan, lane_base, &pair_storage)) {
      loom_amdgpu_vector_fp8_set_decode_action(
          state, register_index, 1u,
          (loom_amdgpu_fp8_decode_action_t){
              .kind = loom_amdgpu_vector_fp8_native_f32_pair_action_kind(
                  descriptor_set, plan->result_element_type),
          });
      continue;
    }
    if (has_native_lane) {
      loom_amdgpu_vector_fp8_set_decode_action(
          state, register_index, 1u,
          (loom_amdgpu_fp8_decode_action_t){
              .kind = loom_amdgpu_vector_fp8_native_f32_lanes_action_kind(
                  descriptor_set, plan->result_element_type),
          });
      continue;
    }
    loom_amdgpu_fp8_decode_action_t packed_action = {0};
    if (lane_count == 2u &&
        loom_amdgpu_vector_fp8_select_packed_pair_action(
            state, LOOM_SCALAR_TYPE_BF16, lane_base, &packed_action)) {
      loom_amdgpu_vector_fp8_set_decode_action(state, register_index, 1u,
                                               packed_action);
      continue;
    }
    loom_amdgpu_vector_fp8_set_decode_action(
        state, register_index, 1u,
        loom_amdgpu_vector_fp8_full_decode_action(state, full_kind, lane_base,
                                                  lane_count));
  }
}

static void loom_amdgpu_vector_fp8_select_scalef32_packed_actions(
    loom_amdgpu_vector_fp8_selection_state_t* state) {
  loom_low_lower_context_t* context = state->context;
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_amdgpu_fp8_decode_action_kind_t direct_result_kind =
      plan->result_element_type == LOOM_SCALAR_TYPE_BF16
          ? LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR
          : LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F16_PAIR;
  if (loom_amdgpu_vector_fp8_has_scalef32_descriptor(
          descriptor_set, plan->descriptor_source_format,
          plan->result_element_type)) {
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      if (loom_amdgpu_vector_fp8_query_storage_pair(plan, register_index * 2u,
                                                    &pair_storage)) {
        loom_amdgpu_vector_fp8_set_decode_action(
            state, register_index, 1u,
            (loom_amdgpu_fp8_decode_action_t){.kind = direct_result_kind});
      }
    }
  }

  if (loom_amdgpu_vector_fp8_has_scalef32_descriptor(
          descriptor_set, plan->descriptor_source_format,
          LOOM_SCALAR_TYPE_F32)) {
    const loom_amdgpu_fp8_decode_action_kind_t f32_pair_kind =
        plan->result_element_type == LOOM_SCALAR_TYPE_BF16 &&
                loom_amdgpu_vector_fp8_has_native_bf16_pack(descriptor_set)
            ? LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F32_PAIR_BF16_PACK
            : LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F32_PAIR;
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      if (plan->strategy.fp8_decode.actions[register_index].kind !=
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE) {
        continue;
      }
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      if (loom_amdgpu_vector_fp8_query_storage_pair(plan, register_index * 2u,
                                                    &pair_storage)) {
        loom_amdgpu_vector_fp8_set_decode_action(
            state, register_index, 1u,
            (loom_amdgpu_fp8_decode_action_t){.kind = f32_pair_kind});
      }
    }
  }

  if (state->missing_action_count == plan->result_register_count) {
    loom_amdgpu_vector_fp8_select_f32_fallback_packed_actions(state);
    return;
  }
  const loom_amdgpu_fp8_decode_action_kind_t full_kind =
      plan->result_element_type == LOOM_SCALAR_TYPE_BF16
          ? LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16
          : LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_F16;
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (plan->strategy.fp8_decode.actions[register_index].kind !=
        LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE) {
      continue;
    }
    const uint32_t lane_base = register_index * 2u;
    const uint32_t lane_count = lane_base + 1u < plan->lane_count ? 2u : 1u;
    loom_amdgpu_vector_fp8_set_decode_action(
        state, register_index, 1u,
        loom_amdgpu_vector_fp8_full_decode_action(state, full_kind, lane_base,
                                                  lane_count));
  }
}

static void loom_amdgpu_vector_fp8_select_unscaled_bf16_actions(
    loom_amdgpu_vector_fp8_selection_state_t* state) {
  loom_low_lower_context_t* context = state->context;
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (loom_amdgpu_vector_fp8_select_e8m0_pk8_actions(state,
                                                     LOOM_SCALAR_TYPE_BF16)) {
    return;
  }

  if (loom_amdgpu_vector_fp8_has_scalef32_descriptor(
          descriptor_set, plan->descriptor_source_format,
          LOOM_SCALAR_TYPE_BF16)) {
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      if (loom_amdgpu_vector_fp8_query_storage_pair(plan, register_index * 2u,
                                                    &pair_storage)) {
        loom_amdgpu_vector_fp8_set_decode_action(
            state, register_index, 1u,
            (loom_amdgpu_fp8_decode_action_t){
                .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR,
            });
      }
    }
  }

  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  loom_amdgpu_vector_fp8_native_descriptor_set_refs(
      descriptor_set, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F32,
      &native_refs);
  const bool has_native_pair =
      native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const bool has_native_lane =
      native_refs.lane != LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (state->missing_action_count == plan->result_register_count) {
    const loom_amdgpu_vector_fp8_packed_bf16_selection_flags_t flags =
        has_native_pair || has_native_lane
            ? LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_REQUIRE_PREFERRED
            : LOOM_AMDGPU_VECTOR_FP8_PACKED_BF16_SELECTION_FLAG_NONE;
    if (loom_amdgpu_vector_fp8_select_all_packed_bf16_actions(state, flags)) {
      return;
    }
  }

  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (plan->strategy.fp8_decode.actions[register_index].kind !=
        LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE) {
      continue;
    }
    const uint32_t lane_base = register_index * 2u;
    const uint32_t lane_count = lane_base + 1u < plan->lane_count ? 2u : 1u;
    loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
    if (has_native_pair && loom_amdgpu_vector_fp8_query_storage_pair(
                               plan, lane_base, &pair_storage)) {
      loom_amdgpu_vector_fp8_set_decode_action(
          state, register_index, 1u,
          (loom_amdgpu_fp8_decode_action_t){
              .kind = loom_amdgpu_vector_fp8_native_f32_pair_action_kind(
                  descriptor_set, LOOM_SCALAR_TYPE_BF16),
          });
      continue;
    }
    if (has_native_lane) {
      loom_amdgpu_vector_fp8_set_decode_action(
          state, register_index, 1u,
          (loom_amdgpu_fp8_decode_action_t){
              .kind = loom_amdgpu_vector_fp8_native_f32_lanes_action_kind(
                  descriptor_set, LOOM_SCALAR_TYPE_BF16),
          });
      continue;
    }
    loom_amdgpu_fp8_decode_action_t packed_action = {0};
    if (loom_amdgpu_vector_fp8_select_packed_pair_action(
            state, LOOM_SCALAR_TYPE_BF16, lane_base, &packed_action)) {
      loom_amdgpu_vector_fp8_set_decode_action(state, register_index, 1u,
                                               packed_action);
      continue;
    }
    loom_amdgpu_vector_fp8_set_decode_action(
        state, register_index, 1u,
        loom_amdgpu_vector_fp8_full_decode_action(
            state, LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16, lane_base,
            lane_count));
  }
}

static bool loom_amdgpu_vector_fp8_has_native_f16_byte_select_family(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fp8_native_descriptor_refs_t* refs) {
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(refs->byte_select); ++i) {
    if (!loom_amdgpu_vector_fp8_descriptor_set_has_ref(descriptor_set,
                                                       refs->byte_select[i])) {
      return false;
    }
  }
  return true;
}

static void loom_amdgpu_vector_fp8_select_unscaled_f16_actions(
    loom_amdgpu_vector_fp8_selection_state_t* state) {
  loom_low_lower_context_t* context = state->context;
  loom_amdgpu_vector_16bit_float_conversion_plan_t* plan = state->plan;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (loom_amdgpu_vector_fp8_select_e8m0_pk8_actions(state,
                                                     LOOM_SCALAR_TYPE_F16)) {
    return;
  }

  loom_amdgpu_fp8_native_descriptor_refs_t f16_refs = {0};
  loom_amdgpu_vector_fp8_native_descriptor_set_refs(
      descriptor_set, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F16,
      &f16_refs);
  const bool has_native_pair = f16_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const bool has_byte_select =
      (plan->lane_count & 1u) != 0 &&
      loom_amdgpu_vector_fp8_has_native_f16_byte_select_family(descriptor_set,
                                                               &f16_refs);
  const uint32_t pair_register_count =
      plan->result_register_count - (has_byte_select ? 1u : 0u);
  if (has_native_pair) {
    for (uint32_t register_index = 0; register_index < pair_register_count;
         ++register_index) {
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      if (loom_amdgpu_vector_fp8_query_storage_pair(plan, register_index * 2u,
                                                    &pair_storage)) {
        loom_amdgpu_vector_fp8_set_decode_action(
            state, register_index, 1u,
            (loom_amdgpu_fp8_decode_action_t){
                .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F16_PAIR,
            });
      }
    }
  }
  if (has_byte_select) {
    loom_amdgpu_vector_fp8_set_decode_action(
        state, plan->result_register_count - 1u, 1u,
        (loom_amdgpu_fp8_decode_action_t){
            .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F16_BYTE_SELECT,
        });
  }
  if (state->missing_action_count == 0) {
    return;
  }

  if (state->missing_action_count == plan->result_register_count &&
      loom_amdgpu_vector_fp8_has_scalef32_descriptor(
          descriptor_set, plan->descriptor_source_format,
          LOOM_SCALAR_TYPE_F16)) {
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      if (loom_amdgpu_vector_fp8_query_storage_pair(plan, register_index * 2u,
                                                    &pair_storage)) {
        loom_amdgpu_vector_fp8_set_decode_action(
            state, register_index, 1u,
            (loom_amdgpu_fp8_decode_action_t){
                .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F16_PAIR,
            });
      }
    }
  }

  loom_amdgpu_fp8_native_descriptor_refs_t f32_refs = {0};
  loom_amdgpu_vector_fp8_native_descriptor_set_refs(
      descriptor_set, plan->descriptor_source_format, LOOM_SCALAR_TYPE_F32,
      &f32_refs);
  if (f32_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    for (uint32_t register_index = 0;
         register_index < plan->result_register_count; ++register_index) {
      if (plan->strategy.fp8_decode.actions[register_index].kind !=
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE) {
        continue;
      }
      loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
      if (loom_amdgpu_vector_fp8_query_storage_pair(plan, register_index * 2u,
                                                    &pair_storage)) {
        loom_amdgpu_vector_fp8_set_decode_action(
            state, register_index, 1u,
            (loom_amdgpu_fp8_decode_action_t){
                .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR,
            });
      }
    }
  }
  if (state->missing_action_count == 0) {
    return;
  }

  if (state->missing_action_count == plan->result_register_count &&
      loom_amdgpu_vector_fp8_select_all_packed_f16_actions(state)) {
    return;
  }
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (plan->strategy.fp8_decode.actions[register_index].kind !=
        LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE) {
      continue;
    }
    loom_amdgpu_fp8_decode_action_t packed_action = {0};
    if (loom_amdgpu_vector_fp8_select_packed_pair_action(
            state, LOOM_SCALAR_TYPE_F16, register_index * 2u, &packed_action)) {
      loom_amdgpu_vector_fp8_set_decode_action(state, register_index, 1u,
                                               packed_action);
    }
  }
  if (state->missing_action_count == 0) {
    return;
  }
  if (state->missing_action_count == plan->result_register_count) {
    loom_amdgpu_vector_fp8_select_f32_fallback_packed_actions(state);
    return;
  }
  for (uint32_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (plan->strategy.fp8_decode.actions[register_index].kind !=
        LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE) {
      continue;
    }
    const uint32_t lane_base = register_index * 2u;
    const uint32_t lane_count = lane_base + 1u < plan->lane_count ? 2u : 1u;
    loom_amdgpu_vector_fp8_set_decode_action(
        state, register_index, 1u,
        loom_amdgpu_vector_fp8_full_decode_action(
            state, LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_F16, lane_base,
            lane_count));
  }
}

void loom_amdgpu_select_vector_fp8_decode_plan(
    loom_low_lower_context_t* context,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  IREE_ASSERT_EQ(plan->strategy_kind,
                 LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_STANDARD);
  plan->strategy_kind = LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_FP8_DECODE;
  plan->strategy.fp8_decode = (loom_amdgpu_vector_fp8_decode_plan_t){0};
  if (loom_amdgpu_vector_fp8_scalef32_is_identity(context, plan)) {
    plan->scale_source = LOOM_VALUE_ID_INVALID;
    plan->scale_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
    plan->scale_group_element_count = 0;
  }

  loom_amdgpu_vector_fp8_selection_state_t state;
  state.context = context;
  state.plan = plan;
  state.missing_action_count = loom_amdgpu_vector_fp8_decode_action_count(plan);
  state.software_resources_initialized = false;

  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(plan)) {
    const bool selected = loom_amdgpu_vector_fp8_select_e8m0_pk8_actions(
        &state, plan->result_element_type);
    IREE_ASSERT_TRUE(selected);
  } else if (loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(
                 plan)) {
    if (plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
      loom_amdgpu_vector_fp8_select_f32_result_actions(&state);
    } else {
      loom_amdgpu_vector_fp8_select_scalef32_packed_actions(&state);
    }
  } else {
    switch (plan->result_element_type) {
      case LOOM_SCALAR_TYPE_F32:
        loom_amdgpu_vector_fp8_select_f32_result_actions(&state);
        break;
      case LOOM_SCALAR_TYPE_BF16:
        loom_amdgpu_vector_fp8_select_unscaled_bf16_actions(&state);
        break;
      case LOOM_SCALAR_TYPE_F16:
        loom_amdgpu_vector_fp8_select_unscaled_f16_actions(&state);
        break;
      default:
        IREE_ASSERT_UNREACHABLE("supported FP8 vector decode result type");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  IREE_ASSERT_EQ(state.missing_action_count, 0u);
}

typedef struct loom_amdgpu_vector_fp8_emit_state_t {
  // Active low-lowering context.
  loom_low_lower_context_t* context;
  // Source operation owning emitted locations and diagnostics.
  const loom_op_t* source_op;
  // Selected vector conversion plan.
  const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan;
  // Materialized source vector in low registers.
  loom_value_id_t low_source;
  // Physical type of one source register.
  loom_type_t source_lane_type;
  // Physical type of one result register.
  loom_type_t result_lane_type;
  // Static packed-lane extraction geometry.
  loom_amdgpu_vector_extract_plan_t extract_plan;
  // Lazily materialized F32 scale or identity constant.
  loom_value_id_t low_f32_scale;
  // Lazily materialized scalar-register type for software decode.
  loom_type_t sgpr_type;
  // Lazily materialized two-register mask type for software decode.
  loom_type_t mask_type;
  // Lazily resolved software decode recipe.
  const loom_amdgpu_fp8_decode_plan_t* decode_plan;
  // Lazily resolved native F32 decoder descriptors.
  const loom_amdgpu_fp8_native_descriptors_t* native_f32_descriptors;
  // Lazily resolved native F16 decoder descriptors.
  const loom_amdgpu_fp8_native_descriptors_t* native_f16_descriptors;
  // Lazily resolved F16/BF16 pack descriptors.
  const loom_amdgpu_float16_pack_descriptors_t* float16_pack_descriptors;
  // Lazily materialized two-register F32 result type.
  loom_type_t f32_pair_type;
} loom_amdgpu_vector_fp8_emit_state_t;

static iree_status_t loom_amdgpu_vector_fp8_emit_state_ensure_software_decode(
    loom_amdgpu_vector_fp8_emit_state_t* state) {
  if (loom_type_kind(state->sgpr_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_type(state->context, &state->sgpr_type));
  }
  if (loom_type_kind(state->mask_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(state->context, 2, &state->mask_type));
  }
  if (state->decode_plan == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_decode_plan(
        state->context, state->plan->source_format,
        state->plan->descriptor_source_format, &state->decode_plan));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_fp8_emit_state_ensure_f32_scale(
    loom_amdgpu_vector_fp8_emit_state_t* state) {
  if (state->low_f32_scale != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(
          state->plan)) {
    return loom_amdgpu_lookup_vector_scale_source(
        state->context, state->source_op, state->plan, &state->low_f32_scale);
  }
  return loom_amdgpu_emit_const_u32(
      state->context, state->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS, state->result_lane_type,
      &state->low_f32_scale);
}

static iree_status_t
loom_amdgpu_vector_fp8_emit_state_ensure_native_f32_descriptors(
    loom_amdgpu_vector_fp8_emit_state_t* state) {
  if (state->native_f32_descriptors != NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
      state->context, state->plan->descriptor_source_format,
      LOOM_SCALAR_TYPE_F32, &state->native_f32_descriptors));
  IREE_ASSERT(state->native_f32_descriptors != NULL);
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_vector_fp8_emit_state_ensure_native_f16_descriptors(
    loom_amdgpu_vector_fp8_emit_state_t* state) {
  if (state->native_f16_descriptors != NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
      state->context, state->plan->descriptor_source_format,
      LOOM_SCALAR_TYPE_F16, &state->native_f16_descriptors));
  IREE_ASSERT(state->native_f16_descriptors != NULL);
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_vector_fp8_emit_state_ensure_float16_pack_descriptors(
    loom_amdgpu_vector_fp8_emit_state_t* state) {
  if (state->float16_pack_descriptors != NULL) {
    return iree_ok_status();
  }
  return loom_amdgpu_get_float16_pack_descriptors(
      state->context, &state->float16_pack_descriptors);
}

static iree_status_t loom_amdgpu_vector_fp8_emit_state_ensure_f32_pair_type(
    loom_amdgpu_vector_fp8_emit_state_t* state) {
  if (loom_type_kind(state->f32_pair_type) != LOOM_TYPE_NONE) {
    return iree_ok_status();
  }
  return loom_amdgpu_make_vgpr_range_type(state->context, 2,
                                          &state->f32_pair_type);
}

static loom_amdgpu_vector_fp8_pair_storage_t
loom_amdgpu_vector_fp8_selected_pair_storage(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    uint32_t lane_index) {
  loom_amdgpu_vector_fp8_pair_storage_t pair_storage;
  const bool has_pair = loom_amdgpu_vector_fp8_query_storage_pair(
      plan, lane_index, &pair_storage);
  IREE_ASSERT_TRUE(has_pair);
  return pair_storage;
}

static iree_status_t loom_amdgpu_vector_fp8_emit_software_f32_lane(
    loom_amdgpu_vector_fp8_emit_state_t* state, uint32_t lane_index,
    loom_amdgpu_fp8_decode_value_flags_t value_flags,
    loom_value_id_t* out_low_lane) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_emit_state_ensure_software_decode(state));
  loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_lane(
      state->context, state->source_op, state->plan, &state->extract_plan,
      state->low_source, state->source_lane_type, lane_index, &low_byte));
  return loom_amdgpu_emit_fp8_to_f32_lane(
      state->context, state->source_op, state->decode_plan, low_byte,
      value_flags, state->result_lane_type, state->sgpr_type, state->mask_type,
      out_low_lane);
}

static iree_status_t loom_amdgpu_vector_fp8_scale_f32_lanes(
    loom_amdgpu_vector_fp8_emit_state_t* state, loom_value_id_t* low_lanes,
    uint32_t lane_count) {
  if (!loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(
          state->plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_emit_state_ensure_f32_scale(state));
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        state->context, state->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
        low_lanes[i], state->low_f32_scale, state->result_lane_type,
        &low_lanes[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_fp8_pack_f32_lanes(
    loom_amdgpu_vector_fp8_emit_state_t* state,
    const loom_value_id_t* low_f32_lanes, uint32_t lane_count,
    loom_value_id_t* out_low_packed) {
  IREE_ASSERT_GE(lane_count, 1u);
  IREE_ASSERT_LE(lane_count, 2u);
  *out_low_packed = LOOM_VALUE_ID_INVALID;
  if (state->plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vector_fp8_emit_state_ensure_float16_pack_descriptors(
            state));
    loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
    if (lane_count == 2u) {
      high_lane = low_f32_lanes[1];
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_const_u32(state->context, state->source_op,
                                     LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                     state->result_lane_type, &high_lane));
    }
    return loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
        state->context, state->source_op, state->float16_pack_descriptors,
        low_f32_lanes[0], high_lane, state->result_lane_type, out_low_packed);
  }

  IREE_ASSERT_EQ(state->plan->result_element_type, LOOM_SCALAR_TYPE_F16);
  for (uint32_t register_lane = 0; register_lane < lane_count;
       ++register_lane) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_pack_f32_lane_to_f16_register(
        state->context, state->source_op, low_f32_lanes[register_lane],
        register_lane, state->result_lane_type, out_low_packed));
  }
  return iree_ok_status();
}

typedef enum loom_amdgpu_vector_fp8_packed_action_family_e {
  LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_NONE = 0,
  LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_BF16 = 1,
  LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_F16 = 2,
} loom_amdgpu_vector_fp8_packed_action_family_t;

static loom_amdgpu_vector_fp8_packed_action_family_t
loom_amdgpu_vector_fp8_packed_action_family(
    loom_amdgpu_fp8_decode_action_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_NORMAL:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_REPAIR:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_VIA_F16:
      return LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_BF16;
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_NORMAL:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_EXACT_REPAIR:
      return LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_F16;
    default:
      return LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_NONE;
  }
}

static bool loom_amdgpu_vector_fp8_decode_actions_form_uniform_batch(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_vector_fp8_packed_action_family_t* out_family,
    const loom_amdgpu_fp8_decode_action_t** out_action) {
  *out_family = LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_NONE;
  *out_action = NULL;
  const uint32_t pair_count = (plan->lane_count + 1u) / 2u;
  if (pair_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }
  const uint32_t action_count =
      loom_amdgpu_vector_fp8_decode_action_count(plan);
  for (uint32_t action_index = 0; action_index < action_count; ++action_index) {
    const loom_amdgpu_fp8_decode_action_t* action =
        &plan->strategy.fp8_decode.actions[action_index];
    if (action->kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_CONTINUATION) {
      continue;
    }
    const loom_amdgpu_vector_fp8_packed_action_family_t family =
        loom_amdgpu_vector_fp8_packed_action_family(action->kind);
    if (family == LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_NONE) {
      return false;
    }
    if (*out_action == NULL) {
      *out_family = family;
      *out_action = action;
    } else if (family != *out_family || action->kind != (*out_action)->kind ||
               action->detail_flags != (*out_action)->detail_flags) {
      return false;
    }
  }
  return *out_action != NULL;
}

static iree_status_t loom_amdgpu_vector_fp8_prepare_pair_sources(
    loom_amdgpu_vector_fp8_emit_state_t* state, uint32_t first_pair_index,
    uint32_t pair_count,
    loom_amdgpu_fp8_packed_u16_pair_source_t* out_pair_sources) {
  loom_amdgpu_vector_fp8_pair_storage_t
      pair_storage[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_ASSERT_LE(pair_count, IREE_ARRAYSIZE(pair_storage));
  for (uint32_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    pair_storage[pair_index] = loom_amdgpu_vector_fp8_selected_pair_storage(
        state->plan, (first_pair_index + pair_index) * 2u);
  }

  loom_value_id_t source_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(source_registers); ++i) {
    source_registers[i] = LOOM_VALUE_ID_INVALID;
  }
  for (uint32_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    const uint32_t source_register_index =
        pair_storage[pair_index].source_register_index;
    if (source_registers[source_register_index] == LOOM_VALUE_ID_INVALID) {
      loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          state->context, state->source_op, state->low_source,
          state->plan->storage_register_count, source_register_index,
          state->source_lane_type, &source_register));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
          state->context, state->source_op, source_register, &source_register));
      source_registers[source_register_index] = source_register;
    }
    out_pair_sources[pair_index] = (loom_amdgpu_fp8_packed_u16_pair_source_t){
        .source_register = source_registers[source_register_index],
        .byte_offset = pair_storage[pair_index].byte_offset,
        .live_lane_count = pair_storage[pair_index].live_lane_count,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_fp8_emit_packed_bf16_pairs(
    loom_amdgpu_vector_fp8_emit_state_t* state,
    const loom_amdgpu_fp8_decode_action_t* action, uint32_t first_pair_index,
    uint32_t pair_count, loom_value_id_t* out_low_packed) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_emit_state_ensure_software_decode(state));
  loom_amdgpu_fp8_packed_u16_pair_source_t
      pair_sources[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_prepare_pair_sources(
      state, first_pair_index, pair_count, pair_sources));
  return loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
      state->context, state->source_op, state->decode_plan, pair_sources,
      pair_count, loom_amdgpu_fp8_decode_action_packed_bf16_strategy(action),
      loom_amdgpu_fp8_decode_action_repairs(action), state->result_lane_type,
      state->sgpr_type, state->mask_type, out_low_packed);
}

static iree_status_t loom_amdgpu_vector_fp8_emit_packed_f16_pairs(
    loom_amdgpu_vector_fp8_emit_state_t* state,
    const loom_amdgpu_fp8_decode_action_t* action, uint32_t first_pair_index,
    uint32_t pair_count, loom_value_id_t* out_low_packed) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_emit_state_ensure_software_decode(state));
  loom_amdgpu_fp8_packed_u16_pair_source_t
      pair_sources[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_prepare_pair_sources(
      state, first_pair_index, pair_count, pair_sources));
  return loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
      state->context, state->source_op, state->decode_plan, pair_sources,
      pair_count, loom_amdgpu_fp8_decode_action_packed_f16_strategy(action),
      loom_amdgpu_fp8_decode_action_repairs(action), state->result_lane_type,
      state->sgpr_type, state->mask_type, out_low_packed);
}

static iree_status_t loom_amdgpu_vector_fp8_finalize_f32_lanes(
    loom_amdgpu_vector_fp8_emit_state_t* state, loom_value_id_t* low_f32_lanes,
    loom_value_id_t* out_low_registers) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_scale_f32_lanes(
      state, low_f32_lanes, state->plan->lane_count));
  if (state->plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
    for (uint32_t lane_index = 0; lane_index < state->plan->lane_count;
         ++lane_index) {
      out_low_registers[lane_index] = low_f32_lanes[lane_index];
    }
    return iree_ok_status();
  }
  for (uint32_t register_index = 0;
       register_index < state->plan->result_register_count; ++register_index) {
    const uint32_t lane_base = register_index * 2u;
    const uint32_t lane_count =
        lane_base + 1u < state->plan->lane_count ? 2u : 1u;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_pack_f32_lanes(
        state, &low_f32_lanes[lane_base], lane_count,
        &out_low_registers[register_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_fp8_emit_uniform_packed_bf16(
    loom_amdgpu_vector_fp8_emit_state_t* state,
    const loom_amdgpu_fp8_decode_action_t* action,
    loom_value_id_t* out_low_registers) {
  const uint32_t pair_count = (state->plan->lane_count + 1u) / 2u;
  loom_value_id_t packed_bf16[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_packed_bf16_pairs(
      state, action, /*first_pair_index=*/0, pair_count, packed_bf16));
  if (state->plan->result_element_type == LOOM_SCALAR_TYPE_BF16 &&
      !loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(
          state->plan)) {
    for (uint32_t i = 0; i < pair_count; ++i) {
      out_low_registers[i] = packed_bf16[i];
    }
    return iree_ok_status();
  }

  loom_value_id_t low_f32_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    const uint32_t lane_base = pair_index * 2u;
    const uint32_t lane_count =
        lane_base + 1u < state->plan->lane_count ? 2u : 1u;
    for (uint32_t register_lane = 0; register_lane < lane_count;
         ++register_lane) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
          state->context, state->source_op, packed_bf16[pair_index],
          register_lane, state->result_lane_type,
          &low_f32_lanes[lane_base + register_lane]));
    }
  }
  return loom_amdgpu_vector_fp8_finalize_f32_lanes(state, low_f32_lanes,
                                                   out_low_registers);
}

static iree_status_t loom_amdgpu_vector_fp8_emit_e8m0_pk8_actions(
    loom_amdgpu_vector_fp8_emit_state_t* state,
    loom_amdgpu_fp8_decode_action_kind_t action_kind,
    loom_value_id_t* out_low_registers) {
  const bool direct_f32 =
      action_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32 &&
      state->plan->result_element_type == LOOM_SCALAR_TYPE_F32 &&
      loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(
          state->plan);
  const bool intermediate_f32 =
      action_kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32 ||
      action_kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32_LANES_PACK ||
      action_kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32_BF16_PACK;
  if (!intermediate_f32 || direct_f32) {
    loom_value_id_t low_e8m0_scale = LOOM_VALUE_ID_INVALID;
    if (loom_amdgpu_vector_16bit_float_conversion_plan_has_e8m0_scale(
            state->plan)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
          state->context, state->source_op, state->plan, &low_e8m0_scale));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_e8m0_pk8(
        state->context, state->source_op, state->plan, state->low_source,
        low_e8m0_scale, state->source_lane_type, state->result_lane_type,
        out_low_registers));
    return iree_ok_status();
  }

  loom_amdgpu_vector_16bit_float_conversion_plan_t f32_plan = *state->plan;
  f32_plan.scale_source = LOOM_VALUE_ID_INVALID;
  f32_plan.scale_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
  f32_plan.scale_group_element_count = 0;
  f32_plan.result_element_type = LOOM_SCALAR_TYPE_F32;
  f32_plan.result_register_count = f32_plan.lane_count;
  loom_value_id_t low_f32_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_e8m0_pk8(
      state->context, state->source_op, &f32_plan, state->low_source,
      LOOM_VALUE_ID_INVALID, state->source_lane_type, state->result_lane_type,
      low_f32_lanes));
  return loom_amdgpu_vector_fp8_finalize_f32_lanes(state, low_f32_lanes,
                                                   out_low_registers);
}

static iree_status_t loom_amdgpu_vector_fp8_emit_scalef32_pair_action(
    loom_amdgpu_vector_fp8_emit_state_t* state, uint32_t register_index,
    loom_scalar_type_t descriptor_result_element_type,
    loom_value_id_t* out_low_registers, uint32_t* out_register_count) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_emit_state_ensure_f32_scale(state));
  const loom_low_lower_resolved_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
      state->context, state->plan->descriptor_source_format,
      descriptor_result_element_type, &descriptor));
  IREE_ASSERT(descriptor != NULL);
  const uint32_t lane_base =
      state->plan->result_element_type == LOOM_SCALAR_TYPE_F32
          ? register_index
          : register_index * 2u;
  const loom_amdgpu_vector_fp8_pair_storage_t pair_storage =
      loom_amdgpu_vector_fp8_selected_pair_storage(state->plan, lane_base);
  if (descriptor_result_element_type != LOOM_SCALAR_TYPE_F32) {
    return loom_amdgpu_emit_vector_fp8_pair_descriptor(
        state->context, state->source_op, state->plan, &pair_storage,
        descriptor, state->low_source, state->source_lane_type,
        &state->low_f32_scale, /*extra_operand_count=*/1,
        state->result_lane_type, &out_low_registers[register_index]);
  }

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_emit_state_ensure_f32_pair_type(state));
  loom_value_id_t converted_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_pair_descriptor(
      state->context, state->source_op, state->plan, &pair_storage, descriptor,
      state->low_source, state->source_lane_type, &state->low_f32_scale,
      /*extra_operand_count=*/1, state->f32_pair_type, &converted_pair));
  if (state->plan->result_element_type == LOOM_SCALAR_TYPE_F32 &&
      state->plan->lane_count == 2u) {
    out_low_registers[0] = converted_pair;
    *out_register_count = 1u;
    return iree_ok_status();
  }
  loom_value_id_t low_f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_native_pair_lanes(
      state->context, state->source_op, converted_pair, state->result_lane_type,
      pair_storage.live_lane_count, low_f32_lanes));
  if (state->plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
    for (uint32_t i = 0; i < pair_storage.live_lane_count; ++i) {
      out_low_registers[lane_base + i] = low_f32_lanes[i];
    }
    return iree_ok_status();
  }
  return loom_amdgpu_vector_fp8_pack_f32_lanes(
      state, low_f32_lanes, pair_storage.live_lane_count,
      &out_low_registers[register_index]);
}

static iree_status_t loom_amdgpu_vector_fp8_emit_native_f32_pair_action(
    loom_amdgpu_vector_fp8_emit_state_t* state, uint32_t register_index,
    loom_value_id_t* out_low_registers) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_emit_state_ensure_native_f32_descriptors(state));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_emit_state_ensure_f32_pair_type(state));
  const uint32_t lane_base =
      state->plan->result_element_type == LOOM_SCALAR_TYPE_F32
          ? register_index
          : register_index * 2u;
  const loom_amdgpu_vector_fp8_pair_storage_t pair_storage =
      loom_amdgpu_vector_fp8_selected_pair_storage(state->plan, lane_base);
  loom_value_id_t converted_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_pair_descriptor(
      state->context, state->source_op, state->plan, &pair_storage,
      &state->native_f32_descriptors->pair_descriptor, state->low_source,
      state->source_lane_type, /*extra_operands=*/NULL,
      /*extra_operand_count=*/0, state->f32_pair_type, &converted_pair));
  loom_value_id_t low_f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_native_pair_lanes(
      state->context, state->source_op, converted_pair, state->result_lane_type,
      pair_storage.live_lane_count, low_f32_lanes));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_scale_f32_lanes(
      state, low_f32_lanes, pair_storage.live_lane_count));
  if (state->plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
    for (uint32_t i = 0; i < pair_storage.live_lane_count; ++i) {
      out_low_registers[lane_base + i] = low_f32_lanes[i];
    }
    return iree_ok_status();
  }
  return loom_amdgpu_vector_fp8_pack_f32_lanes(
      state, low_f32_lanes, pair_storage.live_lane_count,
      &out_low_registers[register_index]);
}

static iree_status_t loom_amdgpu_vector_fp8_emit_native_f32_lanes_action(
    loom_amdgpu_vector_fp8_emit_state_t* state, uint32_t register_index,
    loom_value_id_t* out_low_registers) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_emit_state_ensure_native_f32_descriptors(state));
  const uint32_t lane_base =
      state->plan->result_element_type == LOOM_SCALAR_TYPE_F32
          ? register_index
          : register_index * 2u;
  const uint32_t lane_count =
      state->plan->result_element_type == LOOM_SCALAR_TYPE_F32
          ? 1u
          : (lane_base + 1u < state->plan->lane_count ? 2u : 1u);
  loom_value_id_t low_f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_lane(
        state->context, state->source_op, state->plan, &state->extract_plan,
        state->low_source, state->source_lane_type, lane_base + i, &low_byte));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
        state->context, state->source_op, low_byte, &low_byte));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_unary(
        state->context, state->source_op,
        &state->native_f32_descriptors->lane_descriptor, low_byte,
        state->result_lane_type, &low_f32_lanes[i]));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_scale_f32_lanes(state, low_f32_lanes, lane_count));
  if (state->plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
    out_low_registers[lane_base] = low_f32_lanes[0];
    return iree_ok_status();
  }
  return loom_amdgpu_vector_fp8_pack_f32_lanes(
      state, low_f32_lanes, lane_count, &out_low_registers[register_index]);
}

static iree_status_t loom_amdgpu_vector_fp8_emit_packed_bf16_action(
    loom_amdgpu_vector_fp8_emit_state_t* state,
    const loom_amdgpu_fp8_decode_action_t* action, uint32_t register_index,
    loom_value_id_t* out_low_registers) {
  const uint32_t pair_index =
      state->plan->result_element_type == LOOM_SCALAR_TYPE_F32
          ? register_index / 2u
          : register_index;
  loom_value_id_t packed_bf16 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_packed_bf16_pairs(
      state, action, pair_index, 1u, &packed_bf16));
  if (state->plan->result_element_type == LOOM_SCALAR_TYPE_BF16 &&
      !loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(
          state->plan)) {
    out_low_registers[register_index] = packed_bf16;
    return iree_ok_status();
  }

  const uint32_t lane_base = pair_index * 2u;
  const uint32_t lane_count =
      lane_base + 1u < state->plan->lane_count ? 2u : 1u;
  loom_value_id_t low_f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_bf16_register_lane_as_f32_bits(
        state->context, state->source_op, packed_bf16, i,
        state->result_lane_type, &low_f32_lanes[i]));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_scale_f32_lanes(state, low_f32_lanes, lane_count));
  if (state->plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
    for (uint32_t i = 0; i < lane_count; ++i) {
      out_low_registers[lane_base + i] = low_f32_lanes[i];
    }
    return iree_ok_status();
  }
  return loom_amdgpu_vector_fp8_pack_f32_lanes(
      state, low_f32_lanes, lane_count, &out_low_registers[register_index]);
}

static iree_status_t loom_amdgpu_vector_fp8_emit_full_action(
    loom_amdgpu_vector_fp8_emit_state_t* state,
    const loom_amdgpu_fp8_decode_action_t* action, uint32_t register_index,
    loom_value_id_t* out_low_registers) {
  if (action->kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_F32) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_software_f32_lane(
        state, register_index,
        loom_amdgpu_fp8_decode_action_value_flags(action, 0),
        &out_low_registers[register_index]));
    return loom_amdgpu_vector_fp8_scale_f32_lanes(
        state, &out_low_registers[register_index], 1u);
  }

  const uint32_t lane_base = register_index * 2u;
  const uint32_t lane_count =
      lane_base + 1u < state->plan->lane_count ? 2u : 1u;
  if (action->kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16 &&
      !loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(
          state->plan)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vector_fp8_emit_state_ensure_software_decode(state));
    loom_value_id_t low_bf16_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                         LOOM_VALUE_ID_INVALID};
    for (uint32_t i = 0; i < lane_count; ++i) {
      loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vector_fp8_lane(
          state->context, state->source_op, state->plan, &state->extract_plan,
          state->low_source, state->source_lane_type, lane_base + i,
          &low_byte));
      const loom_amdgpu_fp8_decode_value_flags_t value_flags =
          loom_amdgpu_fp8_decode_action_value_flags(action, i);
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_bf16_lane(
          state->context, state->source_op, state->decode_plan, low_byte,
          value_flags, state->result_lane_type, state->sgpr_type,
          state->mask_type, &low_bf16_lanes[i]));
    }
    if (lane_count == 1u) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          state->context, state->source_op,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, state->result_lane_type,
          &low_bf16_lanes[1]));
    }
    const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
        iree_any_bit_set(state->decode_plan->flags,
                         LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16)
            ? &state->decode_plan->pack_u16_descriptor
            : NULL;
    return loom_amdgpu_emit_packed_u16_lane_pair(
        state->context, state->source_op, pack_u16_descriptor,
        low_bf16_lanes[0], low_bf16_lanes[1], state->result_lane_type,
        &out_low_registers[register_index]);
  }

  loom_value_id_t low_f32_lanes[2] = {LOOM_VALUE_ID_INVALID,
                                      LOOM_VALUE_ID_INVALID};
  for (uint32_t i = 0; i < lane_count; ++i) {
    const loom_amdgpu_fp8_decode_value_flags_t value_flags =
        loom_amdgpu_fp8_decode_action_value_flags(action, i);
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_software_f32_lane(
        state, lane_base + i, value_flags, &low_f32_lanes[i]));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_fp8_scale_f32_lanes(state, low_f32_lanes, lane_count));
  return loom_amdgpu_vector_fp8_pack_f32_lanes(
      state, low_f32_lanes, lane_count, &out_low_registers[register_index]);
}

static iree_status_t loom_amdgpu_vector_fp8_emit_selected_actions(
    loom_amdgpu_vector_fp8_emit_state_t* state,
    loom_value_id_t* out_low_registers, uint32_t* out_register_count) {
  const uint32_t action_count =
      loom_amdgpu_vector_fp8_decode_action_count(state->plan);
  *out_register_count = action_count;
  for (uint32_t i = 0; i < action_count; ++i) {
    out_low_registers[i] = LOOM_VALUE_ID_INVALID;
  }

  const loom_amdgpu_fp8_decode_action_t* first_action =
      &state->plan->strategy.fp8_decode.actions[0];
  if (first_action->kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32 ||
      first_action->kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_BF16 ||
      first_action->kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F16 ||
      first_action->kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32_LANES_PACK ||
      first_action->kind ==
          LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32_BF16_PACK) {
    return loom_amdgpu_vector_fp8_emit_e8m0_pk8_actions(
        state, first_action->kind, out_low_registers);
  }

  loom_amdgpu_vector_fp8_packed_action_family_t uniform_family =
      LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_NONE;
  const loom_amdgpu_fp8_decode_action_t* uniform_action = NULL;
  if (loom_amdgpu_vector_fp8_decode_actions_form_uniform_batch(
          state->plan, &uniform_family, &uniform_action)) {
    if (uniform_family == LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_BF16) {
      return loom_amdgpu_vector_fp8_emit_uniform_packed_bf16(
          state, uniform_action, out_low_registers);
    }
    IREE_ASSERT_EQ(uniform_family,
                   LOOM_AMDGPU_VECTOR_FP8_PACKED_ACTION_FAMILY_F16);
    IREE_ASSERT_EQ(state->plan->result_element_type, LOOM_SCALAR_TYPE_F16);
    IREE_ASSERT_FALSE(
        loom_amdgpu_vector_16bit_float_conversion_plan_has_scale(state->plan));
    return loom_amdgpu_vector_fp8_emit_packed_f16_pairs(
        state, uniform_action, /*first_pair_index=*/0,
        state->plan->result_register_count, out_low_registers);
  }

  for (uint32_t action_index = 0; action_index < action_count; ++action_index) {
    const loom_amdgpu_fp8_decode_action_t* action =
        &state->plan->strategy.fp8_decode.actions[action_index];
    switch (action->kind) {
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_CONTINUATION:
        break;
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_scalef32_pair_action(
            state, action_index, LOOM_SCALAR_TYPE_BF16, out_low_registers,
            out_register_count));
        break;
      }
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F16_PAIR: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_scalef32_pair_action(
            state, action_index, LOOM_SCALAR_TYPE_F16, out_low_registers,
            out_register_count));
        break;
      }
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F32_PAIR:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F32_PAIR_BF16_PACK: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_scalef32_pair_action(
            state, action_index, LOOM_SCALAR_TYPE_F32, out_low_registers,
            out_register_count));
        break;
      }
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F16_PAIR: {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_vector_fp8_emit_state_ensure_native_f16_descriptors(
                state));
        const uint32_t lane_base = action_index * 2u;
        const loom_amdgpu_vector_fp8_pair_storage_t pair_storage =
            loom_amdgpu_vector_fp8_selected_pair_storage(state->plan,
                                                         lane_base);
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_fp8_pair_descriptor(
            state->context, state->source_op, state->plan, &pair_storage,
            &state->native_f16_descriptors->pair_descriptor, state->low_source,
            state->source_lane_type, /*extra_operands=*/NULL,
            /*extra_operand_count=*/0, state->result_lane_type,
            &out_low_registers[action_index]));
        break;
      }
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F16_BYTE_SELECT: {
        loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
        const bool has_refs = loom_amdgpu_fp8_native_descriptor_refs(
            state->plan->descriptor_source_format, LOOM_SCALAR_TYPE_F16,
            &native_refs);
        IREE_ASSERT_TRUE(has_refs);
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_vector_fp8_tail_byte_select_descriptor(
                state->context, state->source_op, state->plan, &native_refs,
                state->low_source, state->source_lane_type,
                state->result_lane_type, &out_low_registers[action_index]));
        break;
      }
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR_BF16_PACK: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_native_f32_pair_action(
            state, action_index, out_low_registers));
        break;
      }
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_LANE:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_LANES_PACK:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_LANES_BF16_PACK: {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_vector_fp8_emit_native_f32_lanes_action(
                state, action_index, out_low_registers));
        break;
      }
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_NORMAL:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_REPAIR:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_VIA_F16: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_packed_bf16_action(
            state, action, action_index, out_low_registers));
        break;
      }
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_NORMAL:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_EXACT_REPAIR: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_packed_f16_pairs(
            state, action, action_index, 1u, &out_low_registers[action_index]));
        break;
      }
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_F32:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_F16: {
        IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_full_action(
            state, action, action_index, out_low_registers));
        break;
      }
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_BF16:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F16:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32_LANES_PACK:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32_BF16_PACK:
      default:
        IREE_ASSERT_UNREACHABLE("selected FP8 vector decode action");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_vector_fp8_mixed_conversion_plan_key(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  const bool scaled =
      loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan);
  switch (plan->result_element_type) {
    case LOOM_SCALAR_TYPE_F32:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_mixed_f32_decode")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_mixed_f32_decode");
    case LOOM_SCALAR_TYPE_BF16:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_mixed_packed_bf16_decode")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_mixed_packed_bf16_decode");
    case LOOM_SCALAR_TYPE_F16:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_mixed_packed_f16_decode")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_mixed_packed_f16_decode");
    default:
      IREE_ASSERT_UNREACHABLE("selected FP8 vector decode result type");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_string_view_t loom_amdgpu_vector_fp8_decode_action_plan_key(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    const loom_amdgpu_fp8_decode_action_t* action) {
  const bool scaled =
      loom_amdgpu_vector_16bit_float_conversion_plan_has_f32_scale(plan);
  switch (action->kind) {
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_e8m0_pk8_f32")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_e8m0_pk8_f32");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32_LANES_PACK:
      if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
        return scaled ? IREE_SV(
                            "amdgpu.vector_16bit_float_conversion.strategy."
                            "fp8_scalef32_e8m0_pk8_f32_manual_bf16_pack")
                      : IREE_SV(
                            "amdgpu.vector_16bit_float_conversion.strategy."
                            "fp8_e8m0_pk8_f32_manual_bf16_pack");
      }
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_e8m0_pk8_f32_manual_f16_pack")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_e8m0_pk8_f32_manual_f16_pack");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F32_BF16_PACK:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_e8m0_pk8_f32_native_bf16_pack")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_e8m0_pk8_f32_native_bf16_pack");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_BF16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy.fp8_e8m0_pk8_bf16");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F16:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy.fp8_e8m0_pk8_f16");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_native_bf16_pair")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_native_bf16_pair");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F16_PAIR:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_native_f16_pair")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_native_f16_pair");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F32_PAIR:
      if (plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "fp8_scalef32_native_f32_pair");
      }
      return plan->result_element_type == LOOM_SCALAR_TYPE_BF16
                 ? IREE_SV(
                       "amdgpu.vector_16bit_float_conversion.strategy."
                       "fp8_scalef32_native_f32_pair_manual_bf16_pack")
                 : IREE_SV(
                       "amdgpu.vector_16bit_float_conversion.strategy."
                       "fp8_scalef32_native_f32_pair_manual_f16_pack");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F32_PAIR_BF16_PACK:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_scalef32_native_f32_pair_native_bf16_pack");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F16_PAIR:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy.fp8_native_f16_pair");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F16_BYTE_SELECT:
      return IREE_SV(
          "amdgpu.vector_16bit_float_conversion.strategy."
          "fp8_native_f16_byte_select");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR:
      if (plan->result_element_type == LOOM_SCALAR_TYPE_F32) {
        return scaled ? IREE_SV(
                            "amdgpu.vector_16bit_float_conversion.strategy."
                            "fp8_scalef32_native_f32_pair")
                      : IREE_SV(
                            "amdgpu.vector_16bit_float_conversion.strategy."
                            "fp8_native_f32_pair");
      }
      if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
        return scaled ? IREE_SV(
                            "amdgpu.vector_16bit_float_conversion.strategy."
                            "fp8_scalef32_native_f32_pair_manual_bf16_pack")
                      : IREE_SV(
                            "amdgpu.vector_16bit_float_conversion.strategy."
                            "fp8_native_f32_pair_manual_bf16_pack");
      }
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_native_f32_pair_manual_f16_pack")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_native_f32_pair_manual_f16_pack");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR_BF16_PACK:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_native_f32_pair_native_bf16_pack")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_native_f32_pair_native_bf16_pack");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_LANE:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_native_f32_lane")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_native_f32_lane");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_LANES_PACK:
      if (plan->result_element_type == LOOM_SCALAR_TYPE_BF16) {
        return scaled ? IREE_SV(
                            "amdgpu.vector_16bit_float_conversion.strategy."
                            "fp8_scalef32_native_f32_lane_manual_bf16_pack")
                      : IREE_SV(
                            "amdgpu.vector_16bit_float_conversion.strategy."
                            "fp8_native_f32_lane_manual_bf16_pack");
      }
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_native_f32_lane_manual_f16_pack")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_native_f32_lane_manual_f16_pack");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_LANES_BF16_PACK:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_native_f32_lane_native_bf16_pack")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_native_f32_lane_native_bf16_pack");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_NORMAL:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_REPAIR:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_VIA_F16:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_software_packed_bf16_decode")
                    : loom_amdgpu_fp8_packed_bf16_strategy_key(
                          loom_amdgpu_fp8_decode_action_packed_bf16_strategy(
                              action),
                          loom_amdgpu_fp8_decode_action_repairs(action));
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_NORMAL:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_EXACT_REPAIR:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_software_packed_f16_decode")
                    : loom_amdgpu_fp8_packed_f16_repair_reason_key(
                          loom_amdgpu_fp8_decode_action_repairs(action));
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_F32:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_software_f32_decode")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_software_f32_decode");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_software_packed_bf16_decode")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_software_packed_bf16_decode");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_F16:
      return scaled ? IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_scalef32_software_packed_f16_decode")
                    : IREE_SV(
                          "amdgpu.vector_16bit_float_conversion.strategy."
                          "fp8_software_packed_f16_decode");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_CONTINUATION:
    default:
      IREE_ASSERT_UNREACHABLE("selected FP8 vector decode action");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_string_view_t loom_amdgpu_vector_fp8_conversion_plan_key(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  IREE_ASSERT_EQ(plan->strategy_kind,
                 LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_FP8_DECODE);
  const uint32_t action_count =
      loom_amdgpu_vector_fp8_decode_action_count(plan);
  iree_string_view_t plan_key = iree_string_view_empty();
  for (uint32_t action_index = 0; action_index < action_count; ++action_index) {
    const loom_amdgpu_fp8_decode_action_t* action =
        &plan->strategy.fp8_decode.actions[action_index];
    if (action->kind == LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_CONTINUATION) {
      continue;
    }
    const iree_string_view_t action_key =
        loom_amdgpu_vector_fp8_decode_action_plan_key(plan, action);
    if (iree_string_view_is_empty(plan_key)) {
      plan_key = action_key;
    } else if (!iree_string_view_equal(plan_key, action_key)) {
      return loom_amdgpu_vector_fp8_mixed_conversion_plan_key(plan);
    }
  }
  return plan_key;
}

iree_status_t loom_amdgpu_lower_vector_fp8_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_value_id_t low_source, loom_type_t source_lane_type,
    loom_type_t result_lane_type) {
  IREE_ASSERT_EQ(plan->strategy_kind,
                 LOOM_AMDGPU_VECTOR_FLOAT_CONVERSION_STRATEGY_FP8_DECODE);
  loom_amdgpu_vector_fp8_emit_state_t state = {
      .context = context,
      .source_op = source_op,
      .plan = plan,
      .low_source = low_source,
      .source_lane_type = source_lane_type,
      .result_lane_type = result_lane_type,
      .extract_plan =
          {
              .source = plan->storage_source,
              .result = plan->result,
              .lane_count = plan->lane_count,
              .register_count = plan->storage_register_count,
              .result_register_count = 1,
              .element_register_count = 1,
              .lane_bit_count = 8,
              .sign_extend_packed_lane = false,
          },
      .low_f32_scale = LOOM_VALUE_ID_INVALID,
      .sgpr_type = loom_type_none(),
      .mask_type = loom_type_none(),
      .f32_pair_type = loom_type_none(),
  };
  loom_value_id_t low_result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint32_t low_result_register_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_fp8_emit_selected_actions(
      &state, low_result_registers, &low_result_register_count));
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             low_result_registers,
                                             low_result_register_count);
}
