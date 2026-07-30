// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/narrow_float/fp4.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/storage.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/float16.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/vector_conversion.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

enum {
  // Packed shifts place each nibble magnitude in a V_PERM selector byte.
  LOOM_AMDGPU_FP4_MAGNITUDE_POSITION_SHIFTS = 0x00040008u,
  // Both packed lanes move their E2M1 sign bit into the F16 sign position.
  LOOM_AMDGPU_FP4_SIGN_POSITION_SHIFTS = 0x00040004u,
  // Selects the three magnitude bits from two positioned E2M1 nibbles.
  LOOM_AMDGPU_FP4_MAGNITUDE_SELECTOR_MASK = 0x07000700u,
  // Selects the two positioned F16 sign bits.
  LOOM_AMDGPU_FP4_SIGN_MASK = 0x80008000u,
  // F16 high bytes for E2M1 magnitudes 0 through 3.
  LOOM_AMDGPU_FP4_F16_MAGNITUDE_TABLE_LOW = 0x3E3C3800u,
  // F16 high bytes for E2M1 magnitudes 4 through 7.
  LOOM_AMDGPU_FP4_F16_MAGNITUDE_TABLE_HIGH = 0x46444240u,
};

struct loom_amdgpu_fp4_decode_recipe_t {
  // Scalar constant descriptor used for lookup words and packed shifts.
  loom_low_lower_resolved_descriptor_t scalar_constant_descriptor;
  // Vector constant descriptor used by table and byte-permute fallbacks.
  loom_low_lower_resolved_descriptor_t vector_constant_descriptor;
  // General byte-permute descriptor used for table lookup.
  loom_low_lower_resolved_descriptor_t permute_descriptor;
  // Optional byte-permute form with zero SRC1 and a literal selector.
  loom_low_lower_resolved_descriptor_t zero_literal_permute_descriptor;
  // Packed 16-bit logical left-shift descriptor.
  loom_low_lower_resolved_descriptor_t packed_shift_descriptor;
  // Literal magnitude/sign mask descriptor.
  loom_low_lower_resolved_descriptor_t mask_descriptor;
  // Packed payload/sign merge descriptor.
  loom_low_lower_resolved_descriptor_t merge_descriptor;
  // Module string ID for descriptor imm32 attributes.
  loom_string_id_t imm32_attr_name_id;
  // Whether VOP3 packets may consume both scalar lookup-table registers.
  bool supports_two_scalar_vop3_sources;
};

typedef struct loom_amdgpu_fp4_decode_recipe_cache_t {
  // Whether recipe has been resolved for this function's target.
  bool initialized;
  // Function-local descriptor resolution shared by every FP4 decode.
  loom_amdgpu_fp4_decode_recipe_t recipe;
} loom_amdgpu_fp4_decode_recipe_cache_t;

static int loom_amdgpu_fp4_decode_recipe_cache_state_key;

static bool loom_amdgpu_fp4_supports_two_scalar_vop3_sources(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  return loom_amdgpu_descriptor_set_info_has_flags(
      descriptor_set_info,
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOP3_TWO_SCALAR_SOURCES);
}

static bool loom_amdgpu_fp4_decode_descriptors_available(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32) ||
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32) ||
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_LSHLREV_B16) ||
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT) ||
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32)) {
    return false;
  }
  const bool has_zero_literal_permute = loom_amdgpu_descriptor_set_has_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32_SRC1_ZERO_SRC2_LIT);
  const bool requires_vector_constant =
      !has_zero_literal_permute ||
      !loom_amdgpu_fp4_supports_two_scalar_vop3_sources(descriptor_set);
  return !requires_vector_constant ||
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32);
}

static iree_status_t loom_amdgpu_initialize_fp4_decode_recipe(
    loom_low_lower_context_t* context,
    loom_amdgpu_fp4_decode_recipe_t* recipe) {
  memset(recipe, 0, sizeof(*recipe));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      &recipe->scalar_constant_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32,
      &recipe->permute_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_LSHLREV_B16,
      &recipe->packed_shift_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      &recipe->mask_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, &recipe->merge_descriptor));

  bool has_zero_literal_permute = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32_SRC1_ZERO_SRC2_LIT,
      &recipe->zero_literal_permute_descriptor, &has_zero_literal_permute));
  recipe->supports_two_scalar_vop3_sources =
      loom_amdgpu_fp4_supports_two_scalar_vop3_sources(
          loom_low_lower_context_descriptor_set(context));
  if (!has_zero_literal_permute || !recipe->supports_two_scalar_vop3_sources) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        &recipe->vector_constant_descriptor));
  }
  return loom_amdgpu_intern(context, IREE_SV("imm32"),
                            &recipe->imm32_attr_name_id);
}

static iree_status_t loom_amdgpu_get_fp4_decode_recipe(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fp4_decode_recipe_t** out_recipe) {
  *out_recipe = NULL;
  loom_amdgpu_fp4_decode_recipe_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fp4_decode_recipe_cache_state_key, sizeof(*cache),
      (void**)&cache));
  if (!cache->initialized) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_initialize_fp4_decode_recipe(context, &cache->recipe));
    cache->initialized = true;
  }
  *out_recipe = &cache->recipe;
  return iree_ok_status();
}

typedef struct loom_amdgpu_fp4_decode_shape_t {
  // Number of packed i32 payload registers.
  uint32_t source_register_count;
  // Number of logical E2M1/F16 lanes represented by the payload.
  uint32_t lane_count;
  // Number of packed F16 result registers.
  uint32_t result_register_count;
} loom_amdgpu_fp4_decode_shape_t;

static bool loom_amdgpu_fp4_decode_shape(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_fp4_decode_shape_t* out_shape) {
  *out_shape = (loom_amdgpu_fp4_decode_shape_t){0};
  if (!loom_vector_decode_isa(source_op)) {
    return false;
  }
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_decode_payload(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_decode_result(source_op));
  const uint32_t source_register_count =
      loom_vector_static_rank1_lane_count(source_type, LOOM_SCALAR_TYPE_I32,
                                          /*maximum_lane_count=*/8);
  if (source_register_count == 0) {
    return false;
  }
  const uint32_t lane_count = source_register_count * 8u;
  const uint32_t result_lane_count = loom_vector_static_rank1_lane_count(
      result_type, LOOM_SCALAR_TYPE_F16,
      LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES);
  uint32_t result_payload_bit_count = 0;
  uint32_t result_register_count = 0;
  if (result_lane_count != lane_count ||
      !loom_amdgpu_type_packed_16bit_float_storage(
          result_type, &result_payload_bit_count, &result_register_count)) {
    return false;
  }
  IREE_ASSERT_EQ(result_payload_bit_count, lane_count * 16u);
  *out_shape = (loom_amdgpu_fp4_decode_shape_t){
      .source_register_count = source_register_count,
      .lane_count = lane_count,
      .result_register_count = result_register_count,
  };
  return true;
}

static bool loom_amdgpu_fp4_unscaled_schema_matches(
    loom_value_fact_encoded_operand_schema_t schema,
    const loom_amdgpu_fp4_decode_shape_t* shape) {
  return !loom_value_fact_encoded_operand_schema_is_unknown(schema) &&
         schema.element_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1 &&
         schema.scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE &&
         schema.secondary_scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE &&
         schema.payload_packing ==
             LOOM_VALUE_FACT_PAYLOAD_PACKING_LITTLE_ENDIAN_NIBBLES &&
         schema.scale_topology == LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE &&
         schema.affine_policy == LOOM_VALUE_FACT_AFFINE_POLICY_NONE &&
         schema.rounding_policy == LOOM_VALUE_FACT_ROUNDING_POLICY_NONE &&
         schema.codebook_policy == LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE &&
         schema.sparsity_policy == LOOM_VALUE_FACT_SPARSITY_POLICY_NONE &&
         schema.flags == 0 &&
         schema.payload_register_count == shape->source_register_count &&
         schema.payload_element_count == shape->lane_count &&
         schema.scale_group_element_count == 0 &&
         schema.scale_operand_count == 0;
}

static bool loom_amdgpu_fp4_e4m3fn_scale16_schema_matches(
    loom_value_fact_encoded_operand_schema_t schema,
    const loom_amdgpu_fp4_decode_shape_t* shape) {
  return shape->lane_count == 16 &&
         !loom_value_fact_encoded_operand_schema_is_unknown(schema) &&
         schema.element_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1 &&
         schema.scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN &&
         schema.secondary_scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE &&
         schema.payload_packing ==
             LOOM_VALUE_FACT_PAYLOAD_PACKING_LITTLE_ENDIAN_NIBBLES &&
         schema.scale_topology == LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D &&
         schema.affine_policy == LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY &&
         schema.rounding_policy == LOOM_VALUE_FACT_ROUNDING_POLICY_NONE &&
         schema.codebook_policy == LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE &&
         schema.sparsity_policy == LOOM_VALUE_FACT_SPARSITY_POLICY_NONE &&
         (schema.flags == 0 ||
          schema.flags ==
              LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK) &&
         schema.payload_register_count == shape->source_register_count &&
         schema.payload_element_count == shape->lane_count &&
         schema.scale_group_element_count == 16 &&
         schema.scale_operand_count == 1;
}

typedef struct loom_amdgpu_fp4_decode_match_t {
  // Physical payload/result shape accepted by the selected route.
  loom_amdgpu_fp4_decode_shape_t shape;
  // Optional encoded scale auxiliary.
  loom_value_id_t scale_source;
  // Exact numeric format of scale_source.
  loom_value_fact_numeric_format_flags_t scale_format;
  // Number of decoded payload lanes sharing scale_source.
  uint32_t scale_group_element_count;
} loom_amdgpu_fp4_decode_match_t;

static bool loom_amdgpu_fp4_decode_match(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_fp4_decode_match_t* out_match) {
  *out_match = (loom_amdgpu_fp4_decode_match_t){
      .scale_source = LOOM_VALUE_ID_INVALID,
  };
  if (fact_table == NULL || descriptor_set == NULL ||
      !loom_amdgpu_fp4_decode_descriptors_available(descriptor_set) ||
      !loom_amdgpu_fp4_decode_shape(module, source_op, &out_match->shape)) {
    return false;
  }

  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table,
                                       loom_vector_decode_schema(source_op)),
          &summary)) {
    return false;
  }
  const bool has_auxiliary =
      loom_vector_decode_auxiliary(source_op).count != 0 ||
      loom_vector_decode_auxiliary_names(source_op).count != 0;
  if (!has_auxiliary &&
      loom_amdgpu_fp4_unscaled_schema_matches(
          summary.storage_schema.encoded_operand, &out_match->shape)) {
    return true;
  }

  loom_value_id_t scale_source = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F16_F32) ||
      !loom_amdgpu_descriptor_set_can_emit_packed_u16_lane_pair(
          descriptor_set) ||
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MUL_F16) ||
      !loom_amdgpu_fp4_e4m3fn_scale16_schema_matches(
          summary.storage_schema.encoded_operand, &out_match->shape) ||
      !loom_amdgpu_vector_decode_scale_source(
          module, source_op, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN,
          &scale_source)) {
    return false;
  }
  out_match->scale_source = scale_source;
  out_match->scale_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN;
  out_match->scale_group_element_count = 16;
  return true;
}

bool loom_amdgpu_vector_decode_can_lower_as_fp4_conversion(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op) {
  loom_amdgpu_fp4_decode_match_t match;
  return loom_amdgpu_fp4_decode_match(module, fact_table, descriptor_set,
                                      source_op, &match);
}

iree_status_t loom_amdgpu_select_fp4_decode_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan,
    bool* out_selected) {
  loom_amdgpu_fp4_decode_match_t match;
  *out_selected = loom_amdgpu_fp4_decode_match(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context),
      loom_low_lower_context_descriptor_set(context), source_op, &match);
  if (!*out_selected) {
    return iree_ok_status();
  }

  const loom_amdgpu_fp4_decode_recipe_t* recipe = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp4_decode_recipe(context, &recipe));

  const loom_value_id_t source = loom_vector_decode_payload(source_op);
  const loom_value_id_t result = loom_vector_decode_result(source_op);
  *out_plan = (loom_amdgpu_vector_16bit_float_conversion_plan_t){
      .source = source,
      .result = result,
      .storage_source = source,
      .content_fact_source = result,
      .scale_source = match.scale_source,
      .scale_format = match.scale_format,
      .scale_group_element_count = match.scale_group_element_count,
      .kind = LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE,
      .source_element_type = LOOM_SCALAR_TYPE_I32,
      .source_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1,
      .descriptor_source_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE,
      .result_element_type = LOOM_SCALAR_TYPE_F16,
      .lane_count = match.shape.lane_count,
      .source_register_count = match.shape.source_register_count,
      .storage_lane_stride = 1,
      .storage_lane_count = match.shape.source_register_count,
      .storage_register_count = match.shape.source_register_count,
      .result_register_count = match.shape.result_register_count,
      .fp4_decode_recipe = recipe,
  };
  return iree_ok_status();
}

typedef struct loom_amdgpu_fp4_decode_emission_state_t {
  // Packed shifts positioning the two magnitudes in V_PERM selector bytes.
  loom_value_id_t magnitude_position_shifts;
  // Packed shifts positioning two sign bits in F16 lanes.
  loom_value_id_t sign_position_shifts;
  // Low four E2M1-to-F16 high-byte lookup entries.
  loom_value_id_t magnitude_table_low;
  // High four E2M1-to-F16 high-byte lookup entries.
  loom_value_id_t magnitude_table_high;
  // Zero VGPR used by targets without a literal byte-duplication packet.
  loom_value_id_t low_zero;
  // Register selectors used by the generic byte-duplication packet.
  loom_value_id_t duplicate_selectors[4];
} loom_amdgpu_fp4_decode_emission_state_t;

static iree_status_t loom_amdgpu_emit_fp4_scalar_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_decode_recipe_t* recipe, uint32_t value,
    loom_type_t scalar_type, loom_value_id_t* out_value) {
  return loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &recipe->scalar_constant_descriptor,
      recipe->imm32_attr_name_id, value, scalar_type, out_value);
}

static iree_status_t loom_amdgpu_initialize_fp4_decode_emission(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_decode_recipe_t* recipe, loom_type_t scalar_type,
    loom_type_t vector_type, loom_amdgpu_fp4_decode_emission_state_t* state) {
  *state = (loom_amdgpu_fp4_decode_emission_state_t){
      .magnitude_position_shifts = LOOM_VALUE_ID_INVALID,
      .sign_position_shifts = LOOM_VALUE_ID_INVALID,
      .magnitude_table_low = LOOM_VALUE_ID_INVALID,
      .magnitude_table_high = LOOM_VALUE_ID_INVALID,
      .low_zero = LOOM_VALUE_ID_INVALID,
      .duplicate_selectors =
          {
              LOOM_VALUE_ID_INVALID,
              LOOM_VALUE_ID_INVALID,
              LOOM_VALUE_ID_INVALID,
              LOOM_VALUE_ID_INVALID,
          },
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
      context, source_op, recipe, LOOM_AMDGPU_FP4_MAGNITUDE_POSITION_SHIFTS,
      scalar_type, &state->magnitude_position_shifts));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
      context, source_op, recipe, LOOM_AMDGPU_FP4_SIGN_POSITION_SHIFTS,
      scalar_type, &state->sign_position_shifts));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
      context, source_op, recipe, LOOM_AMDGPU_FP4_F16_MAGNITUDE_TABLE_HIGH,
      scalar_type, &state->magnitude_table_high));

  if (recipe->supports_two_scalar_vop3_sources) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
        context, source_op, recipe, LOOM_AMDGPU_FP4_F16_MAGNITUDE_TABLE_LOW,
        scalar_type, &state->magnitude_table_low));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
        context, source_op, &recipe->vector_constant_descriptor,
        recipe->imm32_attr_name_id, LOOM_AMDGPU_FP4_F16_MAGNITUDE_TABLE_LOW,
        vector_type, &state->magnitude_table_low));
  }
  if (recipe->zero_literal_permute_descriptor.descriptor != NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &recipe->vector_constant_descriptor,
      recipe->imm32_attr_name_id, 0, vector_type, &state->low_zero));
  state->duplicate_selectors[0] = state->sign_position_shifts;
  for (uint32_t byte_index = 1; byte_index < 4; ++byte_index) {
    const uint32_t source_byte_selector = 4u + byte_index;
    const uint32_t selector =
        source_byte_selector | (source_byte_selector << 16);
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
        context, source_op, recipe, selector, scalar_type,
        &state->duplicate_selectors[byte_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp4_duplicate_byte(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_decode_recipe_t* recipe,
    const loom_amdgpu_fp4_decode_emission_state_t* state,
    loom_value_id_t source_register, uint32_t byte_index,
    loom_type_t vector_type, loom_value_id_t* out_duplicated_byte) {
  const uint32_t source_byte_selector = 4u + byte_index;
  const uint32_t selector = source_byte_selector | (source_byte_selector << 16);
  if (recipe->zero_literal_permute_descriptor.descriptor != NULL) {
    return loom_amdgpu_emit_resolved_vgpr_unary_immediate(
        context, source_op, &recipe->zero_literal_permute_descriptor,
        source_register, selector, vector_type, out_duplicated_byte);
  }
  return loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &recipe->permute_descriptor, source_register,
      state->low_zero, state->duplicate_selectors[byte_index], vector_type,
      out_duplicated_byte);
}

static iree_status_t loom_amdgpu_emit_fp4_pair_as_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_decode_recipe_t* recipe,
    const loom_amdgpu_fp4_decode_emission_state_t* state,
    loom_value_id_t source_register, uint32_t byte_index,
    loom_type_t vector_type, loom_value_id_t* out_packed_f16) {
  loom_value_id_t duplicated_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_duplicate_byte(
      context, source_op, recipe, state, source_register, byte_index,
      vector_type, &duplicated_byte));

  loom_value_id_t positioned_nibbles = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &recipe->packed_shift_descriptor,
      state->magnitude_position_shifts, duplicated_byte, vector_type,
      &positioned_nibbles));
  loom_value_id_t magnitude_selectors = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &recipe->mask_descriptor, positioned_nibbles,
      LOOM_AMDGPU_FP4_MAGNITUDE_SELECTOR_MASK, vector_type,
      &magnitude_selectors));

  loom_value_id_t magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &recipe->permute_descriptor,
      state->magnitude_table_high, state->magnitude_table_low,
      magnitude_selectors, vector_type, &magnitude));

  loom_value_id_t positioned_signs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &recipe->packed_shift_descriptor,
      state->sign_position_shifts, positioned_nibbles, vector_type,
      &positioned_signs));
  loom_value_id_t signs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &recipe->mask_descriptor, positioned_signs,
      LOOM_AMDGPU_FP4_SIGN_MASK, vector_type, &signs));
  return loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &recipe->merge_descriptor, magnitude, signs,
      vector_type, out_packed_f16);
}

static iree_status_t loom_amdgpu_prepare_fp4_block_scale(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_type_t scalar_type, loom_type_t vector_type,
    loom_value_id_t* out_packed_scale,
    const loom_low_lower_resolved_descriptor_t** out_multiply_descriptor) {
  *out_packed_scale = LOOM_VALUE_ID_INVALID;
  *out_multiply_descriptor = NULL;
  if (plan->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(plan->scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN);
  IREE_ASSERT_EQ(plan->scale_group_element_count, 16u);

  loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_vector_scale_source(
      context, source_op, plan, &low_scale));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  const loom_amdgpu_fp8_decode_plan_t* scale_decode_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_decode_plan(
      context, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN,
      LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN, &scale_decode_plan));
  IREE_ASSERT(
      iree_any_bit_set(scale_decode_plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_F16));

  loom_value_id_t f32_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_f32_lane(
      context, source_op, scale_decode_plan, low_scale,
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF, vector_type, scalar_type,
      mask_type, &f32_scale));
  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(scale_decode_plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16)
          ? &scale_decode_plan->pack_u16_descriptor
          : NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_splat_f32_lane_to_packed_f16(
      context, source_op, pack_u16_descriptor, f32_scale, vector_type,
      out_packed_scale));
  *out_multiply_descriptor = &scale_decode_plan->pk_mul_f16_descriptor;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_fp4_decode(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  IREE_ASSERT(plan->fp4_decode_recipe != NULL);
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_source, &low_source));

  loom_type_t scalar_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &scalar_type));
  loom_type_t vector_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vector_type));
  loom_amdgpu_fp4_decode_emission_state_t state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fp4_decode_emission(
      context, source_op, plan->fp4_decode_recipe, scalar_type, vector_type,
      &state));
  loom_value_id_t packed_scale = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_resolved_descriptor_t* multiply_descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_fp4_block_scale(
      context, source_op, plan, scalar_type, vector_type, &packed_scale,
      &multiply_descriptor));

  loom_value_id_t results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint32_t result_index = 0;
  for (uint32_t source_index = 0; source_index < plan->source_register_count;
       ++source_index) {
    loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->source_register_count,
        source_index, vector_type, &source_register));
    for (uint32_t byte_index = 0; byte_index < 4; ++byte_index) {
      loom_value_id_t decoded_pair = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_pair_as_packed_f16(
          context, source_op, plan->fp4_decode_recipe, &state, source_register,
          byte_index, vector_type, &decoded_pair));
      if (multiply_descriptor == NULL) {
        results[result_index++] = decoded_pair;
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
            context, source_op, multiply_descriptor, decoded_pair, packed_scale,
            vector_type, &results[result_index++]));
      }
    }
  }
  IREE_ASSERT_EQ(result_index, plan->result_register_count);
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             results, result_index);
}

iree_string_view_t loom_amdgpu_fp4_decode_plan_key(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  IREE_ASSERT(plan->fp4_decode_recipe != NULL);
  if (plan->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp4_e2m1_e4m3fn_scale16_packed_f16_lookup");
  }
  IREE_ASSERT_EQ(plan->scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
  return IREE_SV(
      "amdgpu.vector_16bit_float_conversion.strategy."
      "fp4_e2m1_packed_f16_lookup");
}
