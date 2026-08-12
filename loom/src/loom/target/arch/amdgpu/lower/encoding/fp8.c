// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8_tables.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

static uint32_t loom_amdgpu_fp8_decode_packed_u16(uint32_t value) {
  return value | (value << 16);
}

static uint32_t loom_amdgpu_fp8_decode_packed_lt_bias(uint32_t threshold) {
  const uint32_t lane_bias = UINT32_C(0x10000) - threshold;
  return loom_amdgpu_fp8_decode_packed_u16(lane_bias);
}

static uint32_t loom_amdgpu_fp8_decode_packed_ge_bias(uint32_t threshold) {
  const uint32_t lane_bias = UINT32_C(0x8000) - threshold;
  return loom_amdgpu_fp8_decode_packed_u16(lane_bias);
}

static uint32_t loom_amdgpu_fp8_decode_subnormal_threshold(
    const loom_scalar_type_fp8_format_t* format) {
  return UINT32_C(1) << format->mantissa_bits;
}

static uint32_t loom_amdgpu_fp8_decode_top_exponent_no_sign(
    const loom_scalar_type_fp8_format_t* format) {
  return ((UINT32_C(1) << format->exponent_bits) - 1u) << format->mantissa_bits;
}

static uint32_t loom_amdgpu_fp8_decode_finite_nan_no_sign(
    const loom_scalar_type_fp8_format_t* format) {
  return loom_amdgpu_fp8_decode_top_exponent_no_sign(format) |
         ((UINT32_C(1) << format->mantissa_bits) - 1u);
}

loom_amdgpu_fp8_decode_value_flags_t
loom_amdgpu_fp8_decode_value_flags_from_facts(loom_value_facts_t facts) {
  loom_amdgpu_fp8_decode_value_flags_t flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (loom_value_facts_is_not_nan(facts)) {
    flags |= LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN;
  }
  if (loom_value_facts_is_not_inf(facts)) {
    flags |= LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF;
  }
  if (loom_value_facts_is_not_subnormal(facts)) {
    flags |= LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL;
  }
  if (loom_value_facts_is_non_zero(facts)) {
    flags |= LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO;
  }
  return flags;
}

loom_value_fact_numeric_format_flags_t loom_amdgpu_fp8_descriptor_source_format(
    loom_value_fact_numeric_format_flags_t exact_source_format,
    loom_value_facts_t content_facts) {
  if (exact_source_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3) {
    return loom_value_facts_is_not_nan(content_facts) &&
                   loom_value_facts_is_not_inf(content_facts)
               ? LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN
               : LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
  }
  uint32_t unused_format_index = 0;
  return loom_amdgpu_fp8_format_index(exact_source_format, &unused_format_index)
             ? exact_source_format
             : LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
}

static_assert(IREE_ARRAYSIZE(kLoomAmdgpuFp8EncodedOperandSchemaRequirements) ==
                  LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_E8M0 + 1,
              "FP8 encoded operand schema requirements cover dense kinds");

static bool loom_amdgpu_fp8_element_format_matches(
    loom_value_fact_numeric_format_flags_t element_format,
    loom_scalar_type_t element_type) {
  if (element_type >= IREE_ARRAYSIZE(kLoomAmdgpuFp8EncodedOperandFormatRows)) {
    return false;
  }
  const loom_amdgpu_fp8_encoded_operand_format_row_t* row =
      &kLoomAmdgpuFp8EncodedOperandFormatRows[element_type];
  if (row->encoded_operand_formats == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
    return false;
  }
  IREE_ASSERT_EQ(row->element_type, element_type);
  return element_format != LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE &&
         (element_format & (element_format - 1)) == 0 &&
         iree_all_bits_set(row->encoded_operand_formats, element_format);
}

static bool loom_amdgpu_fp8_scale_group_matches(
    uint32_t lane_count, uint32_t scale_group_element_count,
    loom_amdgpu_fp8_scale_group_mode_t mode) {
  switch (mode) {
    case LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_NONE:
      return scale_group_element_count == 0;
    case LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_ALL_LANES:
      return scale_group_element_count == lane_count;
    case LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_OCTETS_MAX4: {
      if (lane_count == 0 || (lane_count & 7u) != 0 ||
          scale_group_element_count < 8u ||
          (scale_group_element_count & 7u) != 0) {
        return false;
      }
      const uint32_t scale_group_count =
          scale_group_element_count >= lane_count
              ? 1u
              : (lane_count + scale_group_element_count - 1u) /
                    scale_group_element_count;
      return scale_group_count <= 4u;
    }
    default:
      return false;
  }
}

bool loom_amdgpu_fp8_encoded_operand_schema_matches(
    loom_value_fact_encoded_operand_schema_t schema,
    loom_scalar_type_t element_type, uint32_t lane_count,
    loom_amdgpu_fp8_encoded_operand_schema_kind_t kind) {
  if ((uint32_t)kind >=
      IREE_ARRAYSIZE(kLoomAmdgpuFp8EncodedOperandSchemaRequirements)) {
    return false;
  }
  const loom_amdgpu_fp8_encoded_operand_schema_requirement_t* requirement =
      &kLoomAmdgpuFp8EncodedOperandSchemaRequirements[kind];
  if (loom_value_fact_encoded_operand_schema_is_unknown(schema) ||
      !loom_amdgpu_fp8_element_format_matches(schema.element_format,
                                              element_type) ||
      schema.payload_packing != LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES ||
      schema.scale_format != requirement->scale_format ||
      schema.secondary_scale_format != LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE ||
      schema.scale_topology != requirement->scale_topology ||
      schema.affine_policy != requirement->affine_policy ||
      iree_any_bit_set(schema.rounding_policy,
                       ~LOOM_VALUE_FACT_ROUNDING_POLICY_ALL) ||
      schema.codebook_policy != LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE ||
      schema.sparsity_policy != LOOM_VALUE_FACT_SPARSITY_POLICY_NONE ||
      schema.flags != 0 || schema.payload_register_count != 0 ||
      schema.payload_element_count != lane_count ||
      schema.scale_operand_count != requirement->scale_operand_count ||
      !loom_amdgpu_fp8_scale_group_matches(lane_count,
                                           schema.scale_group.element_count,
                                           requirement->scale_group_mode)) {
    return false;
  }
  return true;
}

static bool loom_amdgpu_fp8_decode_value_is_finite(
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  return iree_all_bits_set(value_flags,
                           LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN |
                               LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF);
}

static bool loom_amdgpu_fp8_decode_plan_has_packed_exact_repair(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_EXACT_REPAIR);
}

static bool loom_amdgpu_fp8_decode_plan_has_packed_zero_repair(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_ZERO_REPAIR);
}

bool loom_amdgpu_fp8_selects_exact_bf16_via_f16(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  return iree_any_bit_set(
             plan->capabilities,
             LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_EXACT_BF16_VIA_F16) &&
         loom_amdgpu_fp8_decode_value_is_finite(value_flags) &&
         !iree_any_bit_set(value_flags,
                           LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL);
}

static bool loom_amdgpu_fp8_decode_plan_has_mask_repair_split(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_MASK_REPAIR_SPLIT);
}

static bool loom_amdgpu_fp8_decode_plan_has_inline_sgpr64_zero_compare(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_INLINE_SGPR64_ZERO_COMPARE);
}

static bool loom_amdgpu_fp8_decode_plan_has_combined_finite_nan_condition(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_COMBINED_FINITE_NAN_CONDITION);
}

static bool loom_amdgpu_fp8_decode_plan_has_combined_non_normal_condition(
    const loom_amdgpu_fp8_decode_plan_t* plan) {
  return iree_any_bit_set(
      plan->capabilities,
      LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_COMBINED_NON_NORMAL_CONDITION);
}

static bool loom_amdgpu_can_emit_fp8_pair_to_packed_u16_finite_path(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags,
    loom_amdgpu_fp8_decode_plan_capabilities_t normal_payload_capability) {
  const loom_amdgpu_fp8_decode_plan_flags_t required_plan_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32;
  if (!iree_all_bits_set(plan->flags, required_plan_flags) ||
      !iree_any_bit_set(plan->capabilities, normal_payload_capability) ||
      !loom_amdgpu_fp8_decode_value_is_finite(value_flags) ||
      !iree_any_bit_set(value_flags,
                        LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL)) {
    return false;
  }
  if (iree_any_bit_set(value_flags,
                       LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO)) {
    return true;
  }
  return loom_amdgpu_fp8_decode_plan_has_packed_zero_repair(plan);
}

bool loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  if (loom_amdgpu_can_emit_fp8_pair_to_packed_u16_finite_path(
          plan, value_flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_F16_PAYLOAD)) {
    return true;
  }
  const loom_amdgpu_fp8_decode_plan_flags_t required_plan_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32;
  return iree_all_bits_set(plan->flags, required_plan_flags) &&
         iree_any_bit_set(
             plan->capabilities,
             LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_F16_PAYLOAD) &&
         loom_amdgpu_fp8_decode_value_is_finite(value_flags) &&
         loom_amdgpu_fp8_decode_plan_has_packed_exact_repair(plan);
}

bool loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  return loom_amdgpu_fp8_pair_to_packed_bf16_missing_requirements(
             plan, value_flags) ==
         LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_NONE;
}

bool loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  if (!loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(plan, value_flags)) {
    return false;
  }
  if (!iree_any_bit_set(plan->flags,
                        LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_F32_PAIR)) {
    return true;
  }
  if (iree_any_bit_set(plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_BF16_PACK)) {
    return false;
  }

  const loom_amdgpu_fp8_packed_u16_repairs_t repairs =
      loom_amdgpu_fp8_pair_to_packed_bf16_repairs(plan, value_flags);
  const loom_amdgpu_fp8_packed_u16_repairs_t top_exponent_repairs =
      LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NAN |
      LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_INF;
  return !iree_any_bit_set(repairs, top_exponent_repairs);
}

typedef enum loom_amdgpu_fp8_packed_u16_repair_query_flag_bits_e {
  LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_NONE = 0u,
  LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_NORMAL_PATH_AVAILABLE = 1u << 0,
  LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_REPAIR_PATH_AVAILABLE = 1u << 1,
  LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_TOP_EXPONENT_REPAIRS = 1u << 2,
} loom_amdgpu_fp8_packed_u16_repair_query_flag_bits_t;
typedef uint32_t loom_amdgpu_fp8_packed_u16_repair_query_flags_t;

static loom_amdgpu_fp8_packed_u16_repairs_t
loom_amdgpu_fp8_pair_to_packed_u16_repairs(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags,
    loom_amdgpu_fp8_packed_u16_repair_query_flags_t query_flags) {
  loom_amdgpu_fp8_packed_u16_repairs_t repairs =
      LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NONE;
  const bool can_use_normal_path = iree_any_bit_set(
      query_flags,
      LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_NORMAL_PATH_AVAILABLE);
  const bool can_use_repair_path = iree_any_bit_set(
      query_flags,
      LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_REPAIR_PATH_AVAILABLE);
  if (!can_use_normal_path && !can_use_repair_path) {
    return repairs;
  }

  const bool value_non_zero =
      iree_any_bit_set(value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO);
  const bool value_not_subnormal = iree_any_bit_set(
      value_flags, LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL);
  if (!can_use_normal_path) {
    if (!value_not_subnormal) {
      repairs |= LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_SUBNORMAL;
    }
    if (!value_non_zero) {
      repairs |= LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_ZERO;
    }
    if (iree_any_bit_set(
            query_flags,
            LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_TOP_EXPONENT_REPAIRS)) {
      if (!iree_any_bit_set(value_flags,
                            LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN)) {
        repairs |= LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NAN;
      }
      if (plan->format.special_policy ==
              LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE &&
          !iree_any_bit_set(value_flags,
                            LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF)) {
        repairs |= LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_INF;
      }
    }
    return repairs;
  }

  if (!value_non_zero) {
    repairs |= LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_ZERO;
  }
  return repairs;
}

loom_amdgpu_fp8_packed_u16_repairs_t
loom_amdgpu_fp8_pair_to_packed_bf16_repairs(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  const bool can_use_normal_path =
      loom_amdgpu_can_emit_fp8_pair_to_packed_u16_finite_path(
          plan, value_flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_BF16_PAYLOAD);
  loom_amdgpu_fp8_packed_u16_repair_query_flags_t query_flags =
      LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_TOP_EXPONENT_REPAIRS;
  if (can_use_normal_path) {
    query_flags |=
        LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_NORMAL_PATH_AVAILABLE;
  }
  if (loom_amdgpu_fp8_decode_plan_has_packed_exact_repair(plan)) {
    query_flags |=
        LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_REPAIR_PATH_AVAILABLE;
  }
  return loom_amdgpu_fp8_pair_to_packed_u16_repairs(plan, value_flags,
                                                    query_flags);
}

iree_string_view_t loom_amdgpu_fp8_packed_bf16_strategy_key(
    bool exact_via_f16, loom_amdgpu_fp8_packed_u16_repairs_t repairs) {
  if (exact_via_f16) {
    return IREE_SV("fp8_packed_bf16_decode_exact_via_f16");
  }
  IREE_ASSERT_LT(repairs,
                 IREE_ARRAYSIZE(kLoomAmdgpuFp8PackedBf16RepairReasons));
  return kLoomAmdgpuFp8PackedBf16RepairReasons[repairs];
}

loom_amdgpu_fp8_packed_u16_repairs_t loom_amdgpu_fp8_pair_to_packed_f16_repairs(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  const bool can_use_normal_path =
      loom_amdgpu_can_emit_fp8_pair_to_packed_u16_finite_path(
          plan, value_flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_F16_PAYLOAD);
  loom_amdgpu_fp8_packed_u16_repair_query_flags_t query_flags =
      LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_REPAIR_PATH_AVAILABLE;
  if (can_use_normal_path) {
    query_flags |=
        LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_QUERY_FLAG_NORMAL_PATH_AVAILABLE;
  }
  return loom_amdgpu_fp8_pair_to_packed_u16_repairs(plan, value_flags,
                                                    query_flags);
}

iree_string_view_t loom_amdgpu_fp8_packed_f16_repair_reason_key(
    loom_amdgpu_fp8_packed_u16_repairs_t repairs) {
  IREE_ASSERT_LT(repairs, IREE_ARRAYSIZE(kLoomAmdgpuFp8PackedF16RepairReasons));
  return kLoomAmdgpuFp8PackedF16RepairReasons[repairs];
}

loom_amdgpu_fp8_packed_bf16_missing_requirements_t
loom_amdgpu_fp8_pair_to_packed_bf16_missing_requirements(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags) {
  loom_amdgpu_fp8_packed_bf16_missing_requirements_t missing_requirements =
      LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_NONE;
  const loom_amdgpu_fp8_decode_plan_flags_t required_plan_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32;
  if (!iree_all_bits_set(plan->flags, required_plan_flags)) {
    missing_requirements |=
        LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_PERMUTE_PACKET;
  }
  if (!iree_any_bit_set(
          plan->capabilities,
          LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_BF16_PAYLOAD)) {
    missing_requirements |=
        LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_PACKED_SHIFT_PACKET;
  }

  const bool has_exact_repair =
      loom_amdgpu_fp8_decode_plan_has_packed_exact_repair(plan);
  if (!has_exact_repair) {
    const loom_amdgpu_fp8_decode_value_flags_t required_value_flags =
        LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN |
        LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF;
    if (!iree_all_bits_set(value_flags, required_value_flags)) {
      missing_requirements |=
          LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_VALUE_FINITE;
    }
    if (!iree_all_bits_set(value_flags,
                           LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL)) {
      missing_requirements |=
          LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_VALUE_NOT_SUBNORMAL;
    }
  }

  if (has_exact_repair ||
      iree_any_bit_set(value_flags,
                       LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO)) {
    return missing_requirements;
  }

  if (!loom_amdgpu_fp8_decode_plan_has_packed_zero_repair(plan)) {
    missing_requirements |=
        LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_ZERO_REPAIR_PACKETS;
  }
  return missing_requirements;
}
