// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <string.h>

#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8_tables.h"

static_assert(IREE_ARRAYSIZE(kLoomAmdgpuFp8SubnormalTableRows) + 1u <= 32,
              "FP8 decode plan cache stores exact rows and one compatibility "
              "variant in one u32 bitset");

static bool loom_amdgpu_fp8_subnormal_table_row_index(
    loom_value_fact_numeric_format_flags_t source_format,
    uint32_t* out_row_index) {
  return loom_amdgpu_fp8_format_index(source_format, out_row_index);
}

static const loom_amdgpu_fp8_subnormal_table_row_t*
loom_amdgpu_find_fp8_subnormal_table_row(
    loom_value_fact_numeric_format_flags_t source_format,
    uint32_t* out_row_index) {
  *out_row_index = 0;
  if (!loom_amdgpu_fp8_subnormal_table_row_index(source_format,
                                                 out_row_index)) {
    return NULL;
  }
  IREE_ASSERT_LT(*out_row_index,
                 IREE_ARRAYSIZE(kLoomAmdgpuFp8SubnormalTableRows));
  const loom_amdgpu_fp8_subnormal_table_row_t* row =
      &kLoomAmdgpuFp8SubnormalTableRows[*out_row_index];
  IREE_ASSERT_EQ(row->source_format, source_format);
  return row;
}

static void loom_amdgpu_initialize_fp8_decode_format_from_row(
    const loom_amdgpu_fp8_subnormal_table_row_t* row,
    loom_amdgpu_fp8_decode_plan_t* plan) {
  if (row == NULL) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 decode");
    IREE_BUILTIN_UNREACHABLE();
  }
  plan->format = row->format;
  memcpy(plan->subnormal_bf16_table_words, row->subnormal_bf16_table_words,
         sizeof(plan->subnormal_bf16_table_words));
  memcpy(plan->subnormal_bf16_byte_table_words,
         row->subnormal_bf16_byte_table_words,
         sizeof(plan->subnormal_bf16_byte_table_words));
  memcpy(plan->subnormal_f16_byte_table_words,
         row->subnormal_f16_byte_table_words,
         sizeof(plan->subnormal_f16_byte_table_words));
}

static void loom_amdgpu_mark_fp8_native_f32_pair_decode_plan_flag(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_amdgpu_fp8_decode_plan_t* plan) {
  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  if (descriptor_set == NULL ||
      !loom_amdgpu_fp8_native_descriptor_refs(
          descriptor_source_format, LOOM_SCALAR_TYPE_F32, &native_refs) ||
      native_refs.pair == LOOM_AMDGPU_DESCRIPTOR_REF_NONE ||
      !loom_amdgpu_descriptor_set_has_ref(descriptor_set, native_refs.pair)) {
    return;
  }
  plan->flags |= LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_F32_PAIR;
}

static void loom_amdgpu_mark_fp8_decode_plan_descriptor_flag(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fp8_decode_plan_descriptor_row_t* row,
    loom_amdgpu_fp8_decode_plan_t* plan) {
  if (descriptor_set != NULL &&
      row->descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set, row->descriptor_ref)) {
    plan->flags |= row->present_flag;
  }
}

static void loom_amdgpu_mark_fp8_decode_plan_descriptor_flags(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_amdgpu_fp8_decode_plan_t* plan) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFp8DecodePlanDescriptorRows); ++i) {
    loom_amdgpu_mark_fp8_decode_plan_descriptor_flag(
        descriptor_set, &kLoomAmdgpuFp8DecodePlanDescriptorRows[i], plan);
  }
  loom_amdgpu_mark_fp8_native_f32_pair_decode_plan_flag(
      descriptor_set, descriptor_source_format, plan);
}

static bool loom_amdgpu_fp8_decode_format_has_packed_exact_repair(
    const loom_scalar_type_fp8_format_t* format) {
  if (format->exponent_bits + format->mantissa_bits != 7 ||
      format->mantissa_bits > 3) {
    return false;
  }
  switch (format->special_policy) {
    case LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN:
      return true;
    case LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE:
      return true;
    default:
      return false;
  }
}

// Returns whether finite values in |format| can be mapped through F16 without
// losing FP8 subnormals. The raw unsigned FP8 payload is first placed in the
// corresponding F16 exponent/mantissa fields and scaled by the bias delta. The
// smallest FP8 subnormal must become a normal F16 value, and the largest finite
// value must remain finite, so the resulting F16 payload can then be rebased to
// BF16 with integer operations.
static bool loom_amdgpu_fp8_decode_format_has_exact_bf16_via_f16(
    const loom_scalar_type_fp8_format_t* format) {
  if (format->exponent_bits + format->mantissa_bits != 7u ||
      format->exponent_bits > 5u || format->mantissa_bits < 2u ||
      format->mantissa_bits > 10u || format->exponent_bias > 15u) {
    return false;
  }

  const int32_t minimum_unbiased_exponent =
      1 - (int32_t)format->exponent_bias - (int32_t)format->mantissa_bits;
  if (minimum_unbiased_exponent < -14) {
    return false;
  }

  uint32_t maximum_finite_exponent =
      (UINT32_C(1) << format->exponent_bits) - 1u;
  if (format->special_policy == LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE) {
    --maximum_finite_exponent;
  } else if (format->special_policy !=
             LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN) {
    return false;
  }
  if (maximum_finite_exponent >= 31u) {
    return false;
  }
  return (int32_t)maximum_finite_exponent - (int32_t)format->exponent_bias <=
         15;
}

static bool loom_amdgpu_fp8_decode_flags_have_packed_normal_payload(
    loom_amdgpu_fp8_decode_plan_flags_t flags, uint32_t packed_exponent_bias) {
  if (packed_exponent_bias != 0 &&
      iree_any_bit_set(flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16)) {
    return true;
  }
  if (!iree_any_bit_set(flags,
                        LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_LSHLREV_B16)) {
    return false;
  }
  return packed_exponent_bias == 0 ||
         iree_any_bit_set(flags,
                          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ADD_U16);
}

static void loom_amdgpu_initialize_fp8_decode_plan_capabilities(
    loom_amdgpu_fp8_decode_plan_t* plan) {
  plan->capabilities = LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_NONE;

  const loom_amdgpu_fp8_decode_plan_flags_t packed_exact_repair_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ADD_U16 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ASHRREV_I16;
  if (loom_amdgpu_fp8_decode_format_has_packed_exact_repair(&plan->format) &&
      iree_all_bits_set(plan->flags, packed_exact_repair_flags)) {
    plan->capabilities |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_EXACT_REPAIR;
  }

  const loom_amdgpu_fp8_decode_plan_flags_t packed_zero_repair_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MIN_U16 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_LO_U16;
  if (iree_all_bits_set(plan->flags, packed_zero_repair_flags)) {
    plan->capabilities |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_ZERO_REPAIR;
  }

  if (iree_any_bit_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_CMP_LG_U64 |
              LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_CMP_LG_U64_SRC1_INLINE)) {
    plan->capabilities |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_MASK_REPAIR_SPLIT;
  }

  if (iree_all_bits_set(
          plan->flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_CMP_LG_U64_SRC1_INLINE)) {
    plan->capabilities |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_INLINE_SGPR64_ZERO_COMPARE;
  }

  if (plan->format.special_policy ==
          LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN &&
      iree_any_bit_set(plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAX_U16 |
                           LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MIN_U16)) {
    plan->capabilities |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_COMBINED_FINITE_NAN_CONDITION;
  }

  const loom_amdgpu_fp8_decode_plan_flags_t non_normal_condition_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ADD_U16 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAX_U16;
  if (iree_all_bits_set(plan->flags, non_normal_condition_flags)) {
    plan->capabilities |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_COMBINED_NON_NORMAL_CONDITION;
  }

  const uint32_t f16_exponent_bias = (15u - plan->format.exponent_bias) << 10;
  const uint32_t packed_f16_exponent_bias =
      f16_exponent_bias | (f16_exponent_bias << 16);
  if (loom_amdgpu_fp8_decode_flags_have_packed_normal_payload(
          plan->flags, packed_f16_exponent_bias)) {
    plan->capabilities |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_F16_PAYLOAD;
  }

  const uint32_t bf16_exponent_bias = (127u - plan->format.exponent_bias) << 7;
  const uint32_t packed_bf16_exponent_bias =
      bf16_exponent_bias | (bf16_exponent_bias << 16);
  if (loom_amdgpu_fp8_decode_flags_have_packed_normal_payload(
          plan->flags, packed_bf16_exponent_bias)) {
    plan->capabilities |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_BF16_PAYLOAD;
  }

  const loom_amdgpu_fp8_decode_plan_flags_t exact_bf16_via_f16_flags =
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MIN_U16 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_LSHRREV_B16 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_F16 |
      LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16;
  if (loom_amdgpu_fp8_decode_format_has_exact_bf16_via_f16(&plan->format) &&
      iree_all_bits_set(plan->flags, exact_bf16_via_f16_flags)) {
    plan->capabilities |=
        LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_EXACT_BF16_VIA_F16;
  }
}

void loom_amdgpu_initialize_fp8_decode_plan_from_descriptor_set(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_numeric_format_flags_t source_format,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_amdgpu_fp8_decode_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  uint32_t unused_row_index = 0;
  const loom_amdgpu_fp8_subnormal_table_row_t* row =
      loom_amdgpu_find_fp8_subnormal_table_row(source_format,
                                               &unused_row_index);
  loom_amdgpu_mark_fp8_decode_plan_descriptor_flags(
      descriptor_set, descriptor_source_format, out_plan);
  loom_amdgpu_initialize_fp8_decode_format_from_row(row, out_plan);
  loom_amdgpu_initialize_fp8_decode_plan_capabilities(out_plan);
}

static iree_status_t loom_amdgpu_resolve_fp8_decode_plan_descriptor(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fp8_decode_plan_descriptor_row_t* row,
    loom_amdgpu_fp8_decode_plan_t* plan) {
  uint8_t* plan_bytes = (uint8_t*)plan;
  loom_low_lower_resolved_descriptor_t* descriptor =
      (loom_low_lower_resolved_descriptor_t*)(plan_bytes +
                                              row->descriptor_offset);
  bool has_descriptor = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, row->descriptor_ref, descriptor, &has_descriptor));
  if (has_descriptor) {
    plan->flags |= row->present_flag;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_resolve_fp8_decode_plan_descriptors(
    loom_low_lower_context_t* context,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_amdgpu_fp8_decode_plan_t* plan) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFp8DecodePlanDescriptorRows); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_fp8_decode_plan_descriptor(
        context, &kLoomAmdgpuFp8DecodePlanDescriptorRows[i], plan));
  }
  loom_amdgpu_mark_fp8_native_f32_pair_decode_plan_flag(
      loom_low_lower_context_descriptor_set(context), descriptor_source_format,
      plan);

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_initialize_fp8_decode_plan(
    loom_low_lower_context_t* context,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    const loom_amdgpu_fp8_subnormal_table_row_t* row,
    loom_amdgpu_fp8_decode_plan_t* plan) {
  memset(plan, 0, sizeof(*plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_fp8_decode_plan_descriptors(
      context, descriptor_source_format, plan));

  loom_amdgpu_initialize_fp8_decode_format_from_row(row, plan);
  loom_amdgpu_initialize_fp8_decode_plan_capabilities(plan);
  return iree_ok_status();
}

typedef struct loom_amdgpu_fp8_decode_plan_cache_t {
  // Plan-variant bits whose entries have been initialized.
  uint32_t initialized_plan_bits;
  // Function-local exact-format plans plus one finite IEEE E4M3 variant.
  loom_amdgpu_fp8_decode_plan_t
      plans[IREE_ARRAYSIZE(kLoomAmdgpuFp8SubnormalTableRows) + 1u];
} loom_amdgpu_fp8_decode_plan_cache_t;

static int loom_amdgpu_fp8_decode_plan_cache_state_key;

iree_status_t loom_amdgpu_get_fp8_decode_plan(
    loom_low_lower_context_t* context,
    loom_value_fact_numeric_format_flags_t source_format,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    const loom_amdgpu_fp8_decode_plan_t** out_plan) {
  *out_plan = NULL;
  uint32_t row_index = 0;
  const loom_amdgpu_fp8_subnormal_table_row_t* row =
      loom_amdgpu_find_fp8_subnormal_table_row(source_format, &row_index);
  if (row == NULL) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 decode");
    IREE_BUILTIN_UNREACHABLE();
  }

  uint32_t plan_index = row_index;
  if (descriptor_source_format != LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE &&
      descriptor_source_format != source_format) {
    IREE_ASSERT_EQ(source_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3);
    IREE_ASSERT_EQ(descriptor_source_format,
                   LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN);
    plan_index = IREE_ARRAYSIZE(kLoomAmdgpuFp8SubnormalTableRows);
  }

  loom_amdgpu_fp8_decode_plan_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fp8_decode_plan_cache_state_key, sizeof(*cache),
      (void**)&cache));
  const uint32_t plan_bit = UINT32_C(1) << plan_index;
  if ((cache->initialized_plan_bits & plan_bit) == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fp8_decode_plan(
        context, descriptor_source_format, row, &cache->plans[plan_index]));
    cache->initialized_plan_bits |= plan_bit;
  }
  *out_plan = &cache->plans[plan_index];
  return iree_ok_status();
}
