// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <string.h>

#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8.h"

typedef struct loom_amdgpu_fp8_subnormal_table_row_t {
  // Encoded FP8 source format used by the decode plan.
  loom_scalar_type_fp8_format_t format;
  // Packed unsigned BF16 payloads for two-bit mantissas.
  uint32_t subnormal_bf16_table_words[2];
  // Packed BF16 payload byte tables for three-bit mantissas.
  uint32_t subnormal_bf16_byte_table_words
      [LOOM_AMDGPU_FP8_BF16_BYTE_COUNT]
      [LOOM_AMDGPU_FP8_BF16_BYTE_TABLE_WORD_COUNT];
  // Packed F16 payload byte tables.
  uint32_t
      subnormal_f16_byte_table_words[LOOM_AMDGPU_FP8_U16_BYTE_COUNT]
                                    [LOOM_AMDGPU_FP8_U16_BYTE_TABLE_WORD_COUNT];
} loom_amdgpu_fp8_subnormal_table_row_t;

#define LOOM_AMDGPU_FP8_SUBNORMAL_TABLE_ROW(                                   \
    row_element_type, row_exponent_bits, row_mantissa_bits, row_exponent_bias, \
    row_special_policy, bf16_table_0, bf16_table_1, bf16_byte_0_0,             \
    bf16_byte_0_1, bf16_byte_1_0, bf16_byte_1_1, f16_byte_0_0, f16_byte_0_1,   \
    f16_byte_1_0, f16_byte_1_1)                                                \
  [row_element_type] = {                                                       \
      .format =                                                                \
          {                                                                    \
              .exponent_bits = row_exponent_bits,                              \
              .mantissa_bits = row_mantissa_bits,                              \
              .exponent_bias = row_exponent_bias,                              \
              .special_policy = row_special_policy,                            \
          },                                                                   \
      .subnormal_bf16_table_words = {bf16_table_0, bf16_table_1},              \
      .subnormal_bf16_byte_table_words =                                       \
          {                                                                    \
              {bf16_byte_0_0, bf16_byte_0_1},                                  \
              {bf16_byte_1_0, bf16_byte_1_1},                                  \
          },                                                                   \
      .subnormal_f16_byte_table_words =                                        \
          {                                                                    \
              {f16_byte_0_0, f16_byte_0_1},                                    \
              {f16_byte_1_0, f16_byte_1_1},                                    \
          },                                                                   \
  }

static const loom_amdgpu_fp8_subnormal_table_row_t
    kLoomAmdgpuFp8SubnormalTableRows[LOOM_SCALAR_TYPE_COUNT_] = {
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8_subnormal_table_rows.inl"
};

#undef LOOM_AMDGPU_FP8_SUBNORMAL_TABLE_ROW

static void loom_amdgpu_initialize_fp8_decode_format(
    loom_scalar_type_t element_type, loom_amdgpu_fp8_decode_plan_t* plan) {
  const loom_amdgpu_fp8_subnormal_table_row_t* row =
      element_type < LOOM_SCALAR_TYPE_COUNT_
          ? &kLoomAmdgpuFp8SubnormalTableRows[element_type]
          : NULL;
  if (row == NULL || row->format.exponent_bits == 0) {
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

typedef struct loom_amdgpu_fp8_decode_plan_descriptor_row_t {
  // Descriptor ref to probe in the active target descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Byte offset of the resolved descriptor field in the decode plan.
  size_t descriptor_offset;
  // Plan flag raised when the descriptor ref is present.
  loom_amdgpu_fp8_decode_plan_flags_t present_flag;
} loom_amdgpu_fp8_decode_plan_descriptor_row_t;

#define LOOM_AMDGPU_FP8_DECODE_PLAN_DESCRIPTOR_ROW(ref, field, flag)       \
  {                                                                        \
      .descriptor_ref = ref,                                               \
      .descriptor_offset = offsetof(loom_amdgpu_fp8_decode_plan_t, field), \
      .present_flag = flag,                                                \
  }

static const loom_amdgpu_fp8_decode_plan_descriptor_row_t
    kLoomAmdgpuFp8DecodePlanDescriptorRows[] = {
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8_decode_plan_descriptor_rows.inl"
};

#undef LOOM_AMDGPU_FP8_DECODE_PLAN_DESCRIPTOR_ROW

static bool loom_amdgpu_fp8_native_f32_pair_decode_plan_descriptor_row(
    loom_scalar_type_t element_type,
    loom_amdgpu_fp8_decode_plan_descriptor_row_t* out_row) {
  *out_row = (loom_amdgpu_fp8_decode_plan_descriptor_row_t){0};
  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  if (!loom_amdgpu_fp8_native_descriptor_refs(
          element_type, LOOM_SCALAR_TYPE_F32, &native_refs)) {
    return false;
  }
  *out_row = (loom_amdgpu_fp8_decode_plan_descriptor_row_t){
      .descriptor_ref = native_refs.pair,
      .descriptor_offset =
          offsetof(loom_amdgpu_fp8_decode_plan_t, native_f32_pair_descriptor),
      .present_flag = LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_F32_PAIR,
  };
  return true;
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
    loom_scalar_type_t element_type, loom_amdgpu_fp8_decode_plan_t* plan) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFp8DecodePlanDescriptorRows); ++i) {
    loom_amdgpu_mark_fp8_decode_plan_descriptor_flag(
        descriptor_set, &kLoomAmdgpuFp8DecodePlanDescriptorRows[i], plan);
  }

  loom_amdgpu_fp8_decode_plan_descriptor_row_t native_pair_row = {0};
  if (loom_amdgpu_fp8_native_f32_pair_decode_plan_descriptor_row(
          element_type, &native_pair_row)) {
    loom_amdgpu_mark_fp8_decode_plan_descriptor_flag(descriptor_set,
                                                     &native_pair_row, plan);
  }
}

void loom_amdgpu_initialize_fp8_decode_plan_from_descriptor_set(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t element_type, loom_amdgpu_fp8_decode_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  loom_amdgpu_mark_fp8_decode_plan_descriptor_flags(descriptor_set,
                                                    element_type, out_plan);
  loom_amdgpu_initialize_fp8_decode_format(element_type, out_plan);
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
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    loom_amdgpu_fp8_decode_plan_t* plan) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFp8DecodePlanDescriptorRows); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_fp8_decode_plan_descriptor(
        context, &kLoomAmdgpuFp8DecodePlanDescriptorRows[i], plan));
  }

  loom_amdgpu_fp8_decode_plan_descriptor_row_t native_pair_row = {0};
  if (loom_amdgpu_fp8_native_f32_pair_decode_plan_descriptor_row(
          element_type, &native_pair_row)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_fp8_decode_plan_descriptor(
        context, &native_pair_row, plan));
  }

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_initialize_fp8_decode_plan(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    loom_amdgpu_fp8_decode_plan_t* plan) {
  memset(plan, 0, sizeof(*plan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_fp8_decode_plan_descriptors(
      context, element_type, plan));

  loom_amdgpu_initialize_fp8_decode_format(element_type, plan);
  return iree_ok_status();
}

typedef struct loom_amdgpu_fp8_decode_plan_cache_t {
  // Scalar-type bits whose plan entries have been initialized.
  uint32_t initialized_type_bits;
  // Function-local decode plans keyed by loom_scalar_type_t.
  loom_amdgpu_fp8_decode_plan_t plans[LOOM_SCALAR_TYPE_COUNT_];
} loom_amdgpu_fp8_decode_plan_cache_t;
static_assert(LOOM_SCALAR_TYPE_COUNT_ <= 32,
              "FP8 decode plan cache stores scalar types in one u32 bitset");

static int loom_amdgpu_fp8_decode_plan_cache_state_key;

iree_status_t loom_amdgpu_get_fp8_decode_plan(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    const loom_amdgpu_fp8_decode_plan_t** out_plan) {
  IREE_ASSERT_LT(element_type, LOOM_SCALAR_TYPE_COUNT_);
  *out_plan = NULL;
  loom_amdgpu_fp8_decode_plan_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fp8_decode_plan_cache_state_key, sizeof(*cache),
      (void**)&cache));
  const uint32_t type_bit = UINT32_C(1) << element_type;
  if ((cache->initialized_type_bits & type_bit) == 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fp8_decode_plan(
        context, element_type, &cache->plans[element_type]));
    cache->initialized_type_bits |= type_bit;
  }
  *out_plan = &cache->plans[element_type];
  return iree_ok_status();
}
