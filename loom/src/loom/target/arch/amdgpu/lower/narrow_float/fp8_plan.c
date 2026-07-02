// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <string.h>

#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8.h"

static uint32_t loom_amdgpu_fp8_subnormal_bf16_payload(
    const loom_scalar_type_fp8_format_t* format, uint32_t mantissa) {
  if (mantissa == 0) {
    return 0;
  }
  uint32_t leading_index = 0;
  for (uint32_t i = 1; i < format->mantissa_bits; ++i) {
    if ((mantissa & (UINT32_C(1) << i)) != 0) {
      leading_index = i;
    }
  }
  const uint32_t exponent =
      128u - format->exponent_bias - format->mantissa_bits + leading_index;
  const uint32_t fraction = (mantissa << (7u - leading_index)) & UINT32_C(0x7F);
  return (exponent << 7) | fraction;
}

static uint32_t loom_amdgpu_fp8_subnormal_f16_payload(
    const loom_scalar_type_fp8_format_t* format, uint32_t mantissa) {
  if (mantissa == 0) {
    return 0;
  }
  const int32_t source_power =
      1 - (int32_t)format->exponent_bias - (int32_t)format->mantissa_bits;
  int32_t leading_index = 0;
  for (uint32_t i = 1; i < format->mantissa_bits; ++i) {
    if ((mantissa & (UINT32_C(1) << i)) != 0) {
      leading_index = (int32_t)i;
    }
  }
  const int32_t exponent = source_power + leading_index;
  const int32_t f16_exponent = exponent + 15;
  if (f16_exponent > 0) {
    const uint32_t fraction =
        (mantissa << (10u - (uint32_t)leading_index)) & UINT32_C(0x3FF);
    return ((uint32_t)f16_exponent << 10) | fraction;
  }

  const int32_t subnormal_shift = source_power + 24;
  IREE_ASSERT_GE(subnormal_shift, 0);
  return mantissa << (uint32_t)subnormal_shift;
}

static uint32_t loom_amdgpu_fp8_subnormal_table_word(
    const loom_scalar_type_fp8_format_t* format, uint32_t mantissa_base) {
  return loom_amdgpu_fp8_subnormal_bf16_payload(format, mantissa_base) |
         (loom_amdgpu_fp8_subnormal_bf16_payload(format, mantissa_base + 1u)
          << 16);
}

static uint32_t loom_amdgpu_fp8_subnormal_byte_table_word(
    const loom_scalar_type_fp8_format_t* format, uint32_t byte_index,
    uint32_t mantissa_base) {
  uint32_t table_word = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t payload =
        loom_amdgpu_fp8_subnormal_bf16_payload(format, mantissa_base + i);
    table_word |= ((payload >> (byte_index * 8u)) & UINT32_C(0xFF)) << (i * 8u);
  }
  return table_word;
}

static uint32_t loom_amdgpu_fp8_subnormal_f16_byte_table_word(
    const loom_scalar_type_fp8_format_t* format, uint32_t byte_index,
    uint32_t mantissa_base) {
  uint32_t table_word = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t payload =
        loom_amdgpu_fp8_subnormal_f16_payload(format, mantissa_base + i);
    table_word |= ((payload >> (byte_index * 8u)) & UINT32_C(0xFF)) << (i * 8u);
  }
  return table_word;
}

static void loom_amdgpu_initialize_fp8_decode_format(
    loom_scalar_type_t element_type, loom_amdgpu_fp8_decode_plan_t* plan) {
  if (!loom_scalar_type_fp8_format(element_type, &plan->format)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 decode");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (plan->format.mantissa_bits == 2) {
    plan->subnormal_bf16_table_words[0] =
        loom_amdgpu_fp8_subnormal_table_word(&plan->format, 0);
    plan->subnormal_bf16_table_words[1] =
        loom_amdgpu_fp8_subnormal_table_word(&plan->format, 2);
  }
  if (plan->format.mantissa_bits <= 3) {
    for (uint32_t byte_index = 0; byte_index < LOOM_AMDGPU_FP8_BF16_BYTE_COUNT;
         ++byte_index) {
      plan->subnormal_bf16_byte_table_words[byte_index][0] =
          loom_amdgpu_fp8_subnormal_byte_table_word(&plan->format, byte_index,
                                                    0);
      plan->subnormal_bf16_byte_table_words[byte_index][1] =
          loom_amdgpu_fp8_subnormal_byte_table_word(&plan->format, byte_index,
                                                    4);
    }
    for (uint32_t byte_index = 0; byte_index < LOOM_AMDGPU_FP8_U16_BYTE_COUNT;
         ++byte_index) {
      plan->subnormal_f16_byte_table_words[byte_index][0] =
          loom_amdgpu_fp8_subnormal_f16_byte_table_word(&plan->format,
                                                        byte_index, 0);
      plan->subnormal_f16_byte_table_words[byte_index][1] =
          loom_amdgpu_fp8_subnormal_f16_byte_table_word(&plan->format,
                                                        byte_index, 4);
    }
  }
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

  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  if (loom_amdgpu_fp8_native_descriptor_refs(element_type, LOOM_SCALAR_TYPE_F32,
                                             &native_refs)) {
    const loom_amdgpu_fp8_decode_plan_descriptor_row_t native_pair_row = {
        .descriptor_ref = native_refs.pair,
        .descriptor_offset =
            offsetof(loom_amdgpu_fp8_decode_plan_t, native_f32_pair_descriptor),
        .present_flag = LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_F32_PAIR,
    };
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

static int loom_amdgpu_fp8_decode_plan_cache_state_key;

iree_status_t loom_amdgpu_get_fp8_decode_plan(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    const loom_amdgpu_fp8_decode_plan_t** out_plan) {
  IREE_ASSERT_LE(LOOM_SCALAR_TYPE_COUNT_, 32);
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
