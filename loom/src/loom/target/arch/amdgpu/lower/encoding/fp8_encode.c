// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/encoding/fp8_encode.h"

#include "loom/ir/attribute.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

#define LOOM_AMDGPU_FP8_E4M3_POSITIVE_MAXIMUM_BITS UINT32_C(0x43E00000)
#define LOOM_AMDGPU_FP8_E4M3_NEGATIVE_MAXIMUM_BITS UINT32_C(0xC3E00000)
#define LOOM_AMDGPU_F32_MAGNITUDE_MASK UINT32_C(0x7FFFFFFF)
#define LOOM_AMDGPU_F32_INFINITY_MAGNITUDE UINT32_C(0x7F800000)
#define LOOM_AMDGPU_FP8_NAN_ENCODING UINT32_C(0x7F)
#define LOOM_AMDGPU_FP8_LOW_PAIR_SIGN_MASK UINT32_C(0x00008080)
#define LOOM_AMDGPU_FP8_HIGH_PAIR_SIGN_MASK UINT32_C(0x80800000)
#define LOOM_AMDGPU_FP8_PACKED_MAGNITUDE_MASK UINT32_C(0x7F7F7F7F)
#define LOOM_AMDGPU_FP8_PACKED_BYTE_HIGH_BITS UINT32_C(0x80808080)
#define LOOM_AMDGPU_FP8_E5M2_NAN_CONDITION_BIAS UINT32_C(0x03030303)
#define LOOM_AMDGPU_FP8_PACKED_F16_SIGN_MASK UINT32_C(0x00800080)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_ROUNDING_SHIFT UINT32_C(0x00070007)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_SIGN_SHIFT UINT32_C(0x00080008)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_CONDITION_MASK_SHIFT \
  UINT32_C(0x000F000F)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_MAXIMUM_MAGNITUDE UINT32_C(0x5F005F00)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_ROUNDING_BIAS UINT32_C(0x003F003F)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_NORMAL_ADJUSTMENT UINT32_C(0xFFC0FFC0)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_SUBNORMAL_MAGIC UINT32_C(0x40004000)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_SUBNORMAL_ADJUSTMENT \
  UINT32_C(0xC000C000)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_SUBNORMAL_CONDITION_BIAS \
  UINT32_C(0xDC00DC00)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_NAN_CONDITION_BIAS UINT32_C(0x03FF03FF)
#define LOOM_AMDGPU_FP8_PACKED_F16_E4M3_NAN_ENCODING UINT32_C(0x007F007F)
#define LOOM_AMDGPU_FP8_PACKED_F16_PACK_SELECTOR UINT32_C(0x02000604)
#define LOOM_AMDGPU_FP8_PACKED_F16_E5M2_LANE_SHIFT UINT32_C(0x00080008)
#define LOOM_AMDGPU_FP8_PACKED_F16_E5M2_ROUNDING_BIAS_AND_NAN \
  UINT32_C(0x007F007F)
#define LOOM_AMDGPU_FP8_PACKED_F16_E5M2_NAN_CONDITION_BIAS UINT32_C(0x03FF03FF)
#define LOOM_AMDGPU_FP8_PACKED_F16_E5M2_CONDITION_MASK_SHIFT \
  UINT32_C(0x000F000F)

static uint32_t loom_amdgpu_fp8_mantissa_mask(
    const loom_scalar_type_fp8_format_t* format) {
  return (UINT32_C(1) << format->mantissa_bits) - 1u;
}

static uint32_t loom_amdgpu_fp8_top_exponent(
    const loom_scalar_type_fp8_format_t* format) {
  return (UINT32_C(1) << format->exponent_bits) - 1u;
}

static uint32_t loom_amdgpu_fp8_maximum_finite_payload(
    const loom_scalar_type_fp8_format_t* format) {
  const uint32_t mantissa_mask = loom_amdgpu_fp8_mantissa_mask(format);
  const uint32_t top_exponent = loom_amdgpu_fp8_top_exponent(format);
  uint32_t exponent = top_exponent;
  uint32_t mantissa = mantissa_mask;
  switch (format->special_policy) {
    case LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE:
      exponent -= 1u;
      break;
    case LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN:
      mantissa -= 1u;
      break;
    case LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN_UNSIGNED_ZERO:
      break;
    default:
      IREE_ASSERT_UNREACHABLE("invalid exact FP8 special-value policy");
      IREE_BUILTIN_UNREACHABLE();
  }
  return (exponent << format->mantissa_bits) | mantissa;
}

static uint32_t loom_amdgpu_fp8_f32_bits_from_normal_payload(
    const loom_scalar_type_fp8_format_t* format, uint32_t payload) {
  const uint32_t mantissa_mask = loom_amdgpu_fp8_mantissa_mask(format);
  const uint32_t exponent = payload >> format->mantissa_bits;
  const uint32_t mantissa = payload & mantissa_mask;
  const uint32_t f32_exponent = exponent + 127u - format->exponent_bias;
  return (f32_exponent << 23) | (mantissa << (23u - format->mantissa_bits));
}

static uint32_t loom_amdgpu_fp8_encode_maximum_magnitude(
    const loom_scalar_type_fp8_format_t* format) {
  // Clamping IEEE inputs to the overflow midpoint lets ordinary RNE produce
  // infinity without allowing larger finite inputs to spill into NaN payloads.
  if (format->special_policy == LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_IEEE) {
    const uint32_t maximum_finite_exponent =
        loom_amdgpu_fp8_top_exponent(format) - 1u;
    const uint32_t f32_exponent =
        maximum_finite_exponent + 127u - format->exponent_bias;
    const uint32_t overflow_midpoint_mantissa =
        (UINT32_C(1) << (format->mantissa_bits + 1u)) - 1u;
    return (f32_exponent << 23) |
           (overflow_midpoint_mantissa << (23u - format->mantissa_bits - 1u));
  }
  return loom_amdgpu_fp8_f32_bits_from_normal_payload(
      format, loom_amdgpu_fp8_maximum_finite_payload(format));
}

static uint32_t loom_amdgpu_fp8_encode_nan_payload(
    const loom_scalar_type_fp8_format_t* format) {
  if (format->special_policy ==
      LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN_UNSIGNED_ZERO) {
    return UINT32_C(0x80);
  }
  return (loom_amdgpu_fp8_top_exponent(format) << format->mantissa_bits) |
         loom_amdgpu_fp8_mantissa_mask(format);
}

static bool loom_amdgpu_fp8_encode_has_refs(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_descriptor_ref_t* descriptor_refs,
    iree_host_size_t descriptor_ref_count) {
  return loom_amdgpu_descriptor_set_has_all_refs(
      descriptor_set, descriptor_refs, descriptor_ref_count);
}

bool loom_amdgpu_fp8_encode_plan_is_software(
    const loom_amdgpu_fp8_encode_plan_t* plan) {
  return plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E4M3 ||
         plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E5M2 ||
         plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2;
}

bool loom_amdgpu_fp8_encode_plan_has_packed_f16_e4m3(
    const loom_amdgpu_fp8_encode_plan_t* plan) {
  return iree_any_bit_set(plan->flags,
                          LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_PACKED_F16_E4M3);
}

bool loom_amdgpu_fp8_encode_plan_has_packed_f16_e5m2(
    const loom_amdgpu_fp8_encode_plan_t* plan) {
  return iree_any_bit_set(plan->flags,
                          LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_PACKED_F16_E5M2);
}

static bool loom_amdgpu_fp8_encode_plan_is_f32_software(
    const loom_amdgpu_fp8_encode_plan_t* plan) {
  return plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E4M3 ||
         plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E5M2;
}

static uint32_t loom_amdgpu_fp8_encode_sign_permute_selector(
    const loom_amdgpu_fp8_encode_plan_t* plan, bool high_pair) {
  const uint32_t sign_byte_index =
      plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2 ? 1u : 3u;
  const uint32_t low_pair_selector =
      (sign_byte_index << 8) | (4u + sign_byte_index);
  return high_pair ? low_pair_selector << 16 : low_pair_selector;
}

bool loom_amdgpu_fp8_encode_plan_is_fnuz_bridge(
    const loom_amdgpu_fp8_encode_plan_t* plan) {
  return plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F32_FNUZ_BRIDGE_E4M3 ||
         plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F32_FNUZ_BRIDGE_E5M2;
}

bool loom_amdgpu_fp8_encode_plan_canonicalizes_native_nan(
    const loom_amdgpu_fp8_encode_plan_t* plan) {
  return iree_any_bit_set(
      plan->flags,
      LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_NATIVE_NAN_CANONICALIZATION);
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_select_fp8_encode_sign_insert_descriptor(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT)) {
    return LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT;
  }
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32)) {
    return LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32;
  }
  return LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
}

static bool loom_amdgpu_select_fnuz_bridge_fp8_encode_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_amdgpu_fp8_encode_plan_t* out_plan) {
  loom_amdgpu_fp8_encode_kind_t kind = LOOM_AMDGPU_FP8_ENCODE_KIND_NONE;
  loom_amdgpu_descriptor_ref_t low_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_descriptor_ref_t high_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (result_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F32_FNUZ_BRIDGE_E4M3;
      low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_FP8_F32_FNUZ_LOW;
      high_descriptor_ref =
          LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_FP8_F32_FNUZ_HIGH;
      break;
    case LOOM_SCALAR_TYPE_F8E5M2:
      kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F32_FNUZ_BRIDGE_E5M2;
      low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF8_F32_FNUZ_LOW;
      high_descriptor_ref =
          LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF8_F32_FNUZ_HIGH;
      break;
    default:
      return false;
  }

  loom_amdgpu_i8_pack_permute_plan_t sign_permute = {0};
  loom_amdgpu_select_i8_pack_permute_plan(descriptor_set, &sign_permute);
  if (sign_permute.kind == LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_NONE) {
    return false;
  }

  const loom_amdgpu_descriptor_ref_t sign_insert_descriptor_ref =
      loom_amdgpu_select_fp8_encode_sign_insert_descriptor(descriptor_set);

  const bool needs_sgpr_constants =
      sign_permute.kind == LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_REGISTER_SELECTOR ||
      sign_insert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32;
  const loom_amdgpu_descriptor_ref_t required_refs[] = {
      low_descriptor_ref,
      high_descriptor_ref,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32_SRC0_INLINE,
      sign_insert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      needs_sgpr_constants ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32
                           : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      source_type == LOOM_SCALAR_TYPE_F16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      source_type == LOOM_SCALAR_TYPE_BF16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      source_type == LOOM_SCALAR_TYPE_BF16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  if (!loom_amdgpu_fp8_encode_has_refs(descriptor_set, required_refs,
                                       IREE_ARRAYSIZE(required_refs))) {
    return false;
  }

  *out_plan = (loom_amdgpu_fp8_encode_plan_t){
      .kind = kind,
      .low_descriptor_ref = low_descriptor_ref,
      .high_descriptor_ref = high_descriptor_ref,
      .sign_insert_descriptor_ref = sign_insert_descriptor_ref,
      .packed_i8_permute = sign_permute,
  };
  return true;
}

static bool loom_amdgpu_descriptor_set_requires_native_fp8_nan_repair(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  return loom_amdgpu_descriptor_set_info_has_flags(
      descriptor_set_info,
      LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_OCP_FP8_NONCANONICAL_NAN);
}

static bool loom_amdgpu_select_native_fp8_nan_canonicalization(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_fp8_encode_kind_t kind,
    loom_amdgpu_fp8_encode_plan_t* inout_plan) {
  loom_amdgpu_i8_pack_permute_plan_t sign_permute = {0};
  loom_amdgpu_select_i8_pack_permute_plan(descriptor_set, &sign_permute);
  if (sign_permute.kind == LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_NONE) {
    return false;
  }

  const loom_amdgpu_descriptor_ref_t sign_insert_descriptor_ref =
      loom_amdgpu_select_fp8_encode_sign_insert_descriptor(descriptor_set);
  if (sign_insert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return false;
  }
  const bool needs_sgpr_constants =
      sign_permute.kind == LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_REGISTER_SELECTOR ||
      sign_insert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32;
  const bool repairs_e5m2 = kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR;
  const loom_amdgpu_descriptor_ref_t required_refs[] = {
      needs_sgpr_constants ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32
                           : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      repairs_e5m2 ? LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT
                   : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      repairs_e5m2 ? LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT
                   : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      repairs_e5m2 ? LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT
                   : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      repairs_e5m2 ? LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32
                   : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  if (!loom_amdgpu_fp8_encode_has_refs(descriptor_set, required_refs,
                                       IREE_ARRAYSIZE(required_refs))) {
    return false;
  }

  inout_plan->flags |=
      LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_NATIVE_NAN_CANONICALIZATION;
  inout_plan->sign_insert_descriptor_ref = sign_insert_descriptor_ref;
  inout_plan->packed_i8_permute = sign_permute;
  return true;
}

static bool loom_amdgpu_select_software_fp8_encode_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_value_fact_numeric_format_flags_t result_format,
    loom_amdgpu_fp8_encode_plan_t* out_plan) {
  const loom_amdgpu_descriptor_ref_t round_add_descriptor_ref =
      loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD3_U32_SRC2_LIT)
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD3_U32_SRC2_LIT
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const loom_amdgpu_descriptor_ref_t sign_insert_descriptor_ref =
      loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT)
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_i8_pack_permute_plan_t packed_i8_permute = {0};
  loom_amdgpu_select_i8_pack_permute_plan(descriptor_set, &packed_i8_permute);

  if (source_type == LOOM_SCALAR_TYPE_F16 &&
      result_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2) {
    const loom_amdgpu_descriptor_ref_t packed_required_refs[] = {
        LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ADD_U16,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ASHRREV_I16,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_LSHRREV_B16,
    };
    const bool has_packed_f16_e5m2 =
        packed_i8_permute.kind ==
            LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_LITERAL_SELECTOR &&
        loom_amdgpu_fp8_encode_has_refs(descriptor_set, packed_required_refs,
                                        IREE_ARRAYSIZE(packed_required_refs));
    const loom_amdgpu_descriptor_ref_t required_refs[] = {
        LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_WIDTH_INLINE,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        round_add_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE
            ? LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT
            : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
        round_add_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE
            ? LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32
            : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
        sign_insert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE
            ? LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32
            : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
        packed_i8_permute.kind == LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_NONE
            ? LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT
            : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
        packed_i8_permute.kind == LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_NONE
            ? LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32
            : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
        packed_i8_permute.kind ==
                LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_REGISTER_SELECTOR
            ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32
            : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
    };
    if (loom_amdgpu_fp8_encode_has_refs(descriptor_set, required_refs,
                                        IREE_ARRAYSIZE(required_refs))) {
      *out_plan = (loom_amdgpu_fp8_encode_plan_t){
          .kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2,
          .flags = has_packed_f16_e5m2
                       ? LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_PACKED_F16_E5M2
                       : LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_NONE,
          .round_add_descriptor_ref = round_add_descriptor_ref,
          .sign_insert_descriptor_ref = sign_insert_descriptor_ref,
          .packed_i8_permute = packed_i8_permute,
      };
      return true;
    }
  }

  const loom_amdgpu_descriptor_ref_t packed_f16_e4m3_required_refs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ADD_F16,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ADD_U16,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ASHRREV_I16,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_LSHRREV_B16,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MIN_U16,
  };
  const bool has_packed_f16_e4m3 =
      source_type == LOOM_SCALAR_TYPE_F16 &&
      result_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN &&
      packed_i8_permute.kind ==
          LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_LITERAL_SELECTOR &&
      loom_amdgpu_fp8_encode_has_refs(
          descriptor_set, packed_f16_e4m3_required_refs,
          IREE_ARRAYSIZE(packed_f16_e4m3_required_refs));

  const loom_amdgpu_descriptor_ref_t required_refs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_WIDTH_INLINE,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      round_add_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      sign_insert_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      packed_i8_permute.kind == LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_NONE ||
              source_type == LOOM_SCALAR_TYPE_BF16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      packed_i8_permute.kind == LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_NONE
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      packed_i8_permute.kind ==
              LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_REGISTER_SELECTOR
          ? LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      source_type == LOOM_SCALAR_TYPE_F16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      result_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ ||
              result_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      result_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FNUZ ||
              result_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2FNUZ
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  if (!loom_amdgpu_fp8_encode_has_refs(descriptor_set, required_refs,
                                       IREE_ARRAYSIZE(required_refs))) {
    return false;
  }

  loom_amdgpu_fp8_encode_kind_t kind = LOOM_AMDGPU_FP8_ENCODE_KIND_NONE;
  switch (result_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E4M3;
      break;
    case LOOM_SCALAR_TYPE_F8E5M2:
      kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E5M2;
      break;
    default:
      return false;
  }

  *out_plan = (loom_amdgpu_fp8_encode_plan_t){
      .kind = kind,
      .flags = has_packed_f16_e4m3
                   ? LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_PACKED_F16_E4M3
                   : LOOM_AMDGPU_FP8_ENCODE_PLAN_FLAG_NONE,
      .round_add_descriptor_ref = round_add_descriptor_ref,
      .sign_insert_descriptor_ref = sign_insert_descriptor_ref,
      .packed_i8_permute = packed_i8_permute,
  };
  return true;
}

static bool loom_amdgpu_select_native_ocp_fp8_encode_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_type,
    loom_value_fact_numeric_format_flags_t result_format,
    loom_amdgpu_fp8_encode_plan_t* out_plan) {
  const bool requires_native_nan_canonicalization =
      loom_amdgpu_descriptor_set_requires_native_fp8_nan_repair(descriptor_set);

  if (!requires_native_nan_canonicalization &&
      source_type == LOOM_SCALAR_TYPE_F16 &&
      result_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2) {
    const loom_amdgpu_descriptor_ref_t direct_refs[] = {
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF8_F16_OCP_LOW,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF8_F16_OCP_HIGH,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
    };
    if (loom_amdgpu_fp8_encode_has_refs(descriptor_set, direct_refs,
                                        IREE_ARRAYSIZE(direct_refs))) {
      *out_plan = (loom_amdgpu_fp8_encode_plan_t){
          .kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F16_PAIR,
          .low_descriptor_ref = direct_refs[0],
          .high_descriptor_ref = direct_refs[1],
      };
      return true;
    }
  }

  loom_amdgpu_descriptor_ref_t low_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_descriptor_ref_t high_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_fp8_encode_kind_t kind = LOOM_AMDGPU_FP8_ENCODE_KIND_NONE;
  switch (result_format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN:
      kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3;
      low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_FP8_F32_OCP_LOW;
      high_descriptor_ref =
          LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_FP8_F32_OCP_HIGH;
      break;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2:
      kind = LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR;
      low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF8_F32_OCP_LOW;
      high_descriptor_ref =
          LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_PK_BF8_F32_OCP_HIGH;
      break;
    default:
      return false;
  }

  const loom_amdgpu_descriptor_ref_t base_refs[] = {
      low_descriptor_ref,
      high_descriptor_ref,
      source_type == LOOM_SCALAR_TYPE_F16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      source_type == LOOM_SCALAR_TYPE_BF16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      source_type == LOOM_SCALAR_TYPE_BF16
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT
          : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  bool can_use_native = loom_amdgpu_fp8_encode_has_refs(
      descriptor_set, base_refs, IREE_ARRAYSIZE(base_refs));
  if (can_use_native &&
      kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3) {
    const loom_amdgpu_descriptor_ref_t saturation_refs[] = {
        LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_MED3_NUM_F32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UNO_F32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
    };
    can_use_native = loom_amdgpu_fp8_encode_has_refs(
        descriptor_set, saturation_refs, IREE_ARRAYSIZE(saturation_refs));
  }

  loom_amdgpu_fp8_encode_plan_t native_plan = {
      .kind = kind,
      .low_descriptor_ref = low_descriptor_ref,
      .high_descriptor_ref = high_descriptor_ref,
  };
  if (can_use_native && requires_native_nan_canonicalization) {
    can_use_native = loom_amdgpu_select_native_fp8_nan_canonicalization(
        descriptor_set, kind, &native_plan);
  }
  if (can_use_native) {
    *out_plan = native_plan;
    return true;
  }

  return false;
}

bool loom_amdgpu_select_fp8_encode_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_value_fact_numeric_format_flags_t result_format,
    loom_amdgpu_fp8_encode_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_fp8_encode_plan_t){0};
  if (source_type != LOOM_SCALAR_TYPE_F16 &&
      source_type != LOOM_SCALAR_TYPE_BF16 &&
      source_type != LOOM_SCALAR_TYPE_F32) {
    return false;
  }

  loom_scalar_type_t format_element_type = LOOM_SCALAR_TYPE_COUNT_;
  loom_scalar_type_fp8_format_t format = {0};
  if (!loom_amdgpu_fp8_format(result_format, &format_element_type, &format) ||
      format_element_type != result_type) {
    return false;
  }
  const bool is_native_ocp_format =
      result_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN ||
      result_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2;
  if (is_native_ocp_format &&
      loom_amdgpu_select_native_ocp_fp8_encode_plan(descriptor_set, source_type,
                                                    result_format, out_plan)) {
    out_plan->format = format;
    return true;
  }

  if (is_native_ocp_format &&
      loom_amdgpu_select_fnuz_bridge_fp8_encode_plan(
          descriptor_set, source_type, result_type, out_plan)) {
    out_plan->format = format;
    return true;
  }

  if (!loom_amdgpu_select_software_fp8_encode_plan(
          descriptor_set, source_type, result_type, result_format, out_plan)) {
    return false;
  }
  out_plan->format = format;
  return true;
}

static iree_status_t loom_amdgpu_initialize_fp8_encode_packed_f16_e4m3_emission(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fp8_encode_packed_f16_e4m3_state_t* out_state) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      &out_state->constant_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      &out_state->magnitude_mask_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MIN_U16,
      &out_state->minimum_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ADD_U16,
      &out_state->integer_add_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ADD_F16,
      &out_state->float_add_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_LSHRREV_B16,
      &out_state->logical_shift_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ASHRREV_I16,
      &out_state->arithmetic_shift_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32,
      &out_state->select_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT,
      &out_state->sign_insert_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32_SRC2_LIT,
      &out_state->pack_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, IREE_SV("imm32"),
                                          &out_state->imm32_attr_name_id));

  loom_type_t constant_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &constant_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E4M3_ROUNDING_SHIFT, constant_type,
      &out_state->rounding_shift));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id, LOOM_AMDGPU_FP8_PACKED_F16_E4M3_SIGN_SHIFT,
      constant_type, &out_state->sign_shift));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E4M3_CONDITION_MASK_SHIFT, constant_type,
      &out_state->condition_mask_shift));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E4M3_MAXIMUM_MAGNITUDE, constant_type,
      &out_state->maximum_magnitude));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E4M3_ROUNDING_BIAS, constant_type,
      &out_state->rounding_bias));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E4M3_NORMAL_ADJUSTMENT, constant_type,
      &out_state->normal_adjustment));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E4M3_SUBNORMAL_MAGIC, constant_type,
      &out_state->subnormal_magic));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E4M3_SUBNORMAL_ADJUSTMENT, constant_type,
      &out_state->subnormal_adjustment));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E4M3_SUBNORMAL_CONDITION_BIAS, constant_type,
      &out_state->subnormal_condition_bias));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E4M3_NAN_CONDITION_BIAS, constant_type,
      &out_state->nan_condition_bias));
  return loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E4M3_NAN_ENCODING, constant_type,
      &out_state->nan_encoding);
}

static iree_status_t loom_amdgpu_initialize_fp8_encode_packed_f16_e5m2_emission(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fp8_encode_packed_f16_e5m2_state_t* out_state) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      &out_state->constant_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      &out_state->magnitude_mask_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ADD_U16,
      &out_state->add_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_LSHRREV_B16,
      &out_state->logical_shift_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ASHRREV_I16,
      &out_state->arithmetic_shift_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32,
      &out_state->select_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT,
      &out_state->sign_insert_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PERM_B32_SRC2_LIT,
      &out_state->pack_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, IREE_SV("imm32"),
                                          &out_state->imm32_attr_name_id));

  loom_type_t constant_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &constant_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id, LOOM_AMDGPU_FP8_PACKED_F16_E5M2_LANE_SHIFT,
      constant_type, &out_state->lane_shift));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E5M2_ROUNDING_BIAS_AND_NAN, constant_type,
      &out_state->rounding_bias_and_nan));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E5M2_NAN_CONDITION_BIAS, constant_type,
      &out_state->nan_condition_bias));
  return loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &out_state->constant_descriptor,
      out_state->imm32_attr_name_id,
      LOOM_AMDGPU_FP8_PACKED_F16_E5M2_CONDITION_MASK_SHIFT, constant_type,
      &out_state->condition_mask_shift);
}

static iree_status_t loom_amdgpu_initialize_fp8_encode_packed_sign_emission(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan, uint32_t encoded_lane_count,
    loom_amdgpu_fp8_encode_emission_state_t* out_state) {
  loom_type_t selector_type = loom_type_none();
  if (plan->packed_i8_permute.kind ==
          LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_REGISTER_SELECTOR ||
      plan->sign_insert_descriptor_ref ==
          LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &selector_type));
  }
  if (plan->packed_i8_permute.kind ==
      LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_REGISTER_SELECTOR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        loom_amdgpu_fp8_encode_sign_permute_selector(plan,
                                                     /*high_pair=*/false),
        selector_type, &out_state->low_sign_permute_selector));
    if (encoded_lane_count > 2u) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
          loom_amdgpu_fp8_encode_sign_permute_selector(plan,
                                                       /*high_pair=*/true),
          selector_type, &out_state->high_sign_permute_selector));
    }
  }
  if (plan->sign_insert_descriptor_ref ==
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        LOOM_AMDGPU_FP8_LOW_PAIR_SIGN_MASK, selector_type,
        &out_state->low_sign_mask));
    if (encoded_lane_count > 2u) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
          LOOM_AMDGPU_FP8_HIGH_PAIR_SIGN_MASK, selector_type,
          &out_state->high_sign_mask));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_initialize_fp8_encode_emission(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan, uint32_t encoded_lane_count,
    loom_type_t lane_type, loom_amdgpu_fp8_encode_emission_state_t* out_state) {
  IREE_ASSERT_NE(encoded_lane_count, 0u);
  *out_state = (loom_amdgpu_fp8_encode_emission_state_t){
      .lane_type = lane_type,
      .positive_maximum = LOOM_VALUE_ID_INVALID,
      .negative_maximum = LOOM_VALUE_ID_INVALID,
      .maximum_magnitude = LOOM_VALUE_ID_INVALID,
      .infinity_magnitude = LOOM_VALUE_ID_INVALID,
      .minimum_normal_magnitude = LOOM_VALUE_ID_INVALID,
      .nan_encoding = LOOM_VALUE_ID_INVALID,
      .fnuz_nan_bridge_magnitude = LOOM_VALUE_ID_INVALID,
      .low_sign_permute_selector = LOOM_VALUE_ID_INVALID,
      .high_sign_permute_selector = LOOM_VALUE_ID_INVALID,
      .low_sign_mask = LOOM_VALUE_ID_INVALID,
      .high_sign_mask = LOOM_VALUE_ID_INVALID,
      .packed_f16_e4m3 =
          {
              .rounding_shift = LOOM_VALUE_ID_INVALID,
              .sign_shift = LOOM_VALUE_ID_INVALID,
              .condition_mask_shift = LOOM_VALUE_ID_INVALID,
              .maximum_magnitude = LOOM_VALUE_ID_INVALID,
              .rounding_bias = LOOM_VALUE_ID_INVALID,
              .normal_adjustment = LOOM_VALUE_ID_INVALID,
              .subnormal_magic = LOOM_VALUE_ID_INVALID,
              .subnormal_adjustment = LOOM_VALUE_ID_INVALID,
              .subnormal_condition_bias = LOOM_VALUE_ID_INVALID,
              .nan_condition_bias = LOOM_VALUE_ID_INVALID,
              .nan_encoding = LOOM_VALUE_ID_INVALID,
          },
      .packed_f16_e5m2 =
          {
              .lane_shift = LOOM_VALUE_ID_INVALID,
              .rounding_bias_and_nan = LOOM_VALUE_ID_INVALID,
              .nan_condition_bias = LOOM_VALUE_ID_INVALID,
              .condition_mask_shift = LOOM_VALUE_ID_INVALID,
          },
  };
  if (encoded_lane_count >= 3u &&
      loom_amdgpu_fp8_encode_plan_has_packed_f16_e4m3(plan)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_initialize_fp8_encode_packed_f16_e4m3_emission(
            context, source_op, &out_state->packed_f16_e4m3));
  }
  if (encoded_lane_count >= 3u &&
      loom_amdgpu_fp8_encode_plan_has_packed_f16_e5m2(plan)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_initialize_fp8_encode_packed_f16_e5m2_emission(
            context, source_op, &out_state->packed_f16_e5m2));
  }
  if (loom_amdgpu_fp8_encode_plan_canonicalizes_native_nan(plan)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fp8_encode_packed_sign_emission(
        context, source_op, plan, encoded_lane_count, out_state));
  }
  if (loom_amdgpu_fp8_encode_plan_is_fnuz_bridge(plan)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &out_state->mask_type));
    uint32_t maximum_magnitude = 0;
    uint32_t nan_bridge_magnitude = 0;
    switch (plan->kind) {
      case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_FNUZ_BRIDGE_E4M3:
        maximum_magnitude = UINT32_C(0x43E00000);
        nan_bridge_magnitude = UINT32_C(0x43F00000);
        break;
      case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_FNUZ_BRIDGE_E5M2:
        maximum_magnitude = UINT32_C(0x47700000);
        nan_bridge_magnitude = UINT32_C(0x47E00000);
        break;
      default:
        IREE_ASSERT_UNREACHABLE("invalid FNUZ bridge FP8 encode kind");
        IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        maximum_magnitude, lane_type, &out_state->maximum_magnitude));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        LOOM_AMDGPU_F32_INFINITY_MAGNITUDE, lane_type,
        &out_state->infinity_magnitude));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        nan_bridge_magnitude, lane_type,
        &out_state->fnuz_nan_bridge_magnitude));
    return loom_amdgpu_initialize_fp8_encode_packed_sign_emission(
        context, source_op, plan, encoded_lane_count, out_state);
  }

  if (loom_amdgpu_fp8_encode_plan_is_software(plan)) {
    if (loom_amdgpu_fp8_encode_plan_has_packed_f16_e4m3(plan)) {
      const uint32_t packed_tail_lane_count = encoded_lane_count % 4u;
      const bool needs_lane_encoding =
          packed_tail_lane_count == 1u || packed_tail_lane_count == 2u;
      if (!needs_lane_encoding) {
        return iree_ok_status();
      }
    }
    if (plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2) {
      out_state->mantissa_shift = 8;
      out_state->rounding_bias = UINT32_C(0x7F);
      const uint32_t packed_tail_lane_count = encoded_lane_count % 4u;
      const bool needs_lane_encoding =
          !loom_amdgpu_fp8_encode_plan_has_packed_f16_e5m2(plan) ||
          packed_tail_lane_count == 1u || packed_tail_lane_count == 2u;
      if (!needs_lane_encoding) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_make_sgpr_range_type(context, 2, &out_state->mask_type));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          UINT32_C(0x7C00), lane_type, &out_state->infinity_magnitude));
      return loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          LOOM_AMDGPU_FP8_NAN_ENCODING, lane_type, &out_state->nan_encoding);
    }

    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &out_state->mask_type));
    const uint32_t maximum_magnitude =
        loom_amdgpu_fp8_encode_maximum_magnitude(&plan->format);
    const uint32_t minimum_normal_magnitude =
        (128u - plan->format.exponent_bias) << 23;
    out_state->mantissa_shift = 23u - plan->format.mantissa_bits;
    out_state->rounding_bias =
        (UINT32_C(1) << (out_state->mantissa_shift - 1u)) - 1u;
    const int32_t normal_adjustment =
        ((int32_t)plan->format.exponent_bias - 127) *
        (int32_t)(UINT32_C(1) << plan->format.mantissa_bits);
    out_state->normal_adjustment = (uint32_t)normal_adjustment;
    // Adding this F32 power of two rounds subnormals at the encoded format's
    // quantum; subtracting its bit pattern leaves the encoded mantissa.
    const uint32_t subnormal_magic_exponent =
        24u - plan->format.exponent_bias - plan->format.mantissa_bits;
    out_state->subnormal_magic = (subnormal_magic_exponent + 127u) << 23;
    out_state->subnormal_adjustment = 0u - out_state->subnormal_magic;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        maximum_magnitude, lane_type, &out_state->maximum_magnitude));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        LOOM_AMDGPU_F32_INFINITY_MAGNITUDE, lane_type,
        &out_state->infinity_magnitude));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        minimum_normal_magnitude, lane_type,
        &out_state->minimum_normal_magnitude));
    return loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        loom_amdgpu_fp8_encode_nan_payload(&plan->format), lane_type,
        &out_state->nan_encoding);
  }

  if (plan->kind != LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &out_state->mask_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_FP8_E4M3_POSITIVE_MAXIMUM_BITS, lane_type,
      &out_state->positive_maximum));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                                    LOOM_AMDGPU_FP8_E4M3_NEGATIVE_MAXIMUM_BITS,
                                    lane_type, &out_state->negative_maximum);
}

static iree_status_t loom_amdgpu_emit_fp8_encode_bfe_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t offset, uint32_t width,
    loom_type_t lane_type, loom_value_id_t* out_value) {
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("offset"), offset, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_append_i64_attr(context, IREE_SV("width"), width, attrs,
                                  IREE_ARRAYSIZE(attrs), &attr_count));
  const loom_value_id_t operands[] = {value};
  loom_op_t* extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_WIDTH_INLINE, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
      &lane_type, 1, &extract_op));
  *out_value = loom_value_slice_get(loom_low_op_results(extract_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_encode_round(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t magnitude, loom_value_id_t rounding_lsb,
    loom_value_id_t* out_rounded) {
  if (plan->round_add_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, plan->round_add_descriptor_ref, &descriptor));
    return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
        context, source_op, &descriptor, magnitude, rounding_lsb,
        state->rounding_bias, state->lane_type, out_rounded);
  }

  loom_value_id_t biased_lsb = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      rounding_lsb, state->rounding_bias, state->lane_type, &biased_lsb));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, magnitude,
      biased_lsb, state->lane_type, out_rounded);
}

static iree_status_t loom_amdgpu_emit_fp8_encode_sign(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source, uint32_t sign_shift, loom_value_id_t encoded,
    loom_value_id_t* out_encoded) {
  loom_value_id_t source_high_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      sign_shift, source, state->lane_type, &source_high_byte));
  if (plan->format.special_policy ==
      LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN_UNSIGNED_ZERO) {
    loom_value_id_t sign = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        source_high_byte, UINT32_C(0x80), state->lane_type, &sign));
    loom_value_id_t signed_encoding = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, encoded, sign,
        state->lane_type, &signed_encoding));
    loom_value_id_t is_nonzero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_compare_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32_SRC1_INLINE, encoded, 0,
        state->lane_type, state->mask_type, &is_nonzero));
    return loom_amdgpu_emit_vgpr_select(context, source_op, encoded,
                                        signed_encoding, is_nonzero,
                                        state->lane_type, out_encoded);
  }

  loom_value_id_t signed_encoding = LOOM_VALUE_ID_INVALID;
  if (plan->sign_insert_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, plan->sign_insert_descriptor_ref, &descriptor));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary_immediate(
        context, source_op, &descriptor, source_high_byte, encoded,
        UINT32_C(0x80), state->lane_type, &signed_encoding));
  } else {
    loom_value_id_t sign = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        source_high_byte, UINT32_C(0x80), state->lane_type, &sign));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, encoded, sign,
        state->lane_type, &signed_encoding));
  }

  *out_encoded = signed_encoding;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_encode_software_f32_magnitude_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source, loom_value_id_t* out_encoded) {
  IREE_ASSERT(loom_amdgpu_fp8_encode_plan_is_f32_software(plan));
  *out_encoded = LOOM_VALUE_ID_INVALID;

  loom_value_id_t original_magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, source,
      LOOM_AMDGPU_F32_MAGNITUDE_MASK, state->lane_type, &original_magnitude));
  loom_value_id_t magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32,
      state->maximum_magnitude, original_magnitude, state->lane_type,
      &magnitude));

  loom_value_id_t rounding_lsb = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_bfe_u32(
      context, source_op, magnitude, state->mantissa_shift, 1, state->lane_type,
      &rounding_lsb));
  loom_value_id_t rounded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_round(
      context, source_op, plan, state, magnitude, rounding_lsb, &rounded));
  loom_value_id_t normal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      state->mantissa_shift, rounded, state->lane_type, &normal));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, normal,
      state->normal_adjustment, state->lane_type, &normal));

  loom_value_id_t subnormal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_F32_LIT, magnitude,
      state->subnormal_magic, state->lane_type, &subnormal));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, subnormal,
      state->subnormal_adjustment, state->lane_type, &subnormal));

  loom_value_id_t is_subnormal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32, magnitude,
      state->minimum_normal_magnitude, state->mask_type, &is_subnormal));
  loom_value_id_t finite_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
      context, source_op, normal, subnormal, is_subnormal, state->lane_type,
      &finite_encoding));

  loom_value_id_t is_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U32,
      original_magnitude, state->infinity_magnitude, state->mask_type,
      &is_nan));
  loom_value_id_t magnitude_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
      context, source_op, finite_encoding, state->nan_encoding, is_nan,
      state->lane_type, &magnitude_encoding));
  *out_encoded = magnitude_encoding;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fp8_encode_software_f32_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source, loom_value_id_t* out_encoded) {
  loom_value_id_t magnitude_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_f32_magnitude_lane(
      context, source_op, plan, state, source, &magnitude_encoding));
  return loom_amdgpu_emit_fp8_encode_sign(context, source_op, plan, state,
                                          source, /*sign_shift=*/24,
                                          magnitude_encoding, out_encoded);
}

static iree_status_t
loom_amdgpu_emit_fp8_encode_software_f16_e5m2_magnitude_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source, loom_value_id_t* out_encoded) {
  IREE_ASSERT_EQ(plan->kind, LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2);
  *out_encoded = LOOM_VALUE_ID_INVALID;

  loom_value_id_t magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, source,
      UINT32_C(0x7FFF), state->lane_type, &magnitude));
  loom_value_id_t rounding_lsb = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_bfe_u32(
      context, source_op, magnitude, state->mantissa_shift, 1, state->lane_type,
      &rounding_lsb));
  loom_value_id_t rounded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_round(
      context, source_op, plan, state, magnitude, rounding_lsb, &rounded));
  loom_value_id_t finite_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      state->mantissa_shift, rounded, state->lane_type, &finite_encoding));

  loom_value_id_t is_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U32, magnitude,
      state->infinity_magnitude, state->mask_type, &is_nan));
  loom_value_id_t magnitude_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
      context, source_op, finite_encoding, state->nan_encoding, is_nan,
      state->lane_type, &magnitude_encoding));
  *out_encoded = magnitude_encoding;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_fp8_encode_software_f16_e5m2_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source, loom_value_id_t* out_encoded) {
  loom_value_id_t magnitude_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fp8_encode_software_f16_e5m2_magnitude_lane(
          context, source_op, plan, state, source, &magnitude_encoding));
  return loom_amdgpu_emit_fp8_encode_sign(context, source_op, plan, state,
                                          source, /*sign_shift=*/8,
                                          magnitude_encoding, out_encoded);
}

static iree_status_t loom_amdgpu_emit_fp8_encode_software_f16_e4m3_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source_pair, loom_value_id_t* out_encoded_pair) {
  const loom_amdgpu_fp8_encode_packed_f16_e4m3_state_t* packed_state =
      &state->packed_f16_e4m3;

  loom_value_id_t shifted_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->logical_shift_descriptor,
      packed_state->sign_shift, source_pair, state->lane_type,
      &shifted_source));
  loom_value_id_t original_magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &packed_state->magnitude_mask_descriptor, source_pair,
      UINT32_C(0x7FFF7FFF), state->lane_type, &original_magnitude));
  loom_value_id_t magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->minimum_descriptor, original_magnitude,
      packed_state->maximum_magnitude, state->lane_type, &magnitude));

  loom_value_id_t shifted_magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->logical_shift_descriptor,
      packed_state->rounding_shift, magnitude, state->lane_type,
      &shifted_magnitude));
  loom_value_id_t rounding_lsb = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &packed_state->magnitude_mask_descriptor,
      shifted_magnitude, UINT32_C(0x00010001), state->lane_type,
      &rounding_lsb));
  loom_value_id_t normal_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->integer_add_descriptor, magnitude,
      rounding_lsb, state->lane_type, &normal_encoding));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->integer_add_descriptor,
      normal_encoding, packed_state->rounding_bias, state->lane_type,
      &normal_encoding));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->logical_shift_descriptor,
      packed_state->rounding_shift, normal_encoding, state->lane_type,
      &normal_encoding));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->integer_add_descriptor,
      normal_encoding, packed_state->normal_adjustment, state->lane_type,
      &normal_encoding));

  loom_value_id_t subnormal_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->float_add_descriptor, magnitude,
      packed_state->subnormal_magic, state->lane_type, &subnormal_encoding));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->integer_add_descriptor,
      subnormal_encoding, packed_state->subnormal_adjustment, state->lane_type,
      &subnormal_encoding));
  loom_value_id_t subnormal_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->integer_add_descriptor, magnitude,
      packed_state->subnormal_condition_bias, state->lane_type,
      &subnormal_condition));
  loom_value_id_t subnormal_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->arithmetic_shift_descriptor,
      packed_state->condition_mask_shift, subnormal_condition, state->lane_type,
      &subnormal_mask));
  loom_value_id_t finite_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &packed_state->select_descriptor, subnormal_mask,
      subnormal_encoding, normal_encoding, state->lane_type, &finite_encoding));

  loom_value_id_t nan_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->integer_add_descriptor,
      original_magnitude, packed_state->nan_condition_bias, state->lane_type,
      &nan_condition));
  loom_value_id_t nan_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->arithmetic_shift_descriptor,
      packed_state->condition_mask_shift, nan_condition, state->lane_type,
      &nan_mask));
  loom_value_id_t magnitude_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &packed_state->select_descriptor, nan_mask,
      packed_state->nan_encoding, finite_encoding, state->lane_type,
      &magnitude_encoding));
  return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
      context, source_op, &packed_state->sign_insert_descriptor, shifted_source,
      magnitude_encoding, LOOM_AMDGPU_FP8_PACKED_F16_SIGN_MASK,
      state->lane_type, out_encoded_pair);
}

iree_status_t loom_amdgpu_emit_fp8_encode_software_f16_e4m3_pairs(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t low_source_pair, loom_value_id_t high_source_pair,
    loom_value_id_t* out_packed) {
  IREE_ASSERT_EQ(plan->kind, LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E4M3);
  IREE_ASSERT(loom_amdgpu_fp8_encode_plan_has_packed_f16_e4m3(plan));

  loom_value_id_t low_encoded_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_f16_e4m3_pair(
      context, source_op, state, low_source_pair, &low_encoded_pair));
  loom_value_id_t high_encoded_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_f16_e4m3_pair(
      context, source_op, state, high_source_pair, &high_encoded_pair));
  return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
      context, source_op, &state->packed_f16_e4m3.pack_descriptor,
      low_encoded_pair, high_encoded_pair,
      LOOM_AMDGPU_FP8_PACKED_F16_PACK_SELECTOR, state->lane_type, out_packed);
}

static iree_status_t loom_amdgpu_emit_fp8_encode_software_f16_e5m2_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source_pair, loom_value_id_t* out_encoded_pair) {
  const loom_amdgpu_fp8_encode_packed_f16_e5m2_state_t* packed_state =
      &state->packed_f16_e5m2;

  loom_value_id_t shifted_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->logical_shift_descriptor,
      packed_state->lane_shift, source_pair, state->lane_type,
      &shifted_source));
  loom_value_id_t magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &packed_state->magnitude_mask_descriptor, source_pair,
      UINT32_C(0x7FFF7FFF), state->lane_type, &magnitude));
  loom_value_id_t rounding_lsb = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &packed_state->magnitude_mask_descriptor,
      shifted_source, UINT32_C(0x00010001), state->lane_type, &rounding_lsb));

  loom_value_id_t rounded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->add_descriptor, magnitude,
      rounding_lsb, state->lane_type, &rounded));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->add_descriptor, rounded,
      packed_state->rounding_bias_and_nan, state->lane_type, &rounded));
  loom_value_id_t finite_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->logical_shift_descriptor,
      packed_state->lane_shift, rounded, state->lane_type, &finite_encoding));

  loom_value_id_t nan_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->add_descriptor, magnitude,
      packed_state->nan_condition_bias, state->lane_type, &nan_condition));
  loom_value_id_t nan_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &packed_state->arithmetic_shift_descriptor,
      packed_state->condition_mask_shift, nan_condition, state->lane_type,
      &nan_mask));
  loom_value_id_t magnitude_encoding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &packed_state->select_descriptor, nan_mask,
      packed_state->rounding_bias_and_nan, finite_encoding, state->lane_type,
      &magnitude_encoding));
  return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
      context, source_op, &packed_state->sign_insert_descriptor, shifted_source,
      magnitude_encoding, LOOM_AMDGPU_FP8_PACKED_F16_SIGN_MASK,
      state->lane_type, out_encoded_pair);
}

iree_status_t loom_amdgpu_emit_fp8_encode_software_f16_e5m2_pairs(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t low_source_pair, loom_value_id_t high_source_pair,
    loom_value_id_t* out_packed) {
  IREE_ASSERT_EQ(plan->kind, LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2);
  IREE_ASSERT(loom_amdgpu_fp8_encode_plan_has_packed_f16_e5m2(plan));

  loom_value_id_t low_encoded_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_f16_e5m2_pair(
      context, source_op, state, low_source_pair, &low_encoded_pair));
  loom_value_id_t high_encoded_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_f16_e5m2_pair(
      context, source_op, state, high_source_pair, &high_encoded_pair));
  return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
      context, source_op, &state->packed_f16_e5m2.pack_descriptor,
      low_encoded_pair, high_encoded_pair,
      LOOM_AMDGPU_FP8_PACKED_F16_PACK_SELECTOR, state->lane_type, out_packed);
}

iree_status_t loom_amdgpu_emit_fp8_encode_duplicate_f16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_type_t lane_type,
    loom_value_id_t* out_packed) {
  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, source,
      UINT32_C(0xFFFF), lane_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      low_lane, lane_type, &high_lane));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_lane,
      high_lane, lane_type, out_packed);
}

static iree_status_t loom_amdgpu_emit_fp8_e4m3_saturate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source, loom_value_id_t* out_value) {
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MED3_NUM_F32, &descriptor));
  loom_value_id_t clamped = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &descriptor, source, state->positive_maximum,
      state->negative_maximum, state->lane_type, &clamped));

  loom_value_id_t is_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UNO_F32, source,
      source, state->mask_type, &is_nan));
  return loom_amdgpu_emit_vgpr_select(context, source_op, clamped, source,
                                      is_nan, state->lane_type, out_value);
}

static iree_status_t loom_amdgpu_prepare_fp8_encode_fnuz_bridge_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t source, loom_value_id_t* out_value) {
  loom_value_id_t original_magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, source,
      LOOM_AMDGPU_F32_MAGNITUDE_MASK, state->lane_type, &original_magnitude));
  loom_value_id_t finite_magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MIN_U32,
      state->maximum_magnitude, original_magnitude, state->lane_type,
      &finite_magnitude));
  loom_value_id_t is_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U32,
      original_magnitude, state->infinity_magnitude, state->mask_type,
      &is_nan));
  loom_value_id_t bridge_magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
      context, source_op, finite_magnitude, state->fnuz_nan_bridge_magnitude,
      is_nan, state->lane_type, &bridge_magnitude));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32_SRC0_INLINE,
      bridge_magnitude, UINT32_C(0x3F000000), state->lane_type, out_value);
}

static iree_status_t loom_amdgpu_prepare_fp8_encode_f32_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t low_source, loom_value_id_t high_source,
    loom_value_id_t* out_low_source, loom_value_id_t* out_high_source) {
  *out_low_source = low_source;
  *out_high_source = high_source;
  if (plan->kind != LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_e4m3_saturate(
      context, source_op, state, low_source, out_low_source));
  return loom_amdgpu_emit_fp8_e4m3_saturate(context, source_op, state,
                                            high_source, out_high_source);
}

static iree_status_t loom_amdgpu_emit_fp8_encode_packed_sign_permute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state, bool high_pair,
    loom_value_id_t low_source, loom_value_id_t high_source,
    loom_value_id_t* out_sign_bytes) {
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, plan->packed_i8_permute.descriptor_ref, &descriptor));
  const uint32_t selector =
      loom_amdgpu_fp8_encode_sign_permute_selector(plan, high_pair);
  switch (plan->packed_i8_permute.kind) {
    case LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_LITERAL_SELECTOR:
      return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
          context, source_op, &descriptor, low_source, high_source, selector,
          state->lane_type, out_sign_bytes);
    case LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_REGISTER_SELECTOR:
      return loom_amdgpu_emit_resolved_vgpr_ternary(
          context, source_op, &descriptor, low_source, high_source,
          high_pair ? state->high_sign_permute_selector
                    : state->low_sign_permute_selector,
          state->lane_type, out_sign_bytes);
    case LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("invalid packed FP8 sign permutation");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_amdgpu_emit_fp8_encode_packed_sign_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state, bool high_pair,
    loom_value_id_t sign_bytes, loom_value_id_t packed,
    loom_value_id_t* out_packed) {
  const uint32_t sign_mask = high_pair ? LOOM_AMDGPU_FP8_HIGH_PAIR_SIGN_MASK
                                       : LOOM_AMDGPU_FP8_LOW_PAIR_SIGN_MASK;
  if (plan->sign_insert_descriptor_ref ==
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32_SRC0_LIT) {
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, plan->sign_insert_descriptor_ref, &descriptor));
    return loom_amdgpu_emit_resolved_vgpr_binary_immediate(
        context, source_op, &descriptor, sign_bytes, packed, sign_mask,
        state->lane_type, out_packed);
  }
  if (plan->sign_insert_descriptor_ref ==
      LOOM_AMDGPU_DESCRIPTOR_REF_V_BFI_B32) {
    loom_low_lower_resolved_descriptor_t descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, plan->sign_insert_descriptor_ref, &descriptor));
    return loom_amdgpu_emit_resolved_vgpr_ternary(
        context, source_op, &descriptor,
        high_pair ? state->high_sign_mask : state->low_sign_mask, sign_bytes,
        packed, state->lane_type, out_packed);
  }

  loom_value_id_t sign_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, sign_bytes,
      sign_mask, state->lane_type, &sign_bits));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, packed,
      sign_bits, state->lane_type, out_packed);
}

static iree_status_t loom_amdgpu_emit_fp8_encode_packed_sign_repair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state, bool high_pair,
    loom_value_id_t low_source, loom_value_id_t high_source,
    loom_value_id_t packed, loom_value_id_t* out_packed) {
  loom_value_id_t sign_bytes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_packed_sign_permute(
      context, source_op, plan, state, high_pair, low_source, high_source,
      &sign_bytes));
  return loom_amdgpu_emit_fp8_encode_packed_sign_insert(
      context, source_op, plan, state, high_pair, sign_bytes, packed,
      out_packed);
}

iree_status_t loom_amdgpu_emit_fp8_encode_software_packed_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    const loom_value_id_t* source_lanes, uint32_t source_lane_count,
    loom_value_id_t* out_packed) {
  IREE_ASSERT(loom_amdgpu_fp8_encode_plan_is_software(plan));
  IREE_ASSERT_EQ(plan->packed_i8_permute.kind,
                 LOOM_AMDGPU_I8_PACK_PERMUTE_KIND_LITERAL_SELECTOR);
  IREE_ASSERT_GE(source_lane_count, 3u);
  IREE_ASSERT_LE(source_lane_count, 4u);

  if (plan->format.special_policy ==
      LOOM_SCALAR_TYPE_FP8_SPECIAL_POLICY_FINITE_NAN_UNSIGNED_ZERO) {
    loom_value_id_t encoded_lanes[4] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
    };
    for (uint32_t lane = 0; lane < source_lane_count; ++lane) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_f32_lane(
          context, source_op, plan, state, source_lanes[lane],
          &encoded_lanes[lane]));
    }
    for (uint32_t lane = source_lane_count; lane < 4u; ++lane) {
      encoded_lanes[lane] = encoded_lanes[source_lane_count - 1u];
    }
    return loom_amdgpu_pack_i8_lanes_with_permute(
        context, source_op, &plan->packed_i8_permute, encoded_lanes,
        IREE_ARRAYSIZE(encoded_lanes), state->lane_type, out_packed);
  }

  loom_value_id_t physical_sources[4] = {
      source_lanes[0],
      source_lanes[1],
      source_lanes[2],
      source_lanes[source_lane_count - 1u],
  };
  loom_value_id_t low_sign_bytes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_packed_sign_permute(
      context, source_op, plan, state, /*high_pair=*/false, physical_sources[0],
      physical_sources[1], &low_sign_bytes));
  loom_value_id_t high_sign_bytes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_packed_sign_permute(
      context, source_op, plan, state, /*high_pair=*/true, physical_sources[2],
      physical_sources[3], &high_sign_bytes));

  loom_value_id_t magnitude_lanes[4] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t lane = 0; lane < source_lane_count; ++lane) {
    if (plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_encode_software_f16_e5m2_magnitude_lane(
              context, source_op, plan, state, physical_sources[lane],
              &magnitude_lanes[lane]));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fp8_encode_software_f32_magnitude_lane(
              context, source_op, plan, state, physical_sources[lane],
              &magnitude_lanes[lane]));
    }
  }
  for (uint32_t lane = source_lane_count; lane < 4u; ++lane) {
    magnitude_lanes[lane] = magnitude_lanes[source_lane_count - 1u];
  }

  loom_value_id_t packed_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_pack_i8_lanes_with_permute(
      context, source_op, &plan->packed_i8_permute, magnitude_lanes,
      IREE_ARRAYSIZE(magnitude_lanes), state->lane_type, &packed_register));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_packed_sign_insert(
      context, source_op, plan, state, /*high_pair=*/false, low_sign_bytes,
      packed_register, &packed_register));
  return loom_amdgpu_emit_fp8_encode_packed_sign_insert(
      context, source_op, plan, state, /*high_pair=*/true, high_sign_bytes,
      packed_register, out_packed);
}

static iree_status_t loom_amdgpu_emit_fp8_encode_raw_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t packed,
    loom_value_id_t low_source, loom_value_id_t high_source,
    loom_value_id_t* out_packed) {
  loom_value_id_t operands[3] = {0};
  iree_host_size_t operand_count = 0;
  if (packed != LOOM_VALUE_ID_INVALID) {
    operands[operand_count++] = packed;
  }
  operands[operand_count++] = low_source;
  if (high_source != LOOM_VALUE_ID_INVALID) {
    operands[operand_count++] = high_source;
  }

  const loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  const loom_tied_result_t* tied_results =
      packed == LOOM_VALUE_ID_INVALID ? NULL : &tied_result;
  const iree_host_size_t tied_result_count =
      packed == LOOM_VALUE_ID_INVALID ? 0 : 1;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_resolve_descriptor_ref(context, descriptor_ref, &descriptor));
  loom_op_t* encode_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_named_attr_slice_empty(), &state->lane_type, 1, tied_results,
      tied_result_count, source_op->location, &encode_op));
  *out_packed = loom_value_slice_get(loom_low_op_results(encode_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp8_encode_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t packed,
    loom_value_id_t low_source, loom_value_id_t high_source,
    loom_value_id_t* out_packed) {
  if (plan->kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F16_PAIR) {
    return loom_amdgpu_emit_fp8_encode_raw_pair(
        context, source_op, state, descriptor_ref, packed, low_source,
        LOOM_VALUE_ID_INVALID, out_packed);
  }

  IREE_ASSERT(!loom_amdgpu_fp8_encode_plan_is_fnuz_bridge(plan));
  loom_value_id_t prepared_low = LOOM_VALUE_ID_INVALID;
  loom_value_id_t prepared_high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_fp8_encode_f32_pair(
      context, source_op, plan, state, low_source, high_source, &prepared_low,
      &prepared_high));
  return loom_amdgpu_emit_fp8_encode_raw_pair(
      context, source_op, state, descriptor_ref, packed, prepared_low,
      prepared_high, out_packed);
}

iree_status_t loom_amdgpu_emit_fp8_encode_fnuz_f32_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    const loom_value_id_t* source_lanes, uint32_t source_lane_count,
    loom_value_id_t* out_packed) {
  IREE_ASSERT(loom_amdgpu_fp8_encode_plan_is_fnuz_bridge(plan));
  IREE_ASSERT(source_lane_count >= 1u && source_lane_count <= 4u);

  loom_value_id_t physical_sources[4] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t prepared_sources[4] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint32_t lane = 0; lane < source_lane_count; ++lane) {
    physical_sources[lane] = source_lanes[lane];
    IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_fp8_encode_fnuz_bridge_lane(
        context, source_op, state, source_lanes[lane],
        &prepared_sources[lane]));
  }
  for (uint32_t lane = source_lane_count; lane < 4u; ++lane) {
    physical_sources[lane] = physical_sources[source_lane_count - 1u];
    prepared_sources[lane] = prepared_sources[source_lane_count - 1u];
  }

  loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_raw_pair(
      context, source_op, state, plan->low_descriptor_ref,
      LOOM_VALUE_ID_INVALID, prepared_sources[0], prepared_sources[1],
      &packed));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_raw_pair(
      context, source_op, state, plan->high_descriptor_ref, packed,
      prepared_sources[2], prepared_sources[3], &packed));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_packed_sign_repair(
      context, source_op, plan, state, /*high_pair=*/false, physical_sources[0],
      physical_sources[1], packed, &packed));
  if (source_lane_count <= 2u) {
    *out_packed = packed;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_fp8_encode_packed_sign_repair(
      context, source_op, plan, state, /*high_pair=*/true, physical_sources[2],
      physical_sources[3], packed, out_packed);
}

iree_status_t loom_amdgpu_emit_fp8_encode_low_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t low_source, loom_value_id_t high_source,
    loom_value_id_t* out_packed) {
  return loom_amdgpu_emit_fp8_encode_pair(
      context, source_op, plan, state, plan->low_descriptor_ref,
      LOOM_VALUE_ID_INVALID, low_source, high_source, out_packed);
}

iree_status_t loom_amdgpu_emit_fp8_encode_high_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t packed, loom_value_id_t low_source,
    loom_value_id_t high_source, loom_value_id_t* out_packed) {
  return loom_amdgpu_emit_fp8_encode_pair(context, source_op, plan, state,
                                          plan->high_descriptor_ref, packed,
                                          low_source, high_source, out_packed);
}

static iree_status_t loom_amdgpu_canonicalize_native_e5m2_nan_payloads(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    loom_value_id_t packed, loom_value_id_t* out_packed) {
  // Clearing each byte's sign bit bounds every magnitude at 0x7f. Adding three
  // cannot carry across byte boundaries and raises the high bit exactly for
  // E5M2 NaN magnitudes 0x7d..0x7f. Move those bits to the payload LSB so all
  // native NaNs use Loom's all-ones canonical magnitude.
  loom_value_id_t magnitudes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, packed,
      LOOM_AMDGPU_FP8_PACKED_MAGNITUDE_MASK, state->lane_type, &magnitudes));
  loom_value_id_t biased_magnitudes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT, magnitudes,
      LOOM_AMDGPU_FP8_E5M2_NAN_CONDITION_BIAS, state->lane_type,
      &biased_magnitudes));
  loom_value_id_t nan_high_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      biased_magnitudes, LOOM_AMDGPU_FP8_PACKED_BYTE_HIGH_BITS,
      state->lane_type, &nan_high_bits));
  loom_value_id_t nan_low_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 7,
      nan_high_bits, state->lane_type, &nan_low_bits));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, magnitudes,
      nan_low_bits, state->lane_type, out_packed);
}

iree_status_t loom_amdgpu_emit_fp8_encode_native_nan_canonicalization(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_encode_plan_t* plan,
    const loom_amdgpu_fp8_encode_emission_state_t* state,
    const loom_value_id_t* source_lanes, uint32_t source_lane_count,
    loom_value_id_t packed, loom_value_id_t* out_packed) {
  IREE_ASSERT(loom_amdgpu_fp8_encode_plan_canonicalizes_native_nan(plan));
  IREE_ASSERT(source_lane_count == 2u || source_lane_count == 4u);

  loom_value_id_t canonical = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, packed, &canonical));
  switch (plan->kind) {
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3: {
      break;
    }
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_canonicalize_native_e5m2_nan_payloads(
          context, source_op, state, canonical, &canonical));
      break;
    }
    default: {
      IREE_ASSERT_UNREACHABLE("invalid native FP8 NaN canonicalization kind");
      IREE_BUILTIN_UNREACHABLE();
    }
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_packed_sign_repair(
      context, source_op, plan, state, /*high_pair=*/false, source_lanes[0],
      source_lanes[1], canonical, &canonical));
  if (source_lane_count == 2u) {
    *out_packed = canonical;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_fp8_encode_packed_sign_repair(
      context, source_op, plan, state, /*high_pair=*/true, source_lanes[2],
      source_lanes[3], canonical, out_packed);
}

iree_string_view_t loom_amdgpu_fp8_encode_plan_key(
    const loom_amdgpu_fp8_encode_plan_t* plan, loom_scalar_type_t source_type) {
  switch (plan->kind) {
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR:
      if (loom_amdgpu_fp8_encode_plan_canonicalizes_native_nan(plan)) {
        switch (source_type) {
          case LOOM_SCALAR_TYPE_F32:
            return IREE_SV(
                "amdgpu.fp8_encode.strategy."
                "f32_pair_native_nan_canonicalized");
          case LOOM_SCALAR_TYPE_F16:
            return IREE_SV(
                "amdgpu.fp8_encode.strategy."
                "f16_via_f32_pair_native_nan_canonicalized");
          case LOOM_SCALAR_TYPE_BF16:
            return IREE_SV(
                "amdgpu.fp8_encode.strategy."
                "bf16_via_f32_pair_native_nan_canonicalized");
          default:
            return iree_string_view_empty();
        }
      }
      switch (source_type) {
        case LOOM_SCALAR_TYPE_F32:
          return IREE_SV("amdgpu.fp8_encode.strategy.f32_pair_native");
        case LOOM_SCALAR_TYPE_F16:
          return IREE_SV("amdgpu.fp8_encode.strategy.f16_via_f32_pair_native");
        case LOOM_SCALAR_TYPE_BF16:
          return IREE_SV("amdgpu.fp8_encode.strategy.bf16_via_f32_pair_native");
        default:
          return iree_string_view_empty();
      }
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_PAIR_SATURATE_E4M3:
      if (loom_amdgpu_fp8_encode_plan_canonicalizes_native_nan(plan)) {
        switch (source_type) {
          case LOOM_SCALAR_TYPE_F32:
            return IREE_SV(
                "amdgpu.fp8_encode.strategy."
                "f32_pair_saturating_native_nan_canonicalized");
          case LOOM_SCALAR_TYPE_F16:
            return IREE_SV(
                "amdgpu.fp8_encode.strategy."
                "f16_via_f32_pair_saturating_native_nan_canonicalized");
          case LOOM_SCALAR_TYPE_BF16:
            return IREE_SV(
                "amdgpu.fp8_encode.strategy."
                "bf16_via_f32_pair_saturating_native_nan_canonicalized");
          default:
            return iree_string_view_empty();
        }
      }
      switch (source_type) {
        case LOOM_SCALAR_TYPE_F32:
          return IREE_SV(
              "amdgpu.fp8_encode.strategy.f32_pair_saturating_native");
        case LOOM_SCALAR_TYPE_F16:
          return IREE_SV(
              "amdgpu.fp8_encode.strategy."
              "f16_via_f32_pair_saturating_native");
        case LOOM_SCALAR_TYPE_BF16:
          return IREE_SV(
              "amdgpu.fp8_encode.strategy."
              "bf16_via_f32_pair_saturating_native");
        default:
          return iree_string_view_empty();
      }
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F16_PAIR:
      return IREE_SV("amdgpu.fp8_encode.strategy.f16_pair_native");
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E4M3:
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_SOFTWARE_E5M2:
      switch (source_type) {
        case LOOM_SCALAR_TYPE_F32:
          return IREE_SV("amdgpu.fp8_encode.strategy.f32_software");
        case LOOM_SCALAR_TYPE_F16:
          if (loom_amdgpu_fp8_encode_plan_has_packed_f16_e4m3(plan)) {
            return IREE_SV("amdgpu.fp8_encode.strategy.f16_e4m3_software");
          }
          return IREE_SV("amdgpu.fp8_encode.strategy.f16_via_f32_software");
        case LOOM_SCALAR_TYPE_BF16:
          return IREE_SV("amdgpu.fp8_encode.strategy.bf16_via_f32_software");
        default:
          return iree_string_view_empty();
      }
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_FNUZ_BRIDGE_E4M3:
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F32_FNUZ_BRIDGE_E5M2:
      switch (source_type) {
        case LOOM_SCALAR_TYPE_F32:
          return IREE_SV("amdgpu.fp8_encode.strategy.f32_fnuz_bridge");
        case LOOM_SCALAR_TYPE_F16:
          return IREE_SV("amdgpu.fp8_encode.strategy.f16_via_f32_fnuz_bridge");
        case LOOM_SCALAR_TYPE_BF16:
          return IREE_SV("amdgpu.fp8_encode.strategy.bf16_via_f32_fnuz_bridge");
        default:
          return iree_string_view_empty();
      }
    case LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2:
      return IREE_SV("amdgpu.fp8_encode.strategy.f16_e5m2_software");
    default:
      return iree_string_view_empty();
  }
}
