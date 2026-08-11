// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/encoding/fp4.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ops/encoding/auxiliary.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/vector/storage.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/float16.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/encoding/vector_conversion.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"

enum {
  // Packed shifts place each nibble magnitude in a V_PERM selector byte.
  LOOM_AMDGPU_FP4_MAGNITUDE_POSITION_SHIFTS = 0x00040008u,
  // Both packed lanes move their E2M1 sign bit into the 16-bit sign position.
  LOOM_AMDGPU_FP4_SIGN_POSITION_SHIFTS = 0x00040004u,
  // Selects the three magnitude bits from two positioned E2M1 nibbles.
  LOOM_AMDGPU_FP4_MAGNITUDE_SELECTOR_MASK = 0x07000700u,
  // Selects the two positioned 16-bit sign bits.
  LOOM_AMDGPU_FP4_SIGN_MASK = 0x80008000u,
  // F16 high bytes for E2M1 magnitudes 0 through 3.
  LOOM_AMDGPU_FP4_F16_MAGNITUDE_TABLE_LOW = 0x3E3C3800u,
  // F16 high bytes for E2M1 magnitudes 4 through 7.
  LOOM_AMDGPU_FP4_F16_MAGNITUDE_TABLE_HIGH = 0x46444240u,
  // Packed shift moving E2M1 magnitude selectors into low result bytes.
  LOOM_AMDGPU_FP4_BF16_SELECTOR_POSITION_SHIFTS = 0x00080008u,
  // Packed E8M0 exponent unbias applied to both 16-bit lanes.
  LOOM_AMDGPU_FP4_BF16_SCALE_UNBIAS = 0xFF81FF81u,
  // Packed shift converting E8M0 exponent deltas to BF16 exponent bits.
  LOOM_AMDGPU_FP4_BF16_SCALE_POSITION_SHIFTS = 0x00070007u,
  // BF16 identity-scale magnitudes 0 and 0.5.
  LOOM_AMDGPU_FP4_BF16_MAGNITUDE_TABLE_01 = 0x3F000000u,
  // BF16 identity-scale magnitudes 1 and 1.5.
  LOOM_AMDGPU_FP4_BF16_MAGNITUDE_TABLE_23 = 0x3FC03F80u,
  // BF16 identity-scale magnitudes 2 and 3.
  LOOM_AMDGPU_FP4_BF16_MAGNITUDE_TABLE_45 = 0x40404000u,
  // BF16 identity-scale magnitudes 4 and 6.
  LOOM_AMDGPU_FP4_BF16_MAGNITUDE_TABLE_67 = 0x40C04080u,
  // BF16 magnitudes 0 and 0.5 scaled by the minimum E8M0 value.
  LOOM_AMDGPU_FP4_BF16_MIN_SCALE_TABLE_01 = 0x00200000u,
  // BF16 magnitudes 1 and 1.5 scaled by the minimum E8M0 value.
  LOOM_AMDGPU_FP4_BF16_MIN_SCALE_TABLE_23 = 0x00600040u,
  // BF16 magnitudes 0 and 0.5 scaled by E8M0 exponent one.
  LOOM_AMDGPU_FP4_BF16_NEXT_SCALE_TABLE_01 = 0x00400000u,
  // Packed positive BF16 infinities used to clamp exponent overflow.
  LOOM_AMDGPU_FP4_BF16_INFINITY_PAIR = 0x7F807F80u,
  // Packed positive quiet BF16 NaNs selected for the E8M0 NaN encoding.
  LOOM_AMDGPU_FP4_BF16_QUIET_NAN_PAIR = 0x7FC07FC0u,
  // V_PERM selector packing low bytes from four packed BF16 pairs.
  LOOM_AMDGPU_FP4_BF16_LOW_BYTE_TABLE_SELECTOR = 0x06040200u,
  // V_PERM selector packing high bytes from four packed BF16 pairs.
  LOOM_AMDGPU_FP4_BF16_HIGH_BYTE_TABLE_SELECTOR = 0x07050301u,
  // Number of E2M1 elements sharing one MXFP4 E8M0 scale.
  LOOM_AMDGPU_FP4_E8M0_BF16_SCALE_GROUP_ELEMENT_COUNT = 32u,
  // Raw E8M0 byte isolated before forming the native F32 scale operand.
  LOOM_AMDGPU_FP4_E8M0_SCALE_BYTE_MASK = 0xFFu,
  // IEEE F32 exponent position carrying the raw E8M0 scale byte.
  LOOM_AMDGPU_FP4_E8M0_F32_EXPONENT_SHIFT = 23u,
  // Number of independently selectable bytes in one payload register.
  LOOM_AMDGPU_FP4_PAYLOAD_BYTES_PER_REGISTER = 4u,
  // Number of independently selectable E8M0 bytes in one scale register.
  LOOM_AMDGPU_FP4_E8M0_SCALE_BYTES_PER_REGISTER = 4u,
  // Number of E2M1 elements consumed by one native packed conversion.
  LOOM_AMDGPU_FP4_PK8_ELEMENT_COUNT = 8u,
  // Number of packed BF16 result registers written by one pk8 conversion.
  LOOM_AMDGPU_FP4_PK8_BF16_RESULT_REGISTER_COUNT = 4u,
  // Maximum E8M0 scale groups in one accepted packed BF16 vector.
  LOOM_AMDGPU_FP4_MAX_E8M0_BF16_SCALE_GROUPS =
      LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES /
      LOOM_AMDGPU_FP4_E8M0_BF16_SCALE_GROUP_ELEMENT_COUNT,
};

static const loom_amdgpu_descriptor_ref_t kLoomAmdgpuFp4NativePairDescriptorRefs
    [LOOM_AMDGPU_FP4_PAYLOAD_BYTES_PER_REGISTER] = {
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALEF32_PK_BF16_FP4_OCP,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALEF32_PK_BF16_FP4_OCP_BYTE1,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALEF32_PK_BF16_FP4_OCP_BYTE2,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALEF32_PK_BF16_FP4_OCP_BYTE3,
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

struct loom_amdgpu_fp4_native_pair_decode_recipe_t {
  // Native scaled pair descriptors selected by source byte.
  loom_low_lower_resolved_descriptor_t
      pair_descriptors[LOOM_AMDGPU_FP4_PAYLOAD_BYTES_PER_REGISTER];
  // Logical right shift selecting an encoded scale byte.
  loom_low_lower_resolved_descriptor_t right_shift_descriptor;
  // Literal mask isolating one encoded E8M0 scale byte.
  loom_low_lower_resolved_descriptor_t scale_mask_descriptor;
  // Logical left shift positioning E8M0 bits as an F32 exponent.
  loom_low_lower_resolved_descriptor_t scale_shift_descriptor;
};

typedef struct loom_amdgpu_fp4_native_pair_decode_recipe_cache_t {
  // Whether the pair recipe has been resolved for this function's target.
  bool initialized;
  // Function-local descriptor resolution shared by native pair decodes.
  loom_amdgpu_fp4_native_pair_decode_recipe_t recipe;
} loom_amdgpu_fp4_native_pair_decode_recipe_cache_t;

static int loom_amdgpu_fp4_native_pair_decode_recipe_cache_state_key;

struct loom_amdgpu_fp4_native_pk8_decode_recipe_t {
  // Native eight-lane E2M1-to-BF16 conversion descriptor.
  loom_low_lower_resolved_descriptor_t conversion_descriptor;
  // Module string ID for the descriptor scale_sel attribute.
  loom_string_id_t scale_selector_attr_name_id;
};

typedef struct loom_amdgpu_fp4_native_pk8_decode_recipe_cache_t {
  // Whether the pk8 recipe has been resolved for this function's target.
  bool initialized;
  // Function-local descriptor resolution shared by native pk8 decodes.
  loom_amdgpu_fp4_native_pk8_decode_recipe_t recipe;
} loom_amdgpu_fp4_native_pk8_decode_recipe_cache_t;

static int loom_amdgpu_fp4_native_pk8_decode_recipe_cache_state_key;

typedef struct loom_amdgpu_fp4_bf16_decode_recipe_t {
  // Vector constant descriptor used by scale-conditioned table values.
  loom_low_lower_resolved_descriptor_t vector_constant_descriptor;
  // Packed unsigned 16-bit add descriptor used to apply exponent deltas.
  loom_low_lower_resolved_descriptor_t packed_add_descriptor;
  // Packed unsigned 16-bit minimum descriptor used to clamp overflow.
  loom_low_lower_resolved_descriptor_t packed_minimum_descriptor;
  // Packed 16-bit logical right-shift descriptor used for table selectors.
  loom_low_lower_resolved_descriptor_t packed_right_shift_descriptor;
} loom_amdgpu_fp4_bf16_decode_recipe_t;

typedef struct loom_amdgpu_fp4_bf16_decode_recipe_cache_t {
  // Whether the portable BF16 recipe has been resolved for this function.
  bool initialized;
  // Function-local descriptor resolution shared by every BF16 FP4 decode.
  loom_amdgpu_fp4_bf16_decode_recipe_t recipe;
} loom_amdgpu_fp4_bf16_decode_recipe_cache_t;

static int loom_amdgpu_fp4_bf16_decode_recipe_cache_state_key;

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

static loom_amdgpu_descriptor_ref_t loom_amdgpu_fp4_immediate_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t inline_ref,
    loom_amdgpu_descriptor_ref_t literal_ref) {
  if (loom_amdgpu_descriptor_set_has_ref(descriptor_set, inline_ref)) {
    return inline_ref;
  }
  return loom_amdgpu_descriptor_set_has_ref(descriptor_set, literal_ref)
             ? literal_ref
             : LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
}

static bool loom_amdgpu_fp4_native_e8m0_bf16_pair_descriptors_available(
    const loom_low_descriptor_set_t* descriptor_set) {
  // All source-byte variants are generated from the same instruction overlay,
  // so the base ref represents availability of the complete family.
  return loom_amdgpu_descriptor_set_has_ref(
             descriptor_set,
             LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALEF32_PK_BF16_FP4_OCP) &&
         loom_amdgpu_fp4_immediate_descriptor_ref(
             descriptor_set,
             LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_SRC0_INLINE,
             LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT) !=
             LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT) &&
         loom_amdgpu_fp4_immediate_descriptor_ref(
             descriptor_set,
             LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_SRC0_INLINE,
             LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT) !=
             LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
}

static bool loom_amdgpu_fp4_native_e8m0_bf16_pk8_descriptor_available(
    const loom_low_descriptor_set_t* descriptor_set) {
  return loom_amdgpu_descriptor_set_has_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALE_PK8_BF16_FP4_OCP);
}

static iree_status_t loom_amdgpu_initialize_fp4_native_pair_decode_recipe(
    loom_low_lower_context_t* context,
    loom_amdgpu_fp4_native_pair_decode_recipe_t* recipe) {
  memset(recipe, 0, sizeof(*recipe));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_amdgpu_descriptor_ref_t right_shift_ref =
      loom_amdgpu_fp4_immediate_descriptor_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_SRC0_INLINE,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT);
  const loom_amdgpu_descriptor_ref_t scale_shift_ref =
      loom_amdgpu_fp4_immediate_descriptor_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_SRC0_INLINE,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT);
  IREE_ASSERT_NE(right_shift_ref, LOOM_AMDGPU_DESCRIPTOR_REF_NONE);
  IREE_ASSERT_NE(scale_shift_ref, LOOM_AMDGPU_DESCRIPTOR_REF_NONE);
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFp4NativePairDescriptorRefs); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, kLoomAmdgpuFp4NativePairDescriptorRefs[i],
        &recipe->pair_descriptors[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, right_shift_ref, &recipe->right_shift_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      &recipe->scale_mask_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, scale_shift_ref, &recipe->scale_shift_descriptor));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_get_fp4_native_pair_decode_recipe(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fp4_native_pair_decode_recipe_t** out_recipe) {
  *out_recipe = NULL;
  loom_amdgpu_fp4_native_pair_decode_recipe_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fp4_native_pair_decode_recipe_cache_state_key,
      sizeof(*cache), (void**)&cache));
  if (!cache->initialized) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fp4_native_pair_decode_recipe(
        context, &cache->recipe));
    cache->initialized = true;
  }
  *out_recipe = &cache->recipe;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_initialize_fp4_native_pk8_decode_recipe(
    loom_low_lower_context_t* context,
    loom_amdgpu_fp4_native_pk8_decode_recipe_t* recipe) {
  memset(recipe, 0, sizeof(*recipe));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_SCALE_PK8_BF16_FP4_OCP,
      &recipe->conversion_descriptor));
  return loom_amdgpu_intern(context, IREE_SV("scale_sel"),
                            &recipe->scale_selector_attr_name_id);
}

static iree_status_t loom_amdgpu_get_fp4_native_pk8_decode_recipe(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fp4_native_pk8_decode_recipe_t** out_recipe) {
  *out_recipe = NULL;
  loom_amdgpu_fp4_native_pk8_decode_recipe_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fp4_native_pk8_decode_recipe_cache_state_key,
      sizeof(*cache), (void**)&cache));
  if (!cache->initialized) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fp4_native_pk8_decode_recipe(
        context, &cache->recipe));
    cache->initialized = true;
  }
  *out_recipe = &cache->recipe;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_initialize_fp4_bf16_decode_recipe(
    loom_low_lower_context_t* context,
    loom_amdgpu_fp4_bf16_decode_recipe_t* recipe) {
  memset(recipe, 0, sizeof(*recipe));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      &recipe->vector_constant_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ADD_U16,
      &recipe->packed_add_descriptor));
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MIN_U16,
      &recipe->packed_minimum_descriptor));
  return loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_LSHRREV_B16,
      &recipe->packed_right_shift_descriptor);
}

static iree_status_t loom_amdgpu_get_fp4_bf16_decode_recipe(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fp4_bf16_decode_recipe_t** out_recipe) {
  *out_recipe = NULL;
  loom_amdgpu_fp4_bf16_decode_recipe_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fp4_bf16_decode_recipe_cache_state_key,
      sizeof(*cache), (void**)&cache));
  if (!cache->initialized) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_initialize_fp4_bf16_decode_recipe(context, &cache->recipe));
    cache->initialized = true;
  }
  *out_recipe = &cache->recipe;
  return iree_ok_status();
}

typedef struct loom_amdgpu_fp4_decode_shape_t {
  // Number of packed i32 payload registers.
  uint32_t source_register_count;
  // Number of logical E2M1/result lanes represented by the payload.
  uint32_t lane_count;
  // Number of packed 16-bit result registers.
  uint32_t result_register_count;
  // Scalar element type of the packed 16-bit result lanes.
  loom_scalar_type_t result_element_type;
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
  const loom_scalar_type_t result_element_type =
      loom_type_element_type(result_type);
  if (result_element_type != LOOM_SCALAR_TYPE_F16 &&
      result_element_type != LOOM_SCALAR_TYPE_BF16) {
    return false;
  }
  const uint32_t result_lane_count = loom_vector_static_rank1_lane_count(
      result_type, result_element_type,
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
      .result_element_type = result_element_type,
  };
  return true;
}

static bool loom_amdgpu_fp4_unscaled_schema_matches(
    loom_value_fact_encoded_operand_schema_t schema,
    const loom_amdgpu_fp4_decode_shape_t* shape) {
  return shape->result_element_type == LOOM_SCALAR_TYPE_F16 &&
         !loom_value_fact_encoded_operand_schema_is_unknown(schema) &&
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
         schema.scale_group.element_count == 0 &&
         schema.scale_operand_count == 0;
}

typedef struct loom_amdgpu_fp4_scale_shape_t {
  // Number of logical payload lanes sharing one scale.
  uint32_t group_element_count;
  // Number of scale values covering the decoded payload.
  uint32_t scale_count;
  // Number of packed i32 registers carrying scale_count encoded scale bytes.
  uint32_t register_count;
} loom_amdgpu_fp4_scale_shape_t;

static loom_value_fact_numeric_format_flags_t
loom_amdgpu_fp4_result_scale_format(
    const loom_amdgpu_fp4_decode_shape_t* shape) {
  switch (shape->result_element_type) {
    case LOOM_SCALAR_TYPE_F16:
      return LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN;
    case LOOM_SCALAR_TYPE_BF16:
      return LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0;
    default:
      return LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE;
  }
}

static bool loom_amdgpu_fp4_scaled_schema_matches(
    loom_value_fact_encoded_operand_schema_t schema,
    const loom_amdgpu_fp4_decode_shape_t* shape,
    loom_amdgpu_fp4_scale_shape_t* out_scale_shape) {
  *out_scale_shape = (loom_amdgpu_fp4_scale_shape_t){0};
  const loom_value_fact_numeric_format_flags_t scale_format =
      loom_amdgpu_fp4_result_scale_format(shape);
  if (loom_value_fact_encoded_operand_schema_is_unknown(schema) ||
      schema.element_format != LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1 ||
      schema.scale_format != scale_format ||
      schema.secondary_scale_format != LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE ||
      schema.payload_packing !=
          LOOM_VALUE_FACT_PAYLOAD_PACKING_LITTLE_ENDIAN_NIBBLES ||
      schema.scale_topology != LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D ||
      schema.affine_policy != LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY ||
      schema.rounding_policy != LOOM_VALUE_FACT_ROUNDING_POLICY_NONE ||
      schema.codebook_policy != LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE ||
      schema.sparsity_policy != LOOM_VALUE_FACT_SPARSITY_POLICY_NONE ||
      (schema.flags != 0 &&
       schema.flags !=
           LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK) ||
      schema.payload_register_count != shape->source_register_count ||
      schema.payload_element_count != shape->lane_count ||
      schema.scale_operand_count != 1) {
    return false;
  }

  const uint32_t group_element_count = schema.scale_group.element_count;
  if (group_element_count < 2 || (group_element_count & 1u) != 0 ||
      (group_element_count & (group_element_count - 1u)) != 0 ||
      group_element_count > shape->lane_count ||
      shape->lane_count % group_element_count != 0) {
    return false;
  }
  if (scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0 &&
      group_element_count !=
          LOOM_AMDGPU_FP4_E8M0_BF16_SCALE_GROUP_ELEMENT_COUNT) {
    return false;
  }
  const uint8_t group_rank =
      loom_value_fact_encoded_operand_scale_group_rank(&schema);
  if (group_rank > 1 ||
      (group_rank == 1 && schema.scale_group.shape[0] != group_element_count)) {
    return false;
  }

  const uint32_t scale_count = shape->lane_count / group_element_count;
  const uint32_t register_count = (scale_count + 3u) / 4u;
  if (register_count == 0 ||
      register_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return false;
  }
  *out_scale_shape = (loom_amdgpu_fp4_scale_shape_t){
      .group_element_count = group_element_count,
      .scale_count = scale_count,
      .register_count = register_count,
  };
  return true;
}

static bool loom_amdgpu_fp4_scale_source(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_amdgpu_fp4_scale_shape_t* scale_shape,
    loom_value_id_t* out_scale_source) {
  *out_scale_source = LOOM_VALUE_ID_INVALID;
  loom_encoding_auxiliary_view_t auxiliary_view = {0};
  iree_string_view_t unknown_key = iree_string_view_empty();
  if (!loom_encoding_auxiliary_view_resolve(
          module, loom_vector_decode_auxiliary(source_op),
          loom_vector_decode_auxiliary_names(source_op), &auxiliary_view,
          &unknown_key) ||
      auxiliary_view.present_keys !=
          loom_encoding_auxiliary_key_flag(LOOM_ENCODING_AUXILIARY_KEY_SCALE)) {
    return false;
  }
  const loom_value_id_t scale_source =
      auxiliary_view.values[LOOM_ENCODING_AUXILIARY_KEY_SCALE];
  if (scale_source == LOOM_VALUE_ID_INVALID ||
      scale_source >= module->values.count ||
      loom_vector_static_rank1_lane_count(
          loom_module_value_type(module, scale_source), LOOM_SCALAR_TYPE_I32,
          LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) !=
          scale_shape->register_count) {
    return false;
  }
  *out_scale_source = scale_source;
  return true;
}

static bool loom_amdgpu_fp4_scaled_descriptors_available(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fp4_decode_shape_t* shape,
    const loom_amdgpu_fp4_scale_shape_t* scale_shape) {
  if (scale_shape->scale_count != 1 &&
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT)) {
    return false;
  }
  if (shape->result_element_type == LOOM_SCALAR_TYPE_F16) {
    return loom_amdgpu_descriptor_set_has_ref(
               descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F16_F32) &&
           loom_amdgpu_descriptor_set_can_emit_packed_u16_lane_pair(
               descriptor_set) &&
           loom_amdgpu_descriptor_set_has_ref(
               descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MUL_F16);
  }
  IREE_ASSERT_EQ(shape->result_element_type, LOOM_SCALAR_TYPE_BF16);
  const loom_amdgpu_descriptor_ref_t required_refs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_ADD_U16,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MIN_U16,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_LSHRREV_B16,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
  };
  return loom_amdgpu_descriptor_set_has_all_refs(
             descriptor_set, required_refs, IREE_ARRAYSIZE(required_refs)) &&
         loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
             UINT32_C(0xFF)) &&
         loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
             UINT32_C(0xFFFF0000)) &&
         loom_amdgpu_descriptor_set_can_emit_vgpr_compare_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
             LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32_SRC1_INLINE, 0) &&
         loom_amdgpu_descriptor_set_can_emit_vgpr_compare_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
             LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32_SRC1_INLINE, 1) &&
         loom_amdgpu_descriptor_set_can_emit_vgpr_compare_immediate(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
             LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32_SRC1_INLINE,
             UINT32_C(0xFF));
}

typedef struct loom_amdgpu_fp4_decode_match_t {
  // Physical payload/result shape accepted by the selected route.
  loom_amdgpu_fp4_decode_shape_t shape;
  // Optional encoded scale auxiliary.
  loom_value_id_t scale_source;
  // Exact numeric format of scale_source.
  loom_value_fact_numeric_format_flags_t scale_format;
  // Physical encoded-scale shape accepted by the selected route.
  loom_amdgpu_fp4_scale_shape_t scale_shape;
} loom_amdgpu_fp4_decode_match_t;

static bool loom_amdgpu_fp4_decode_match(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op, loom_amdgpu_fp4_decode_match_t* out_match) {
  *out_match = (loom_amdgpu_fp4_decode_match_t){
      .scale_source = LOOM_VALUE_ID_INVALID,
  };
  if (fact_table == NULL ||
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

  loom_amdgpu_fp4_scale_shape_t scale_shape;
  if (!loom_amdgpu_fp4_scaled_schema_matches(
          summary.storage_schema.encoded_operand, &out_match->shape,
          &scale_shape)) {
    return false;
  }
  loom_value_id_t scale_source = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_fp4_scale_source(module, source_op, &scale_shape,
                                    &scale_source)) {
    return false;
  }
  out_match->scale_source = scale_source;
  out_match->scale_format = summary.storage_schema.encoded_operand.scale_format;
  out_match->scale_shape = scale_shape;
  return true;
}

static loom_amdgpu_fp4_decode_kind_t loom_amdgpu_fp4_decode_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fp4_decode_match_t* match) {
  if (descriptor_set == NULL) {
    return LOOM_AMDGPU_FP4_DECODE_KIND_NONE;
  }
  if (match->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0 &&
      loom_amdgpu_fp4_native_e8m0_bf16_pk8_descriptor_available(
          descriptor_set)) {
    return LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_E8M0_PK8;
  }
  if (match->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0 &&
      loom_amdgpu_fp4_native_e8m0_bf16_pair_descriptors_available(
          descriptor_set)) {
    return LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_SCALEF32_PAIR;
  }
  if (!loom_amdgpu_fp4_decode_descriptors_available(descriptor_set)) {
    return LOOM_AMDGPU_FP4_DECODE_KIND_NONE;
  }
  if (match->scale_format != LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
    if (!loom_amdgpu_fp4_scaled_descriptors_available(
            descriptor_set, &match->shape, &match->scale_shape)) {
      return LOOM_AMDGPU_FP4_DECODE_KIND_NONE;
    }
  }
  return LOOM_AMDGPU_FP4_DECODE_KIND_PORTABLE_LOOKUP;
}

bool loom_amdgpu_vector_decode_can_lower_as_fp4_conversion(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_op_t* source_op) {
  loom_amdgpu_fp4_decode_match_t match;
  return loom_amdgpu_fp4_decode_match(module, fact_table, source_op, &match) &&
         loom_amdgpu_fp4_decode_kind(descriptor_set, &match) !=
             LOOM_AMDGPU_FP4_DECODE_KIND_NONE;
}

iree_status_t loom_amdgpu_select_fp4_decode_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_16bit_float_conversion_plan_t* out_plan,
    bool* out_selected) {
  loom_amdgpu_fp4_decode_match_t match;
  const bool matched = loom_amdgpu_fp4_decode_match(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context), source_op, &match);
  const loom_amdgpu_fp4_decode_kind_t decode_kind =
      matched ? loom_amdgpu_fp4_decode_kind(
                    loom_low_lower_context_descriptor_set(context), &match)
              : LOOM_AMDGPU_FP4_DECODE_KIND_NONE;
  *out_selected = decode_kind != LOOM_AMDGPU_FP4_DECODE_KIND_NONE;
  if (decode_kind == LOOM_AMDGPU_FP4_DECODE_KIND_NONE) {
    return iree_ok_status();
  }

  const loom_amdgpu_fp4_decode_recipe_t* portable_recipe = NULL;
  const loom_amdgpu_fp4_native_pair_decode_recipe_t* native_pair_recipe = NULL;
  const loom_amdgpu_fp4_native_pk8_decode_recipe_t* native_pk8_recipe = NULL;
  if (decode_kind == LOOM_AMDGPU_FP4_DECODE_KIND_PORTABLE_LOOKUP) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_get_fp4_decode_recipe(context, &portable_recipe));
  } else if (decode_kind == LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_SCALEF32_PAIR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp4_native_pair_decode_recipe(
        context, &native_pair_recipe));
  } else {
    IREE_ASSERT_EQ(decode_kind, LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_E8M0_PK8);
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp4_native_pk8_decode_recipe(
        context, &native_pk8_recipe));
  }

  const loom_value_id_t source = loom_vector_decode_payload(source_op);
  const loom_value_id_t result = loom_vector_decode_result(source_op);
  *out_plan = (loom_amdgpu_vector_16bit_float_conversion_plan_t){
      .source = source,
      .result = result,
      .storage_source = source,
      .content_fact_source = result,
      .scale_source = match.scale_source,
      .scale_format = match.scale_format,
      .scale_group_element_count = match.scale_shape.group_element_count,
      .scale_count = match.scale_shape.scale_count,
      .scale_register_count = match.scale_shape.register_count,
      .kind = LOOM_AMDGPU_VECTOR_16BIT_FLOAT_CONVERSION_KIND_DECODE,
      .source_element_type = LOOM_SCALAR_TYPE_I32,
      .source_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1,
      .descriptor_source_format = LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE,
      .result_element_type = match.shape.result_element_type,
      .lane_count = match.shape.lane_count,
      .source_register_count = match.shape.source_register_count,
      .storage_lane_stride = 1,
      .storage_lane_count = match.shape.source_register_count,
      .storage_register_count = match.shape.source_register_count,
      .result_register_count = match.shape.result_register_count,
      .fp4_decode =
          {
              .kind = decode_kind,
              .portable_recipe = portable_recipe,
              .native_pair_recipe = native_pair_recipe,
              .native_pk8_recipe = native_pk8_recipe,
          },
  };
  return iree_ok_status();
}

typedef struct loom_amdgpu_fp4_decode_emission_state_t {
  // Packed shifts positioning the two magnitudes in V_PERM selector bytes.
  loom_value_id_t magnitude_position_shifts;
  // Packed shifts positioning two sign bits in 16-bit result lanes.
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
    const loom_amdgpu_fp4_decode_recipe_t* recipe,
    loom_scalar_type_t result_element_type, loom_type_t scalar_type,
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
  if (result_element_type == LOOM_SCALAR_TYPE_F16) {
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
  } else {
    IREE_ASSERT_EQ(result_element_type, LOOM_SCALAR_TYPE_BF16);
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

static iree_status_t loom_amdgpu_emit_fp4_pair_selectors(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_decode_recipe_t* recipe,
    const loom_amdgpu_fp4_decode_emission_state_t* state,
    loom_value_id_t source_register, uint32_t byte_index,
    loom_type_t vector_type, loom_value_id_t* out_positioned_nibbles,
    loom_value_id_t* out_magnitude_selectors) {
  loom_value_id_t duplicated_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_duplicate_byte(
      context, source_op, recipe, state, source_register, byte_index,
      vector_type, &duplicated_byte));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &recipe->packed_shift_descriptor,
      state->magnitude_position_shifts, duplicated_byte, vector_type,
      out_positioned_nibbles));
  return loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &recipe->mask_descriptor, *out_positioned_nibbles,
      LOOM_AMDGPU_FP4_MAGNITUDE_SELECTOR_MASK, vector_type,
      out_magnitude_selectors);
}

static iree_status_t loom_amdgpu_emit_fp4_pair_signs(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_decode_recipe_t* recipe,
    const loom_amdgpu_fp4_decode_emission_state_t* state,
    loom_value_id_t positioned_nibbles, loom_type_t vector_type,
    loom_value_id_t* out_signs) {
  loom_value_id_t positioned_signs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &recipe->packed_shift_descriptor,
      state->sign_position_shifts, positioned_nibbles, vector_type,
      &positioned_signs));
  return loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &recipe->mask_descriptor, positioned_signs,
      LOOM_AMDGPU_FP4_SIGN_MASK, vector_type, out_signs);
}

static iree_status_t loom_amdgpu_emit_fp4_pair_as_packed_f16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_decode_recipe_t* recipe,
    const loom_amdgpu_fp4_decode_emission_state_t* state,
    loom_value_id_t source_register, uint32_t byte_index,
    loom_type_t vector_type, loom_value_id_t* out_packed_f16) {
  loom_value_id_t positioned_nibbles = LOOM_VALUE_ID_INVALID;
  loom_value_id_t magnitude_selectors = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_pair_selectors(
      context, source_op, recipe, state, source_register, byte_index,
      vector_type, &positioned_nibbles, &magnitude_selectors));

  loom_value_id_t magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &recipe->permute_descriptor,
      state->magnitude_table_high, state->magnitude_table_low,
      magnitude_selectors, vector_type, &magnitude));

  loom_value_id_t signs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fp4_pair_signs(context, source_op, recipe, state,
                                      positioned_nibbles, vector_type, &signs));
  return loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &recipe->merge_descriptor, magnitude, signs,
      vector_type, out_packed_f16);
}

typedef struct loom_amdgpu_fp4_bf16_lookup_table_t {
  // Low BF16 bytes for magnitudes 0-3 and 4-7.
  loom_value_id_t low_bytes[2];
  // High BF16 bytes for magnitudes 0-3 and 4-7.
  loom_value_id_t high_bytes[2];
  // Per-workitem predicate selecting signed NaNs for E8M0 scale 0xFF.
  loom_value_id_t nan_mask;
} loom_amdgpu_fp4_bf16_lookup_table_t;

typedef struct loom_amdgpu_fp4_bf16_emission_constants_t {
  // Packed shift moving magnitude selectors into low result bytes.
  loom_value_id_t selector_position_shifts;
  // Packed E8M0 exponent unbias.
  loom_value_id_t scale_unbias;
  // Packed shift moving exponent deltas into BF16 exponent fields.
  loom_value_id_t scale_position_shifts;
  // Identity-scale packed BF16 magnitude pairs.
  loom_value_id_t identity_tables[4];
  // Minimum-scale special packed BF16 magnitude pairs.
  loom_value_id_t minimum_scale_tables[2];
  // Exponent-one special packed BF16 magnitudes 0 and 0.5.
  loom_value_id_t next_scale_table_01;
  // Packed positive BF16 infinities.
  loom_value_id_t infinity_pair;
  // Packed positive quiet BF16 NaNs.
  loom_value_id_t quiet_nan_pair;
  // Selector packing low bytes from four packed BF16 pairs.
  loom_value_id_t low_byte_table_selector;
  // Selector packing high bytes from four packed BF16 pairs.
  loom_value_id_t high_byte_table_selector;
} loom_amdgpu_fp4_bf16_emission_constants_t;

typedef struct loom_amdgpu_fp4_scale_emission_state_t {
  // Packed i32 registers carrying the encoded scale bytes.
  loom_value_id_t low_source;
  // Exact numeric format of the encoded scale bytes.
  loom_value_fact_numeric_format_flags_t scale_format;
  // Number of scale values covering the decoded payload.
  uint32_t scale_count;
  // Number of packed registers in low_source.
  uint32_t source_register_count;
  // FP8 lane recipe used to decode one E4M3FN scale byte to F32.
  const loom_amdgpu_fp8_decode_plan_t* decode_plan;
  // SGPR pair type used by encoded-scale compare/select packets.
  loom_type_t mask_type;
  // Optional native u16 pair pack used to splat each decoded scale.
  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor;
  // Packed F16 multiply applying a decoded scale to one E2M1 pair.
  const loom_low_lower_resolved_descriptor_t* multiply_descriptor;
  // Packed integer recipe used by the portable E8M0-to-BF16 path.
  const loom_amdgpu_fp4_bf16_decode_recipe_t* bf16_decode_recipe;
  // Constants shared by every E8M0 scale group in this conversion.
  loom_amdgpu_fp4_bf16_emission_constants_t bf16_constants;
} loom_amdgpu_fp4_scale_emission_state_t;

static iree_status_t loom_amdgpu_initialize_fp4_scale_emission(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan,
    loom_amdgpu_fp4_scale_emission_state_t* state) {
  *state = (loom_amdgpu_fp4_scale_emission_state_t){
      .low_source = LOOM_VALUE_ID_INVALID,
      .scale_format = plan->scale_format,
      .mask_type = loom_type_none(),
  };
  if (plan->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
    return iree_ok_status();
  }
  IREE_ASSERT_GE(plan->scale_group_element_count, 2u);
  IREE_ASSERT_EQ(plan->scale_group_element_count & 1u, 0u);
  IREE_ASSERT_EQ(plan->lane_count % plan->scale_group_element_count, 0u);
  state->scale_count = plan->scale_count;
  state->source_register_count = plan->scale_register_count;
  IREE_ASSERT_GT(state->scale_count, 0u);
  IREE_ASSERT_GT(state->source_register_count, 0u);
  IREE_ASSERT_LE(state->scale_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->scale_source,
                                                   &state->low_source));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, state->low_source, &state->low_source));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &state->mask_type));
  if (plan->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_decode_plan(
        context, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN,
        LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN, &state->decode_plan));
    IREE_ASSERT(
        iree_any_bit_set(state->decode_plan->flags,
                         LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_F16));
    state->pack_u16_descriptor =
        iree_any_bit_set(state->decode_plan->flags,
                         LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16)
            ? &state->decode_plan->pack_u16_descriptor
            : NULL;
    state->multiply_descriptor = &state->decode_plan->pk_mul_f16_descriptor;
    return iree_ok_status();
  }

  IREE_ASSERT_EQ(plan->scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0);
  IREE_ASSERT_LE(state->scale_count,
                 LOOM_AMDGPU_FP4_MAX_E8M0_BF16_SCALE_GROUPS);
  return loom_amdgpu_get_fp4_bf16_decode_recipe(context,
                                                &state->bf16_decode_recipe);
}

static iree_status_t loom_amdgpu_extract_fp4_scale_byte(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_scale_emission_state_t* state, uint32_t scale_index,
    loom_type_t vector_type, loom_value_id_t* out_low_scale_byte) {
  IREE_ASSERT_LT(scale_index, state->scale_count);
  const uint32_t source_register_index = scale_index / 4u;
  const uint32_t source_byte_index = scale_index & 3u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, state->low_source, state->source_register_count,
      source_register_index, vector_type, &source_register));
  *out_low_scale_byte = source_register;
  if (source_byte_index != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        source_byte_index * 8u, source_register, vector_type,
        out_low_scale_byte));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp4_e4m3fn_packed_scale(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_scale_emission_state_t* state, uint32_t scale_index,
    loom_type_t scalar_type, loom_type_t vector_type,
    loom_value_id_t* out_packed_scale) {
  IREE_ASSERT_EQ(state->scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN);
  IREE_ASSERT_LT(scale_index, state->scale_count);

  loom_value_id_t low_scale_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_fp4_scale_byte(
      context, source_op, state, scale_index, vector_type, &low_scale_byte));

  loom_value_id_t f32_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_f32_lane(
      context, source_op, state->decode_plan, low_scale_byte,
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF, vector_type, scalar_type,
      state->mask_type, &f32_scale));
  return loom_amdgpu_splat_f32_lane_to_packed_f16(
      context, source_op, state->pack_u16_descriptor, f32_scale, vector_type,
      out_packed_scale);
}

static iree_status_t loom_amdgpu_initialize_fp4_bf16_emission_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_decode_recipe_t* decode_recipe,
    const loom_amdgpu_fp4_bf16_decode_recipe_t* bf16_recipe,
    loom_type_t scalar_type, loom_type_t vector_type,
    loom_amdgpu_fp4_bf16_emission_constants_t* constants) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
      context, source_op, decode_recipe,
      LOOM_AMDGPU_FP4_BF16_SELECTOR_POSITION_SHIFTS, scalar_type,
      &constants->selector_position_shifts));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
      context, source_op, decode_recipe, LOOM_AMDGPU_FP4_BF16_SCALE_UNBIAS,
      scalar_type, &constants->scale_unbias));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
      context, source_op, decode_recipe,
      LOOM_AMDGPU_FP4_BF16_SCALE_POSITION_SHIFTS, scalar_type,
      &constants->scale_position_shifts));

  static const uint32_t identity_table_values[] = {
      LOOM_AMDGPU_FP4_BF16_MAGNITUDE_TABLE_01,
      LOOM_AMDGPU_FP4_BF16_MAGNITUDE_TABLE_23,
      LOOM_AMDGPU_FP4_BF16_MAGNITUDE_TABLE_45,
      LOOM_AMDGPU_FP4_BF16_MAGNITUDE_TABLE_67,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(identity_table_values); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
        context, source_op, decode_recipe, identity_table_values[i],
        scalar_type, &constants->identity_tables[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
      context, source_op, decode_recipe, LOOM_AMDGPU_FP4_BF16_INFINITY_PAIR,
      scalar_type, &constants->infinity_pair));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
      context, source_op, decode_recipe,
      LOOM_AMDGPU_FP4_BF16_LOW_BYTE_TABLE_SELECTOR, scalar_type,
      &constants->low_byte_table_selector));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_scalar_constant(
      context, source_op, decode_recipe,
      LOOM_AMDGPU_FP4_BF16_HIGH_BYTE_TABLE_SELECTOR, scalar_type,
      &constants->high_byte_table_selector));

  static const uint32_t minimum_scale_table_values[] = {
      LOOM_AMDGPU_FP4_BF16_MIN_SCALE_TABLE_01,
      LOOM_AMDGPU_FP4_BF16_MIN_SCALE_TABLE_23,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(minimum_scale_table_values); ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
        context, source_op, &bf16_recipe->vector_constant_descriptor,
        decode_recipe->imm32_attr_name_id, minimum_scale_table_values[i],
        vector_type, &constants->minimum_scale_tables[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &bf16_recipe->vector_constant_descriptor,
      decode_recipe->imm32_attr_name_id,
      LOOM_AMDGPU_FP4_BF16_NEXT_SCALE_TABLE_01, vector_type,
      &constants->next_scale_table_01));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, &bf16_recipe->vector_constant_descriptor,
      decode_recipe->imm32_attr_name_id, LOOM_AMDGPU_FP4_BF16_QUIET_NAN_PAIR,
      vector_type, &constants->quiet_nan_pair));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fp4_e8m0_bf16_lookup_table(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_decode_recipe_t* decode_recipe,
    const loom_amdgpu_fp4_decode_emission_state_t* decode_state,
    const loom_amdgpu_fp4_scale_emission_state_t* scale_state,
    uint32_t scale_index, loom_type_t vector_type,
    loom_amdgpu_fp4_bf16_lookup_table_t* table) {
  IREE_ASSERT_EQ(scale_state->scale_format,
                 LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0);
  IREE_ASSERT_LT(scale_index, scale_state->scale_count);
  const loom_amdgpu_fp4_bf16_emission_constants_t* constants =
      &scale_state->bf16_constants;

  loom_value_id_t low_scale_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_fp4_scale_byte(
      context, source_op, scale_state, scale_index, vector_type,
      &low_scale_byte));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      low_scale_byte, UINT32_C(0xFF), vector_type, &low_scale_byte));

  loom_value_id_t packed_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_duplicate_byte(
      context, source_op, decode_recipe, decode_state, low_scale_byte, 0,
      vector_type, &packed_scale));
  loom_value_id_t unbiased_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op,
      &scale_state->bf16_decode_recipe->packed_add_descriptor, packed_scale,
      constants->scale_unbias, vector_type, &unbiased_scale));
  loom_value_id_t exponent_delta = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &decode_recipe->packed_shift_descriptor,
      constants->scale_position_shifts, unbiased_scale, vector_type,
      &exponent_delta));

  loom_value_id_t magnitude_pairs[4];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(magnitude_pairs); ++i) {
    loom_value_id_t scaled_pair = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
        context, source_op,
        &scale_state->bf16_decode_recipe->packed_add_descriptor,
        constants->identity_tables[i], exponent_delta, vector_type,
        &scaled_pair));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
        context, source_op,
        &scale_state->bf16_decode_recipe->packed_minimum_descriptor,
        scaled_pair, constants->infinity_pair, vector_type,
        &magnitude_pairs[i]));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      magnitude_pairs[0], UINT32_C(0xFFFF0000), vector_type,
      &magnitude_pairs[0]));

  loom_value_id_t is_minimum_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_compare_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32_SRC1_INLINE, low_scale_byte, 0,
      vector_type, scale_state->mask_type, &is_minimum_scale));
  loom_value_id_t is_next_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_compare_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32_SRC1_INLINE, low_scale_byte, 1,
      vector_type, scale_state->mask_type, &is_next_scale));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_compare_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32_SRC1_INLINE, low_scale_byte,
      UINT32_C(0xFF), vector_type, scale_state->mask_type, &table->nan_mask));

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
      context, source_op, magnitude_pairs[0],
      constants->minimum_scale_tables[0], is_minimum_scale, vector_type,
      &magnitude_pairs[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
      context, source_op, magnitude_pairs[0], constants->next_scale_table_01,
      is_next_scale, vector_type, &magnitude_pairs[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_select(
      context, source_op, magnitude_pairs[1],
      constants->minimum_scale_tables[1], is_minimum_scale, vector_type,
      &magnitude_pairs[1]));

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &decode_recipe->permute_descriptor,
      magnitude_pairs[1], magnitude_pairs[0],
      constants->low_byte_table_selector, vector_type, &table->low_bytes[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &decode_recipe->permute_descriptor,
      magnitude_pairs[1], magnitude_pairs[0],
      constants->high_byte_table_selector, vector_type, &table->high_bytes[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &decode_recipe->permute_descriptor,
      magnitude_pairs[3], magnitude_pairs[2],
      constants->low_byte_table_selector, vector_type, &table->low_bytes[1]));
  return loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &decode_recipe->permute_descriptor,
      magnitude_pairs[3], magnitude_pairs[2],
      constants->high_byte_table_selector, vector_type, &table->high_bytes[1]);
}

static iree_status_t loom_amdgpu_emit_fp4_pair_as_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_decode_recipe_t* decode_recipe,
    const loom_amdgpu_fp4_decode_emission_state_t* decode_state,
    const loom_amdgpu_fp4_scale_emission_state_t* scale_state,
    const loom_amdgpu_fp4_bf16_lookup_table_t* table,
    loom_value_id_t source_register, uint32_t byte_index,
    loom_type_t vector_type, loom_value_id_t* out_packed_bf16) {
  loom_value_id_t positioned_nibbles = LOOM_VALUE_ID_INVALID;
  loom_value_id_t magnitude_selectors = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_pair_selectors(
      context, source_op, decode_recipe, decode_state, source_register,
      byte_index, vector_type, &positioned_nibbles, &magnitude_selectors));

  loom_value_id_t high_bytes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &decode_recipe->permute_descriptor,
      table->high_bytes[1], table->high_bytes[0], magnitude_selectors,
      vector_type, &high_bytes));
  loom_value_id_t low_byte_selectors = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op,
      &scale_state->bf16_decode_recipe->packed_right_shift_descriptor,
      scale_state->bf16_constants.selector_position_shifts, magnitude_selectors,
      vector_type, &low_byte_selectors));
  loom_value_id_t low_bytes = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_ternary(
      context, source_op, &decode_recipe->permute_descriptor,
      table->low_bytes[1], table->low_bytes[0], low_byte_selectors, vector_type,
      &low_bytes));
  loom_value_id_t magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &decode_recipe->merge_descriptor, high_bytes,
      low_bytes, vector_type, &magnitude));

  loom_value_id_t signs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_pair_signs(
      context, source_op, decode_recipe, decode_state, positioned_nibbles,
      vector_type, &signs));
  loom_value_id_t signed_magnitude = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &decode_recipe->merge_descriptor, magnitude, signs,
      vector_type, &signed_magnitude));
  loom_value_id_t signed_nan = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
      context, source_op, &decode_recipe->merge_descriptor,
      scale_state->bf16_constants.quiet_nan_pair, signs, vector_type,
      &signed_nan));
  return loom_amdgpu_emit_vgpr_select(context, source_op, signed_magnitude,
                                      signed_nan, table->nan_mask, vector_type,
                                      out_packed_bf16);
}

static iree_status_t loom_amdgpu_emit_fp4_native_scale_low_byte(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_native_pair_decode_recipe_t* recipe,
    loom_value_id_t source_register, uint32_t byte_index,
    loom_type_t vector_type, loom_value_id_t* out_low_byte) {
  *out_low_byte = source_register;
  if (byte_index == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &recipe->right_shift_descriptor, source_register,
      byte_index * 8u, vector_type, out_low_byte);
}

static iree_status_t loom_amdgpu_emit_fp4_native_e8m0_f32_scale(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp4_native_pair_decode_recipe_t* recipe,
    loom_value_id_t low_scale_source, uint32_t scale_register_count,
    uint32_t scale_index, loom_type_t vector_type,
    loom_value_id_t* out_f32_scale) {
  const uint32_t source_register_index =
      scale_index / LOOM_AMDGPU_FP4_E8M0_SCALE_BYTES_PER_REGISTER;
  const uint32_t source_byte_index =
      scale_index % LOOM_AMDGPU_FP4_E8M0_SCALE_BYTES_PER_REGISTER;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_scale_source, scale_register_count,
      source_register_index, vector_type, &source_register));

  loom_value_id_t low_scale_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_native_scale_low_byte(
      context, source_op, recipe, source_register, source_byte_index,
      vector_type, &low_scale_byte));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &recipe->scale_mask_descriptor, low_scale_byte,
      LOOM_AMDGPU_FP4_E8M0_SCALE_BYTE_MASK, vector_type, &low_scale_byte));
  return loom_amdgpu_emit_resolved_vgpr_unary_immediate(
      context, source_op, &recipe->scale_shift_descriptor, low_scale_byte,
      LOOM_AMDGPU_FP4_E8M0_F32_EXPONENT_SHIFT, vector_type, out_f32_scale);
}

static iree_status_t loom_amdgpu_lower_vector_fp4_decode_native_pair(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  IREE_ASSERT_EQ(plan->fp4_decode.kind,
                 LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_SCALEF32_PAIR);
  IREE_ASSERT_EQ(plan->result_element_type, LOOM_SCALAR_TYPE_BF16);
  IREE_ASSERT_EQ(plan->scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0);
  IREE_ASSERT_EQ(plan->scale_group_element_count,
                 LOOM_AMDGPU_FP4_E8M0_BF16_SCALE_GROUP_ELEMENT_COUNT);
  const loom_amdgpu_fp4_native_pair_decode_recipe_t* recipe =
      plan->fp4_decode.native_pair_recipe;
  IREE_ASSERT(recipe != NULL);

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_source, &low_source));
  loom_value_id_t low_scale_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->scale_source,
                                                   &low_scale_source));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_scale_source, &low_scale_source));

  loom_type_t vector_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vector_type));
  loom_value_id_t results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  const uint32_t result_pairs_per_scale = plan->scale_group_element_count / 2u;
  uint32_t result_pairs_until_next_scale = 0;
  uint32_t scale_index = 0;
  uint32_t result_index = 0;
  loom_value_id_t active_f32_scale = LOOM_VALUE_ID_INVALID;
  for (uint32_t source_index = 0; source_index < plan->source_register_count;
       ++source_index) {
    loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->source_register_count,
        source_index, vector_type, &source_register));
    for (uint32_t byte_index = 0;
         byte_index < LOOM_AMDGPU_FP4_PAYLOAD_BYTES_PER_REGISTER;
         ++byte_index) {
      if (result_pairs_until_next_scale == 0) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_native_e8m0_f32_scale(
            context, source_op, recipe, low_scale_source,
            plan->scale_register_count, scale_index++, vector_type,
            &active_f32_scale));
        result_pairs_until_next_scale = result_pairs_per_scale;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
          context, source_op, &recipe->pair_descriptors[byte_index],
          source_register, active_f32_scale, vector_type,
          &results[result_index++]));
      --result_pairs_until_next_scale;
    }
  }
  IREE_ASSERT_EQ(scale_index, plan->scale_count);
  IREE_ASSERT_EQ(result_index, plan->result_register_count);
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             results, result_index);
}

static iree_status_t loom_amdgpu_lower_vector_fp4_decode_native_pk8(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  IREE_ASSERT_EQ(plan->fp4_decode.kind,
                 LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_E8M0_PK8);
  IREE_ASSERT_EQ(plan->result_element_type, LOOM_SCALAR_TYPE_BF16);
  IREE_ASSERT_EQ(plan->scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0);
  IREE_ASSERT_EQ(plan->scale_group_element_count,
                 LOOM_AMDGPU_FP4_E8M0_BF16_SCALE_GROUP_ELEMENT_COUNT);
  IREE_ASSERT_EQ(
      plan->source_register_count * LOOM_AMDGPU_FP4_PK8_ELEMENT_COUNT,
      plan->lane_count);
  IREE_ASSERT_EQ(plan->source_register_count *
                     LOOM_AMDGPU_FP4_PK8_BF16_RESULT_REGISTER_COUNT,
                 plan->result_register_count);
  const loom_amdgpu_fp4_native_pk8_decode_recipe_t* recipe =
      plan->fp4_decode.native_pk8_recipe;
  IREE_ASSERT(recipe != NULL);

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_source, &low_source));
  loom_value_id_t low_scale_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->scale_source,
                                                   &low_scale_source));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_scale_source, &low_scale_source));

  loom_type_t vector_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vector_type));
  loom_type_t result_packet_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, LOOM_AMDGPU_FP4_PK8_BF16_RESULT_REGISTER_COUNT,
      &result_packet_type));

  loom_value_id_t result_packets[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint32_t active_scale_register_index = UINT32_MAX;
  loom_value_id_t active_scale_register = LOOM_VALUE_ID_INVALID;
  for (uint32_t source_index = 0; source_index < plan->source_register_count;
       ++source_index) {
    loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->source_register_count,
        source_index, vector_type, &source_register));

    const uint32_t scale_index =
        (source_index * LOOM_AMDGPU_FP4_PK8_ELEMENT_COUNT) /
        plan->scale_group_element_count;
    const uint32_t scale_register_index =
        scale_index / LOOM_AMDGPU_FP4_E8M0_SCALE_BYTES_PER_REGISTER;
    const uint32_t scale_selector =
        scale_index % LOOM_AMDGPU_FP4_E8M0_SCALE_BYTES_PER_REGISTER;
    if (scale_register_index != active_scale_register_index) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
          context, source_op, low_scale_source, plan->scale_register_count,
          scale_register_index, vector_type, &active_scale_register));
      active_scale_register_index = scale_register_index;
    }

    loom_named_attr_t scale_selector_attr = {
        .name_id = recipe->scale_selector_attr_name_id,
        .value = loom_attr_i64(scale_selector),
    };
    const loom_value_id_t operands[] = {source_register, active_scale_register};
    loom_op_t* convert_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &recipe->conversion_descriptor, operands,
        IREE_ARRAYSIZE(operands),
        scale_selector == 0
            ? loom_named_attr_slice_empty()
            : loom_make_named_attr_slice(&scale_selector_attr, 1),
        &result_packet_type, 1, /*tied_results=*/NULL,
        /*tied_result_count=*/0, source_op->location, &convert_op));
    result_packets[source_index] =
        loom_value_slice_get(loom_low_op_results(convert_op), 0);
  }

  if (plan->source_register_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, result_packets[0]);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, plan->result_register_count, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context),
                            result_packets, plan->source_register_count,
                            result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_vector_fp4_decode_portable(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  IREE_ASSERT_EQ(plan->fp4_decode.kind,
                 LOOM_AMDGPU_FP4_DECODE_KIND_PORTABLE_LOOKUP);
  const loom_amdgpu_fp4_decode_recipe_t* decode_recipe =
      plan->fp4_decode.portable_recipe;
  IREE_ASSERT(decode_recipe != NULL);
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
      context, source_op, decode_recipe, plan->result_element_type, scalar_type,
      vector_type, &state));
  loom_amdgpu_fp4_scale_emission_state_t scale_state;
  IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fp4_scale_emission(
      context, source_op, plan, &scale_state));
  uint32_t active_scale_index = UINT32_MAX;
  loom_value_id_t active_packed_f16_scale = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_fp4_bf16_lookup_table_t active_bf16_lookup_table;
  if (scale_state.scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN &&
      scale_state.scale_count == 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_e4m3fn_packed_scale(
        context, source_op, &scale_state, 0, scalar_type, vector_type,
        &active_packed_f16_scale));
    active_scale_index = 0;
  }
  if (scale_state.scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fp4_bf16_emission_constants(
        context, source_op, decode_recipe, scale_state.bf16_decode_recipe,
        scalar_type, vector_type, &scale_state.bf16_constants));
  }

  loom_value_id_t results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  uint32_t result_index = 0;
  for (uint32_t source_index = 0; source_index < plan->source_register_count;
       ++source_index) {
    loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
        context, source_op, low_source, plan->source_register_count,
        source_index, vector_type, &source_register));
    for (uint32_t byte_index = 0; byte_index < 4; ++byte_index) {
      const uint32_t scale_index =
          scale_state.scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE
              ? 0
              : (result_index * 2u) / plan->scale_group_element_count;
      if (scale_state.scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0) {
        if (scale_index != active_scale_index) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_e8m0_bf16_lookup_table(
              context, source_op, decode_recipe, &state, &scale_state,
              scale_index, vector_type, &active_bf16_lookup_table));
          active_scale_index = scale_index;
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_pair_as_packed_bf16(
            context, source_op, decode_recipe, &state, &scale_state,
            &active_bf16_lookup_table, source_register, byte_index, vector_type,
            &results[result_index++]));
      } else {
        loom_value_id_t decoded_pair = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_pair_as_packed_f16(
            context, source_op, decode_recipe, &state, source_register,
            byte_index, vector_type, &decoded_pair));
        if (scale_state.scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE) {
          results[result_index++] = decoded_pair;
          continue;
        }
        IREE_ASSERT_EQ(scale_state.scale_format,
                       LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN);
        if (scale_index != active_scale_index) {
          IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp4_e4m3fn_packed_scale(
              context, source_op, &scale_state, scale_index, scalar_type,
              vector_type, &active_packed_f16_scale));
          active_scale_index = scale_index;
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_vgpr_binary(
            context, source_op, scale_state.multiply_descriptor, decoded_pair,
            active_packed_f16_scale, vector_type, &results[result_index++]));
      }
    }
  }
  IREE_ASSERT_EQ(result_index, plan->result_register_count);
  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             results, result_index);
}

iree_status_t loom_amdgpu_lower_vector_fp4_decode(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  switch (plan->fp4_decode.kind) {
    case LOOM_AMDGPU_FP4_DECODE_KIND_PORTABLE_LOOKUP:
      return loom_amdgpu_lower_vector_fp4_decode_portable(context, source_op,
                                                          plan);
    case LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_SCALEF32_PAIR:
      return loom_amdgpu_lower_vector_fp4_decode_native_pair(context, source_op,
                                                             plan);
    case LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_E8M0_PK8:
      return loom_amdgpu_lower_vector_fp4_decode_native_pk8(context, source_op,
                                                            plan);
    default:
      IREE_ASSERT_UNREACHABLE("unselected packed FP4 decode");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_string_view_t loom_amdgpu_fp4_decode_plan_key(
    const loom_amdgpu_vector_16bit_float_conversion_plan_t* plan) {
  IREE_ASSERT_NE(plan->fp4_decode.kind, LOOM_AMDGPU_FP4_DECODE_KIND_NONE);
  if (plan->fp4_decode.kind ==
      LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_SCALEF32_PAIR) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp4_e2m1_e8m0_scale32_native_bf16_pair");
  }
  if (plan->fp4_decode.kind == LOOM_AMDGPU_FP4_DECODE_KIND_NATIVE_E8M0_PK8) {
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp4_e2m1_e8m0_scale32_native_bf16_pk8");
  }
  IREE_ASSERT_EQ(plan->fp4_decode.kind,
                 LOOM_AMDGPU_FP4_DECODE_KIND_PORTABLE_LOOKUP);
  if (plan->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN) {
    switch (plan->scale_group_element_count) {
      case 2:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "fp4_e2m1_e4m3fn_scale2_packed_f16_lookup");
      case 4:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "fp4_e2m1_e4m3fn_scale4_packed_f16_lookup");
      case 8:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "fp4_e2m1_e4m3fn_scale8_packed_f16_lookup");
      case 16:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "fp4_e2m1_e4m3fn_scale16_packed_f16_lookup");
      case 32:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "fp4_e2m1_e4m3fn_scale32_packed_f16_lookup");
      case 64:
        return IREE_SV(
            "amdgpu.vector_16bit_float_conversion.strategy."
            "fp4_e2m1_e4m3fn_scale64_packed_f16_lookup");
      default:
        IREE_ASSERT_UNREACHABLE("unsupported packed FP4 scale group");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  if (plan->scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0) {
    IREE_ASSERT_EQ(plan->scale_group_element_count,
                   LOOM_AMDGPU_FP4_E8M0_BF16_SCALE_GROUP_ELEMENT_COUNT);
    return IREE_SV(
        "amdgpu.vector_16bit_float_conversion.strategy."
        "fp4_e2m1_e8m0_scale32_packed_bf16_lookup");
  }
  IREE_ASSERT_EQ(plan->scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE);
  return IREE_SV(
      "amdgpu.vector_16bit_float_conversion.strategy."
      "fp4_e2m1_packed_f16_lookup");
}
