// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix/contract.h"

#include "loom/target/arch/amdgpu/matrix/contract_tables.h"
#include "loom/target/arch/amdgpu/target_info.h"

// Generated target-owned fragment layout rows.
extern const loom_amdgpu_matrix_fragment_layout_t
    kLoomAmdgpuMatrixFragmentLayouts[LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT];

static const iree_string_view_t kAmdgpuMatrixFamilyNames[] = {
    [LOOM_AMDGPU_MATRIX_FAMILY_UNKNOWN] = IREE_SVL("unknown"),
    [LOOM_AMDGPU_MATRIX_FAMILY_MFMA] = IREE_SVL("mfma"),
    [LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC] = IREE_SVL("smfmac"),
    [LOOM_AMDGPU_MATRIX_FAMILY_WMMA] = IREE_SVL("wmma"),
    [LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC] = IREE_SVL("swmmac"),
};

static const iree_string_view_t kAmdgpuMatrixNumericTypeNames[] = {
    [LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN] = IREE_SVL("unknown"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_F64] = IREE_SVL("f64"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_F32] = IREE_SVL("f32"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_F16] = IREE_SVL("f16"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_BF16] = IREE_SVL("bf16"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_XF32] = IREE_SVL("xf32"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_I32] = IREE_SVL("i32"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_I8] = IREE_SVL("i8"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_IU8] = IREE_SVL("iu8"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_I4] = IREE_SVL("i4"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_IU4] = IREE_SVL("iu4"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_FP8] = IREE_SVL("fp8"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_BF8] = IREE_SVL("bf8"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_FP6] = IREE_SVL("fp6"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_BF6] = IREE_SVL("bf6"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_FP4] = IREE_SVL("fp4"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_F8] = IREE_SVL("f8"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_F6] = IREE_SVL("f6"),
    [LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4] = IREE_SVL("f8f6f4"),
};

static const iree_string_view_t kAmdgpuMatrixScaleKindNames[] = {
    [LOOM_AMDGPU_MATRIX_SCALE_NONE] = IREE_SVL("none"),
    [LOOM_AMDGPU_MATRIX_SCALE_32] = IREE_SVL("scale32"),
    [LOOM_AMDGPU_MATRIX_SCALE_16] = IREE_SVL("scale16"),
};

static const loom_amdgpu_matrix_feature_info_t kAmdgpuMatrixFeatureInfos[] = {
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
        .name = IREE_SVL("mfma-gfx908"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A,
        .name = IREE_SVL("mfma-gfx908-gfx90a"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K,
        .name = IREE_SVL("mfma-gfx90a-bf16-1k"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64,
        .name = IREE_SVL("mfma-gfx90a-f64"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8,
        .name = IREE_SVL("mfma-gfx940-fp8"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950,
        .name = IREE_SVL("mfma-gfx950"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4,
        .name = IREE_SVL("mfma-gfx950-scale-f8f6f4"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940,
        .name = IREE_SVL("smfmac-gfx940"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX950,
        .name = IREE_SVL("smfmac-gfx950"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
        .name = IREE_SVL("wmma-gfx11"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12,
        .name = IREE_SVL("wmma-gfx12"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12,
        .name = IREE_SVL("swmmac-gfx12"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250,
        .name = IREE_SVL("wmma-gfx1250"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4,
        .name = IREE_SVL("wmma-gfx1250-scale-f8f6f4"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250,
        .name = IREE_SVL("swmmac-gfx1250"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_XF32,
        .name = IREE_SVL("mfma-gfx940-xf32"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940_FP8,
        .name = IREE_SVL("smfmac-gfx940-fp8"),
    },
    {
        .feature_bit = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_I8,
        .name = IREE_SVL("mfma-gfx940-i8"),
    },
};

iree_string_view_t loom_amdgpu_matrix_family_name(
    loom_amdgpu_matrix_family_t family) {
  if ((iree_host_size_t)family >= IREE_ARRAYSIZE(kAmdgpuMatrixFamilyNames)) {
    return kAmdgpuMatrixFamilyNames[LOOM_AMDGPU_MATRIX_FAMILY_UNKNOWN];
  }
  return kAmdgpuMatrixFamilyNames[family];
}

iree_string_view_t loom_amdgpu_matrix_numeric_type_name(
    loom_amdgpu_matrix_numeric_type_t numeric_type) {
  if ((iree_host_size_t)numeric_type >=
      IREE_ARRAYSIZE(kAmdgpuMatrixNumericTypeNames)) {
    return kAmdgpuMatrixNumericTypeNames[LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN];
  }
  return kAmdgpuMatrixNumericTypeNames[numeric_type];
}

iree_string_view_t loom_amdgpu_matrix_scale_kind_name(
    loom_amdgpu_matrix_scale_kind_t scale_kind) {
  if ((iree_host_size_t)scale_kind >=
      IREE_ARRAYSIZE(kAmdgpuMatrixScaleKindNames)) {
    return IREE_SV("unknown");
  }
  return kAmdgpuMatrixScaleKindNames[scale_kind];
}

bool loom_amdgpu_matrix_scale_format_selector_from_numeric_format(
    loom_value_fact_numeric_format_flags_t format, int64_t* out_value) {
  *out_value = 0;
  const loom_numeric_format_info_t* info = NULL;
  if (!loom_numeric_format_info(format, &info)) {
    return false;
  }
  switch (info->float_family) {
    case LOOM_NUMERIC_FLOAT_FAMILY_F8_E8M0:
      *out_value = LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_E8M0;
      return true;
    case LOOM_NUMERIC_FLOAT_FAMILY_FP8:
      *out_value = LOOM_AMDGPU_MATRIX_SCALE_FORMAT_SELECTOR_FP8_E4M3;
      return true;
    default:
      return false;
  }
}

const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_matrix_fragment_layout_for_kind(
    loom_amdgpu_matrix_fragment_layout_kind_t kind) {
  if (kind <= LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN ||
      kind >= LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT) {
    return NULL;
  }
  return &kLoomAmdgpuMatrixFragmentLayouts[kind];
}

const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_matrix_contract_descriptor_fragment_layout(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor) {
  if (descriptor == NULL) {
    return NULL;
  }
  return loom_amdgpu_matrix_fragment_layout_for_kind(
      descriptor->fragment_layout_kind);
}

bool loom_amdgpu_matrix_feature_bits_from_profile(
    loom_amdgpu_matrix_feature_profile_t profile,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits) {
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  if ((iree_host_size_t)profile <
      IREE_ARRAYSIZE(kLoomAmdgpuMatrixFeatureBitsByProfile)) {
    feature_bits = kLoomAmdgpuMatrixFeatureBitsByProfile[profile];
  }
  *out_feature_bits = feature_bits;
  return feature_bits != 0;
}

iree_host_size_t loom_amdgpu_matrix_feature_info_count(void) {
  return IREE_ARRAYSIZE(kAmdgpuMatrixFeatureInfos);
}

const loom_amdgpu_matrix_feature_info_t* loom_amdgpu_matrix_feature_info_at(
    iree_host_size_t index) {
  if (index >= IREE_ARRAYSIZE(kAmdgpuMatrixFeatureInfos)) {
    return NULL;
  }
  return &kAmdgpuMatrixFeatureInfos[index];
}

iree_status_t loom_amdgpu_matrix_feature_bits_from_processor(
    iree_string_view_t processor,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits) {
  processor = iree_string_view_trim(processor);
  const loom_amdgpu_processor_info_t* processor_info = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_processor(processor, &processor_info));
  if (loom_amdgpu_matrix_feature_bits_from_profile(
          processor_info->properties.features.matrix, out_feature_bits)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU processor '%.*s' has no matrix feature profile",
      (int)processor.size, processor.data);
}

iree_host_size_t loom_amdgpu_matrix_contract_descriptor_count(void) {
  return kLoomAmdgpuMatrixContractDescriptorCount;
}

const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_descriptor_at(iree_host_size_t index) {
  if (index >= kLoomAmdgpuMatrixContractDescriptorCount) {
    return NULL;
  }
  return &kLoomAmdgpuMatrixContractDescriptors[index];
}

const loom_amdgpu_matrix_contract_realization_choices_t*
loom_amdgpu_matrix_contract_realization_choices_at(iree_host_size_t index) {
  if (index >= kLoomAmdgpuMatrixContractDescriptorCount) {
    return NULL;
  }
  return &kLoomAmdgpuMatrixContractRealizationChoices[index];
}

const loom_amdgpu_matrix_result_representation_t*
loom_amdgpu_matrix_result_representation_at(
    loom_amdgpu_matrix_result_representation_id_t representation_id) {
  if (representation_id == LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE ||
      representation_id >= LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_COUNT) {
    return NULL;
  }
  return &kLoomAmdgpuMatrixResultRepresentations[representation_id];
}

const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_wait_state_descriptor_for_low_descriptor_ref(
    loom_amdgpu_descriptor_ref_t low_descriptor_ref) {
  if (low_descriptor_ref >= LOOM_AMDGPU_DESCRIPTOR_REF_COUNT) {
    return NULL;
  }
  const uint16_t* contract_ordinals =
      kLoomAmdgpuMatrixWaitStateContractOrdinalsByDescriptorRef;
  const uint16_t contract_ordinal = contract_ordinals[low_descriptor_ref];
  if (contract_ordinal == UINT16_MAX) {
    return NULL;
  }
  IREE_ASSERT_LT((iree_host_size_t)contract_ordinal,
                 kLoomAmdgpuMatrixContractDescriptorCount);
  return &kLoomAmdgpuMatrixContractDescriptors[contract_ordinal];
}

bool loom_amdgpu_matrix_contract_is_available(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size) {
  if (descriptor == NULL) {
    return false;
  }
  if ((feature_bits & descriptor->required_feature_bits) !=
      descriptor->required_feature_bits) {
    return false;
  }
  if (wave_size == 0) {
    return true;
  }
  loom_amdgpu_matrix_wave_size_bits_t wave_size_bits = 0;
  if (wave_size == 32) {
    wave_size_bits = LOOM_AMDGPU_MATRIX_WAVE_SIZE_32;
  } else if (wave_size == 64) {
    wave_size_bits = LOOM_AMDGPU_MATRIX_WAVE_SIZE_64;
  } else {
    return false;
  }
  return (descriptor->wave_size_bits & wave_size_bits) != 0;
}

static bool loom_amdgpu_matrix_contract_family_matches(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_amdgpu_matrix_contract_match_request_t* request) {
  return request->family == LOOM_AMDGPU_MATRIX_FAMILY_UNKNOWN ||
         descriptor->family == request->family;
}

static bool loom_amdgpu_matrix_contract_tile_shape_matches(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_amdgpu_matrix_contract_match_request_t* request) {
  return descriptor->tile_shape.block_count ==
             request->tile_shape.block_count &&
         descriptor->tile_shape.result_row_count ==
             request->tile_shape.result_row_count &&
         descriptor->tile_shape.result_column_count ==
             request->tile_shape.result_column_count &&
         descriptor->tile_shape.reduction_count ==
             request->tile_shape.reduction_count;
}

static bool loom_amdgpu_matrix_contract_numeric_type_matches(
    loom_amdgpu_matrix_numeric_type_t descriptor_numeric_type,
    loom_amdgpu_matrix_numeric_type_t request_numeric_type,
    loom_amdgpu_matrix_contract_flags_t descriptor_flags) {
  if (descriptor_numeric_type == request_numeric_type) {
    return true;
  }
  if (descriptor_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_F8) {
    return request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP8 ||
           request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_BF8;
  }
  if (descriptor_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_F6) {
    return request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP6 ||
           request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_BF6;
  }
  if (descriptor_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4) {
    return request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP8 ||
           request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_BF8 ||
           request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP6 ||
           request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_BF6 ||
           request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP4;
  }
  if (!iree_any_bit_set(descriptor_flags,
                        LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT)) {
    return false;
  }
  return (descriptor_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_IU8 &&
          request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_I8) ||
         (descriptor_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_IU4 &&
          request_numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_I4);
}

static bool loom_amdgpu_matrix_contract_payload_matches(
    loom_amdgpu_matrix_payload_shape_t descriptor_payload,
    loom_amdgpu_matrix_payload_shape_t request_payload,
    loom_amdgpu_matrix_contract_flags_t descriptor_flags) {
  if (request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN) {
    return false;
  }
  if (!loom_amdgpu_matrix_contract_numeric_type_matches(
          descriptor_payload.numeric_type, request_payload.numeric_type,
          descriptor_flags)) {
    return false;
  }
  if (request_payload.register_count != 0 &&
      descriptor_payload.register_count != 0 &&
      descriptor_payload.register_count != request_payload.register_count) {
    return false;
  }
  if (request_payload.element_count != 0 &&
      descriptor_payload.element_count != 0 &&
      descriptor_payload.element_count != request_payload.element_count) {
    return false;
  }
  return true;
}

static loom_amdgpu_matrix_contract_rejection_bits_t
loom_amdgpu_matrix_contract_payload_rejection_bits(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_amdgpu_matrix_contract_match_request_t* request) {
  loom_amdgpu_matrix_contract_rejection_bits_t rejection_bits =
      LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  if (!loom_amdgpu_matrix_contract_payload_matches(
          descriptor->lhs_payload, request->lhs_payload, descriptor->flags)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_LHS_PAYLOAD;
  }
  if (!loom_amdgpu_matrix_contract_payload_matches(
          descriptor->rhs_payload, request->rhs_payload, descriptor->flags)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RHS_PAYLOAD;
  }
  if (!loom_amdgpu_matrix_contract_payload_matches(
          descriptor->accumulator_payload, request->accumulator_payload, 0)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_ACCUMULATOR_PAYLOAD;
  }
  if (!loom_amdgpu_matrix_contract_payload_matches(
          descriptor->result_payload, request->result_payload, 0)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RESULT_PAYLOAD;
  }
  return rejection_bits;
}

static loom_amdgpu_matrix_contract_rejection_bits_t
loom_amdgpu_matrix_contract_missing_flag_rejection_bits(
    loom_amdgpu_matrix_contract_flags_t missing_flags) {
  loom_amdgpu_matrix_contract_rejection_bits_t rejection_bits =
      LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SPARSE;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS) != 0) {
    rejection_bits |=
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_REUSE;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_CLAMP;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SIGN_SELECT;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS) != 0) {
    rejection_bits |=
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_AB_MODIFIERS;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_C_MODIFIER;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_OPSEL) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_OPSEL;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS) != 0) {
    rejection_bits |=
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE_FORMATS;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK) !=
      0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_REQUIRED_FLAGS;
  }
  return rejection_bits;
}

static loom_amdgpu_matrix_contract_rejection_bits_t
loom_amdgpu_matrix_contract_flag_rejection_bits(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_amdgpu_matrix_contract_match_request_t* request) {
  const loom_amdgpu_matrix_contract_flags_t required_abi_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_OPSEL |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS;
  const loom_amdgpu_matrix_contract_flags_t missing_available_flags =
      descriptor->flags & required_abi_flags & ~request->available_flags;
  loom_amdgpu_matrix_contract_rejection_bits_t rejection_bits =
      loom_amdgpu_matrix_contract_missing_flag_rejection_bits(
          missing_available_flags);
  const loom_amdgpu_matrix_contract_flags_t missing_required_flags =
      request->required_flags & ~descriptor->flags;
  if (missing_required_flags != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_REQUIRED_FLAGS;
    rejection_bits |= loom_amdgpu_matrix_contract_missing_flag_rejection_bits(
        missing_required_flags);
  }
  return rejection_bits;
}

static bool loom_amdgpu_matrix_contract_scale_format_matches(
    loom_amdgpu_matrix_scale_format_selector_bits_t request_bits,
    loom_amdgpu_matrix_scale_format_selector_bits_t descriptor_bits) {
  return request_bits == 0 || descriptor_bits == 0 ||
         iree_all_bits_set(descriptor_bits, request_bits);
}

static loom_amdgpu_matrix_contract_rejection_bits_t
loom_amdgpu_matrix_contract_scale_format_rejection_bits(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_amdgpu_matrix_contract_match_request_t* request) {
  if (iree_any_bit_set(descriptor->flags,
                       LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS)) {
    return LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  }
  if (loom_amdgpu_matrix_contract_scale_format_matches(
          request->lhs_scale_format_selector_bits,
          descriptor->implicit_scale_format_selector_bits) &&
      loom_amdgpu_matrix_contract_scale_format_matches(
          request->rhs_scale_format_selector_bits,
          descriptor->implicit_scale_format_selector_bits)) {
    return LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  }
  return LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_SCALE_FORMAT;
}

const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_select(
    const loom_amdgpu_matrix_contract_match_request_t* request,
    uint16_t* out_descriptor_ordinal,
    loom_amdgpu_matrix_contract_match_diagnostic_t* out_diagnostic) {
  if (out_descriptor_ordinal != NULL) {
    *out_descriptor_ordinal = LOOM_AMDGPU_MATRIX_CONTRACT_ORDINAL_NONE;
  }
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {
      .descriptor_count = kLoomAmdgpuMatrixContractDescriptorCount,
  };
  if (request == NULL) {
    diagnostic.rejection_bits =
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_INVALID_REQUEST;
    if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
    return NULL;
  }

  loom_amdgpu_matrix_contract_rejection_bits_t payload_rejection_bits =
      LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  loom_amdgpu_matrix_contract_rejection_bits_t flag_rejection_bits =
      LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  loom_amdgpu_matrix_contract_rejection_bits_t scale_format_rejection_bits =
      LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  for (iree_host_size_t i = 0; i < kLoomAmdgpuMatrixContractDescriptorCount;
       ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        &kLoomAmdgpuMatrixContractDescriptors[i];
    if (!loom_amdgpu_matrix_contract_family_matches(descriptor, request)) {
      continue;
    }
    ++diagnostic.family_candidate_count;

    if (!loom_amdgpu_matrix_contract_tile_shape_matches(descriptor, request)) {
      continue;
    }
    ++diagnostic.shape_candidate_count;

    const loom_amdgpu_matrix_contract_rejection_bits_t payload_rejection =
        loom_amdgpu_matrix_contract_payload_rejection_bits(descriptor, request);
    if (payload_rejection != LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE) {
      payload_rejection_bits |= payload_rejection;
      continue;
    }
    ++diagnostic.payload_candidate_count;

    if (descriptor->scale_kind != request->scale_kind) {
      continue;
    }
    ++diagnostic.scale_candidate_count;

    const loom_amdgpu_matrix_contract_rejection_bits_t flag_rejection =
        loom_amdgpu_matrix_contract_flag_rejection_bits(descriptor, request);
    if (flag_rejection != LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE) {
      flag_rejection_bits |= flag_rejection;
      continue;
    }
    ++diagnostic.flag_candidate_count;

    const loom_amdgpu_matrix_contract_rejection_bits_t scale_format_rejection =
        loom_amdgpu_matrix_contract_scale_format_rejection_bits(descriptor,
                                                                request);
    if (scale_format_rejection != LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE) {
      scale_format_rejection_bits |= scale_format_rejection;
      continue;
    }
    ++diagnostic.scale_format_candidate_count;

    if ((request->feature_bits & descriptor->required_feature_bits) !=
        descriptor->required_feature_bits) {
      continue;
    }
    ++diagnostic.feature_candidate_count;

    if (!loom_amdgpu_matrix_contract_is_available(
            descriptor, request->feature_bits, request->wave_size)) {
      continue;
    }
    ++diagnostic.wave_candidate_count;

    if (out_descriptor_ordinal != NULL) {
      IREE_ASSERT_LE(i, UINT16_MAX);
      *out_descriptor_ordinal = (uint16_t)i;
    }
    if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
    return descriptor;
  }

  if (diagnostic.family_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FAMILY;
  } else if (diagnostic.shape_candidate_count == 0) {
    diagnostic.rejection_bits =
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_TILE_SHAPE;
  } else if (diagnostic.payload_candidate_count == 0) {
    diagnostic.rejection_bits = payload_rejection_bits;
  } else if (diagnostic.scale_candidate_count == 0) {
    diagnostic.rejection_bits =
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_SCALE_KIND;
  } else if (diagnostic.flag_candidate_count == 0) {
    diagnostic.rejection_bits = flag_rejection_bits;
  } else if (diagnostic.scale_format_candidate_count == 0) {
    diagnostic.rejection_bits = scale_format_rejection_bits;
  } else if (diagnostic.feature_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FEATURES;
  } else if (diagnostic.wave_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_WAVE_SIZE;
  }
  if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
  return NULL;
}

#undef MATRIX_DESCRIPTOR
#undef MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32
#undef MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X4_F32
#undef MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16
#undef MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_W64
#undef MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_HALF_16X16X16
#undef MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_HALF_16X16X16_W64
#undef MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_F32
#undef MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_HALF_16X16X16
#undef MATRIX_FRAGMENT_LAYOUT_RDNA4_WMMA_HALF_16X16X32
#undef MATRIX_PAYLOAD
#undef MATRIX_TILE_SHAPE
#undef MFMA_GFX940_FP8_FEATURES
#undef MFMA_GFX950_FEATURES
#undef MFMA_GFX950_SCALE_FEATURES
#undef SMFMAC_GFX940_FEATURES
#undef SMFMAC_GFX950_FEATURES
#undef WMMA_GFX12_FEATURES
#undef WMMA_GFX1250_FEATURES
#undef WMMA_GFX1250_SCALE_FEATURES
#undef SWMMAC_GFX12_FEATURES
#undef SWMMAC_GFX1250_FEATURES
#undef SWMMAC_GFX12_IU_FLAGS
#undef WMMA_GFX1250_MODS_ALL_FLAGS
#undef WMMA_GFX1250_MODS_C_FLAGS
#undef WMMA_GFX1250_SCALE_F4_FLAGS
#undef SWMMAC_GFX1250_AB_FLAGS
#undef SWMMAC_GFX1250_IU8_FLAGS
