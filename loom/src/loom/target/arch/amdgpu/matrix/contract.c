// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix/contract.h"

#include "loom/target/arch/amdgpu/matrix/contract_tables.h"
#include "loom/target/arch/amdgpu/target_info.h"

#define MATRIX_TILE_SHAPE(row_count_value, column_count_value, \
                          reduction_count_value)               \
  (loom_amdgpu_matrix_tile_shape_t) {                          \
    .result_row_count = (row_count_value),                     \
    .result_column_count = (column_count_value),               \
    .reduction_count = (reduction_count_value),                \
  }

#define MATRIX_ROLE_LAYOUT(role_value, map_kind_value, register_count_value, \
                           elements_per_register_value,                      \
                           element_bit_count_value, coordinate_flags_value)  \
  (loom_amdgpu_matrix_fragment_role_layout_t) {                              \
    .role = (role_value), .map_kind = (map_kind_value),                      \
    .register_count = (register_count_value),                                \
    .elements_per_register = (elements_per_register_value),                  \
    .element_bit_count = (element_bit_count_value),                          \
    .coordinate_flags = (coordinate_flags_value),                            \
  }

#define MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16(kind_value,           \
                                                         name_value)           \
  {                                                                            \
      .kind = (kind_value),                                                    \
      .name = IREE_SVL(name_value),                                            \
      .wave_size = 32,                                                         \
      .tile_shape = MATRIX_TILE_SHAPE(16, 16, 16),                             \
      .lhs = MATRIX_ROLE_LAYOUT(                                               \
          LOOM_CONTRACT_OPERAND_ROLE_LHS,                                      \
          LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION, 8, 2, 16,    \
          LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |                                \
              LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION),                      \
      .rhs = MATRIX_ROLE_LAYOUT(                                               \
          LOOM_CONTRACT_OPERAND_ROLE_RHS,                                      \
          LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION, 8, 2, 16, \
          LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN |                             \
              LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION),                      \
      .accumulator = MATRIX_ROLE_LAYOUT(                                       \
          LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,                              \
          LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN, 8, 1, 32,  \
          LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |                                \
              LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN),                         \
      .result = MATRIX_ROLE_LAYOUT(                                            \
          LOOM_CONTRACT_OPERAND_ROLE_RESULT,                                   \
          LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN, 8, 1, 32,  \
          LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |                                \
              LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN),                         \
  }

#define MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16(kind_value, name_value)   \
  {                                                                             \
      .kind = (kind_value),                                                     \
      .name = IREE_SVL(name_value),                                             \
      .wave_size = 64,                                                          \
      .tile_shape = MATRIX_TILE_SHAPE(16, 16, 16),                              \
      .lhs = MATRIX_ROLE_LAYOUT(                                                \
          LOOM_CONTRACT_OPERAND_ROLE_LHS,                                       \
          LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION,    \
          2, 2, 16,                                                             \
          LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |                                 \
              LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION),                       \
      .rhs = MATRIX_ROLE_LAYOUT(                                                \
          LOOM_CONTRACT_OPERAND_ROLE_RHS,                                       \
          LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION, \
          2, 2, 16,                                                             \
          LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN |                              \
              LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION),                       \
      .accumulator = MATRIX_ROLE_LAYOUT(                                        \
          LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,                               \
          LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN, 4, 1, 32,    \
          LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |                                 \
              LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN),                          \
      .result = MATRIX_ROLE_LAYOUT(                                             \
          LOOM_CONTRACT_OPERAND_ROLE_RESULT,                                    \
          LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN, 4, 1, 32,    \
          LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |                                 \
              LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN),                          \
  }

#define MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X4_F32(kind_value,            \
                                                         name_value)            \
  {                                                                             \
      .kind = (kind_value),                                                     \
      .name = IREE_SVL(name_value),                                             \
      .wave_size = 64,                                                          \
      .tile_shape = MATRIX_TILE_SHAPE(16, 16, 4),                               \
      .lhs = MATRIX_ROLE_LAYOUT(                                                \
          LOOM_CONTRACT_OPERAND_ROLE_LHS,                                       \
          LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION,    \
          1, 1, 32,                                                             \
          LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |                                 \
              LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION),                       \
      .rhs = MATRIX_ROLE_LAYOUT(                                                \
          LOOM_CONTRACT_OPERAND_ROLE_RHS,                                       \
          LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION, \
          1, 1, 32,                                                             \
          LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN |                              \
              LOOM_MATRIX_FRAGMENT_COORDINATE_REDUCTION),                       \
      .accumulator = MATRIX_ROLE_LAYOUT(                                        \
          LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,                               \
          LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN, 4, 1, 32,    \
          LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |                                 \
              LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN),                          \
      .result = MATRIX_ROLE_LAYOUT(                                             \
          LOOM_CONTRACT_OPERAND_ROLE_RESULT,                                    \
          LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN, 4, 1, 32,    \
          LOOM_MATRIX_FRAGMENT_COORDINATE_ROW |                                 \
              LOOM_MATRIX_FRAGMENT_COORDINATE_COLUMN),                          \
  }

static const loom_amdgpu_matrix_fragment_layout_t kMatrixFragmentLayouts[] = {
    [LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16] =
        MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16(
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16,
            "rdna3.wmmar3.f32.16x16x16.f16"),
    [LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_BF16] =
        MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16(
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_BF16,
            "rdna3.wmmar3.f32.16x16x16.bf16"),
    [LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_F16] =
        MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16(
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_F16,
            "cdna.mfma.f32.16x16x16.f16"),
    [LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_BF16] =
        MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16(
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16_BF16,
            "cdna.mfma.f32.16x16x16.bf16"),
    [LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X4_F32] =
        MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X4_F32(
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X4_F32,
            "cdna.mfma.f32.16x16x4.f32"),
};

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
    [LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4] = IREE_SVL("f8f6f4"),
};

static const iree_string_view_t kAmdgpuMatrixScaleKindNames[] = {
    [LOOM_AMDGPU_MATRIX_SCALE_NONE] = IREE_SVL("none"),
    [LOOM_AMDGPU_MATRIX_SCALE_32] = IREE_SVL("scale32"),
    [LOOM_AMDGPU_MATRIX_SCALE_16] = IREE_SVL("scale16"),
};

static const loom_amdgpu_matrix_feature_bits_t
    kAmdgpuMatrixFeatureBitsByProfile[] = {
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE] = 0,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908] =
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A] =
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908_GFX90A |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940] =
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64 |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8 |
            LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950] =
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64 |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8 |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950 |
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4 |
            LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940 |
            LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX950,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11] =
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12] =
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 |
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12 |
            LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250] =
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 |
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12 |
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250 |
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4 |
            LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12 |
            LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250,
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

const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_matrix_fragment_layout_for_kind(
    loom_amdgpu_matrix_fragment_layout_kind_t kind) {
  if (kind <= LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN ||
      (iree_host_size_t)kind >= IREE_ARRAYSIZE(kMatrixFragmentLayouts)) {
    return NULL;
  }
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      &kMatrixFragmentLayouts[kind];
  return layout->kind == kind ? layout : NULL;
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

const loom_amdgpu_matrix_fragment_role_layout_t*
loom_amdgpu_matrix_fragment_role_layout(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role) {
  return loom_matrix_fragment_role_layout(layout, role);
}

bool loom_amdgpu_matrix_fragment_coordinate(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, uint16_t lane, uint16_t register_index,
    uint16_t element_index,
    loom_amdgpu_matrix_fragment_coordinate_t* out_coordinate) {
  return loom_matrix_fragment_coordinate(layout, role, lane, register_index,
                                         element_index, out_coordinate);
}

bool loom_amdgpu_matrix_feature_bits_from_profile(
    loom_amdgpu_matrix_feature_profile_t profile,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits) {
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  if ((iree_host_size_t)profile <
      IREE_ARRAYSIZE(kAmdgpuMatrixFeatureBitsByProfile)) {
    feature_bits = kAmdgpuMatrixFeatureBitsByProfile[profile];
  }
  *out_feature_bits = feature_bits;
  return feature_bits != 0;
}

iree_status_t loom_amdgpu_matrix_feature_bits_from_processor(
    iree_string_view_t processor,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits) {
  processor = iree_string_view_trim(processor);
  const loom_amdgpu_processor_info_t* processor_info = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_processor(processor, &processor_info));
  if (loom_amdgpu_matrix_feature_bits_from_profile(
          processor_info->features.matrix, out_feature_bits)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
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
  return descriptor->tile_shape.result_row_count ==
             request->tile_shape.result_row_count &&
         descriptor->tile_shape.result_column_count ==
             request->tile_shape.result_column_count &&
         descriptor->tile_shape.reduction_count ==
             request->tile_shape.reduction_count;
}

static bool loom_amdgpu_matrix_contract_payload_matches(
    loom_amdgpu_matrix_payload_shape_t descriptor_payload,
    loom_amdgpu_matrix_payload_shape_t request_payload) {
  if (request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN) {
    return false;
  }
  bool numeric_type_matches =
      descriptor_payload.numeric_type == request_payload.numeric_type;
  if (descriptor_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4) {
    // Selector-family descriptors accept exact low-bit requests when the
    // request also proves that matrix-format selector operands are available.
    numeric_type_matches =
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP8 ||
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_BF8 ||
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP6 ||
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_BF6 ||
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP4 ||
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4;
  }
  if (!numeric_type_matches) {
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
  if (!loom_amdgpu_matrix_contract_payload_matches(descriptor->lhs_payload,
                                                   request->lhs_payload)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_LHS_PAYLOAD;
  }
  if (!loom_amdgpu_matrix_contract_payload_matches(descriptor->rhs_payload,
                                                   request->rhs_payload)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RHS_PAYLOAD;
  }
  if (!loom_amdgpu_matrix_contract_payload_matches(
          descriptor->accumulator_payload, request->accumulator_payload)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_ACCUMULATOR_PAYLOAD;
  }
  if (!loom_amdgpu_matrix_contract_payload_matches(descriptor->result_payload,
                                                   request->result_payload)) {
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

const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_select(
    const loom_amdgpu_matrix_contract_match_request_t* request,
    loom_amdgpu_matrix_contract_match_diagnostic_t* out_diagnostic) {
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
  } else if (diagnostic.feature_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FEATURES;
  } else if (diagnostic.wave_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_WAVE_SIZE;
  }
  if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
  return NULL;
}

#undef MATRIX_DESCRIPTOR
#undef MATRIX_FRAGMENT_LAYOUT_CDNA_MFMA_F32_16X16X16
#undef MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16
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
