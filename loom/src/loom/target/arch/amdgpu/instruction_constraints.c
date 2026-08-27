// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/instruction_constraints.h"

#include "loom/target/arch/amdgpu/refs/target_refs.h"

static const loom_amdgpu_instruction_constraint_info_t kDsPairConstraint = {
    .constraint =
        LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT,
    .kind = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_OPERAND_ENCODING,
    .resolution = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPLIT_DS_PAIR,
    .constraint_key = IREE_SVL("amdgpu.ds.paired_address_alignment"),
};

static const loom_amdgpu_instruction_constraint_info_t kDsAddtidAddressConstraint = {
    .constraint =
        LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION,
    .kind = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_OPERAND_ENCODING,
    .resolution =
        LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_MATERIALIZE_ADDTID_ADDRESS,
    .constraint_key = IREE_SVL("amdgpu.ds.addtid_address_materialization"),
};

static const loom_amdgpu_instruction_constraint_info_t
    kClusterMulticastConstraint = {
        .constraint =
            LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION,
        .kind = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_SURROUNDING_SEQUENCE,
        .resolution =
            LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_PRESERVE_CLUSTER_MASK,
        .constraint_key =
            IREE_SVL("amdgpu.cluster.multicast_mask_preservation"),
};

static const loom_amdgpu_instruction_constraint_info_t kTensorMulticastConstraint = {
    .constraint =
        LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION,
    .kind = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_SURROUNDING_SEQUENCE,
    .resolution =
        LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_PRESERVE_TENSOR_MULTICAST_MASK,
    .constraint_key = IREE_SVL("amdgpu.tensor.multicast_mask_preservation"),
};

static const loom_amdgpu_instruction_constraint_info_t
    kWmmaK64ScalePrefixConstraint = {
        .constraint =
            LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX,
        .kind = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_SURROUNDING_SEQUENCE,
        .resolution =
            LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_PREFIX_NEUTRAL_REGULAR_SCALE,
        .constraint_key = IREE_SVL("amdgpu.wmma.fp8_bf8_k64_scale_prefix"),
};

static const loom_amdgpu_instruction_constraint_info_t kWmmaK128Constraint = {
    .constraint = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT,
    .kind = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_INSTRUCTION_FORM,
    .resolution = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPLIT_WMMA_K128,
    .constraint_key = IREE_SVL("amdgpu.wmma.fp8_bf8_k128_split"),
};

static const loom_amdgpu_instruction_constraint_info_t kWmma32x16F4Constraint =
    {
        .constraint = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT,
        .kind = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_INSTRUCTION_FORM,
        .resolution =
            LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPLIT_WMMA_32X16,
        .constraint_key = IREE_SVL("amdgpu.wmma.f4_32x16_split"),
};

static const loom_amdgpu_instruction_constraint_info_t
    kWmmaScaleEncodingConstraint = {
        .constraint = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING,
        .kind = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_OPERAND_ENCODING,
        .resolution =
            LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_LEGALIZE_SCALED_WMMA,
        .constraint_key = IREE_SVL("amdgpu.wmma.scale_encoding"),
};

static const loom_amdgpu_instruction_constraint_info_t
    kLowPrecisionSwmmacConstraint = {
        .constraint =
            LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING,
        .kind = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_INSTRUCTION_FORM,
        .resolution =
            LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_LOWER_LOW_PRECISION_SWMMAC,
        .constraint_key = IREE_SVL("amdgpu.swmmac.low_precision_lowering"),
};

static const loom_amdgpu_instruction_constraint_info_t
    kIntegerMatrixSpacingConstraint = {
        .constraint =
            LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING,
        .kind = LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_COEXECUTION_HAZARD,
        .resolution =
            LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPACE_INTEGER_MATRIX,
        .constraint_key = IREE_SVL("amdgpu.matrix.integer_coexecution_spacing"),
};

loom_amdgpu_instruction_constraint_bits_t
loom_amdgpu_instruction_constraints_for_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      loom_amdgpu_descriptor_ref_for_descriptor(descriptor_set, descriptor);
  switch (descriptor_ref) {
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ2_B32:
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ2_B64:
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ2ST64_B32:
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ2ST64_B64:
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE2_B32:
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE2_B64:
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE2ST64_B32:
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE2ST64_B64:
      return LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT;

    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_ADDTID_B32:
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_ADDTID_B32:
      return LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION;

    case LOOM_AMDGPU_DESCRIPTOR_REF_CLUSTER_LOAD_ASYNC_TO_LDS_B8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_CLUSTER_LOAD_ASYNC_TO_LDS_B32:
    case LOOM_AMDGPU_DESCRIPTOR_REF_CLUSTER_LOAD_ASYNC_TO_LDS_B64:
    case LOOM_AMDGPU_DESCRIPTOR_REF_CLUSTER_LOAD_ASYNC_TO_LDS_B128:
      return LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION;

    case LOOM_AMDGPU_DESCRIPTOR_REF_TENSOR_LOAD_TO_LDS_D2:
    case LOOM_AMDGPU_DESCRIPTOR_REF_TENSOR_LOAD_TO_LDS_D4:
      return LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X64_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X64_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X64_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X64_FP8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X64_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X64_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X64_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X64_FP8_FP8:
      return LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X128_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X128_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X128_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X128_FP8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X128_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X128_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X128_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X128_FP8_FP8:
      return LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_32X16X128_F4:
      return LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE16_F32_16X16X128_F8F6F4_F8_F8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE16_F32_32X16X128_F4:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE_F32_16X16X128_F8F6F4_F8_F8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE_F32_32X16X128_F4:
      return LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F16_16X16X128_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F16_16X16X128_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F16_16X16X128_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F16_16X16X128_FP8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X32_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X32_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X32_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X32_FP8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X128_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X128_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X128_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X128_FP8_FP8:
      return LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_I32_16X16X128_IU8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_I32_16X16X32_IU4:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_I32_16X16X32_IU8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_I32_16X16X64_IU4:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4_ACC_ZERO:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4_W64:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4_W64_ACC_ZERO:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8_ACC_ZERO:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8_W64:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8_W64_ACC_ZERO:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X32_IU4:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X32_IU4_ACC_ZERO:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X64_IU8:
      return LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING;

    default:
      return 0;
  }
}

loom_amdgpu_instruction_constraint_info_t
loom_amdgpu_instruction_constraint_info(
    loom_amdgpu_instruction_constraint_bit_t constraint) {
  switch (constraint) {
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT:
      return kDsPairConstraint;
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION:
      return kDsAddtidAddressConstraint;
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION:
      return kClusterMulticastConstraint;
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION:
      return kTensorMulticastConstraint;
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX:
      return kWmmaK64ScalePrefixConstraint;
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT:
      return kWmmaK128Constraint;
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT:
      return kWmma32x16F4Constraint;
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING:
      return kWmmaScaleEncodingConstraint;
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING:
      return kLowPrecisionSwmmacConstraint;
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING:
      return kIntegerMatrixSpacingConstraint;
    default:
      IREE_CHECK_UNREACHABLE("unknown AMDGPU instruction constraint");
      return (loom_amdgpu_instruction_constraint_info_t){0};
  }
}

iree_string_view_t loom_amdgpu_instruction_constraint_kind_name(
    loom_amdgpu_instruction_constraint_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_OPERAND_ENCODING:
      return IREE_SV("operand_encoding");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_INSTRUCTION_FORM:
      return IREE_SV("instruction_form");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_SURROUNDING_SEQUENCE:
      return IREE_SV("surrounding_sequence");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_COEXECUTION_HAZARD:
      return IREE_SV("coexecution_hazard");
    default:
      IREE_CHECK_UNREACHABLE("unknown AMDGPU instruction constraint kind");
      return iree_string_view_empty();
  }
}

iree_string_view_t loom_amdgpu_instruction_constraint_resolution_name(
    loom_amdgpu_instruction_constraint_resolution_t resolution) {
  switch (resolution) {
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPLIT_DS_PAIR:
      return IREE_SV("split_ds2_to_single_address");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_MATERIALIZE_ADDTID_ADDRESS:
      return IREE_SV("materialize_addtid_address");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_PRESERVE_CLUSTER_MASK:
      return IREE_SV("clear_and_restore_cluster_mask");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_PRESERVE_TENSOR_MULTICAST_MASK:
      return IREE_SV("clear_and_restore_tensor_multicast_mask");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_PREFIX_NEUTRAL_REGULAR_SCALE:
      return IREE_SV("prefix_wmma_k64_with_neutral_regular_scale");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPLIT_WMMA_K128:
      return IREE_SV("split_wmma_k128_to_regular_scale_k64");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPLIT_WMMA_32X16:
      return IREE_SV("split_wmma_32x16_to_16x16");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_LEGALIZE_SCALED_WMMA:
      return IREE_SV("legalize_scaled_wmma");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_LOWER_LOW_PRECISION_SWMMAC:
      return IREE_SV("lower_low_precision_swmmac");
    case LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPACE_INTEGER_MATRIX:
      return IREE_SV("enforce_integer_matrix_spacing");
    default:
      IREE_CHECK_UNREACHABLE(
          "unknown AMDGPU instruction constraint resolution");
      return iree_string_view_empty();
  }
}
