// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/gfx1250_a0_errata.h"

#include "loom/target/arch/amdgpu/refs/target_refs.h"

static const loom_amdgpu_gfx1250_a0_erratum_t kDs2Erratum = {
    .erratum_key = IREE_SVL("gfx1250.a0.ds2_alignment"),
    .legalization_key = IREE_SVL("split_ds2_to_single_address"),
};

static const loom_amdgpu_gfx1250_a0_erratum_t kDsAddtidErratum = {
    .erratum_key = IREE_SVL("gfx1250.a0.ds_addtid_address"),
    .legalization_key = IREE_SVL("materialize_addtid_address"),
};

static const loom_amdgpu_gfx1250_a0_erratum_t kClusterMulticastErratum = {
    .erratum_key = IREE_SVL("gfx1250.a0.cluster_multicast"),
    .legalization_key = IREE_SVL("clear_and_restore_cluster_mask"),
};

static const loom_amdgpu_gfx1250_a0_erratum_t kTensorMulticastErratum = {
    .erratum_key = IREE_SVL("gfx1250.a0.tensor_multicast"),
    .legalization_key = IREE_SVL("clear_and_restore_tensor_multicast_mask"),
};

static const loom_amdgpu_gfx1250_a0_erratum_t kWmmaK64Erratum = {
    .erratum_key = IREE_SVL("gfx1250.a0.wmma_k64_fp8_bf8"),
    .legalization_key = IREE_SVL("prefix_wmma_k64_with_neutral_regular_scale"),
};

static const loom_amdgpu_gfx1250_a0_erratum_t kWmmaK128Erratum = {
    .erratum_key = IREE_SVL("gfx1250.a0.wmma_k128_fp8_bf8"),
    .legalization_key = IREE_SVL("split_wmma_k128_to_regular_scale_k64"),
};

static const loom_amdgpu_gfx1250_a0_erratum_t kWmma32x16F4Erratum = {
    .erratum_key = IREE_SVL("gfx1250.a0.wmma_32x16_f4"),
    .legalization_key = IREE_SVL("split_wmma_32x16_to_16x16"),
};

static const loom_amdgpu_gfx1250_a0_erratum_t kWmmaScaleErratum = {
    .erratum_key = IREE_SVL("gfx1250.a0.wmma_scale_encoding"),
    .legalization_key = IREE_SVL("legalize_scaled_wmma"),
};

static const loom_amdgpu_gfx1250_a0_erratum_t kLowPrecisionSwmmacErratum = {
    .erratum_key = IREE_SVL("gfx1250.a0.swmmac_fp8_bf8"),
    .legalization_key = IREE_SVL("lower_low_precision_swmmac_for_a0"),
};

static const loom_amdgpu_gfx1250_a0_erratum_t kIntegerMatrixSpacingErratum = {
    .erratum_key = IREE_SVL("gfx1250.a0.integer_matrix_coexecution"),
    .legalization_key = IREE_SVL("enforce_integer_matrix_spacing"),
};

const loom_amdgpu_gfx1250_a0_erratum_t*
loom_amdgpu_gfx1250_a0_erratum_for_descriptor(
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
      return &kDs2Erratum;

    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_ADDTID_B32:
    case LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_ADDTID_B32:
      return &kDsAddtidErratum;

    case LOOM_AMDGPU_DESCRIPTOR_REF_CLUSTER_LOAD_ASYNC_TO_LDS_B8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_CLUSTER_LOAD_ASYNC_TO_LDS_B32:
    case LOOM_AMDGPU_DESCRIPTOR_REF_CLUSTER_LOAD_ASYNC_TO_LDS_B64:
    case LOOM_AMDGPU_DESCRIPTOR_REF_CLUSTER_LOAD_ASYNC_TO_LDS_B128:
      return &kClusterMulticastErratum;

    case LOOM_AMDGPU_DESCRIPTOR_REF_TENSOR_LOAD_TO_LDS_D2:
    case LOOM_AMDGPU_DESCRIPTOR_REF_TENSOR_LOAD_TO_LDS_D4:
      return &kTensorMulticastErratum;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X64_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X64_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X64_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X64_FP8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X64_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X64_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X64_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X64_FP8_FP8:
      return &kWmmaK64Erratum;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X128_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X128_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X128_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X128_FP8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X128_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X128_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X128_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X128_FP8_FP8:
      return &kWmmaK128Erratum;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_32X16X128_F4:
      return &kWmma32x16F4Erratum;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE16_F32_16X16X128_F8F6F4_F8_F8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE16_F32_32X16X128_F4:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE_F32_16X16X128_F8F6F4_F8_F8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_SCALE_F32_32X16X128_F4:
      return &kWmmaScaleErratum;

    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F16_16X16X128_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F16_16X16X128_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F16_16X16X128_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F16_16X16X128_FP8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X128_BF8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X128_BF8_FP8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X128_FP8_BF8:
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_SWMMAC_F32_16X16X128_FP8_FP8:
      return &kLowPrecisionSwmmacErratum;

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
      return &kIntegerMatrixSpacingErratum;

    default:
      return NULL;
  }
}
