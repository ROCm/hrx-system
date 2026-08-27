// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/matrix_wait_states.h"

#define LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(profile_, use_, pass_, cycles_) \
  {                                                                        \
      .profile = profile_,                                                 \
      .use = use_,                                                         \
      .pass_count = pass_,                                                 \
      .cycle_count = cycles_,                                              \
  }

static const loom_amdgpu_matrix_wait_result_row_t
    kAmdgpuMatrixWaitResultRows[] = {
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 2, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 4, 7),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 8, 11),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 16, 19),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 32, 35),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 2, 2),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 4, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 8, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 16, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 32, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 2, 3),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 4, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 8, 9),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 16, 17),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 32, 33),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 2, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 4, 7),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 8, 11),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 16, 19),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 32, 35),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 2, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 4, 8),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 8, 12),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 16, 20),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, 32, 36),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 2, 2),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 4, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 8, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 16, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, 32, 0),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 2, 3),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 4, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 8, 9),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 16, 17),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, 32, 33),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 2, 5),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 4, 8),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 8, 12),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 16, 20),
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW(
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
            LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB, 32, 36),
};

#undef LOOM_AMDGPU_MATRIX_WAIT_RESULT_ROW

enum {
  LOOM_AMDGPU_MATRIX_WAIT_PASS_2_INDEX = 0,
  LOOM_AMDGPU_MATRIX_WAIT_PASS_4_INDEX,
  LOOM_AMDGPU_MATRIX_WAIT_PASS_8_INDEX,
  LOOM_AMDGPU_MATRIX_WAIT_PASS_16_INDEX,
  LOOM_AMDGPU_MATRIX_WAIT_PASS_32_INDEX,
  LOOM_AMDGPU_MATRIX_WAIT_PASS_COUNT_,
};

static_assert(IREE_ARRAYSIZE(kAmdgpuMatrixWaitResultRows) ==
                  (LOOM_AMDGPU_MATRIX_WAIT_PROFILE_COUNT_ - 1u) *
                      (LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_COUNT_ - 1u) *
                      LOOM_AMDGPU_MATRIX_WAIT_PASS_COUNT_,
              "matrix wait rows must cover each concrete profile/use/pass");

static const iree_string_view_t kAmdgpuMatrixWaitProfileNames[] = {
    [LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN] = IREE_SVL("unknown"),
    [LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950] =
        IREE_SVL("mfma_pre_gfx950"),
    [LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950] = IREE_SVL("mfma_gfx950"),
};

static const iree_string_view_t kAmdgpuMatrixWaitResultUseNames[] = {
    [LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN] = IREE_SVL("unknown"),
    [LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX] = IREE_SVL("non_matrix"),
    [LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT] =
        IREE_SVL("matrix_srcc_exact"),
    [LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP] =
        IREE_SVL("matrix_srcc_overlap"),
    [LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB] =
        IREE_SVL("matrix_src_ab"),
};

static const loom_amdgpu_matrix_wait_profile_t
    kAmdgpuMatrixWaitProfilesByFeatureProfile[] = {
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE] =
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908] =
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A] =
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940] =
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950] =
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC] =
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11] =
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12] =
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250] =
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
        [LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC] =
            LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
};

iree_string_view_t loom_amdgpu_matrix_wait_profile_name(
    loom_amdgpu_matrix_wait_profile_t profile) {
  if ((iree_host_size_t)profile >=
      IREE_ARRAYSIZE(kAmdgpuMatrixWaitProfileNames)) {
    return kAmdgpuMatrixWaitProfileNames
        [LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN];
  }
  return kAmdgpuMatrixWaitProfileNames[profile];
}

iree_string_view_t loom_amdgpu_matrix_wait_result_use_name(
    loom_amdgpu_matrix_wait_result_use_t use) {
  if ((iree_host_size_t)use >=
      IREE_ARRAYSIZE(kAmdgpuMatrixWaitResultUseNames)) {
    return kAmdgpuMatrixWaitResultUseNames
        [LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN];
  }
  return kAmdgpuMatrixWaitResultUseNames[use];
}

bool loom_amdgpu_matrix_wait_profile_from_feature_profile(
    loom_amdgpu_matrix_feature_profile_t feature_profile,
    loom_amdgpu_matrix_wait_profile_t* out_profile) {
  loom_amdgpu_matrix_wait_profile_t profile =
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN;
  if ((iree_host_size_t)feature_profile <
      IREE_ARRAYSIZE(kAmdgpuMatrixWaitProfilesByFeatureProfile)) {
    profile = kAmdgpuMatrixWaitProfilesByFeatureProfile[feature_profile];
  }
  *out_profile = profile;
  return profile != LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN;
}

iree_host_size_t loom_amdgpu_matrix_wait_result_row_count(void) {
  return IREE_ARRAYSIZE(kAmdgpuMatrixWaitResultRows);
}

const loom_amdgpu_matrix_wait_result_row_t*
loom_amdgpu_matrix_wait_result_row_at(iree_host_size_t index) {
  if (index >= IREE_ARRAYSIZE(kAmdgpuMatrixWaitResultRows)) {
    return NULL;
  }
  return &kAmdgpuMatrixWaitResultRows[index];
}

static bool loom_amdgpu_matrix_wait_pass_index(
    uint16_t pass_count, iree_host_size_t* out_pass_index) {
  switch (pass_count) {
    case 2:
      *out_pass_index = LOOM_AMDGPU_MATRIX_WAIT_PASS_2_INDEX;
      return true;
    case 4:
      *out_pass_index = LOOM_AMDGPU_MATRIX_WAIT_PASS_4_INDEX;
      return true;
    case 8:
      *out_pass_index = LOOM_AMDGPU_MATRIX_WAIT_PASS_8_INDEX;
      return true;
    case 16:
      *out_pass_index = LOOM_AMDGPU_MATRIX_WAIT_PASS_16_INDEX;
      return true;
    case 32:
      *out_pass_index = LOOM_AMDGPU_MATRIX_WAIT_PASS_32_INDEX;
      return true;
    default:
      return false;
  }
}

const loom_amdgpu_matrix_wait_result_row_t* loom_amdgpu_matrix_wait_result_find(
    loom_amdgpu_matrix_wait_profile_t profile, uint16_t pass_count,
    loom_amdgpu_matrix_wait_result_use_t use) {
  if (profile == LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN ||
      profile >= LOOM_AMDGPU_MATRIX_WAIT_PROFILE_COUNT_ ||
      use == LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN ||
      use >= LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_COUNT_) {
    return NULL;
  }
  iree_host_size_t pass_index = 0;
  if (!loom_amdgpu_matrix_wait_pass_index(pass_count, &pass_index)) {
    return NULL;
  }
  const iree_host_size_t row_index =
      (((iree_host_size_t)profile - 1u) *
           (LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_COUNT_ - 1u) +
       ((iree_host_size_t)use - 1u)) *
          LOOM_AMDGPU_MATRIX_WAIT_PASS_COUNT_ +
      pass_index;
  IREE_ASSERT_LT(row_index, IREE_ARRAYSIZE(kAmdgpuMatrixWaitResultRows));
  const loom_amdgpu_matrix_wait_result_row_t* row =
      &kAmdgpuMatrixWaitResultRows[row_index];
  IREE_ASSERT_EQ(row->profile, profile);
  IREE_ASSERT_EQ(row->use, use);
  IREE_ASSERT_EQ(row->pass_count, pass_count);
  return row;
}

bool loom_amdgpu_matrix_wait_result_cycle_count(
    loom_amdgpu_matrix_wait_profile_t profile, uint16_t pass_count,
    loom_amdgpu_matrix_wait_result_use_t use, uint16_t* out_cycle_count) {
  *out_cycle_count = 0;
  const loom_amdgpu_matrix_wait_result_row_t* row =
      loom_amdgpu_matrix_wait_result_find(profile, pass_count, use);
  if (row == NULL) {
    return false;
  }
  *out_cycle_count = row->cycle_count;
  return true;
}
