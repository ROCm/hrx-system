// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/matrix_wait_states.h"

#include "iree/testing/gtest.h"

namespace {

TEST(AmdgpuMatrixWaitStatesTest, MapsFeatureProfiles) {
  loom_amdgpu_matrix_wait_profile_t wait_profile =
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN;
  EXPECT_TRUE(loom_amdgpu_matrix_wait_profile_from_feature_profile(
      LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940, &wait_profile));
  EXPECT_EQ(wait_profile, LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950);
  EXPECT_TRUE(loom_amdgpu_matrix_wait_profile_from_feature_profile(
      LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950, &wait_profile));
  EXPECT_EQ(wait_profile, LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950);
  EXPECT_TRUE(loom_amdgpu_matrix_wait_profile_from_feature_profile(
      LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC, &wait_profile));
  EXPECT_EQ(wait_profile, LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950);
  EXPECT_FALSE(loom_amdgpu_matrix_wait_profile_from_feature_profile(
      LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12, &wait_profile));
  EXPECT_EQ(wait_profile, LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN);
}

TEST(AmdgpuMatrixWaitStatesTest, NamesTableIds) {
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_matrix_wait_profile_name(
          LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950),
      IREE_SV("mfma_pre_gfx950")));
  EXPECT_TRUE(
      iree_string_view_equal(loom_amdgpu_matrix_wait_profile_name(
                                 LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950),
                             IREE_SV("mfma_gfx950")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_matrix_wait_result_use_name(
          LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP),
      IREE_SV("matrix_srcc_overlap")));
}

TEST(AmdgpuMatrixWaitStatesTest, LooksUpMatrixResultRows) {
  uint16_t cycle_count = 0;
  EXPECT_TRUE(loom_amdgpu_matrix_wait_result_cycle_count(
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950, 4,
      LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, &cycle_count));
  EXPECT_EQ(cycle_count, 7);
  EXPECT_TRUE(loom_amdgpu_matrix_wait_result_cycle_count(
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950, 4,
      LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, &cycle_count));
  EXPECT_EQ(cycle_count, 8);
  EXPECT_TRUE(loom_amdgpu_matrix_wait_result_cycle_count(
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950, 16,
      LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP, &cycle_count));
  EXPECT_EQ(cycle_count, 17);
  EXPECT_TRUE(loom_amdgpu_matrix_wait_result_cycle_count(
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950, 4,
      LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT, &cycle_count));
  EXPECT_EQ(cycle_count, 0);
  EXPECT_TRUE(loom_amdgpu_matrix_wait_result_cycle_count(
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950, 32,
      LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, &cycle_count));
  EXPECT_EQ(cycle_count, 36);
}

TEST(AmdgpuMatrixWaitStatesTest, RejectsUnsupportedRows) {
  uint16_t cycle_count = 1;
  EXPECT_FALSE(loom_amdgpu_matrix_wait_result_cycle_count(
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950, 64,
      LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, &cycle_count));
  EXPECT_EQ(cycle_count, 0);
  EXPECT_FALSE(loom_amdgpu_matrix_wait_result_cycle_count(
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN, 4,
      LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX, &cycle_count));
  EXPECT_EQ(cycle_count, 0);
  EXPECT_FALSE(loom_amdgpu_matrix_wait_result_cycle_count(
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_GFX950, 4,
      LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN, &cycle_count));
  EXPECT_EQ(cycle_count, 0);
}

TEST(AmdgpuMatrixWaitStatesTest, ExposesRowsByOrdinal) {
  ASSERT_GT(loom_amdgpu_matrix_wait_result_row_count(), 0u);
  ASSERT_NE(loom_amdgpu_matrix_wait_result_row_at(0), nullptr);
  EXPECT_EQ(loom_amdgpu_matrix_wait_result_row_at(
                loom_amdgpu_matrix_wait_result_row_count()),
            nullptr);
}

TEST(AmdgpuMatrixWaitStatesTest, RowsRoundTripThroughLookup) {
  const iree_host_size_t row_count = loom_amdgpu_matrix_wait_result_row_count();
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    const loom_amdgpu_matrix_wait_result_row_t* row =
        loom_amdgpu_matrix_wait_result_row_at(i);
    ASSERT_NE(row, nullptr);
    EXPECT_EQ(loom_amdgpu_matrix_wait_result_find(row->profile, row->pass_count,
                                                  row->use),
              row);
  }
}

}  // namespace
