// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/tsan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

TEST(TsanTest, DefaultArenaPreservesShaderAbi) {
  iree_hal_amd_gpu_tsan_options_t options;
  iree_hal_amd_gpu_tsan_options_initialize(&options);
  iree_hal_amd_gpu_tsan_config_t config;
  IREE_ASSERT_OK(iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
  EXPECT_EQ(config.record_length, 32u);
  EXPECT_EQ(config.abi_version, 0u);
  EXPECT_EQ(config.flags, 3u);
  EXPECT_EQ(config.reserved0, 0u);
  EXPECT_EQ(config.shadow_size, UINT64_C(512) * 1024 * 1024);
  EXPECT_EQ(config.shadow_base, 0u);
}

TEST(TsanTest, ExplicitByteBudgetIsIndependentOfWorkgroupGeometry) {
  const iree_hal_amd_gpu_tsan_options_t options = {.shadow_size = 2176};
  iree_hal_amd_gpu_tsan_config_t config;
  IREE_ASSERT_OK(iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
  EXPECT_EQ(config.shadow_size, 2176u);
}

TEST(TsanTest, ArenaMustFitCompilerByteOffsets) {
  iree_hal_amd_gpu_tsan_options_t options = {};
  iree_hal_amd_gpu_tsan_config_t config;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
  EXPECT_EQ(config.record_length, 0u);
  options.shadow_size = UINT64_C(1) << 32;
  IREE_ASSERT_OK(iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
  EXPECT_EQ(config.shadow_size, UINT64_C(1) << 32);
  for (uint64_t size : {(UINT64_C(1) << 32) + 1, UINT64_MAX}) {
    options.shadow_size = size;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
    EXPECT_EQ(config.record_length, 0u);
  }
}

}  // namespace
}  // namespace iree::hal::amd::gpu
