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
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_tsan_calculate_config(&options, 65536, &config));
  EXPECT_EQ(config.record_length, 96u);
  EXPECT_EQ(config.abi_version, 0u);
  EXPECT_EQ(config.flags, 3u);
  EXPECT_EQ(config.memory_granule_shift, 2u);
  EXPECT_EQ(config.shadow_entry_size, 8u);
  EXPECT_EQ(config.workgroup_shadow_stride, 8u + 16384u * 8u);
  EXPECT_EQ(config.dispatch_shadow_stride, config.shadow_size);
  EXPECT_EQ(config.shadow_slot_count, 1u);
  EXPECT_EQ(config.shadow_size, (8u + 16384u * 8u) * 256u * 16u);
  EXPECT_EQ(config.shadow_base, 0u);
  EXPECT_EQ(config.queue_state_base, 0u);
  EXPECT_EQ(config.queue_aql_base, 0u);
  EXPECT_EQ(config.queue_aql_slot_mask, 0u);
}

TEST(TsanTest, ByteGranulesAndCoverageOverridePreserveWorkgroupHeaders) {
  iree_hal_amd_gpu_tsan_options_t options = {
      .workgroup_local_memory_size = 256,
      .workgroup_capacity = 3,
  };
  iree_hal_amd_gpu_tsan_config_t config;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_tsan_calculate_config(&options, 65536, &config));
  EXPECT_EQ(config.workgroup_shadow_stride, 8u + 256u * 8u);
  EXPECT_EQ(config.dispatch_shadow_stride, (8u + 256u * 8u) * 3u);
  EXPECT_EQ(config.shadow_size, (8u + 256u * 8u) * 3u);
}

TEST(TsanTest, PublicGeometryRejectsInvalidCoverage) {
  iree_hal_amd_gpu_tsan_options_t defaults;
  iree_hal_amd_gpu_tsan_options_initialize(&defaults);
  iree_hal_amd_gpu_tsan_config_t config;
  auto options = defaults;
  options.memory_granule_shift = 17;
  EXPECT_EQ(
      Status(iree_hal_amd_gpu_tsan_calculate_config(&options, 65536, &config))
          .code(),
      StatusCode::kInvalidArgument);
  options = defaults;
  options.workgroup_capacity = 0;
  EXPECT_EQ(
      Status(iree_hal_amd_gpu_tsan_calculate_config(&options, 65536, &config))
          .code(),
      StatusCode::kInvalidArgument);
  options = defaults;
  options.workgroup_local_memory_size = 3;
  EXPECT_EQ(
      Status(iree_hal_amd_gpu_tsan_calculate_config(&options, 65536, &config))
          .code(),
      StatusCode::kInvalidArgument);
  EXPECT_EQ(
      Status(iree_hal_amd_gpu_tsan_calculate_config(&defaults, 0, &config))
          .code(),
      StatusCode::kInvalidArgument);
}

TEST(TsanTest, RejectsMultiplicationOverflowBeforeNativeAllocation) {
  iree_hal_amd_gpu_tsan_options_t options = {
      .workgroup_local_memory_size = UINT32_C(1) << 31,
      .workgroup_capacity = UINT32_MAX,
  };
  iree_hal_amd_gpu_tsan_config_t config;
  EXPECT_EQ(
      Status(iree_hal_amd_gpu_tsan_calculate_config(&options, 65536, &config))
          .code(),
      StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace iree::hal::amd::gpu
