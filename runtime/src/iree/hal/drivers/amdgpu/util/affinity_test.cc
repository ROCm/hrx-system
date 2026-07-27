// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/affinity.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace iree::hal::amdgpu {
namespace {

TEST(DeviceAffinityTest, IteratesEmptyMask) {
  int iteration_count = 0;
  IREE_HAL_AMDGPU_FOR_PHYSICAL_DEVICE(0) { ++iteration_count; }
  EXPECT_EQ(iteration_count, 0);
}

TEST(DeviceAffinityTest, IteratesSparseMask) {
  int expected_ordinals[] = {1, 4, 7};
  int iteration_count = 0;
  IREE_HAL_AMDGPU_FOR_PHYSICAL_DEVICE(0x92ull) {
    ASSERT_LT(iteration_count, 3);
    EXPECT_EQ(device_count, 3);
    EXPECT_EQ(device_index, iteration_count);
    EXPECT_EQ(device_ordinal, expected_ordinals[iteration_count]);
    ++iteration_count;
  }
  EXPECT_EQ(iteration_count, 3);
}

TEST(DeviceAffinityTest, IteratesFinalBitWithoutTrailingZeroScan) {
  int iteration_count = 0;
  IREE_HAL_AMDGPU_FOR_PHYSICAL_DEVICE(UINT64_C(1) << 63) {
    EXPECT_EQ(device_count, 1);
    EXPECT_EQ(device_index, 0);
    EXPECT_EQ(device_ordinal, 63);
    ++iteration_count;
  }
  EXPECT_EQ(iteration_count, 1);
}

}  // namespace
}  // namespace iree::hal::amdgpu
