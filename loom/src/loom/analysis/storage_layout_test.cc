// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/storage_layout.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(StorageLayoutTest, AppendsAlignedRanges) {
  uint64_t byte_extent = 0;
  uint64_t byte_offset = UINT64_MAX;
  ASSERT_TRUE(loom_storage_layout_append(
      /*byte_length=*/12, /*byte_alignment=*/4, &byte_extent, &byte_offset));
  EXPECT_EQ(byte_offset, 0u);
  EXPECT_EQ(byte_extent, 12u);

  ASSERT_TRUE(loom_storage_layout_append(
      /*byte_length=*/4, /*byte_alignment=*/16, &byte_extent, &byte_offset));
  EXPECT_EQ(byte_offset, 16u);
  EXPECT_EQ(byte_extent, 20u);
}

TEST(StorageLayoutTest, AlignmentOverflowPreservesExtent) {
  uint64_t byte_extent = UINT64_MAX - 2;
  uint64_t byte_offset = UINT64_MAX;
  EXPECT_FALSE(loom_storage_layout_append(
      /*byte_length=*/1, /*byte_alignment=*/4, &byte_extent, &byte_offset));
  EXPECT_EQ(byte_offset, 0u);
  EXPECT_EQ(byte_extent, UINT64_MAX - 2);
}

TEST(StorageLayoutTest, ExtentOverflowPreservesExtent) {
  uint64_t byte_extent = UINT64_MAX - 1;
  uint64_t byte_offset = UINT64_MAX;
  EXPECT_FALSE(loom_storage_layout_append(
      /*byte_length=*/2, /*byte_alignment=*/1, &byte_extent, &byte_offset));
  EXPECT_EQ(byte_offset, 0u);
  EXPECT_EQ(byte_extent, UINT64_MAX - 1);
}

}  // namespace
}  // namespace loom
