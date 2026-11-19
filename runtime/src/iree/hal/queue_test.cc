// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/queue.h"

#include "iree/base/internal/debugging.h"
#include "iree/testing/gtest.h"

namespace {

TEST(TransferGeometryTest, BoundedSlicesPreserveEveryAddress) {
  // The oracle enumerates addressed byte pairs, independently of how the
  // implementation partitions whole slices, rows, and partial rows.
  const iree_hal_transfer_region_t regions[] = {
      {7, 5, 3, {7, 35}, {7, 35}},
      {7, 5, 3, {11, 61}, {13, 73}},
      {7, 5, 3, {0, 0}, {13, 73}},
      {7, 1, 3, {UINT64_MAX, 11}, {UINT64_MAX, 13}},
      {7, 5, 1, {11, UINT64_MAX}, {13, UINT64_MAX}},
  };
  for (const auto& region : regions) {
    for (uint64_t capacity : {1, 6, 7, 8, 34, 35, 36, 69, 70, 105, 128}) {
      const uint64_t length =
          region.row_length * region.row_count * region.slice_count;
      uint64_t cursor = 0;
      while (cursor < length) {
        uint64_t source_offset = 0, target_offset = 0;
        const auto chunk = iree_hal_transfer_region_slice(
            &region, cursor, capacity, &source_offset, &target_offset);
        const uint64_t count =
            chunk.row_length * chunk.row_count * chunk.slice_count;
        ASSERT_GT(count, 0u);
        ASSERT_LE(count, capacity);
        ASSERT_LE(count, length - cursor);
        for (uint64_t z = 0; z < chunk.slice_count; ++z) {
          for (uint64_t y = 0; y < chunk.row_count; ++y) {
            for (uint64_t x = 0; x < chunk.row_length; ++x, ++cursor) {
              const uint64_t column = cursor % region.row_length;
              const uint64_t row =
                  (cursor / region.row_length) % region.row_count;
              const uint64_t slice =
                  cursor / (region.row_length * region.row_count);
              EXPECT_EQ(source_offset + x +
                            y * chunk.source_strides.row_stride +
                            z * chunk.source_strides.slice_stride,
                        column + row * region.source_strides.row_stride +
                            slice * region.source_strides.slice_stride);
              EXPECT_EQ(target_offset + x +
                            y * chunk.target_strides.row_stride +
                            z * chunk.target_strides.slice_stride,
                        column + row * region.target_strides.row_stride +
                            slice * region.target_strides.slice_stride);
            }
          }
        }
      }
    }
  }
}

TEST(TransferGeometryTest, IndependentPaddedWindows) {
  const iree_hal_transfer_region_t region = {
      .row_length = 256,
      .row_count = 128,
      .slice_count = 1,
      .source_strides = {512, 0},
      .target_strides = {1024, 0},
  };
  EXPECT_TRUE(iree_hal_transfer_range_fits(17, region.row_length,
                                           region.row_count, region.slice_count,
                                           &region.source_strides, 17 + 65280));
  EXPECT_FALSE(iree_hal_transfer_range_fits(
      17, region.row_length, region.row_count, region.slice_count,
      &region.source_strides, 17 + 65279));
  EXPECT_TRUE(iree_hal_transfer_range_fits(
      31, region.row_length, region.row_count, region.slice_count,
      &region.target_strides, 31 + 130304));
  EXPECT_FALSE(iree_hal_transfer_range_fits(
      31, region.row_length, region.row_count, region.slice_count,
      &region.target_strides, 31 + 130303));
}

TEST(TransferGeometryTest, EmptyIgnoresOriginsAndStrides) {
  EXPECT_TRUE(iree_hal_transfer_range_fits(UINT64_MAX, 0, UINT64_MAX,
                                           UINT64_MAX, nullptr, 0));
  EXPECT_TRUE(iree_hal_transfer_range_fits(UINT64_MAX, UINT64_MAX, 0,
                                           UINT64_MAX, nullptr, 0));
  EXPECT_TRUE(iree_hal_transfer_range_fits(UINT64_MAX, UINT64_MAX, UINT64_MAX,
                                           0, nullptr, 0));
}

TEST(TransferGeometryTest, UnusedStridesAreNotRead) {
  iree_hal_transfer_strides_t strides = {0, 0};
  IREE_ASAN_POISON_MEMORY_REGION(&strides, sizeof(strides));
  const bool fits = iree_hal_transfer_range_fits(19, 90, 1, 1, &strides, 109);
  IREE_ASAN_UNPOISON_MEMORY_REGION(&strides, sizeof(strides));
  EXPECT_TRUE(fits);

  strides.row_stride = 128;
  IREE_ASAN_POISON_MEMORY_REGION(&strides.slice_stride, sizeof(uint64_t));
  const bool rows_fit =
      iree_hal_transfer_range_fits(19, 90, 2, 1, &strides, 237);
  IREE_ASAN_UNPOISON_MEMORY_REGION(&strides.slice_stride, sizeof(uint64_t));
  EXPECT_TRUE(rows_fit);
}

TEST(TransferGeometryTest, FullWidthBoundariesAndOverflow) {
  iree_hal_transfer_strides_t strides = {1, 1};
  EXPECT_TRUE(
      iree_hal_transfer_range_fits(0, UINT64_MAX, 1, 1, &strides, UINT64_MAX));
  EXPECT_TRUE(iree_hal_transfer_range_fits(UINT64_MAX - 1, 1, 1, 1, &strides,
                                           UINT64_MAX));
  EXPECT_FALSE(
      iree_hal_transfer_range_fits(UINT64_MAX, 1, 1, 1, &strides, UINT64_MAX));
  EXPECT_TRUE(
      iree_hal_transfer_range_fits(0, 1, UINT64_MAX, 1, &strides, UINT64_MAX));
  EXPECT_TRUE(
      iree_hal_transfer_range_fits(0, 1, 1, UINT64_MAX, &strides, UINT64_MAX));
  EXPECT_FALSE(
      iree_hal_transfer_range_fits(0, 1, UINT64_MAX, 2, &strides, UINT64_MAX));
  strides.row_stride = UINT64_C(1) << 63;
  EXPECT_FALSE(iree_hal_transfer_range_fits(0, 1, 3, 1, &strides, UINT64_MAX));
  strides.slice_stride = UINT64_C(1) << 63;
  EXPECT_FALSE(iree_hal_transfer_range_fits(0, 1, 1, 3, &strides, UINT64_MAX));
}

TEST(TransferGeometryTest, RepeatedReadsDoNotRequireDenseBacking) {
  const iree_hal_transfer_strides_t strides = {0, 0};
  EXPECT_TRUE(
      iree_hal_transfer_range_fits(7, 9, UINT64_MAX, UINT64_MAX, &strides, 16));
  EXPECT_FALSE(
      iree_hal_transfer_range_fits(7, 9, UINT64_MAX, UINT64_MAX, &strides, 15));
}

TEST(TransferGeometryTest, MatchesAddressedBytesAcrossSmallLayouts) {
  // Enumerating actual bytes is an independent oracle for the constant-time
  // range check, including transposed dimensions and repeated source rows.
  for (uint64_t row_length = 0; row_length <= 4; ++row_length) {
    for (uint64_t rows = 0; rows <= 3; ++rows) {
      for (uint64_t slices = 0; slices <= 3; ++slices) {
        for (uint64_t row_stride : {0u, 2u, 5u}) {
          for (uint64_t slice_stride : {0u, 3u, 17u}) {
            const iree_hal_transfer_strides_t strides = {row_stride,
                                                         slice_stride};
            for (uint64_t length : {0u, 3u, 11u, 24u, 47u}) {
              bool fits = true;
              for (uint64_t z = 0; z < slices; ++z) {
                for (uint64_t y = 0; y < rows; ++y) {
                  for (uint64_t x = 0; x < row_length; ++x) {
                    fits &= 3 + z * slice_stride + y * row_stride + x < length;
                  }
                }
              }
              EXPECT_EQ(iree_hal_transfer_range_fits(3, row_length, rows,
                                                     slices, &strides, length),
                        fits);
            }
          }
        }
      }
    }
  }
}

}  // namespace
