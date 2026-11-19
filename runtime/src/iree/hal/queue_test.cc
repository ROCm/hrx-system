// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/queue.h"

#include "iree/base/internal/debugging.h"
#include "iree/testing/gtest.h"

namespace {

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
