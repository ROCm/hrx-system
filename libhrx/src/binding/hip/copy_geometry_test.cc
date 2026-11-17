// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/copy_geometry.h"

#include "iree/testing/gtest.h"

namespace {

TEST(CopyGeometryTest, IndependentOriginsAndSlicePadding) {
  hipMemcpy3DParms params = {
      .srcPos = {3, 2, 1},
      .srcPtr = {nullptr, 16, 16, 5},
      .dstPos = {7, 1, 2},
      .dstPtr = {nullptr, 32, 32, 4},
      .extent = {8, 2, 3},
  };
  iree_hip_copy_geometry_t geometry;
  ASSERT_TRUE(iree_hip_copy_geometry_calculate(&params, &geometry));
  EXPECT_EQ(geometry.length, 48u);
  EXPECT_EQ(geometry.source.offset, 115u);
  EXPECT_EQ(geometry.source.span, 184u);
  EXPECT_EQ(geometry.source.row_pitch, 16u);
  EXPECT_EQ(geometry.source.slice_pitch, 80u);
  EXPECT_EQ(geometry.target.offset, 295u);
  EXPECT_EQ(geometry.target.span, 296u);
  EXPECT_EQ(geometry.target.row_pitch, 32u);
  EXPECT_EQ(geometry.target.slice_pitch, 128u);
  const auto compact = iree_hip_copy_geometry_compact_layout(&geometry);
  EXPECT_EQ(compact.offset, 0u);
  EXPECT_EQ(compact.span, 48u);
  EXPECT_EQ(compact.row_pitch, 8u);
  EXPECT_EQ(compact.slice_pitch, 16u);
}

TEST(CopyGeometryTest, ImplicitSliceHeightIncludesOrigin) {
  hipMemcpy3DParms params = {
      .srcPos = {1, 2, 0},
      .srcPtr = {nullptr, 8, 8, 0},
      .dstPos = {2, 1, 0},
      .dstPtr = {nullptr, 16, 16, 0},
      .extent = {4, 3, 2},
  };
  iree_hip_copy_geometry_t geometry;
  ASSERT_TRUE(iree_hip_copy_geometry_calculate(&params, &geometry));
  EXPECT_EQ(geometry.length, 24u);
  EXPECT_EQ(geometry.source.offset, 17u);
  EXPECT_EQ(geometry.source.span, 60u);
  EXPECT_EQ(geometry.source.slice_pitch, 40u);
  EXPECT_EQ(geometry.target.offset, 18u);
  EXPECT_EQ(geometry.target.span, 100u);
  EXPECT_EQ(geometry.target.slice_pitch, 64u);
}

TEST(CopyGeometryTest, OverflowRejectsEitherEndpointBeforePointerUse) {
  // Only byte geometry is evaluated: none of these descriptor pointers needs
  // backing storage, and no device or native memory operation is involved.
  const hipMemcpy3DParms valid = {
      .srcPtr = {nullptr, 8, 8, 3},
      .dstPtr = {nullptr, 8, 8, 3},
      .extent = {4, 2, 2},
  };
  for (bool target : {false, true}) {
    for (int overflow = 0; overflow < 5; ++overflow) {
      SCOPED_TRACE(target);
      SCOPED_TRACE(overflow);
      auto params = valid;
      auto& pointer = target ? params.dstPtr : params.srcPtr;
      switch (overflow) {
        case 0:  // Effective height cannot include the origin.
          pointer.ysize = 0;
          if (target) {
            params.dstPos.y = SIZE_MAX;
          } else {
            params.srcPos.y = SIZE_MAX;
          }
          break;
        case 1:  // The slice pitch cannot be represented.
          pointer.ysize = SIZE_MAX;
          break;
        case 2:  // The z origin cannot be represented as a byte offset.
          if (target) {
            params.dstPos.z = SIZE_MAX;
          } else {
            params.srcPos.z = SIZE_MAX;
          }
          break;
        case 3:  // The copied span exceeds the addressable range.
          pointer.ysize = SIZE_MAX / 8;
          params.extent.depth = 3;
          break;
        case 4:  // Offset and span each fit, but their sum does not.
          pointer.ysize = SIZE_MAX / 16;
          if (target) {
            params.dstPos.z = 2;
          } else {
            params.srcPos.z = 2;
          }
          break;
      }
      iree_hip_copy_geometry_t geometry;
      EXPECT_FALSE(iree_hip_copy_geometry_calculate(&params, &geometry));
    }
  }
  auto params = valid;
  params.extent.width = SIZE_MAX;
  iree_hip_copy_geometry_t geometry;
  EXPECT_FALSE(iree_hip_copy_geometry_calculate(&params, &geometry));
}

}  // namespace
