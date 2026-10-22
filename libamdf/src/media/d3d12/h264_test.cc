// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/h264.h"

#include <cstring>

#include "gtest/gtest.h"

namespace amdf::media::d3d12 {
namespace {

class H264PictureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session_.info.configuration = {AMDF_MEDIA_CODEC_H264,
                                   AMDF_MEDIA_DIRECTION_DECODE,
                                   AMDF_MEDIA_PIXEL_FORMAT_NV12,
                                   320,
                                   192,
                                   4};
    session_.h264.profile = AMDF_MEDIA_H264_PROFILE_HIGH;
    session_.h264.level_idc = 31;
    picture_.sequence.maximum_reference_frame_count = 3;
    picture_.frame_number = 7;
    picture_.scaling_matrices = &matrices_;
    picture_.slices = slices_;
    picture_.slice_count = 2;
    picture_.reference_count = 2;
    picture_.references = references_;
    references_[0].frame = references_[1].frame = &frame_;
    references_[0].frame_number = 6;
    references_[0].top_field_order_count =
        references_[0].bottom_field_order_count = 12;
    references_[1].frame_number = 1;
    references_[1].long_term = 1;
    references_[1].top_field_order_count =
        references_[1].bottom_field_order_count = -2;
    std::memset(&matrices_, 16, sizeof(matrices_));
  }

  // Admitted stream envelope; these syntax tests need no native device.
  Session session_ = {};
  // Parsed picture supplied at the public preparation boundary.
  amdf_media_h264_decode_picture_info_t picture_ = {};
  // Caller-owned frame identity validated separately by command preparation.
  amdf_media_frame_t frame_ = {};
  // Short-term and long-term markings whose identities must remain distinct.
  amdf_media_h264_reference_t references_[2] = {};
  // Two complete slices occupying disjoint compressed ranges.
  amdf_media_h264_slice_t slices_[2] = {{0, 128}, {128, 128}};
  // Raster-ordered matrices, converted at the native boundary.
  amdf_media_h264_scaling_matrices_t matrices_ = {};
};

TEST_F(H264PictureTest, DenseIndicesPreserveReferenceMarkingsAndPictureOrder) {
  ASSERT_EQ(ValidateH264(session_, picture_, 256), AMDF_STATUS_OK);
  DXVA_PicParams_H264 native;
  DXVA_Qmatrix_H264 matrices;
  DXVA_Slice_H264_Short slices[2];
  LowerH264(session_, picture_, &native, &matrices, slices);
  EXPECT_EQ(native.CurrPic.bPicEntry, 2u);
  EXPECT_EQ(native.RefFrameList[0].bPicEntry, 0u);
  EXPECT_EQ(native.RefFrameList[1].bPicEntry, 0x81u);
  EXPECT_EQ(native.FrameNumList[0], 6u);
  EXPECT_EQ(native.FrameNumList[1], 1u);
  EXPECT_EQ(native.FieldOrderCntList[1][0], -2);
  EXPECT_EQ(native.FieldOrderCntList[1][1], -2);
  EXPECT_EQ(native.UsedForReferenceFlags, 15u);
  for (uint32_t index = 2; index < 16; ++index) {
    EXPECT_EQ(native.RefFrameList[index].bPicEntry, 0xffu);
  }
  EXPECT_EQ(slices[1].BSNALunitDataLocation, 128u);
  EXPECT_EQ(slices[1].SliceBytesInBuffer, 128u);
  EXPECT_EQ(slices[1].wBadSliceChopping, 0u);
}

TEST_F(H264PictureTest, SliceRangesAndFrameNumbersRejectBeforeNativeRecording) {
  slices_[1].byte_offset = 127;
  EXPECT_EQ(ValidateH264(session_, picture_, 256),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  slices_[1].byte_offset = 128;
  EXPECT_EQ(ValidateH264(session_, picture_, 255),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  picture_.frame_number = 16;
  EXPECT_EQ(ValidateH264(session_, picture_, 256),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  picture_.sequence.log2_max_frame_num_minus4 = 1;
  EXPECT_EQ(ValidateH264(session_, picture_, 256), AMDF_STATUS_OK);
}

TEST_F(H264PictureTest, ScalingMatricesTraverseAlternatingRasterDiagonals) {
  for (auto& matrix : matrices_.scaling_4x4) {
    for (uint32_t index = 0; index < 16; ++index) {
      matrix[index] = index + 1;
    }
  }
  for (auto& matrix : matrices_.scaling_8x8) {
    for (uint32_t index = 0; index < 64; ++index) {
      matrix[index] = index + 1;
    }
  }
  ASSERT_EQ(ValidateH264(session_, picture_, 256), AMDF_STATUS_OK);
  DXVA_PicParams_H264 native;
  DXVA_Qmatrix_H264 matrices;
  DXVA_Slice_H264_Short slices[2];
  LowerH264(session_, picture_, &native, &matrices, slices);
  auto check = [](const uint8_t* values, uint32_t width) {
    uint32_t index = 0;
    for (uint32_t diagonal = 0; diagonal < 2 * width - 1; ++diagonal) {
      for (uint32_t step = 0; step < width; ++step) {
        const uint32_t y = diagonal % 2 == 0 ? width - step - 1 : step;
        if (diagonal < y || diagonal - y >= width) {
          continue;
        }
        EXPECT_EQ(values[index++], y * width + diagonal - y + 1);
      }
    }
    EXPECT_EQ(index, width * width);
  };
  for (const auto& matrix : matrices.bScalingLists4x4) {
    check(matrix, 4);
  }
  for (const auto& matrix : matrices.bScalingLists8x8) {
    check(matrix, 8);
  }
}

}  // namespace
}  // namespace amdf::media::d3d12
