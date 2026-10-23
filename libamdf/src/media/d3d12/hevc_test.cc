// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/media/d3d12/decode.h"

namespace amdf::media::d3d12 {
namespace {

class HevcPictureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session_.info.configuration = {AMDF_MEDIA_CODEC_HEVC,
                                   AMDF_MEDIA_DIRECTION_DECODE,
                                   AMDF_MEDIA_PIXEL_FORMAT_P010,
                                   320,
                                   192,
                                   5};
    session_.hevc.profile = AMDF_MEDIA_HEVC_PROFILE_MAIN10;
    session_.hevc.sample_bit_depth = 10;
    picture_.type = AMDF_STRUCTURE_TYPE_MEDIA_HEVC_DECODE_PICTURE_INFO;
    picture_.structure_size = sizeof(picture_);
    picture_.sequence.maximum_decoded_picture_buffer_count_minus1 = 4;
    picture_.sequence.log2_diff_max_min_luma_coding_block_size = 3;
    picture_.sequence.log2_diff_max_min_luma_transform_block_size = 3;
    picture_.sequence.long_term_references_present = 1;
    picture_.picture_order_count = 10;
    references_[0] = {{&frames_[0], 3}, 12, 0};
    references_[1] = {{&frames_[1], 1}, -2, 0};
    references_[2] = {{&frames_[2], 0}, 4, 1};
    picture_.references = references_;
    picture_.reference_count = 3;
    picture_.short_term_before = {1, {1}};
    picture_.short_term_after = {1, {0}};
    picture_.long_term = {1, {2}};
    picture_.slice_count = 2;
    picture_.slices = slices_;
  }

  amdf_status_t Prepare() {
    return PrepareDecode(session_, &picture_, 256, 4, 4, 2, native_slices_,
                         &native_);
  }

  // Admitted precision and bounded native reference-index space.
  Session session_ = {};
  // Public caller-authored syntax, consumed by the actual preparation boundary.
  amdf_media_hevc_decode_picture_info_t picture_ = {};
  // Distinct logical frame identities; native image validation is CTS-covered.
  amdf_media_frame_t frames_[3] = {};
  // Deliberately shuffled physical slots with short- and long-term markings.
  amdf_media_hevc_reference_t references_[3] = {};
  // Two complete disjoint slices in one compressed range.
  amdf_media_hevc_slice_t slices_[2] = {{0, 128}, {128, 128}};
  // Provisioned native slice storage, with no allocation in preparation.
  DXVA_Slice_HEVC_Short native_slices_[2] = {};
  // Prepared arguments handed directly to the native recording operation.
  DecodePicture native_ = {};
};

TEST_F(HevcPictureTest, ReferenceListsRemainDistinctFromStableSurfaceSlots) {
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  const auto& parameters = native_.parameters.hevc;
  EXPECT_EQ(parameters.CurrPic.bPicEntry, 4u);
  EXPECT_EQ(parameters.RefPicList[0].bPicEntry, 3u);
  EXPECT_EQ(parameters.RefPicList[1].bPicEntry, 1u);
  EXPECT_EQ(parameters.RefPicList[2].bPicEntry, 0x80u);
  EXPECT_EQ(parameters.RefPicSetStCurrBefore[0], 1u);
  EXPECT_EQ(parameters.RefPicSetStCurrAfter[0], 0u);
  EXPECT_EQ(parameters.RefPicSetLtCurr[0], 2u);
  EXPECT_EQ(parameters.PicOrderCntValList[1], -2);
  EXPECT_EQ(native_.references[0].frame, &frames_[0]);
  EXPECT_EQ(native_.references[0].slot_index, 3u);
  for (uint32_t index = 3; index < 15; ++index) {
    EXPECT_EQ(parameters.RefPicList[index].bPicEntry, 0xffu);
  }
  for (uint32_t index = 1; index < 8; ++index) {
    EXPECT_EQ(parameters.RefPicSetStCurrBefore[index], 0xffu);
    EXPECT_EQ(parameters.RefPicSetStCurrAfter[index], 0xffu);
    EXPECT_EQ(parameters.RefPicSetLtCurr[index], 0xffu);
  }
}

TEST_F(HevcPictureTest, ConflictingSlotsAndReferenceSetsRejectBeforeRecording) {
  references_[0].binding.slot_index = 4;
  EXPECT_EQ(Prepare(), amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  references_[0].binding.slot_index = 1;
  EXPECT_EQ(Prepare(), amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  references_[0].binding.slot_index = 5;
  EXPECT_EQ(Prepare(), amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  references_[0].binding.slot_index = 3;
  picture_.short_term_after.indices[0] = 1;
  EXPECT_EQ(Prepare(), amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  picture_.short_term_after.indices[0] = 3;
  EXPECT_EQ(Prepare(), amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  picture_.short_term_after.indices[0] = 0;
  EXPECT_EQ(Prepare(), AMDF_STATUS_OK);
}

TEST_F(HevcPictureTest, PrecisionAndTileBoundsPreserveTheDeclaredPicture) {
  picture_.picture.initial_qp_minus26 = -38;
  picture_.picture.tiles_enabled = 1;
  picture_.picture.tile_column_count_minus1 = 1;
  picture_.picture.tile_row_count_minus1 = 1;
  picture_.picture.tile_column_width_minus1[0] = 1;
  picture_.picture.tile_row_height_minus1[0] = 0;
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  EXPECT_EQ(native_.parameters.hevc.bit_depth_luma_minus8, 2u);
  EXPECT_EQ(native_.parameters.hevc.column_width_minus1[0], 1u);
  session_.hevc.sample_bit_depth = 8;
  EXPECT_EQ(Prepare(), amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  picture_.picture.initial_qp_minus26 = -26;
  EXPECT_EQ(Prepare(), AMDF_STATUS_OK);
  picture_.picture.tile_column_width_minus1[0] = 4;
  EXPECT_EQ(Prepare(), amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  picture_.picture.tile_column_width_minus1[0] = UINT32_MAX;
  EXPECT_EQ(Prepare(), amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
}

TEST_F(HevcPictureTest, AsymmetricScalingGridsUseTheHevcDiagonalScan) {
  amdf_media_hevc_scaling_matrices_t matrices;
  std::memset(&matrices, 16, sizeof(matrices));
  for (auto& matrix : matrices.scaling_8x8) {
    for (uint32_t index = 0; index < 64; ++index) {
      matrix[index] = index + 1;
    }
  }
  picture_.scaling_matrices = &matrices;
  picture_.sequence.scaling_list_enabled = 1;
  ASSERT_EQ(Prepare(), AMDF_STATUS_OK);
  // A coordinate's diagonal is x+y. All smaller diagonals precede it, then
  // coordinates on this diagonal are ordered by x. This is independent of
  // the production scan's diagonal walk.
  for (uint32_t y = 0; y < 8; ++y) {
    for (uint32_t x = 0; x < 8; ++x) {
      uint32_t position = 0;
      for (uint32_t other = 0; other < 64; ++other) {
        const uint32_t other_x = other % 8, other_y = other / 8;
        position += other_x + other_y < x + y ||
                    (other_x + other_y == x + y && other_x < x);
      }
      for (const auto& matrix : native_.matrices.hevc.ucScalingLists1) {
        EXPECT_EQ(matrix[position], y * 8 + x + 1);
      }
    }
  }
}

}  // namespace
}  // namespace amdf::media::d3d12
