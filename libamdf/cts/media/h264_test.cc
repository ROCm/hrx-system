// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <utility>

#include "libamdf/cts/media/decode_fixture.h"
#include "libamdf/cts/media/h264_fixture.h"

namespace {

class MediaH264Test : public amdf::cts::media::MediaDecodeTest,
                      public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    const amdf_media_h264_configuration_t syntax = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION,
        .structure_size = sizeof(syntax),
        .profile = AMDF_MEDIA_H264_PROFILE_MAIN,
        .level_idc = 13,
    };
    ASSERT_NO_FATAL_FAILURE(
        SetUpMedia({AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_DIRECTION_DECODE,
                    AMDF_MEDIA_PIXEL_FORMAT_NV12, 256, 256, 4},
                   &syntax, GetParam()));
    if (IsSkipped()) {
      return;
    }
    ASSERT_NO_FATAL_FAILURE(amdf::cts::media::LoadH264Fixture(&fixture_));
    std::memset(&matrices_, 16, sizeof(matrices_));
    for (uint32_t index = 0; index < fixture_.size(); ++index) {
      auto& entry = fixture_[index];
      entry.picture.scaling_matrices = &matrices_;
      for (uint32_t i = 0; i < entry.picture.reference_count; ++i) {
        ASSERT_LT(entry.reference_slots[i], references_.size());
        entry.references[i].binding = {&references_[entry.reference_slots[i]],
                                       entry.reference_slots[i]};
      }
      inputs_[index] = {entry.bytes, entry.byte_length, entry.reference_slot,
                        &entry.picture,
                        amdf::cts::media::kH264PixelHashes[index]};
    }
  }

  void PreparePicture(uint32_t index, uint32_t slot) {
    ASSERT_EQ(MediaDecodeTest::PreparePicture(inputs_[index], slot),
              AMDF_STATUS_OK);
  }

  // Parsed corpus shared with the Windows GPU consumer.
  std::array<amdf::cts::media::H264FixturePicture, 12> fixture_ = {};
  // Explicit raster grids borrowed during preparation.
  amdf_media_h264_scaling_matrices_t matrices_ = {};
  // Common submission view; no native syntax or hidden reference state.
  std::array<amdf::cts::media::DecodeInput, 12> inputs_ = {};
};

TEST_P(MediaH264Test, ReusesRecordingsAndReferenceSlotsWithExactPixels) {
  ASSERT_NO_FATAL_FAILURE(RunPictures(inputs_, 24));
}

TEST_P(MediaH264Test, LifecycleRequiresExplicitAcceptedBeginAndEnd) {
  ASSERT_NO_FATAL_FAILURE(CreatePictures());
  ASSERT_NO_FATAL_FAILURE(PreparePicture(0, 0));
  EXPECT_EQ(Submit(commands_[0]),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  ASSERT_NO_FATAL_FAILURE(Begin());
  ASSERT_NO_FATAL_FAILURE(Join());
  EXPECT_EQ(Submit(control_),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  // Release all recording borrows: the active session still owns its native
  // lifetime and cannot disappear without an ordered END.
  for (auto& command : commands_) {
    ASSERT_EQ(media_->command_destroy(std::exchange(command, nullptr)),
              AMDF_STATUS_OK);
  }
  ASSERT_EQ(media_->command_destroy(std::exchange(control_, nullptr)),
            AMDF_STATUS_OK);
  EXPECT_EQ(media_->session_destroy(session_),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  ASSERT_NO_FATAL_FAILURE(CreateCommand(&control_));
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_NO_FATAL_FAILURE(CreateCommand(&commands_[0]));
  ASSERT_NO_FATAL_FAILURE(PreparePicture(0, 0));
  EXPECT_EQ(Submit(commands_[0]),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
}

INSTANTIATE_TEST_SUITE_P(NativeAndLinear, MediaH264Test, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& parameter) {
                           return parameter.param ? "Planes" : "Image";
                         });

}  // namespace
