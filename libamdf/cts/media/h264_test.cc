// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

#include "libamdf/cts/media/decode_fixture.h"
#include "libamdf/cts/media/h264_fixture.h"
#include "libamdf/cts/media/h264_profile_fixture.h"

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
                        &entry.picture, entry.pixel_hash};
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

TEST_P(MediaH264Test, WritesReferenceBeforeDependentPicture) {
  ASSERT_NO_FATAL_FAILURE(CreatePictures());
  ASSERT_NO_FATAL_FAILURE(Begin());
  for (uint32_t index = 0; index < 2; ++index) {
    std::fill(output_memory_.bytes().begin(), output_memory_.bytes().end(),
              0xa5);
    ASSERT_NO_FATAL_FAILURE(
        Cache(output_memory_, AMDF_HOST_CACHE_OPERATION_FLUSH));
    ASSERT_NO_FATAL_FAILURE(PreparePicture(index, index));
    ASSERT_NO_FATAL_FAILURE(Cache(input_, AMDF_HOST_CACHE_OPERATION_FLUSH));
    ASSERT_EQ(Submit(commands_[index]), AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(Join());
    ASSERT_NO_FATAL_FAILURE(CheckGuards(output_memory_, output_layouts_));
    ASSERT_NO_FATAL_FAILURE(CheckReferenceGuards());
    EXPECT_EQ(HashFramePixels(output_memory_, output_layouts_[index]),
              inputs_[index].pixel_hash)
        << "output picture=" << index;
    ASSERT_NO_FATAL_FAILURE(
        CheckReferenceWritten(inputs_[index].reference_slot, 0xa5));
  }
  ASSERT_NO_FATAL_FAILURE(End());
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

INSTANTIATE_TEST_SUITE_P(NativeAndLinearOutputs, MediaH264Test,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& parameter) {
                           return parameter.param ? "OutputPlanes" : "Images";
                         });

class MediaH264GeometryTest : public amdf::cts::media::MediaDecodeTest,
                              public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(amdf::cts::media::LoadH264GeometryFixture(
        &fixture_, &configuration_, &matrices_));
    ASSERT_NO_FATAL_FAILURE(
        SetUpMedia({AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_DIRECTION_DECODE,
                    AMDF_MEDIA_PIXEL_FORMAT_NV12, 320, 208, 2},
                   &configuration_, GetParam()));
    if (IsSkipped()) {
      return;
    }
    for (uint32_t index = 0; index < fixture_.size(); ++index) {
      auto& entry = fixture_[index];
      for (uint32_t reference = 0; reference < entry.picture.reference_count;
           ++reference) {
        const uint32_t slot = entry.reference_slots[reference];
        entry.references[reference].binding = {&references_[slot], slot};
      }
      inputs_[index] = {entry.bytes, entry.byte_length, entry.reference_slot,
                        &entry.picture, entry.pixel_hash};
    }
  }

  std::array<amdf::cts::media::H264FixturePicture, 2> fixture_ = {};
  amdf_media_h264_configuration_t configuration_ = {};
  amdf_media_h264_scaling_matrices_t matrices_ = {};
  std::array<amdf::cts::media::DecodeInput, 2> inputs_ = {};
};

TEST_P(MediaH264GeometryTest, DecodesDependentPictureAcrossTileBoundaries) {
  ASSERT_NO_FATAL_FAILURE(RunPictures(inputs_, 4));
}

INSTANTIATE_TEST_SUITE_P(NativeAndLinearOutputs, MediaH264GeometryTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& parameter) {
                           return parameter.param ? "OutputPlanes" : "Images";
                         });

struct H264ProfileCase {
  // Exact admitted syntax profile represented by the compressed picture.
  amdf_media_h264_profile_t profile;
  // Stable test suffix describing the syntax mechanism under test.
  const char* name;
};

class MediaH264ProfileTest
    : public amdf::cts::media::MediaDecodeTest,
      public ::testing::WithParamInterface<H264ProfileCase> {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(amdf::cts::media::LoadH264ProfileFixture(
        GetParam().profile, &fixture_));
    ASSERT_NO_FATAL_FAILURE(
        SetUpMedia({AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_DIRECTION_DECODE,
                    AMDF_MEDIA_PIXEL_FORMAT_NV12, 256, 256, 1},
                   &fixture_.configuration, false));
    if (IsSkipped()) {
      return;
    }
    input_ = {fixture_.bytes, fixture_.byte_length, 0, &fixture_.picture,
              fixture_.pixel_hash};
  }

  // One independently encoded profile-specific IDR and its parsed syntax.
  amdf::cts::media::H264ProfileFixture fixture_ = {};
  // Borrowed submission view reused only after native retirement.
  amdf::cts::media::DecodeInput input_ = {};
};

TEST_P(MediaH264ProfileTest, DecodesProfileSpecificSyntaxWithExactPixels) {
  ASSERT_NO_FATAL_FAILURE(RunPictures(std::span(&input_, 1), 4));
}

INSTANTIATE_TEST_SUITE_P(
    AdvertisedProfiles, MediaH264ProfileTest,
    ::testing::Values(
        H264ProfileCase{AMDF_MEDIA_H264_PROFILE_CONSTRAINED_BASELINE,
                        "ConstrainedBaseline"},
        H264ProfileCase{AMDF_MEDIA_H264_PROFILE_HIGH, "High"}),
    [](const ::testing::TestParamInfo<H264ProfileCase>& parameter) {
      return std::string(parameter.param.name);
    });

}  // namespace
