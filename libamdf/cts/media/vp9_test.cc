// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>

#include "libamdf/cts/media/decode_fixture.h"
#include "libamdf/cts/media/vp9_fixture.h"

namespace {

struct Vp9Case {
  // Pass explicit planes instead of a native image identity.
  bool linear;
  // Embedded stream selecting precision and coded geometry.
  const char* filename;
  // Stable conformance selector for required-test qualification.
  const char* name;
};

class MediaVp9Test : public amdf::cts::media::MediaDecodeTest,
                     public ::testing::WithParamInterface<Vp9Case> {
 protected:
  void SetUp() override {
    const auto& test = GetParam();
    ASSERT_NO_FATAL_FAILURE(
        amdf::cts::media::LoadVp9Fixture(test.filename, &fixture_));
    const amdf_media_vp9_configuration_t syntax = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_VP9_CONFIGURATION,
        .structure_size = sizeof(syntax),
        .profile = static_cast<amdf_media_vp9_profile_t>(
            fixture_.depth == 8 ? AMDF_MEDIA_VP9_PROFILE_0
                                : AMDF_MEDIA_VP9_PROFILE_2),
        .sample_bit_depth = fixture_.depth,
    };
    ASSERT_NO_FATAL_FAILURE(
        SetUpMedia({AMDF_MEDIA_CODEC_VP9, AMDF_MEDIA_DIRECTION_DECODE,
                    static_cast<amdf_media_pixel_format_t>(
                        fixture_.depth == 8 ? AMDF_MEDIA_PIXEL_FORMAT_NV12
                                            : AMDF_MEDIA_PIXEL_FORMAT_P010),
                    fixture_.width, fixture_.height, 9},
                   &syntax, test.linear));
    if (IsSkipped()) {
      return;
    }
    for (uint32_t i = 0; i < fixture_.pictures.size(); ++i) {
      auto& entry = fixture_.pictures[i];
      for (uint32_t r = 0; r < entry.picture.reference_count; ++r) {
        auto& binding = entry.references[r];
        ASSERT_LT(binding.slot_index, references_.size());
        binding.frame = &references_[binding.slot_index];
      }
      inputs_[i] = {entry.bytes, entry.byte_length, entry.reference_slot,
                    &entry.picture, entry.pixel_hash};
    }
  }

  // Independently decoded compressed pictures and resolved syntax.
  amdf::cts::media::Vp9Fixture fixture_ = {};
  // Submission views bound to the application's stable reference images.
  std::array<amdf::cts::media::DecodeInput, 12> inputs_ = {};
};

TEST_P(MediaVp9Test, ReusesAdaptiveContextsAndAliasedLogicalReferences) {
  ASSERT_NO_FATAL_FAILURE(RunPictures(inputs_, 24));
}

TEST_P(MediaVp9Test, RejectsInvalidMapAndPartitionsBeforeReplacingRecording) {
  ASSERT_NO_FATAL_FAILURE(CreatePictures());
  for (uint32_t i = 0; i < 4; ++i) {
    ASSERT_EQ(PreparePicture(inputs_[i], i), AMDF_STATUS_OK);
  }
  auto& picture = fixture_.pictures[1].picture;
  const auto preceding = picture;
  picture.active_reference_indices[0] = 8;
  EXPECT_EQ(PreparePicture(inputs_[1], 1),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  picture = preceding;
  picture.reference_map[0] = static_cast<uint8_t>(picture.reference_count);
  EXPECT_EQ(PreparePicture(inputs_[1], 1),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  picture = preceding;
  picture.compressed_header_byte_length = inputs_[1].byte_length;
  EXPECT_EQ(PreparePicture(inputs_[1], 1),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  picture = preceding;
  ASSERT_NO_FATAL_FAILURE(Cache(input_, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(
      Cache(output_memory_, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(Begin());
  for (auto* command : commands_) {
    ASSERT_EQ(Submit(command), AMDF_STATUS_OK);
  }
  ASSERT_NO_FATAL_FAILURE(Join());
  ASSERT_NO_FATAL_FAILURE(CheckPixels(inputs_, 0));
  ASSERT_NO_FATAL_FAILURE(End());
}

INSTANTIATE_TEST_SUITE_P(
    PrecisionAndStorage, MediaVp9Test,
    ::testing::Values(
        Vp9Case{false, "vp9_8.packet", "Profile0Image"},
        Vp9Case{true, "vp9_8.packet", "Profile0Planes"},
        Vp9Case{false, "vp9_10.packet", "Profile2Image"},
        Vp9Case{true, "vp9_10.packet", "Profile2Planes"},
        Vp9Case{false, "vp9_16x16_8.packet", "Profile0MinimumImage"},
        Vp9Case{true, "vp9_16x16_10.packet", "Profile2MinimumPlanes"}),
    [](const ::testing::TestParamInfo<Vp9Case>& parameter) {
      return std::string(parameter.param.name);
    });

class MediaVp9AdmissionTest : public amdf::cts::media::MediaDecodeTest {
 protected:
  void SetUp() override {
    const amdf_media_vp9_configuration_t syntax = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_VP9_CONFIGURATION,
        .structure_size = sizeof(syntax),
        .profile = AMDF_MEDIA_VP9_PROFILE_2,
        .sample_bit_depth = 10,
    };
    ASSERT_NO_FATAL_FAILURE(
        SetUpMedia({AMDF_MEDIA_CODEC_VP9, AMDF_MEDIA_DIRECTION_DECODE,
                    AMDF_MEDIA_PIXEL_FORMAT_P010, 520, 200, 9},
                   &syntax, false));
  }
};

TEST_F(MediaVp9AdmissionTest, PreservesTheExactCodedAndPrecisionEnvelope) {
  amdf_media_vp9_configuration_t syntax = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_VP9_CONFIGURATION,
      .structure_size = sizeof(syntax),
      .profile = AMDF_MEDIA_VP9_PROFILE_2,
      .sample_bit_depth = 10,
  };
  amdf_media_configuration_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
      .structure_size = sizeof(query),
      .next = &syntax,
      .queue_family_ordinal = family_,
      .configuration = configuration_.configuration,
  };
  auto info = configuration_;
  EXPECT_EQ(info.configuration.coded_height, 200u);
  EXPECT_GE(info.reference.description.height, 256u);
  query.configuration.format = AMDF_MEDIA_PIXEL_FORMAT_NV12;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_FORMAT);
  EXPECT_EQ(info.output.byte_length, 0u);
  query.configuration.format = AMDF_MEDIA_PIXEL_FORMAT_P010;
  syntax.sample_bit_depth = 12;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);
  syntax.sample_bit_depth = 10;
  query.configuration.coded_height = 201;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS);
  EXPECT_EQ(info.configuration.coded_height, 201u);
  const auto preceding = info;
  query.next = nullptr;
  EXPECT_EQ(media_->device_query_configuration(device_, &query, &info),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(std::memcmp(&preceding, &info, sizeof(info)), 0);
}

}  // namespace
