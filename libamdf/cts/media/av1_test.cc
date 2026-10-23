// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>

#include "libamdf/cts/media/av1_fixture.h"
#include "libamdf/cts/media/decode_fixture.h"

namespace {

struct Av1Case {
  // Pass explicit planes instead of a native image identity.
  bool linear;
  // Embedded stream selecting precision and coded geometry.
  const char* filename;
  // Stable conformance selector for required-test qualification.
  const char* name;
};

class MediaAv1Test : public amdf::cts::media::MediaDecodeTest,
                     public ::testing::WithParamInterface<Av1Case> {
 protected:
  void SetUp() override {
    const auto& test = GetParam();
    ASSERT_NO_FATAL_FAILURE(
        amdf::cts::media::LoadAv1Fixture(test.filename, &fixture_));
    const amdf_media_av1_configuration_t syntax = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION,
        .structure_size = sizeof(syntax),
        .profile = 0,
        .sample_bit_depth = fixture_.depth,
    };
    ASSERT_NO_FATAL_FAILURE(
        SetUpMedia({AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_DIRECTION_DECODE,
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
  amdf::cts::media::Av1Fixture fixture_ = {};
  // Submission views bound to the application's stable reference images.
  std::array<amdf::cts::media::DecodeInput, 12> inputs_ = {};
};

TEST_P(MediaAv1Test, ReusesReferencesWithIndependentOutputAndGrain) {
  ASSERT_NO_FATAL_FAILURE(RunPictures(inputs_, 24));
}

TEST_P(MediaAv1Test, RejectsInvalidMapTilesAndGrainBeforeReplacingRecording) {
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
  picture.context_update_tile =
      picture.tile_column_count * picture.tile_row_count;
  EXPECT_EQ(PreparePicture(inputs_[1], 1),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  picture = preceding;
  picture.sequence.film_grain = 1;
  picture.film_grain.apply = 1;
  picture.film_grain.luma_point_count = 15;
  EXPECT_EQ(PreparePicture(inputs_[1], 1),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
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
    PrecisionAndStorage, MediaAv1Test,
    ::testing::Values(
        Av1Case{false, "av1_baseline_8.packet", "Baseline8Image"},
        Av1Case{true, "av1_baseline_8.packet", "Baseline8Planes"},
        Av1Case{false, "av1_baseline_10.packet", "Baseline10Image"},
        Av1Case{true, "av1_baseline_10.packet", "Baseline10Planes"},
        Av1Case{false, "av1_extended_8.packet", "Extended8Image"},
        Av1Case{true, "av1_extended_8.packet", "Extended8Planes"},
        Av1Case{false, "av1_extended_10.packet", "Extended10Image"},
        Av1Case{true, "av1_extended_10.packet", "Extended10Planes"},
        Av1Case{false, "av1_tools_8.packet", "Tools8Image"},
        Av1Case{true, "av1_tools_10.packet", "Tools10Planes"},
        Av1Case{false, "av1_lossless_8.packet", "Lossless8MinimumImage"},
        Av1Case{true, "av1_lossless_10.packet", "Lossless10MinimumPlanes"}),
    [](const ::testing::TestParamInfo<Av1Case>& parameter) {
      return std::string(parameter.param.name);
    });

class MediaAv1AdmissionTest : public amdf::cts::media::MediaDecodeTest {
 protected:
  void SetUp() override {
    const amdf_media_av1_configuration_t syntax = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION,
        .structure_size = sizeof(syntax),
        .profile = 0,
        .sample_bit_depth = 10,
    };
    ASSERT_NO_FATAL_FAILURE(
        SetUpMedia({AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_DIRECTION_DECODE,
                    AMDF_MEDIA_PIXEL_FORMAT_P010, 520, 200, 9},
                   &syntax, false));
  }
};

TEST_F(MediaAv1AdmissionTest, PreservesTheExactCodedAndPrecisionEnvelope) {
  amdf_media_av1_configuration_t syntax = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION,
      .structure_size = sizeof(syntax),
      .profile = 0,
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
