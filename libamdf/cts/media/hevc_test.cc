// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>

#include "libamdf/cts/media/decode_fixture.h"
#include "libamdf/cts/media/hevc_fixture.h"

namespace {

struct HevcCase {
  // Pass explicit planes instead of a native image identity.
  bool linear;
  // Actual luma/chroma precision, independent of the profile declaration.
  uint32_t depth;
  // Declared syntax envelope; Main10 can carry eight-bit syntax.
  amdf_media_hevc_profile_t profile;
};

class MediaHevcTest : public amdf::cts::media::MediaDecodeTest,
                      public ::testing::WithParamInterface<HevcCase> {
 protected:
  void SetUp() override {
    const auto& test = GetParam();
    const amdf_media_hevc_configuration_t syntax = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION,
        .structure_size = sizeof(syntax),
        .profile = test.profile,
        .sample_bit_depth = test.depth,
    };
    ASSERT_NO_FATAL_FAILURE(
        SetUpMedia({AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_DIRECTION_DECODE,
                    static_cast<amdf_media_pixel_format_t>(
                        test.depth == 8 ? AMDF_MEDIA_PIXEL_FORMAT_NV12
                                        : AMDF_MEDIA_PIXEL_FORMAT_P010),
                    256, 200, 5},
                   &syntax, test.linear));
    if (IsSkipped()) {
      return;
    }
    ASSERT_NO_FATAL_FAILURE(
        amdf::cts::media::LoadHevcFixture(test.depth, &fixture_));
    for (uint32_t index = 0; index < fixture_.pictures.size(); ++index) {
      auto& entry = fixture_.pictures[index];
      for (uint32_t reference = 0; reference < entry.picture.reference_count;
           ++reference) {
        auto& binding = entry.references[reference].binding;
        ASSERT_LT(binding.slot_index, references_.size());
        binding.frame = &references_[binding.slot_index];
      }
      inputs_[index] = {entry.bytes, entry.byte_length, entry.reference_slot,
                        &entry.picture, entry.pixel_hash};
    }
  }

  // Resolved sequence/picture syntax, nonuniform grids, and independent pixels.
  amdf::cts::media::HevcFixture fixture_ = {};
  // Borrowed submission views over the fixed parsed corpus.
  std::array<amdf::cts::media::DecodeInput, 12> inputs_ = {};
};

TEST_P(MediaHevcTest, ReusesRecordingsAndReferencesWithExactPrecision) {
  ASSERT_NO_FATAL_FAILURE(RunPictures(inputs_, 24));
}

INSTANTIATE_TEST_SUITE_P(
    PrecisionAndStorage, MediaHevcTest,
    ::testing::Values(HevcCase{false, 8, AMDF_MEDIA_HEVC_PROFILE_MAIN},
                      HevcCase{true, 8, AMDF_MEDIA_HEVC_PROFILE_MAIN},
                      HevcCase{false, 10, AMDF_MEDIA_HEVC_PROFILE_MAIN10},
                      HevcCase{true, 10, AMDF_MEDIA_HEVC_PROFILE_MAIN10},
                      HevcCase{false, 8, AMDF_MEDIA_HEVC_PROFILE_MAIN10}),
    [](const ::testing::TestParamInfo<HevcCase>& parameter) {
      const auto& test = parameter.param;
      return std::string(test.profile == AMDF_MEDIA_HEVC_PROFILE_MAIN ? "Main"
                         : test.depth == 10 ? "Main10"
                                            : "Main10EightBit") +
             (test.linear ? "Planes" : "Image");
    });

class MediaHevcAdmissionTest : public amdf::cts::media::MediaDecodeTest {
 protected:
  void SetUp() override {
    const amdf_media_hevc_configuration_t syntax = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION,
        .structure_size = sizeof(syntax),
        .profile = AMDF_MEDIA_HEVC_PROFILE_MAIN10,
        .sample_bit_depth = 10,
    };
    ASSERT_NO_FATAL_FAILURE(
        SetUpMedia({AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_DIRECTION_DECODE,
                    AMDF_MEDIA_PIXEL_FORMAT_P010, 256, 200, 5},
                   &syntax, false));
  }
};

TEST_F(MediaHevcAdmissionTest,
       UnsupportedTuplePreservesPrecisionAndDimensions) {
  uint32_t count = 0;
  ASSERT_EQ(
      media_->device_enumerate_codecs(device_, family_, 0, nullptr, &count),
      AMDF_STATUS_OK);
  if (count > 1) {
    // Capacity zero is a count query; a nonzero short array remains unchanged.
    std::vector<amdf_media_codec_info_t> codecs(
        count - 1, {AMDF_MEDIA_CODEC_JPEG, AMDF_MEDIA_DIRECTION_ENCODE});
    const auto preceding = codecs;
    EXPECT_EQ(media_->device_enumerate_codecs(
                  device_, family_, static_cast<uint32_t>(codecs.size()),
                  codecs.data(), &count),
              amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
    EXPECT_EQ(std::memcmp(preceding.data(), codecs.data(),
                          codecs.size() * sizeof(codecs[0])),
              0);
  }
  amdf_media_hevc_configuration_t syntax = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION,
      .structure_size = sizeof(syntax),
      .profile = AMDF_MEDIA_HEVC_PROFILE_MAIN10,
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
  query.configuration.format = AMDF_MEDIA_PIXEL_FORMAT_NV12;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_FORMAT);
  EXPECT_EQ(info.output.byte_length, 0u);
  query.configuration.format = AMDF_MEDIA_PIXEL_FORMAT_P010;
  syntax.profile = AMDF_MEDIA_HEVC_PROFILE_MAIN;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);
  syntax.profile = AMDF_MEDIA_HEVC_PROFILE_MAIN10;
  query.configuration.coded_width = 258;
  ASSERT_EQ(media_->device_query_configuration(device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS);
  EXPECT_EQ(info.configuration.coded_width, 258u);
  const auto preceding = info;
  query.next = nullptr;
  EXPECT_EQ(media_->device_query_configuration(device_, &query, &info),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(std::memcmp(&preceding, &info, sizeof(info)), 0);
}

TEST_F(MediaHevcAdmissionTest, RejectedPreparationPreservesExecutablePicture) {
  ASSERT_NO_FATAL_FAILURE(CreatePictures());
  amdf::cts::media::HevcFixture fixture = {};
  ASSERT_NO_FATAL_FAILURE(amdf::cts::media::LoadHevcFixture(10, &fixture));
  std::array<amdf::cts::media::DecodeInput, 4> inputs;
  for (uint32_t index = 0; index < inputs.size(); ++index) {
    auto& entry = fixture.pictures[index];
    for (uint32_t reference = 0; reference < entry.picture.reference_count;
         ++reference) {
      auto& binding = entry.references[reference].binding;
      binding.frame = &references_[binding.slot_index];
    }
    inputs[index] = {entry.bytes, entry.byte_length, entry.reference_slot,
                     &entry.picture, entry.pixel_hash};
    ASSERT_EQ(PreparePicture(inputs[index], index), AMDF_STATUS_OK);
  }
  // The one-slice recording must reject this count before walking its array.
  // Only the preceding complete recording may subsequently reach the engine.
  auto& picture = fixture.pictures[0].picture;
  picture.slice_count = 2;
  EXPECT_EQ(PreparePicture(inputs[0], 0),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  picture.slice_count = 1;
  ASSERT_NO_FATAL_FAILURE(Cache(input_, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(
      Cache(output_memory_, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(Begin());
  for (auto* command : commands_) {
    ASSERT_EQ(Submit(command), AMDF_STATUS_OK);
  }
  ASSERT_NO_FATAL_FAILURE(Join());
  ASSERT_NO_FATAL_FAILURE(CheckPixels(inputs, 0));
  ASSERT_NO_FATAL_FAILURE(End());
}

}  // namespace
