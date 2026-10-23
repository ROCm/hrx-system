// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <vector>

#include "libamdf/cts/interop/gpu/media/decode_fixture.h"
#include "libamdf/cts/media/av1_fixture.h"

namespace {

class MediaAv1InteropTest : public MediaDecodeInteropTest {
 protected:
  void RunAv1(const char* filename) {
    amdf::cts::media::Av1Fixture fixture = {};
    ASSERT_NO_FATAL_FAILURE(
        amdf::cts::media::LoadAv1Fixture(filename, &fixture));
    const uint32_t depth = fixture.depth;
    const amdf_media_av1_configuration_t syntax = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION,
        .structure_size = sizeof(syntax),
        .profile = 0,
        .sample_bit_depth = depth,
    };
    ASSERT_NO_FATAL_FAILURE(
        CreateSession({AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_DIRECTION_DECODE,
                       static_cast<amdf_media_pixel_format_t>(
                           depth == 8 ? AMDF_MEDIA_PIXEL_FORMAT_NV12
                                      : AMDF_MEDIA_PIXEL_FORMAT_P010),
                       fixture.width, fixture.height, 9},
                      &syntax));
    std::array<DecodePictureInput, 12> inputs = {};
    for (uint32_t i = 0; i < fixture.pictures.size(); ++i) {
      auto& entry = fixture.pictures[i];
      for (uint32_t r = 0; r < entry.picture.reference_count; ++r) {
        auto& binding = entry.references[r];
        binding.frame = &reference_frames_[binding.slot_index];
      }
      inputs[i] = {entry.bytes, entry.byte_length, entry.reference_slot,
                   &entry.picture, entry.pixel_hash};
    }
    ASSERT_NO_FATAL_FAILURE(RunPictures(inputs, 24, 4));
  }
};

TEST_F(MediaAv1InteropTest, Baseline8ProducesExactGpuSamples) {
  ASSERT_NO_FATAL_FAILURE(RunAv1("av1_baseline_8.packet"));
}

TEST_F(MediaAv1InteropTest, Baseline10ProducesExactGpuSamples) {
  ASSERT_NO_FATAL_FAILURE(RunAv1("av1_baseline_10.packet"));
}

TEST_F(MediaAv1InteropTest, Extended8ProducesExactGpuSamples) {
  ASSERT_NO_FATAL_FAILURE(RunAv1("av1_extended_8.packet"));
}

TEST_F(MediaAv1InteropTest, Extended10ProducesExactGpuSamples) {
  ASSERT_NO_FATAL_FAILURE(RunAv1("av1_extended_10.packet"));
}

TEST_F(MediaAv1InteropTest, SegmentationAndMatricesPreserveEightBitSamples) {
  ASSERT_NO_FATAL_FAILURE(RunAv1("av1_tools_8.packet"));
}

TEST_F(MediaAv1InteropTest, SegmentationAndMatricesPreserveTenBitSamples) {
  ASSERT_NO_FATAL_FAILURE(RunAv1("av1_tools_10.packet"));
}

TEST_F(MediaAv1InteropTest, LosslessEightBitPreservesEverySample) {
  ASSERT_NO_FATAL_FAILURE(RunAv1("av1_lossless_crop_8.packet"));
}

TEST_F(MediaAv1InteropTest, LosslessTenBitPreservesEverySample) {
  ASSERT_NO_FATAL_FAILURE(RunAv1("av1_lossless_crop_10.packet"));
}

TEST_F(MediaAv1InteropTest, AdmissionSeparatesStoragePaddingAndCodedExtent) {
  uint32_t count = 0;
  ASSERT_EQ(
      media_->device_enumerate_codecs(media_device_, 0, 0, nullptr, &count),
      AMDF_STATUS_OK);
  ASSERT_GT(count, 1u);
  std::vector<amdf_media_codec_info_t> codecs(
      count - 1, {AMDF_MEDIA_CODEC_JPEG, AMDF_MEDIA_DIRECTION_ENCODE});
  const auto preceding_codecs = codecs;
  EXPECT_EQ(media_->device_enumerate_codecs(
                media_device_, 0, static_cast<uint32_t>(codecs.size()),
                codecs.data(), &count),
            amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
  EXPECT_EQ(std::memcmp(preceding_codecs.data(), codecs.data(),
                        codecs.size() * sizeof(codecs[0])),
            0);
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
      .configuration = {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_DIRECTION_DECODE,
                        AMDF_MEDIA_PIXEL_FORMAT_P010, 520, 200, 9},
  };
  amdf_media_configuration_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            AMDF_STATUS_OK);
  ASSERT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED);
  EXPECT_EQ(info.configuration.coded_height, 200u);
  EXPECT_GE(info.reference.description.height, 208u);
  const auto preceding = info;
  query.next = nullptr;
  EXPECT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(std::memcmp(&preceding, &info, sizeof(info)), 0);
  query.next = &syntax;
  syntax.sample_bit_depth = 12;
  ASSERT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);
  EXPECT_EQ(info.output.byte_length, 0u);
}

}  // namespace
