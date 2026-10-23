// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "amdf/media/hevc.h"

#include <array>
#include <cstring>

#include "libamdf/cts/interop/gpu/media/decode_fixture.h"
#include "libamdf/cts/media/hevc_fixture.h"

namespace {

class MediaHevcInteropTest : public MediaDecodeInteropTest {
 protected:
  void RunHevc(uint32_t depth, uint32_t frame_count, uint32_t recording_count,
               amdf_media_hevc_profile_t profile = 0) {
    const amdf_media_hevc_configuration_t hevc = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION,
        .structure_size = sizeof(hevc),
        .profile = profile != 0 ? profile
                   : depth == 8 ? AMDF_MEDIA_HEVC_PROFILE_MAIN
                                : AMDF_MEDIA_HEVC_PROFILE_MAIN10,
        .sample_bit_depth = depth,
    };
    ASSERT_NO_FATAL_FAILURE(
        CreateSession({AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_DIRECTION_DECODE,
                       static_cast<amdf_media_pixel_format_t>(
                           depth == 8 ? AMDF_MEDIA_PIXEL_FORMAT_NV12
                                      : AMDF_MEDIA_PIXEL_FORMAT_P010),
                       256, 200, 5},
                      &hevc));
    amdf::cts::media::HevcFixture fixture = {};
    ASSERT_NO_FATAL_FAILURE(amdf::cts::media::LoadHevcFixture(depth, &fixture));
    std::array<DecodePictureInput, 12> inputs = {};
    for (uint32_t i = 0; i < fixture.pictures.size(); ++i) {
      auto& entry = fixture.pictures[i];
      for (uint32_t reference = 0; reference < entry.picture.reference_count;
           ++reference) {
        auto& binding = entry.references[reference].binding;
        binding.frame = &reference_frames_[binding.slot_index];
      }
      inputs[i] = {entry.bytes, entry.byte_length, entry.reference_slot,
                   &entry.picture, entry.pixel_hash};
    }
    ASSERT_NO_FATAL_FAILURE(RunPictures(inputs, frame_count, recording_count));
  }
};

TEST_F(MediaHevcInteropTest, MainReferencesProduceExactGpuSamples) {
  ASSERT_NO_FATAL_FAILURE(RunHevc(8, 12, 12));
}

TEST_F(MediaHevcInteropTest, Main10ReferencesPreserveAllTenBits) {
  ASSERT_NO_FATAL_FAILURE(RunHevc(10, 12, 12));
}

TEST_F(MediaHevcInteropTest, ReusesRecordingsAndReferencesAcrossRandomAccess) {
  ASSERT_NO_FATAL_FAILURE(RunHevc(10, 32, 4));
}

TEST_F(MediaHevcInteropTest, Main10ProfileAdmitsCompatibleEightBitSyntax) {
  ASSERT_NO_FATAL_FAILURE(RunHevc(8, 12, 12, AMDF_MEDIA_HEVC_PROFILE_MAIN10));
}

TEST_F(MediaHevcInteropTest, AdmissionDoesNotConvertPrecisionOrPadDimensions) {
  amdf_media_hevc_configuration_t hevc = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION,
      .structure_size = sizeof(hevc),
      .profile = AMDF_MEDIA_HEVC_PROFILE_MAIN10,
      .sample_bit_depth = 10,
  };
  amdf_media_configuration_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
      .structure_size = sizeof(query),
      .next = &hevc,
      .configuration = {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_DIRECTION_DECODE,
                        AMDF_MEDIA_PIXEL_FORMAT_P010, 256, 200, 5},
  };
  amdf_media_configuration_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            AMDF_STATUS_OK);
  ASSERT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED);

  query.configuration.format = AMDF_MEDIA_PIXEL_FORMAT_NV12;
  ASSERT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_FORMAT);
  EXPECT_EQ(info.output.byte_length, 0u);

  query.configuration.format = AMDF_MEDIA_PIXEL_FORMAT_P010;
  hevc.profile = AMDF_MEDIA_HEVC_PROFILE_MAIN;
  ASSERT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);

  hevc.profile = AMDF_MEDIA_HEVC_PROFILE_MAIN10;
  query.configuration.coded_width = 258;
  ASSERT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS);
  EXPECT_EQ(info.configuration.coded_width, 258u);

  // An invalid schema is a failed call, distinct from a valid unsupported
  // envelope. Failure preserves the preceding query result.
  const auto preceding = info;
  query.next = nullptr;
  EXPECT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(std::memcmp(&preceding, &info, sizeof(info)), 0);
}

}  // namespace
