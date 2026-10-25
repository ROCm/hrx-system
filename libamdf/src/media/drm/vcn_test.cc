// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/vcn.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/media/drm/queue.h"

namespace {

TEST(VcnProtocolTest, SelectsGenerationFactsWithoutReinterpretingVersions) {
  struct ProtocolCase {
    uint32_t version;
    amdf_media_drm_command_protocol_t command_protocol;
    amdf_media_drm_av1_protocol_t av1_protocol;
    uint32_t decode_array_mode;
    uint32_t av1_encode_width_alignment;
    uint32_t av1_encode_height_alignment;
    uint32_t decode_reference_swizzle_mode;
    uint32_t encode_reconstruction_swizzle_mode;
    uint32_t reference_height_alignment;
  };
  constexpr ProtocolCase kCases[] = {
      {0x040000, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN4,
       AMDF_MEDIA_DRM_AV1_PROTOCOL_V1, 3, 64, 16, 0, 0, 32},
      {0x040003, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN4,
       AMDF_MEDIA_DRM_AV1_PROTOCOL_V1, 1, 64, 16, 0, 0, 32},
      {0x040006, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN4,
       AMDF_MEDIA_DRM_AV1_PROTOCOL_V1, 3, 64, 16, 0, 0, 32},
      {0x050000, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN5,
       AMDF_MEDIA_DRM_AV1_PROTOCOL_V2, 3, 8, 2,
       AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D,
       AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D, 64},
      {0x050001, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN5,
       AMDF_MEDIA_DRM_AV1_PROTOCOL_V2, 1, 8, 2,
       AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D,
       AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D, 64},
      {0x050300, AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN5,
       AMDF_MEDIA_DRM_AV1_PROTOCOL_V2, 3, 8, 2,
       AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D,
       AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D, 64},
  };
  for (const ProtocolCase& test_case : kCases) {
    SCOPED_TRACE(test_case.version);
    amdf_media_drm_vcn_protocol_info_t protocol = {};
    ASSERT_TRUE(amdf_media_drm_vcn_protocol(test_case.version, &protocol));
    EXPECT_EQ(protocol.command_protocol, test_case.command_protocol);
    EXPECT_EQ(protocol.av1_protocol, test_case.av1_protocol);
    EXPECT_EQ(protocol.decode_array_mode, test_case.decode_array_mode);
    EXPECT_EQ(protocol.av1_encode_width_alignment,
              test_case.av1_encode_width_alignment);
    EXPECT_EQ(protocol.av1_encode_height_alignment,
              test_case.av1_encode_height_alignment);
    EXPECT_EQ(protocol.decode_reference_swizzle_mode,
              test_case.decode_reference_swizzle_mode);
    EXPECT_EQ(protocol.encode_reconstruction_swizzle_mode,
              test_case.encode_reconstruction_swizzle_mode);
    EXPECT_EQ(protocol.reference_height_alignment,
              test_case.reference_height_alignment);
  }

  amdf_media_drm_vcn_protocol_info_t protocol = {
      .command_protocol = AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_DIRECT,
      .av1_protocol = AMDF_MEDIA_DRM_AV1_PROTOCOL_V2,
      .decode_array_mode = 7,
      .av1_encode_width_alignment = 32,
      .av1_encode_height_alignment = 8,
      .decode_reference_swizzle_mode = 9,
      .encode_reconstruction_swizzle_mode = 10,
      .reference_height_alignment = 128,
  };
  EXPECT_FALSE(amdf_media_drm_vcn_protocol(0x030000, &protocol));
  EXPECT_EQ(protocol.command_protocol,
            AMDF_MEDIA_DRM_COMMAND_PROTOCOL_JPEG_DIRECT);
  EXPECT_EQ(protocol.av1_protocol, AMDF_MEDIA_DRM_AV1_PROTOCOL_V2);
  EXPECT_EQ(protocol.decode_array_mode, 7u);
  EXPECT_EQ(protocol.av1_encode_width_alignment, 32u);
  EXPECT_EQ(protocol.av1_encode_height_alignment, 8u);
  EXPECT_EQ(protocol.decode_reference_swizzle_mode, 9u);
  EXPECT_EQ(protocol.encode_reconstruction_swizzle_mode, 10u);
  EXPECT_EQ(protocol.reference_height_alignment, 128u);
}

TEST(VcnProtocolTest, ReferenceGeometryTracksTheSelectedGeneration) {
  amdf_media_drm_device_t device = {
      .address_begin = 4096,
      .address_end = UINT64_C(1) << 47,
  };
  const amdf_media_image_description_t reference = {
      .format = AMDF_MEDIA_PIXEL_FORMAT_NV12,
      .width = 256,
      .height = 96,
      .array_layer_count = 1,
      .roles = AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE,
  };
  amdf_media_image_requirements_t requirements = {};

  device.decode_reference_height_alignment = 32;
  ASSERT_EQ(amdf_media_drm_image_requirements(&device.base.base, &reference,
                                              &requirements),
            AMDF_STATUS_OK);
  EXPECT_EQ(requirements.byte_length, UINT64_C(256) * (96 + 48));

  device.decode_reference_height_alignment = 64;
  ASSERT_EQ(amdf_media_drm_image_requirements(&device.base.base, &reference,
                                              &requirements),
            AMDF_STATUS_OK);
  EXPECT_EQ(requirements.byte_length, UINT64_C(256) * (128 + 64));

  amdf_media_image_description_t output = reference;
  output.roles = AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT;
  ASSERT_EQ(amdf_media_drm_image_requirements(&device.base.base, &output,
                                              &requirements),
            AMDF_STATUS_OK);
  EXPECT_EQ(requirements.byte_length, UINT64_C(256) * (96 + 48));
}

TEST(VcnProtocolTest, TiledReferencesCannotAlsoBeLinearOutputs) {
  amdf_media_drm_device_t device = {
      .address_begin = 4096,
      .address_end = UINT64_C(1) << 47,
      .decode_reference_height_alignment = 64,
  };
  amdf_media_image_description_t description = {
      .format = AMDF_MEDIA_PIXEL_FORMAT_NV12,
      .width = 320,
      .height = 208,
      .array_layer_count = 1,
      .roles = AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT |
               AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE,
  };
  amdf_media_image_requirements_t requirements = {};

  device.decode_reference_swizzle_mode =
      AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D;
  EXPECT_EQ(amdf_media_drm_image_requirements(&device.base.base, &description,
                                              &requirements),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));

  device.decode_reference_swizzle_mode = 0;
  device.decode_reference_height_alignment = 32;
  ASSERT_EQ(amdf_media_drm_image_requirements(&device.base.base, &description,
                                              &requirements),
            AMDF_STATUS_OK);
  EXPECT_EQ(requirements.byte_length, UINT64_C(512) * (224 + 112));
}

TEST(VcnProtocolTest, TiledReferencesExposeOnlyTheirNativeIdentity) {
  amdf_memory_t memory = {};
  memory.info.access_count = 1;
  amdf_media_drm_image_t image = {
      .requirements =
          {
              .description =
                  {
                      .format = AMDF_MEDIA_PIXEL_FORMAT_NV12,
                      .width = 256,
                      .height = 256,
                      .array_layer_count = 1,
                      .roles = AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE,
                  },
              .byte_length = 98304,
              .byte_offset_alignment = 256,
          },
      .memory = &memory,
      .byte_offset = 256,
      .swizzle_mode = AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D,
  };
  const amdf_media_image_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY,
      .structure_size = sizeof(query),
  };
  amdf_media_linear_image_layout_t layout = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_LINEAR_IMAGE_LAYOUT,
      .structure_size = sizeof(layout),
  };
  amdf_media_image_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO,
      .structure_size = sizeof(info),
      .next = &layout,
  };
  EXPECT_EQ(amdf_media_drm_image_query_info(&image.base, 0, &query, &info),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  info.next = nullptr;
  ASSERT_EQ(amdf_media_drm_image_query_info(&image.base, 0, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.memory, &memory);
  EXPECT_EQ(info.byte_offset, 256u);
  EXPECT_EQ(info.requirements.byte_length, 98304u);
}

TEST(VcnProtocolTest, TiledReferencesRejectLinearPlaneBindings) {
  amdf_media_drm_device_t device = {
      .decode_reference_swizzle_mode =
          AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D,
  };
  const amdf_media_frame_plane_t planes[2] = {};
  const amdf_media_frame_t frame = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_FRAME,
      .structure_size = sizeof(frame),
      .format = AMDF_MEDIA_PIXEL_FORMAT_NV12,
      .width = 256,
      .height = 256,
      .storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_LINEAR_PLANES,
      .storage = {.linear = {2, planes}},
  };
  const amdf_media_image_description_t description = {
      .format = AMDF_MEDIA_PIXEL_FORMAT_NV12,
      .width = 256,
      .height = 256,
      .array_layer_count = 1,
      .roles = AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE,
  };
  amdf_media_drm_frame_t prepared = {};
  EXPECT_EQ(
      amdf_media_drm_frame_prepare(&device, &frame, &description, &prepared),
      amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
}

template <size_t N>
void ExpectWords(const uint32_t* actual,
                 const std::array<uint32_t, N>& expected) {
  for (size_t i = 0; i < N; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i], expected[i]);
  }
}

class VcnDecodePacketTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_.engines[0].command_protocol = AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN5;
    device_.engines[0].decode_array_mode = 3;
    queue_.base.device = &device_.base.base;
    queue_.base.info.queue_family_ordinal = 0;
    session_.base.queue = &queue_.base;
    session_.storage = &state_;
    session_.requirements.configuration = {AMDF_MEDIA_CODEC_H264,
                                           AMDF_MEDIA_DIRECTION_DECODE,
                                           AMDF_MEDIA_PIXEL_FORMAT_NV12,
                                           256,
                                           96,
                                           17};
    session_.codec.h264.profile = AMDF_MEDIA_H264_PROFILE_CONSTRAINED_BASELINE;
    session_.codec.h264.level_idc = 13;
    session_.context_byte_length = 0x3456;
    session_.stream_handle = 0x89abcdef;
    state_.address = kSessionAddress;
    command_.address = kCommandAddress;
    command_.host_pointer = words_.data();

    output_.addresses[0] = kOutputAddress;
    output_.addresses[1] = kOutputAddress + 512 * 96;
    output_.row_pitches[0] = 512;
    output_.row_pitches[1] = 512;
    output_.row_counts[0] = 96;
    output_.row_counts[1] = 48;
    SetReferenceFrame(kCurrentAddress, &current_);
    for (uint32_t i = 0; i < references_.size(); ++i) {
      SetReferenceFrame(kReferenceAddress + i * 0x20000, &references_[i]);
      picture_references_[i].binding.slot_index = i;
    }
    std::memset(&matrices_, 0x5a, sizeof(matrices_));
    picture_.scaling_matrices = &matrices_;
    picture_.references = picture_references_.data();
  }

  static void SetReferenceFrame(uint64_t address,
                                amdf_media_drm_frame_t* frame) {
    frame->addresses[0] = address;
    frame->addresses[1] = address + 512 * 128;
    frame->row_pitches[0] = 512;
    frame->row_pitches[1] = 512;
    frame->row_counts[0] = 128;
    frame->row_counts[1] = 64;
  }

  uint32_t Emit(amdf_media_drm_command_protocol_t protocol,
                uint32_t reference_count, uint32_t current_slot) {
    words_.fill(0xcdcdcdcd);
    device_.engines[0].command_protocol = protocol;
    const uint32_t swizzle_mode =
        protocol == AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN5
            ? AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D
            : 0;
    current_.swizzle_mode = swizzle_mode;
    for (auto& reference : references_) {
      reference.swizzle_mode = swizzle_mode;
    }
    picture_.reference_count = reference_count;
    return amdf_media_drm_vcn_h264(&command_, &session_, kCompressedAddress,
                                   2048, &output_, &current_,
                                   references_.data(), current_slot, &picture_);
  }

  static constexpr uint64_t kCommandAddress = UINT64_C(0x1234567800000000);
  static constexpr uint64_t kSessionAddress = UINT64_C(0x2345678900000000);
  static constexpr uint64_t kCompressedAddress = UINT64_C(0x3456789a00001000);
  static constexpr uint64_t kOutputAddress = UINT64_C(0x456789ab00000000);
  static constexpr uint64_t kCurrentAddress = UINT64_C(0x56789abc00000000);
  static constexpr uint64_t kReferenceAddress = UINT64_C(0x6789abcd00000000);
  std::array<uint32_t, 1024> words_ = {};
  amdf_media_drm_buffer_t command_ = {};
  amdf_media_drm_buffer_t state_ = {};
  amdf_media_drm_device_t device_ = {};
  amdf_media_drm_queue_t queue_ = {};
  amdf_media_drm_session_t session_ = {};
  amdf_media_drm_frame_t output_ = {};
  amdf_media_drm_frame_t current_ = {};
  std::array<amdf_media_drm_frame_t, 16> references_ = {};
  std::array<amdf_media_h264_reference_t, 16> picture_references_ = {};
  amdf_media_h264_scaling_matrices_t matrices_ = {};
  amdf_media_h264_decode_picture_info_t picture_ = {};
};

TEST_F(VcnDecodePacketTest, Vcn5UsesMesaTier2MessageAndFixedPacketExtent) {
  picture_references_[0].binding.slot_index = 2;
  ASSERT_EQ(Emit(AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN5, 1, 7), 156u);
  EXPECT_EQ(words_[3], 156u);
  EXPECT_EQ(words_[6], 0x00100a0du);
  EXPECT_EQ(words_[39], 0xcdcdcdcdu);
  const uint32_t* message =
      words_.data() + AMDF_MEDIA_DRM_VCN_MESSAGE_OFFSET / 4;
  constexpr std::array<uint32_t, 18> kExpectedHeader = {
      40, 1552, 3,   1,    0x89abcdef, 0,    2,   72,  188,
      0,  6,    428, 1124, 0,          0x10, 260, 168, 0};
  ExpectWords(message, kExpectedHeader);
  EXPECT_EQ(message[18], 7u);
  EXPECT_EQ(message[19], 1u);
  EXPECT_EQ(message[18 + 17], 0xa0du);
  EXPECT_EQ(message[18 + 21], 1u);
  const uint32_t* dynamic = message + 18 + 47;
  EXPECT_EQ(dynamic[1], 512u);
  EXPECT_EQ(dynamic[2], 128u);
  EXPECT_EQ(dynamic[7], 1u);
  EXPECT_EQ(dynamic[8], static_cast<uint32_t>(kCurrentAddress));
  EXPECT_EQ(dynamic[9], kCurrentAddress >> 32);
  EXPECT_EQ(dynamic[10], static_cast<uint32_t>(kReferenceAddress));
  EXPECT_EQ(dynamic[26], kReferenceAddress >> 32);
  EXPECT_EQ(words_[AMDF_MEDIA_DRM_VCN_SCALING_OFFSET / 4], 0x5a5a5a5au);

  const auto vcn5_words = words_;
  ASSERT_EQ(Emit(AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN4, 1, 7), 156u);
  EXPECT_EQ(words_[AMDF_MEDIA_DRM_VCN_MESSAGE_OFFSET / 4 + 18 + 21], 0u);
  words_[AMDF_MEDIA_DRM_VCN_MESSAGE_OFFSET / 4 + 18 + 21] =
      AMDF_MEDIA_DRM_VCN_REFERENCE_SWIZZLE_256B_D;
  EXPECT_EQ(words_, vcn5_words);
}

}  // namespace
