// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/vcn_encode.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "libamdf/src/media/drm/queue.h"

namespace {

uint64_t Fnv1a(const void* data, uint32_t byte_length) {
  uint64_t hash = UINT64_C(0xcbf29ce484222325);
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (uint32_t i = 0; i < byte_length; ++i) {
    hash = (hash ^ bytes[i]) * UINT64_C(0x100000001b3);
  }
  return hash;
}

struct PacketBlock {
  const uint32_t* payload = nullptr;
  uint32_t payload_word_count = 0;
};

PacketBlock FindPacketBlock(const uint32_t* words, uint32_t byte_length,
                            uint32_t operation) {
  for (uint32_t index = 15; index < byte_length / sizeof(uint32_t);) {
    const uint32_t block_byte_length = words[index];
    if (block_byte_length < 2 * sizeof(uint32_t) ||
        block_byte_length % sizeof(uint32_t) != 0 ||
        index + block_byte_length / sizeof(uint32_t) >
            byte_length / sizeof(uint32_t)) {
      ADD_FAILURE() << "malformed packet block at word " << index;
      return {};
    }
    if (words[index + 1] == operation) {
      return {
          words + index + 2,
          static_cast<uint32_t>(block_byte_length / sizeof(uint32_t) - 2),
      };
    }
    index += block_byte_length / sizeof(uint32_t);
  }
  ADD_FAILURE() << "missing packet operation 0x" << std::hex << operation;
  return {};
}

class VcnEncodePacketTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_.engines[0].information.ip_discovery_version = 0x040006;
    device_.engines[0].command_protocol = AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN4;
    queue_.base.device = &device_.base.base;
    queue_.base.info.queue_family_ordinal = 0;
    session_.base.queue = &queue_.base;
    session_.storage = &state_;
    state_.address = kSessionAddress;
    command_.host_pointer = words_;
    session_.requirements.configuration = {AMDF_MEDIA_CODEC_H264,
                                           AMDF_MEDIA_DIRECTION_ENCODE,
                                           AMDF_MEDIA_PIXEL_FORMAT_NV12,
                                           320,
                                           192,
                                           2};
    session_.encode.rate_control.mode = AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP;
    session_.encode.rate_control.frame_rate_numerator = 30;
    session_.encode.rate_control.frame_rate_denominator = 1;
    session_.encode.rate_control.control.cqp = {20, 20};
    resources_.input.addresses[0] = kSourceAddress;
    resources_.input.addresses[1] = kSourceAddress + 512 * 192;
    resources_.input.row_pitches[0] = 512;
    resources_.input.row_pitches[1] = 512;
    resources_.reconstruction_address = kReconstructionAddress;
    resources_.reconstruction_layer_byte_length = 512 * 256 * 3 / 2;
    resources_.reconstruction_luma_byte_length = 512 * 256;
    resources_.reconstruction_row_pitch = 512;
    resources_.reconstruction_slot_count = 2;
    resources_.compressed_address = kCompressedAddress;
    resources_.compressed_byte_length = 131072;
    resources_.feedback_address = kFeedbackAddress;
  }

  void ConfigureSampleBitDepth(uint32_t sample_bit_depth) {
    const uint32_t sample_byte_count = sample_bit_depth == 10 ? 2 : 1;
    session_.requirements.configuration.format =
        sample_bit_depth == 10 ? AMDF_MEDIA_PIXEL_FORMAT_P010
                               : AMDF_MEDIA_PIXEL_FORMAT_NV12;
    resources_.input.addresses[1] =
        kSourceAddress + 512 * 192 * sample_byte_count;
    resources_.input.row_pitches[0] = 512 * sample_byte_count;
    resources_.input.row_pitches[1] = 512 * sample_byte_count;
    resources_.reconstruction_layer_byte_length =
        512 * 256 * 3 / 2 * sample_byte_count;
    resources_.reconstruction_luma_byte_length = 512 * 256 * sample_byte_count;
    resources_.reconstruction_row_pitch = 512 * sample_byte_count;
  }

  void ConfigureVcn5() {
    device_.engines[0].information.ip_discovery_version = 0x050000;
    device_.engines[0].command_protocol = AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN5;
  }

  static constexpr uint64_t kSessionAddress = UINT64_C(0x123400000000);
  static constexpr uint64_t kReconstructionAddress = UINT64_C(0x223400000000);
  static constexpr uint64_t kSourceAddress = UINT64_C(0x323400000000);
  static constexpr uint64_t kCompressedAddress = UINT64_C(0x423400000000);
  static constexpr uint64_t kFeedbackAddress = UINT64_C(0x523400000000);
  uint32_t words_[2048] = {};
  amdf_media_drm_buffer_t command_ = {};
  amdf_media_drm_buffer_t state_ = {};
  amdf_media_drm_device_t device_ = {};
  amdf_media_drm_queue_t queue_ = {};
  amdf_media_drm_session_t session_ = {};
  amdf_media_drm_vcn_encode_resources_t resources_ = {};
};

TEST_F(VcnEncodePacketTest, H264MatchesQualifiedNativePackets) {
  session_.codec.h264.level_idc = 30;
  uint32_t length = amdf_media_drm_vcn_encode_control(
      &command_, &session_, AMDF_MEDIA_COMMAND_OPERATION_BEGIN);
  EXPECT_EQ(length, 316u);
  EXPECT_EQ(Fnv1a(words_, length), UINT64_C(0x891222953808d69c));

  amdf_media_h264_encode_picture_info_t picture = {
      .frame_type = AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR,
      .used_as_reference = 1,
  };
  length = amdf_media_drm_vcn_encode_h264(&command_, &session_, &resources_, 0,
                                          &picture);
  EXPECT_EQ(length, 1844u);
  EXPECT_EQ(Fnv1a(words_, length), UINT64_C(0xf594489fda3cbb27));

  amdf_media_frame_t frame = {};
  amdf_media_h264_encode_reference_t reference = {
      .binding = {&frame, 0},
      .frame_type = AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR,
  };
  picture.frame_type = AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_P;
  picture.frame_number = 1;
  picture.picture_order_count = 2;
  picture.frame_decoding_order_number = 1;
  picture.reference_count = 1;
  picture.references = &reference;
  length = amdf_media_drm_vcn_encode_h264(&command_, &session_, &resources_, 1,
                                          &picture);
  EXPECT_EQ(length, 1844u);
  EXPECT_EQ(Fnv1a(words_, length), UINT64_C(0xd2c52ce50dd373b8));

  length = amdf_media_drm_vcn_encode_control(&command_, &session_,
                                             AMDF_MEDIA_COMMAND_OPERATION_END);
  EXPECT_EQ(length, 68u);
  EXPECT_EQ(Fnv1a(words_, length), UINT64_C(0xf1375b730de6143c));
}

TEST_F(VcnEncodePacketTest, H264Vcn5UsesDirectTier2References) {
  ConfigureVcn5();
  session_.codec.h264.level_idc = 30;
  uint32_t length = amdf_media_drm_vcn_encode_control(
      &command_, &session_, AMDF_MEDIA_COMMAND_OPERATION_BEGIN);
  EXPECT_EQ(length, 312u);
  EXPECT_EQ(Fnv1a(words_, length), UINT64_C(0x93b78668cdf49f15));
  EXPECT_EQ(words_[6], 0x0001000au);

  amdf_media_h264_encode_picture_info_t picture = {
      .frame_type = AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR,
      .used_as_reference = 1,
  };
  length = amdf_media_drm_vcn_encode_h264(&command_, &session_, &resources_, 0,
                                          &picture);
  EXPECT_EQ(length, 5052u);
  EXPECT_EQ(Fnv1a(words_, length), UINT64_C(0x427dcee169434969));
  EXPECT_LE(length, sizeof(words_));

  PacketBlock block = FindPacketBlock(words_, length, 0x200002);
  ASSERT_NE(block.payload, nullptr);
  EXPECT_EQ(block.payload_word_count, 10u);
  EXPECT_EQ(block.payload[3], 0u);
  EXPECT_EQ(block.payload[4], 1u);
  EXPECT_EQ(block.payload[6], 77u);

  block = FindPacketBlock(words_, length, 0x0f);
  ASSERT_NE(block.payload, nullptr);
  EXPECT_EQ(block.payload_word_count, 10u);
  EXPECT_EQ(block.payload[8], 0u);
  EXPECT_EQ(block.payload[9], 0u);

  block = FindPacketBlock(words_, length, 0x200003);
  ASSERT_NE(block.payload, nullptr);
  ASSERT_EQ(block.payload_word_count, 75u);
  EXPECT_EQ(block.payload[2], 1u);
  EXPECT_EQ(block.payload[37], 0u);
  EXPECT_EQ(block.payload[71], 0u);
  EXPECT_EQ(block.payload[72], UINT32_MAX);
  EXPECT_EQ(block.payload[74], UINT32_MAX);

  block = FindPacketBlock(words_, length, 0x11);
  ASSERT_NE(block.payload, nullptr);
  ASSERT_EQ(block.payload_word_count, 1029u);
  EXPECT_EQ(block.payload[0], 0u);
  EXPECT_EQ(block.payload[1], 0u);
  EXPECT_EQ(block.payload[2], 2u);
  EXPECT_EQ(block.payload[3],
            static_cast<uint32_t>(kReconstructionAddress >> 32));
  EXPECT_EQ(block.payload[4], static_cast<uint32_t>(kReconstructionAddress));
  EXPECT_EQ(block.payload[5], 512u);
  EXPECT_EQ(block.payload[6],
            static_cast<uint32_t>(kReconstructionAddress >> 32));
  EXPECT_EQ(block.payload[7],
            static_cast<uint32_t>(kReconstructionAddress + 512 * 256));
  EXPECT_EQ(block.payload[8], 256u);
  EXPECT_EQ(block.payload[12], 0u);
  const uint64_t frame_context_address =
      kReconstructionAddress + 2 * (512 * 256 * 3 / 2);
  EXPECT_EQ(block.payload[13],
            static_cast<uint32_t>(frame_context_address >> 32));
  EXPECT_EQ(block.payload[14], static_cast<uint32_t>(frame_context_address));
  EXPECT_EQ(block.payload[15], UINT32_MAX);
  EXPECT_EQ(block.payload[17], 0u);
  EXPECT_EQ(block.payload[1023], 512u);
  EXPECT_EQ(block.payload[1024], 256u);

  amdf_media_frame_t frame = {};
  amdf_media_h264_encode_reference_t reference = {
      .binding = {&frame, 0},
      .frame_type = AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR,
  };
  picture.frame_type = AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_P;
  picture.frame_number = 1;
  picture.picture_order_count = 2;
  picture.frame_decoding_order_number = 1;
  picture.reference_count = 1;
  picture.references = &reference;
  length = amdf_media_drm_vcn_encode_h264(&command_, &session_, &resources_, 1,
                                          &picture);
  EXPECT_EQ(length, 5052u);
  EXPECT_EQ(Fnv1a(words_, length), UINT64_C(0x9876aa3ec98ca8f8));
  block = FindPacketBlock(words_, length, 0x200003);
  ASSERT_NE(block.payload, nullptr);
  ASSERT_EQ(block.payload_word_count, 75u);
  EXPECT_EQ(block.payload[1], 2u);
  EXPECT_EQ(block.payload[5], 0u);
  EXPECT_EQ(block.payload[37], 1u);
  EXPECT_EQ(block.payload[71], 0u);
  EXPECT_EQ(block.payload[72], 0u);
  EXPECT_EQ(block.payload[74], UINT32_MAX);

  length = amdf_media_drm_vcn_encode_control(&command_, &session_,
                                             AMDF_MEDIA_COMMAND_OPERATION_END);
  EXPECT_EQ(length, 68u);
  EXPECT_EQ(Fnv1a(words_, length), UINT64_C(0x16e12386c181e8f1));
}

TEST_F(VcnEncodePacketTest, HevcMatchesQualifiedNativePackets) {
  session_.requirements.configuration.codec = AMDF_MEDIA_CODEC_HEVC;
  session_.encode_codec.hevc.minimum_luma_coding_unit_log2 = 3;
  session_.encode_codec.hevc.log2_max_picture_order_count_lsb_minus4 = 0;

  const uint64_t expected_begin_hash = UINT64_C(0x47042be7d5ed9719);
  const uint64_t expected_picture_hashes[2][2] = {
      {UINT64_C(0xde574b56fe3cd338), UINT64_C(0x11ab14843aa80034)},
      {UINT64_C(0x0a146f6cba11a741), UINT64_C(0x3cff02a7876c595d)},
  };
  for (uint32_t depth_index = 0; depth_index < 2; ++depth_index) {
    ConfigureSampleBitDepth(depth_index == 0 ? 8 : 10);
    uint32_t length = amdf_media_drm_vcn_encode_control(
        &command_, &session_, AMDF_MEDIA_COMMAND_OPERATION_BEGIN);
    EXPECT_EQ(length, 316u);
    EXPECT_EQ(Fnv1a(words_, length), expected_begin_hash);

    amdf_media_hevc_encode_picture_info_t picture = {
        .frame_type = AMDF_MEDIA_HEVC_ENCODE_FRAME_TYPE_IDR,
        .used_as_reference = 1,
    };
    length = amdf_media_drm_vcn_encode_hevc(&command_, &session_, &resources_,
                                            0, &picture);
    EXPECT_EQ(length, 1776u);
    EXPECT_EQ(Fnv1a(words_, length), expected_picture_hashes[depth_index][0]);

    amdf_media_frame_t frame = {};
    amdf_media_hevc_encode_reference_t reference = {
        .binding = {&frame, 0},
    };
    picture.frame_type = AMDF_MEDIA_HEVC_ENCODE_FRAME_TYPE_P;
    picture.picture_order_count = 1;
    picture.reference_count = 1;
    picture.references = &reference;
    length = amdf_media_drm_vcn_encode_hevc(&command_, &session_, &resources_,
                                            1, &picture);
    EXPECT_EQ(length, 1776u);
    EXPECT_EQ(Fnv1a(words_, length), expected_picture_hashes[depth_index][1]);
  }
}

TEST_F(VcnEncodePacketTest, Av1MatchesPixelFirstQualifiedNativePackets) {
  session_.requirements.configuration.codec = AMDF_MEDIA_CODEC_AV1;
  session_.encode_codec.av1.order_hint_bit_count = 4;

  const uint64_t expected_begin_hash = UINT64_C(0xe780f5dfdc1a3b97);
  const uint64_t expected_picture_hashes[2][2] = {
      {UINT64_C(0xece6917fba00c207), UINT64_C(0xb368817b52fffd51)},
      {UINT64_C(0xb4831cf345e8842c), UINT64_C(0x27e9da6ee72f0a4a)},
  };
  for (uint32_t depth_index = 0; depth_index < 2; ++depth_index) {
    ConfigureSampleBitDepth(depth_index == 0 ? 8 : 10);
    uint32_t length = amdf_media_drm_vcn_encode_control(
        &command_, &session_, AMDF_MEDIA_COMMAND_OPERATION_BEGIN);
    EXPECT_EQ(length, 316u);
    EXPECT_EQ(Fnv1a(words_, length), expected_begin_hash);

    amdf_media_av1_encode_picture_info_t picture = {
        .frame_type = 0,
        .picture_index = 0,
        .primary_reference = 7,
        .refresh_frame_flags = 255,
    };
    length = amdf_media_drm_vcn_encode_av1(&command_, &session_, &resources_, 0,
                                           &picture);
    EXPECT_EQ(length, 1732u);
    EXPECT_EQ(Fnv1a(words_, length), expected_picture_hashes[depth_index][0]);

    amdf_media_frame_t frame = {};
    amdf_media_av1_encode_reference_t reference = {
        .binding = {&frame, 0},
    };
    picture.frame_type = 1;
    picture.order_hint = 1;
    picture.picture_index = 1;
    picture.primary_reference = 0;
    picture.reference_count = 1;
    picture.references = &reference;
    for (uint32_t index = 0; index < 7; ++index) {
      picture.reference_indices[index] = 0;
    }
    length = amdf_media_drm_vcn_encode_av1(&command_, &session_, &resources_, 1,
                                           &picture);
    EXPECT_EQ(length, 1752u);
    EXPECT_EQ(Fnv1a(words_, length), expected_picture_hashes[depth_index][1]);
  }
}

TEST_F(VcnEncodePacketTest, RetiredFeedbackProducesOneBoundedRegion) {
  amdf_media_encoded_region_t region = {};
  amdf_media_encode_result_info_t result = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_RESULT_INFO,
      .structure_size = sizeof(result),
      .region_capacity = 1,
      .regions = &region,
  };
  uint32_t feedback[10] = {};
  feedback[1] = 1;
  feedback[6] = 4096;
  feedback[8] = 128;
  EXPECT_EQ(
      amdf_media_drm_vcn_encode_result(&session_, feedback, 8192, &result),
      AMDF_STATUS_OK);
  EXPECT_EQ(result.written_byte_length, 4096u);
  EXPECT_EQ(result.region_count, 1u);
  EXPECT_EQ(region.kind, AMDF_MEDIA_ENCODED_REGION_H264_ANNEX_B_SLICE);
  EXPECT_EQ(region.byte_offset, 0u);
  EXPECT_EQ(region.byte_length, 3968u);

  result.region_capacity = 0;
  result.regions = nullptr;
  EXPECT_EQ(
      amdf_media_drm_vcn_encode_result(&session_, feedback, 8192, &result),
      amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
  EXPECT_EQ(result.written_byte_length, 4096u);
  EXPECT_EQ(result.region_count, 1u);

  feedback[6] = 8193;
  EXPECT_EQ(
      amdf_media_drm_vcn_encode_result(&session_, feedback, 8192, &result),
      amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
  EXPECT_EQ(result.written_byte_length, 0u);
  EXPECT_EQ(result.region_count, 0u);

  feedback[6] = 4096;
  feedback[8] = 4096;
  EXPECT_EQ(
      amdf_media_drm_vcn_encode_result(&session_, feedback, 8192, &result),
      amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
  EXPECT_EQ(result.written_byte_length, 0u);
  EXPECT_EQ(result.region_count, 0u);
}

TEST_F(VcnEncodePacketTest, Av1ResultExtensionClearsForCompleteFrameObu) {
  session_.requirements.configuration.codec = AMDF_MEDIA_CODEC_AV1;
  amdf_media_av1_encode_result_info_t av1 = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_RESULT_INFO,
      .structure_size = sizeof(av1),
      .primary_reference = 6,
  };
  amdf_media_encoded_region_t region = {};
  amdf_media_encode_result_info_t result = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_RESULT_INFO,
      .structure_size = sizeof(result),
      .next = &av1,
      .written_byte_length = 99,
      .region_capacity = 1,
      .region_count = 1,
      .regions = &region,
  };
  EXPECT_EQ(amdf_media_drm_vcn_encode_result_clear(&session_, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(result.written_byte_length, 0u);
  EXPECT_EQ(result.region_count, 0u);
  EXPECT_EQ(result.region_capacity, 1u);
  EXPECT_EQ(result.regions, &region);
  EXPECT_EQ(av1.primary_reference, 0u);

  uint32_t feedback[10] = {};
  feedback[1] = 1;
  feedback[6] = 512;
  feedback[8] = 16;
  EXPECT_EQ(
      amdf_media_drm_vcn_encode_result(&session_, feedback, 1024, &result),
      AMDF_STATUS_OK);
  EXPECT_EQ(region.kind, AMDF_MEDIA_ENCODED_REGION_AV1_FRAME_OBU);
}

}  // namespace
