// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/encode.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "libamdf/src/media/av1.h"
#include "libamdf/src/media/h264.h"
#include "libamdf/src/media/hevc.h"

namespace {

template <typename T>
void Initialize(T* value, amdf_structure_type_t type) {
  value->type = type;
  value->structure_size = sizeof(*value);
}

class EncodeValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    requirements_.configuration = {AMDF_MEDIA_CODEC_H264,
                                   AMDF_MEDIA_DIRECTION_ENCODE,
                                   AMDF_MEDIA_PIXEL_FORMAT_NV12,
                                   320,
                                   180,
                                   2};
    requirements_.compressed_byte_offset_alignment = 256;
    requirements_.compressed_byte_length_alignment = 256;
    requirements_.minimum_compressed_byte_length = 92160;

    Initialize(&encode_, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION);
    encode_.intra_picture_period = 30;
    encode_.inter_picture_period = 1;
    encode_.rate_control.mode = AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP;
    encode_.rate_control.frame_rate_numerator = 30;
    encode_.rate_control.frame_rate_denominator = 1;
    encode_.rate_control.control.cqp = {22, 24};

    Initialize(&picture_, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_PICTURE_INFO);
    picture_.input = &input_;
    picture_.reconstruction = &reconstruction_;
    picture_.reconstruction_slot_index = 1;
    picture_.compressed.memory = reinterpret_cast<amdf_memory_t*>(1);
    picture_.compressed.byte_offset = 256;
    picture_.compressed.byte_length = 92160;
  }

  amdf_media_configuration_info_t requirements_ = {};
  amdf_media_encode_configuration_t encode_ = {};
  amdf_media_encode_picture_info_t picture_ = {};
  amdf_media_frame_t input_ = {};
  amdf_media_frame_t reconstruction_ = {};
  amdf_media_frame_t reference_frame_ = {};
};

TEST_F(EncodeValidationTest, RateControlAndCompressedOutputAreExactContracts) {
  EXPECT_EQ(amdf_media_encode_validate_configuration(
                &requirements_.configuration, &encode_),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_media_encode_validate_picture(&requirements_, &picture_),
            AMDF_STATUS_OK);

  picture_.compressed.byte_length = 91904;
  EXPECT_EQ(amdf_media_encode_validate_picture(&requirements_, &picture_),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  picture_.compressed.byte_length = 92160;
  picture_.compressed.byte_offset = 128;
  EXPECT_EQ(amdf_media_encode_validate_picture(&requirements_, &picture_),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));

  encode_.rate_control.mode = AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR;
  encode_.rate_control.control.bit_rate = {10,      40,      0,      4000000,
                                           4000000, 8000000, 4000000};
  EXPECT_EQ(amdf_media_encode_validate_configuration(
                &requirements_.configuration, &encode_),
            AMDF_STATUS_OK);
  encode_.rate_control.control.bit_rate.peak_bit_rate = 5000000;
  EXPECT_EQ(amdf_media_encode_validate_configuration(
                &requirements_.configuration, &encode_),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
}

TEST_F(EncodeValidationTest, H264IPicturesUseStableReconstructionSlots) {
  amdf_media_h264_encode_configuration_t h264_encode = {};
  Initialize(&h264_encode, AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_CONFIGURATION);
  h264_encode.picture_order_count_type = 2;
  amdf_media_h264_configuration_t h264 = {};
  Initialize(&h264, AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION);
  h264.next = &encode_;
  h264.profile = AMDF_MEDIA_H264_PROFILE_MAIN;
  h264.level_idc = 31;
  encode_.next = &h264_encode;
  EXPECT_EQ(amdf_media_h264_validate_encode_configuration(
                &requirements_.configuration, &h264, &encode_, &h264_encode),
            AMDF_STATUS_OK);

  amdf_media_h264_encode_picture_info_t h264_picture = {};
  Initialize(&h264_picture, AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_PICTURE_INFO);
  h264_picture.frame_type = AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR;
  h264_picture.used_as_reference = 1;
  picture_.next = &h264_picture;
  EXPECT_EQ(amdf_media_h264_validate_encode_picture(
                &requirements_, &picture_, &h264_encode, &h264_picture),
            AMDF_STATUS_OK);

  const amdf_media_h264_encode_reference_t reference = {
      {&reference_frame_, 0}, 0, 0, 0};
  h264_picture.frame_type = AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_P;
  h264_picture.frame_number = 1;
  h264_picture.picture_order_count = 2;
  h264_picture.frame_decoding_order_number = 1;
  h264_picture.reference_count = 1;
  h264_picture.references = &reference;
  EXPECT_EQ(amdf_media_h264_validate_encode_picture(
                &requirements_, &picture_, &h264_encode, &h264_picture),
            AMDF_STATUS_OK);

  picture_.reconstruction_slot_index = 0;
  EXPECT_EQ(amdf_media_h264_validate_encode_picture(
                &requirements_, &picture_, &h264_encode, &h264_picture),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
}

TEST_F(EncodeValidationTest, HevcMain10AndGeometryRemainOneAdmittedEnvelope) {
  requirements_.configuration.codec = AMDF_MEDIA_CODEC_HEVC;
  requirements_.configuration.format = AMDF_MEDIA_PIXEL_FORMAT_P010;
  amdf_media_hevc_encode_configuration_t hevc_encode = {};
  Initialize(&hevc_encode, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_ENCODE_CONFIGURATION);
  hevc_encode.level_idc = 120;
  hevc_encode.tier = AMDF_MEDIA_HEVC_TIER_MAIN;
  hevc_encode.minimum_luma_coding_unit_log2 = 3;
  hevc_encode.maximum_luma_coding_unit_log2 = 6;
  hevc_encode.minimum_luma_transform_unit_log2 = 2;
  hevc_encode.maximum_luma_transform_unit_log2 = 5;
  hevc_encode.maximum_transform_hierarchy_depth_inter = 4;
  hevc_encode.maximum_transform_hierarchy_depth_intra = 4;
  amdf_media_hevc_configuration_t hevc = {};
  Initialize(&hevc, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION);
  hevc.next = &encode_;
  hevc.profile = AMDF_MEDIA_HEVC_PROFILE_MAIN10;
  hevc.sample_bit_depth = 10;
  encode_.next = &hevc_encode;
  EXPECT_EQ(amdf_media_hevc_validate_encode_configuration(
                &requirements_.configuration, &hevc, &encode_, &hevc_encode),
            AMDF_STATUS_OK);

  hevc_encode.maximum_luma_transform_unit_log2 = 6;
  EXPECT_EQ(amdf_media_hevc_validate_encode_configuration(
                &requirements_.configuration, &hevc, &encode_, &hevc_encode),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  hevc_encode.maximum_luma_transform_unit_log2 = 5;
  hevc.profile = AMDF_MEDIA_HEVC_PROFILE_MAIN;
  EXPECT_EQ(amdf_media_hevc_validate_encode_configuration(
                &requirements_.configuration, &hevc, &encode_, &hevc_encode),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
}

TEST_F(EncodeValidationTest, Av1ReferencesPreserveLogicalAndPhysicalIdentity) {
  requirements_.configuration.codec = AMDF_MEDIA_CODEC_AV1;
  amdf_media_av1_encode_configuration_t av1_encode = {};
  Initialize(&av1_encode, AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_CONFIGURATION);
  av1_encode.level = 4;
  av1_encode.tier = AMDF_MEDIA_AV1_TIER_MAIN;
  av1_encode.feature_flags = AMDF_MEDIA_AV1_ENCODE_FEATURE_ORDER_HINT |
                             AMDF_MEDIA_AV1_ENCODE_FEATURE_CDEF;
  av1_encode.order_hint_bit_count = 8;
  amdf_media_av1_configuration_t av1 = {};
  Initialize(&av1, AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION);
  av1.next = &encode_;
  av1.profile = 0;
  av1.sample_bit_depth = 8;
  encode_.next = &av1_encode;
  EXPECT_EQ(amdf_media_av1_validate_encode_configuration(
                &requirements_.configuration, &av1, &encode_, &av1_encode),
            AMDF_STATUS_OK);

  amdf_media_av1_encode_picture_info_t av1_picture = {};
  Initialize(&av1_picture, AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_PICTURE_INFO);
  av1_picture.primary_reference = 7;
  av1_picture.refresh_frame_flags = 255;
  av1_picture.quantization.matrix_levels[0] = 255;
  av1_picture.quantization.matrix_levels[1] = 255;
  av1_picture.quantization.matrix_levels[2] = 255;
  for (uint32_t index = 0; index < 7; ++index) {
    av1_picture.reference_indices[index] = 255;
  }
  picture_.next = &av1_picture;
  EXPECT_EQ(amdf_media_av1_validate_encode_picture(&requirements_, &picture_,
                                                   &av1_encode, &av1_picture),
            AMDF_STATUS_OK);

  const amdf_media_av1_encode_reference_t reference = {
      {&reference_frame_, 0}, 0, 0, 0};
  av1_picture.frame_type = 1;
  av1_picture.order_hint = 1;
  av1_picture.primary_reference = 0;
  av1_picture.reference_count = 1;
  av1_picture.references = &reference;
  for (uint32_t index = 0; index < 7; ++index) {
    av1_picture.reference_indices[index] = 0;
  }
  EXPECT_EQ(amdf_media_av1_validate_encode_picture(&requirements_, &picture_,
                                                   &av1_encode, &av1_picture),
            AMDF_STATUS_OK);

  av1_picture.reference_indices[3] = 1;
  EXPECT_EQ(amdf_media_av1_validate_encode_picture(&requirements_, &picture_,
                                                   &av1_encode, &av1_picture),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
}

}  // namespace
