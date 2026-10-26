// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/encode.h"

#include <algorithm>

#include "gtest/gtest.h"

namespace amdf::media::d3d12 {
namespace {

template <typename T>
void Initialize(T* value, amdf_structure_type_t type) {
  value->type = type;
  value->structure_size = sizeof(*value);
}

class TemporalEncodePictureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session_.info.configuration = {AMDF_MEDIA_CODEC_H264,
                                   AMDF_MEDIA_DIRECTION_ENCODE,
                                   AMDF_MEDIA_PIXEL_FORMAT_NV12,
                                   320,
                                   192,
                                   2};
    session_.requirements.configuration = session_.info.configuration;
    session_.requirements.compressed_byte_offset_alignment = 256;
    session_.requirements.compressed_byte_length_alignment = 256;
    session_.requirements.minimum_compressed_byte_length = 65536;
    session_.encode.rate_control.mode = AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP;
    session_.encode.rate_control.temporal_layer_count = 2;
    session_.encode.rate_control.temporal_layers = session_.rate_control_layers;
    session_.rate_control_layers[0] = {15, 1, {{20, 22}}};
    session_.rate_control_layers[1] = {15, 1, {{30, 32}}};
    for (uint32_t index = 0; index < 2; ++index) {
      auto& native = session_.native_encode.temporal_layer_rate_controls[index];
      native.cqp = {20 + index * 10, 22 + index * 10, 22 + index * 10};
      native.descriptor.Mode = D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_CQP;
      native.descriptor.ConfigParams = {sizeof(native.cqp)};
      native.descriptor.ConfigParams.pConfiguration_CQP = &native.cqp;
      native.descriptor.TargetFrameRate = {15, 1};
    }

    Initialize(&encode_, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_PICTURE_INFO);
    encode_.input = &input_;
    encode_.reconstruction = &reconstruction_;
    encode_.reconstruction_slot_index = 1;
    encode_.temporal_layer_index = 1;
    encode_.compressed.memory = reinterpret_cast<amdf_memory_t*>(1);
    encode_.compressed.byte_offset = 256;
    encode_.compressed.byte_length = 65536;
  }

  void ExpectEnhancementLayerRateControl(const EncodePicture& native) {
    const auto& rate = native.arguments.SequenceControlDesc.RateControl;
    EXPECT_EQ(rate.Mode, D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_CQP);
    EXPECT_EQ(rate.ConfigParams.pConfiguration_CQP,
              &session_.native_encode.temporal_layer_rate_controls[1].cqp);
    EXPECT_EQ(rate.ConfigParams.pConfiguration_CQP
                  ->ConstantQP_InterPredictedFrame_PrevRefOnly,
              32u);
    EXPECT_EQ(rate.TargetFrameRate.Numerator, 15u);
    EXPECT_EQ(rate.TargetFrameRate.Denominator, 1u);
  }

  Session session_ = {};
  amdf_media_encode_picture_info_t encode_ = {};
  amdf_media_frame_t input_ = {};
  amdf_media_frame_t reconstruction_ = {};
  amdf_media_frame_t reference_frame_ = {};
};

TEST_F(TemporalEncodePictureTest, H264CarriesCurrentAndReferenceLayerIdentity) {
  session_.native_encode.codec = D3D12_VIDEO_ENCODER_CODEC_H264;
  amdf_media_h264_encode_reference_t reference = {
      {&reference_frame_, 0},
      0,
      AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR,
      7,
      0,
      41,
  };
  amdf_media_h264_encode_picture_info_t h264 = {};
  Initialize(&h264, AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_PICTURE_INFO);
  h264.frame_type = AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_P;
  h264.frame_number = 1;
  h264.picture_order_count = 2;
  h264.frame_decoding_order_number = 42;
  h264.used_as_reference = 1;
  h264.reference_count = 1;
  h264.references = &reference;
  encode_.next = &h264;

  EncodePicture native = {};
  ASSERT_EQ(PrepareEncode(session_, encode_, 8, &native), AMDF_STATUS_OK);
  ExpectEnhancementLayerRateControl(native);
  EXPECT_EQ(native.h264.FrameDecodingOrderNumber, 1u);
  EXPECT_EQ(native.h264.TemporalLayerIndex, 1u);
  ASSERT_EQ(native.h264.ReferenceFramesReconPictureDescriptorsCount, 1u);
  EXPECT_EQ(native.h264_references[0].FrameDecodingOrderNumber, 7u);
  EXPECT_EQ(native.h264_references[0].TemporalLayerIndex, 0u);
}

TEST_F(TemporalEncodePictureTest, HevcCarriesCurrentAndReferenceLayerIdentity) {
  session_.info.configuration.codec = AMDF_MEDIA_CODEC_HEVC;
  session_.requirements.configuration.codec = AMDF_MEDIA_CODEC_HEVC;
  session_.native_encode.codec = D3D12_VIDEO_ENCODER_CODEC_HEVC;
  amdf_media_hevc_encode_reference_t reference = {
      {&reference_frame_, 0},
      0,
      0,
      0,
  };
  amdf_media_hevc_encode_picture_info_t hevc = {};
  Initialize(&hevc, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_ENCODE_PICTURE_INFO);
  hevc.frame_type = AMDF_MEDIA_HEVC_ENCODE_FRAME_TYPE_P;
  hevc.picture_order_count = 1;
  hevc.used_as_reference = 1;
  hevc.reference_count = 1;
  hevc.references = &reference;
  encode_.next = &hevc;

  EncodePicture native = {};
  ASSERT_EQ(PrepareEncode(session_, encode_, 8, &native), AMDF_STATUS_OK);
  ExpectEnhancementLayerRateControl(native);
  EXPECT_EQ(native.hevc.TemporalLayerIndex, 1u);
  ASSERT_EQ(native.hevc.ReferenceFramesReconPictureDescriptorsCount, 1u);
  EXPECT_EQ(native.hevc_references[0].TemporalLayerIndex, 0u);
}

TEST_F(TemporalEncodePictureTest, Av1UsesOneBasedNativeScalabilityIdentity) {
  session_.info.configuration.codec = AMDF_MEDIA_CODEC_AV1;
  session_.requirements.configuration.codec = AMDF_MEDIA_CODEC_AV1;
  session_.native_encode.codec = D3D12_VIDEO_ENCODER_CODEC_AV1;
  session_.av1_encode.order_hint_bit_count = 8;
  amdf_media_av1_encode_reference_t reference = {
      {&reference_frame_, 0}, 0, 0, 0, 0,
  };
  amdf_media_av1_encode_picture_info_t av1 = {};
  Initialize(&av1, AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_PICTURE_INFO);
  av1.frame_type = 1;
  av1.order_hint = 1;
  av1.picture_index = 1;
  av1.primary_reference = 0;
  av1.refresh_frame_flags = 1;
  av1.quantization.matrix_levels[0] = UINT8_MAX;
  av1.quantization.matrix_levels[1] = UINT8_MAX;
  av1.quantization.matrix_levels[2] = UINT8_MAX;
  av1.reference_count = 1;
  av1.references = &reference;
  std::fill(std::begin(av1.reference_map), std::end(av1.reference_map),
            uint8_t{0});
  std::fill(std::begin(av1.reference_indices), std::end(av1.reference_indices),
            uint8_t{7});
  encode_.next = &av1;

  EncodePicture native = {};
  ASSERT_EQ(PrepareEncode(session_, encode_, 8, &native), AMDF_STATUS_OK);
  ExpectEnhancementLayerRateControl(native);
  EXPECT_EQ(native.av1.TemporalLayerIndexPlus1, 2u);
  for (uint32_t index = 0; index < 8; ++index) {
    EXPECT_EQ(native.av1.ReferenceFramesReconPictureDescriptors[index]
                  .ReconstructedPictureResourceIndex,
              0u);
    EXPECT_EQ(native.av1.ReferenceFramesReconPictureDescriptors[index]
                  .TemporalLayerIndexPlus1,
              1u);
  }
  for (uint32_t index = 0; index < 7; ++index) {
    EXPECT_EQ(native.av1.ReferenceIndices[index], 7u);
  }

  session_.encode.rate_control.temporal_layer_count = 1;
  encode_.temporal_layer_index = 0;
  ASSERT_EQ(PrepareEncode(session_, encode_, 8, &native), AMDF_STATUS_OK);
  EXPECT_EQ(native.av1.TemporalLayerIndexPlus1, 0u);
  for (uint32_t index = 0; index < 8; ++index) {
    EXPECT_EQ(native.av1.ReferenceFramesReconPictureDescriptors[index]
                  .TemporalLayerIndexPlus1,
              0u);
  }
}

}  // namespace
}  // namespace amdf::media::d3d12
