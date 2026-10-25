// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "libamdf/cts/interop/gpu/media/encode_fixture.h"

namespace {

struct EncodeMatrixCase {
  const char* name;
  EncodeExecutionCase execution;
  amdf_media_encode_rate_control_mode_t rate_control_mode;
  uint32_t width;
  uint32_t height;
};

class MediaEncodeMatrixTest
    : public MediaEncodeInteropTest,
      public testing::WithParamInterface<EncodeMatrixCase> {};

TEST_F(MediaEncodeInteropTest, H264DependentPicture) {
  ASSERT_NO_FATAL_FAILURE(
      RunPictures({AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
                  AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP, 320, 192, 2, 2));
}

TEST_P(MediaEncodeMatrixTest, PipelinesGpuInputsAndReusesRecordings) {
  const auto& test_case = GetParam();
  ASSERT_NO_FATAL_FAILURE(
      RunPictures(test_case.execution, test_case.rate_control_mode,
                  test_case.width, test_case.height, 12, 4));
}

INSTANTIATE_TEST_SUITE_P(
    EncodeFamilies, MediaEncodeMatrixTest,
    testing::Values(
        EncodeMatrixCase{
            "H264Cqp",
            {AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP,
            320,
            192},
        EncodeMatrixCase{
            "H264Cbr",
            {AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR,
            320,
            192},
        EncodeMatrixCase{
            "H264Vbr",
            {AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_VBR,
            320,
            192},
        EncodeMatrixCase{
            "HevcMainCqp",
            {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP,
            320,
            192},
        EncodeMatrixCase{
            "HevcMainCbr",
            {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR,
            320,
            192},
        EncodeMatrixCase{
            "HevcMainVbr",
            {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_VBR,
            320,
            192},
        EncodeMatrixCase{
            "HevcMain10Cqp",
            {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_PIXEL_FORMAT_P010, 10},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP,
            320,
            192},
        EncodeMatrixCase{
            "HevcMain10Cbr",
            {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_PIXEL_FORMAT_P010, 10},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR,
            320,
            192},
        EncodeMatrixCase{
            "HevcMain10Vbr",
            {AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_PIXEL_FORMAT_P010, 10},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_VBR,
            320,
            192},
        EncodeMatrixCase{
            "Av1Main8Cqp",
            {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP,
            320,
            208},
        EncodeMatrixCase{
            "Av1Main8Cbr",
            {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR,
            320,
            208},
        EncodeMatrixCase{
            "Av1Main8Vbr",
            {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_PIXEL_FORMAT_NV12, 8},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_VBR,
            320,
            208},
        EncodeMatrixCase{
            "Av1Main10Cqp",
            {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_PIXEL_FORMAT_P010, 10},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP,
            320,
            208},
        EncodeMatrixCase{
            "Av1Main10Cbr",
            {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_PIXEL_FORMAT_P010, 10},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR,
            320,
            208},
        EncodeMatrixCase{
            "Av1Main10Vbr",
            {AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_PIXEL_FORMAT_P010, 10},
            AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_VBR,
            320,
            208}),
    [](const testing::TestParamInfo<EncodeMatrixCase>& info) {
      return std::string(info.param.name);
    });

}  // namespace
