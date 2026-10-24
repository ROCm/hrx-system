// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/audio_source.h"

#include <limits>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(AudioSourceTest, RawEncodingsPreserveChannelsAndIndependentReadRanges) {
  for (uint32_t channels : {1, 2, 6}) {
    for (bool floating : {false, true}) {
      const size_t sample_bytes = floating ? 4 : 2;
      // Deliberately unaligned to exercise the public byte representation.
      std::vector<uint8_t> bytes(1 + 7 * channels * sample_bytes);
      std::vector<float> expected(7 * channels);
      for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = (static_cast<int>(i % 7) - 3) / 8.0f;
        if (floating) {
          iree_unaligned_store_le_f32(bytes.data() + 1 + i * sample_bytes,
                                      expected[i]);
        } else {
          iree_unaligned_store_le_u16(
              bytes.data() + 1 + i * sample_bytes,
              static_cast<int16_t>(expected[i] * 32768));
        }
      }
      const std::string mime = "Audio/PCM; rate=44100; CHANNELS=\"" +
                               std::to_string(channels) + "\"; format=\"" +
                               (floating ? "f32le" : "s16le") + "\"";
      loom_serve_audio_source_t source;
      IREE_ASSERT_OK(loom_serve_audio_source_parse(
          {mime.data(), mime.size()}, {bytes.data() + 1, bytes.size() - 1},
          &source));
      EXPECT_EQ(source.sample_rate, 44100u);
      EXPECT_EQ(source.channels, channels);
      EXPECT_EQ(source.frame_count, 7u);
      EXPECT_EQ(source.data.data, bytes.data() + 1);
      std::vector<float> decoded(expected.size());
      IREE_ASSERT_OK(
          loom_serve_audio_source_read(&source, 0, 3, decoded.data()));
      IREE_ASSERT_OK(loom_serve_audio_source_read(
          &source, 3, 4, decoded.data() + 3 * channels));
      EXPECT_EQ(decoded, expected);
    }
  }
}

TEST(AudioSourceTest, RejectsMalformedGeometryAndAmbiguousPcmEncoding) {
  const uint8_t bytes[16] = {0};
  for (const auto mime : {
           IREE_SV("audio/pcm"),
           IREE_SV("audio/pcm;rate=24000"),
           IREE_SV("audio/pcm;rate=0;channels=1"),
           IREE_SV("audio/pcm;rate=24000;channels=0"),
           IREE_SV("audio/pcm;rate=4294967296;channels=1"),
           IREE_SV("audio/pcm;rate=24000;channels=3"),
           IREE_SV("audio/pcm;rate=24000;channels=1;format=f32be"),
           IREE_SV("audio/pcm;rate=24000;channels=1;format="),
           IREE_SV("audio/pcm;rate=24000;channels=1;format=f32le;FORMAT=s16le"),
           IREE_SV("audio/pcm;rate=24000;RATE=16000;channels=1"),
           IREE_SV("audio/pcm;rate=\"24\\\";channels=1"),
           IREE_SV("audio/pcm;rate=24000;channels=1;endian=little"),
       }) {
    loom_serve_audio_source_t source;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_audio_source_parse(mime, {bytes, 16}, &source));
    EXPECT_EQ(source.data.data, nullptr);
  }
  loom_serve_audio_source_t source;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_audio_source_parse(
          IREE_SV("audio/pcm;rate=24000;channels=1;format=f32le"), {bytes, 6},
          &source));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_audio_source_parse(IREE_SV("audio/wav"), {bytes, 0}, &source));
}

TEST(AudioSourceTest, RejectsNonfiniteOrUnnormalizedFloatingSamples) {
  uint8_t bytes[4];
  loom_serve_audio_source_t source;
  IREE_ASSERT_OK(loom_serve_audio_source_parse(
      IREE_SV("audio/pcm;rate=24000;channels=1;format=f32le"), {bytes, 4},
      &source));
  float output = 0;
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(), 1.01f, -1.01f}) {
    iree_unaligned_store_le_f32(bytes, value);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_audio_source_read(&source, 0, 1, &output));
  }
  for (float value : {-1.0f, -0.0f, 0.5f, 1.0f}) {
    iree_unaligned_store_le_f32(bytes, value);
    IREE_ASSERT_OK(loom_serve_audio_source_read(&source, 0, 1, &output));
    EXPECT_EQ(output, value);
  }
}

TEST(AudioSourceTest, ContainerGeometryComesFromBytes) {
  const uint8_t bytes[] = {1, 2, 3, 4};
  for (const auto mime :
       {IREE_SV(""), IREE_SV("audio/wav"), IREE_SV("audio/flac"),
        IREE_SV("audio/mpeg"), IREE_SV("application/octet-stream")}) {
    loom_serve_audio_source_t source;
    IREE_ASSERT_OK(loom_serve_audio_source_parse(mime, {bytes, 4}, &source));
    EXPECT_EQ(source.encoding, LOOM_SERVE_AUDIO_ENCODING_CONTAINER);
    EXPECT_EQ(source.sample_rate, 0u);
    EXPECT_EQ(source.channels, 0u);
    EXPECT_EQ(source.frame_count, 0u);
  }
}

TEST(AudioSourceTest, ExactDurationAvoidsIntermediateOverflow) {
  uint64_t count = 0;
  IREE_ASSERT_OK(
      loom_serve_audio_frame_count(UINT64_MAX, UINT32_MAX, UINT32_MAX, &count));
  EXPECT_EQ(count, UINT64_MAX);
  IREE_ASSERT_OK(loom_serve_audio_frame_count(44101, 44100, 24000, &count));
  EXPECT_EQ(count, 24001u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_serve_audio_frame_count(UINT64_MAX, 1, 2, &count));
  EXPECT_EQ(count, 0u);
}

}  // namespace
