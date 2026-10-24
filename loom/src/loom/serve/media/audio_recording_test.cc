// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/audio_recording.h"

#include <limits>
#include <memory>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/test_allocator.h"

namespace {

using Recording = std::unique_ptr<loom_serve_audio_recording_t,
                                  decltype(&loom_serve_audio_recording_free)>;

std::vector<uint8_t> Wave(size_t frames, uint32_t channels = 2) {
  std::vector<uint8_t> result(44 + frames * channels * 4);
  memcpy(result.data(), "RIFF", 4);
  iree_unaligned_store_le_u32(result.data() + 4, result.size() - 8);
  memcpy(result.data() + 8, "WAVEfmt ", 8);
  iree_unaligned_store_le_u32(result.data() + 16, 16);
  iree_unaligned_store_le_u16(result.data() + 20, 3);
  iree_unaligned_store_le_u16(result.data() + 22, channels);
  iree_unaligned_store_le_u32(result.data() + 24, 44100);
  iree_unaligned_store_le_u32(result.data() + 28, 44100 * channels * 4);
  iree_unaligned_store_le_u16(result.data() + 32, channels * 4);
  iree_unaligned_store_le_u16(result.data() + 34, 32);
  memcpy(result.data() + 36, "data", 4);
  iree_unaligned_store_le_u32(result.data() + 40, result.size() - 44);
  for (size_t i = 0; i < frames * channels; ++i) {
    iree_unaligned_store_le_f32(result.data() + 44 + i * 4,
                                (static_cast<int>(i % 31) - 15) / 32.0f);
  }
  return result;
}

iree_status_t Read(const std::vector<uint8_t>& bytes, bool encoded,
                   size_t limit, iree_allocator_t allocator,
                   Recording* output) {
  const auto payload =
      encoded ? iree_make_const_byte_span(bytes.data(), bytes.size())
              : iree_make_const_byte_span(bytes.data() + 44, bytes.size() - 44);
  loom_serve_audio_source_t source;
  IREE_RETURN_IF_ERROR(loom_serve_audio_source_parse(
      encoded ? IREE_SV("audio/wav")
              : IREE_SV("audio/pcm;format=f32le;rate=44100;channels=2"),
      payload, &source));
  loom_serve_audio_recording_t* recording = nullptr;
  auto status = loom_serve_audio_recording_allocate(&source, limit, allocator,
                                                    &recording);
  output->reset(recording);
  return status;
}

TEST(AudioRecordingTest, BoundedGrowthAndExactEofPreserveOriginalGeometry) {
  for (size_t frames : {1, 4096, 4097, 8192, 12345}) {
    const auto wave = Wave(frames);
    for (bool encoded : {false, true}) {
      Recording recording(nullptr, loom_serve_audio_recording_free);
      const size_t size = frames * 8;
      if (encoded && !LOOM_TEST_AUDIO_WAV) {
        IREE_EXPECT_NOT_OK(
            Read(wave, true, size, iree_allocator_system(), &recording));
        EXPECT_EQ(recording, nullptr);
        continue;
      }
      for (size_t limit : {size, size + 3, size * 2}) {
        IREE_ASSERT_OK(
            Read(wave, encoded, limit, iree_allocator_system(), &recording));
        const auto source = loom_serve_audio_recording_source(recording.get());
        EXPECT_EQ(source.encoding, LOOM_SERVE_AUDIO_ENCODING_F32LE);
        EXPECT_EQ(source.frame_count, frames);
        EXPECT_EQ(source.sample_rate, 44100u);
        EXPECT_EQ(source.channels, 2u);
        ASSERT_EQ(source.data.data_length, size);
        EXPECT_EQ(memcmp(source.data.data, wave.data() + 44, size), 0);
      }
      for (size_t limit : {size_t{0}, size - 1}) {
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_INVALID_ARGUMENT,
            Read(wave, encoded, limit, iree_allocator_system(), &recording));
        EXPECT_EQ(recording, nullptr);
      }
    }
  }
}

TEST(AudioRecordingTest, AllocationFailuresRetainNoPartialRecording) {
  const auto wave = Wave(12345);
  for (bool encoded : {false, true}) {
    if (encoded && !LOOM_TEST_AUDIO_WAV) {
      continue;
    }
    iree::vm::testing::CountingAllocator baseline;
    Recording recording(nullptr, loom_serve_audio_recording_free);
    IREE_ASSERT_OK(
        Read(wave, encoded, 12345 * 8, baseline.allocator(), &recording));
    recording.reset();
    for (size_t fail_at = 1; fail_at <= baseline.allocation_count();
         ++fail_at) {
      SCOPED_TRACE(fail_at);
      iree::vm::testing::CountingAllocator allocator(fail_at);
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          Read(wave, encoded, 12345 * 8, allocator.allocator(), &recording));
      EXPECT_EQ(recording, nullptr);
    }
  }
}

TEST(AudioRecordingTest, NonfiniteSamplesNeverReachModelProcessing) {
  for (float value : {std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    auto wave = Wave(17);
    iree_unaligned_store_le_f32(wave.data() + 44 + 5 * 4, value);
    for (bool encoded : {false, true}) {
      if (encoded && !LOOM_TEST_AUDIO_WAV) {
        continue;
      }
      Recording recording(nullptr, loom_serve_audio_recording_free);
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          Read(wave, encoded, 1024, iree_allocator_system(), &recording));
      EXPECT_EQ(recording, nullptr);
    }
  }
}

}  // namespace
