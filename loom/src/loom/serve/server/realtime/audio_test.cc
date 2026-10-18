// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/audio.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "iree/base/internal/base64.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class Audio {
 public:
  ~Audio() {
    loom_serve_realtime_audio_deinitialize(&value);
    loom_serve_audio_resampler_filter_deinitialize(&filter);
  }
  // Immutable coefficients retained through every borrowing cursor.
  loom_serve_audio_resampler_filter_t filter = {};
  // Production cursor, reset independently of coefficient ownership.
  loom_serve_realtime_audio_t value = {};
};

void ConvertBlock(loom_serve_realtime_audio_t* audio,
                  const loom_serve_speech_block_t& block, std::string* pcm) {
  bool consumed = false;
  while (!consumed) {
    iree_string_builder_t packet;
    iree_string_builder_initialize(iree_allocator_system(), &packet);
    auto status =
        loom_serve_realtime_audio_convert(audio, &block, &packet, &consumed);
    std::string decoded(
        iree_base64_decoded_size(iree_string_builder_view(&packet)), '\0');
    iree_host_size_t length = 0;
    if (iree_status_is_ok(status)) {
      status = iree_base64_decode(
          iree_string_builder_view(&packet),
          iree_make_byte_span(decoded.data(), decoded.size()), &length);
    }
    iree_string_builder_deinitialize(&packet);
    IREE_ASSERT_OK(status);
    decoded.resize(length);
    pcm->append(decoded);
  }
}

void Convert(Audio* audio, uint32_t rate, uint32_t channels,
             const std::vector<float>& samples,
             const std::vector<size_t>& partitions, bool separate_eof,
             std::string* pcm) {
  if (!audio->filter.denominator) {
    IREE_ASSERT_OK(loom_serve_realtime_audio_filter_initialize(
        {rate, channels}, iree_allocator_system(), &audio->filter));
  }
  IREE_ASSERT_OK(
      loom_serve_realtime_audio_begin({rate, channels}, &audio->filter,
                                      iree_allocator_system(), &audio->value));
  pcm->clear();
  std::vector<uint8_t> encoded_samples(samples.size() * sizeof(float));
  for (size_t i = 0; i < samples.size(); ++i) {
    iree_unaligned_store_le_f32(encoded_samples.data() + i * sizeof(float),
                                samples[i]);
  }
  const size_t frames = samples.size() / channels;
  size_t position = 0, partition = 0;
  bool complete = false;
  while (!complete) {
    const size_t count = std::min(frames - position,
                                  partitions[partition++ % partitions.size()]);
    complete =
        position == frames || (!separate_eof && position + count == frames);
    const loom_serve_speech_block_t block = {
        position, count,
        iree_make_const_byte_span(
            encoded_samples.data() + position * channels * sizeof(float),
            count * channels * sizeof(float)),
        complete};
    ASSERT_NO_FATAL_FAILURE(ConvertBlock(&audio->value, block, pcm));
    position += count;
  }
  EXPECT_EQ(pcm->size() / 2, (frames * 24000 + rate - 1) / rate);
  EXPECT_EQ(audio->value.frames, pcm->size() / 2);
}

TEST(RealtimeAudioTest, DownmixClockAndQuantizationPreserveIndependentSignal) {
  constexpr double kPi = 3.14159265358979323846264338327950288;
  for (uint32_t rate :
       {8000u, 16000u, 22050u, 24000u, 44100u, 48000u, 96000u, 16001u}) {
    for (uint32_t channels : {1u, 2u}) {
      SCOPED_TRACE(::testing::Message()
                   << rate << " Hz, " << channels << " channels");
      const auto signal = [&](double time) {
        return 0.05 + 0.3 * std::sin(2 * kPi * 701 * time);
      };
      std::vector<float> samples((rate / 4 + 17) * channels);
      for (size_t i = 0; i < samples.size() / channels; ++i) {
        for (uint32_t channel = 0; channel < channels; ++channel) {
          const double offset = channels == 1 ? 0 : (channel ? -0.125 : 0.125);
          samples[i * channels + channel] = static_cast<float>(
              signal(static_cast<double>(i) / rate) + offset);
        }
      }
      Audio audio;
      std::string expected, actual;
      ASSERT_NO_FATAL_FAILURE(Convert(&audio, rate, channels, samples,
                                      {samples.size()}, false, &expected));
      ASSERT_NO_FATAL_FAILURE(Convert(&audio, rate, channels, samples,
                                      {1, 7, 137, 8193}, true, &actual));
      EXPECT_EQ(actual, expected);
      // The sinc oracle admits 3e-5 absolute error; PCM rounding adds half a
      // quantization unit. Two integer units bound their combined error.
      for (size_t i = 480; i + 480 < actual.size() / 2; ++i) {
        const int32_t observed = static_cast<int16_t>(
            iree_unaligned_load_le_u16(actual.data() + i * 2));
        const int32_t reference = static_cast<int32_t>(
            std::round(signal(static_cast<double>(i) / 24000) * 32768));
        ASSERT_LE(std::abs(observed - reference), 2) << "frame " << i;
      }
      ASSERT_NO_FATAL_FAILURE(
          Convert(&audio, rate, channels, {}, {1}, true, &actual));
      EXPECT_TRUE(actual.empty());
    }
  }
}

TEST(RealtimeAudioTest, SharedCoefficientsPreserveInterleavedCursorLifetimes) {
  for (uint32_t rate : {16000u, 24000u, 44100u, 16001u}) {
    SCOPED_TRACE(rate);
    std::vector<float> first(997), second(4807);
    for (size_t i = 0; i < first.size(); ++i) {
      first[i] = static_cast<float>(0.3 * std::sin(i * 0.17));
    }
    for (size_t i = 0; i < second.size(); ++i) {
      second[i] = static_cast<float>(-0.2 + 0.4 * std::cos(i * 0.13));
    }
    Audio owner, peer;
    std::string expected_first, expected_second;
    ASSERT_NO_FATAL_FAILURE(Convert(&owner, rate, 1, first, {first.size()},
                                    false, &expected_first));
    ASSERT_NO_FATAL_FAILURE(Convert(&owner, rate, 1, second, {second.size()},
                                    false, &expected_second));
    const auto* coefficients = owner.filter.coefficients;
    IREE_ASSERT_OK(loom_serve_realtime_audio_begin(
        {rate, 1}, &owner.filter, iree_allocator_system(), &owner.value));
    IREE_ASSERT_OK(loom_serve_realtime_audio_begin(
        {rate, 1}, &owner.filter, iree_allocator_system(), &peer.value));
    for (size_t i = 0; i < first.size(); ++i) {
      iree_unaligned_store_le_f32(first.data() + i, first[i]);
    }
    for (size_t i = 0; i < second.size(); ++i) {
      iree_unaligned_store_le_f32(second.data() + i, second[i]);
    }
    std::string actual_first, actual_second;
    size_t first_position = 0, second_position = 0;
    while (second_position < second.size()) {
      if (first_position < first.size()) {
        const size_t count =
            std::min<size_t>(137, first.size() - first_position);
        const loom_serve_speech_block_t block = {
            first_position, count,
            iree_make_const_byte_span(first.data() + first_position,
                                      count * sizeof(float)),
            first_position + count == first.size()};
        ASSERT_NO_FATAL_FAILURE(
            ConvertBlock(&owner.value, block, &actual_first));
        first_position += count;
        if (first_position == first.size()) {
          loom_serve_realtime_audio_deinitialize(&owner.value);
        }
      }
      const size_t count =
          std::min<size_t>(113, second.size() - second_position);
      const loom_serve_speech_block_t block = {
          second_position, count,
          iree_make_const_byte_span(second.data() + second_position,
                                    count * sizeof(float)),
          second_position + count == second.size()};
      ASSERT_NO_FATAL_FAILURE(ConvertBlock(&peer.value, block, &actual_second));
      second_position += count;
    }
    EXPECT_EQ(owner.filter.coefficients, coefficients);
    EXPECT_EQ(actual_first, expected_first);
    EXPECT_EQ(actual_second, expected_second);
  }
}

TEST(RealtimeAudioTest, SameRateSaturationAndNonfiniteFailureAreExplicit) {
  Audio audio;
  std::string pcm;
  ASSERT_NO_FATAL_FAILURE(Convert(
      &audio, 24000, 1, {-2, -1, -0.5, 0, 0.5, 1, 2}, {1, 3}, false, &pcm));
  const int16_t expected[] = {-32768, -32768, -16384, 0, 16384, 32767, 32767};
  ASSERT_EQ(pcm.size(), sizeof(expected));
  for (size_t i = 0; i < std::size(expected); ++i) {
    EXPECT_EQ(
        static_cast<int16_t>(iree_unaligned_load_le_u16(pcm.data() + 2 * i)),
        expected[i]);
  }
  for (float value : {std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    IREE_ASSERT_OK(loom_serve_realtime_audio_begin(
        {24000, 1}, &audio.filter, iree_allocator_system(), &audio.value));
    uint8_t sample[4];
    iree_unaligned_store_le_f32(sample, value);
    const loom_serve_speech_block_t block = {
        0, 1, iree_make_const_byte_span(sample, sizeof(sample)), true};
    char storage[64];
    iree_string_builder_t output;
    iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                                &output);
    bool consumed = false;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                          loom_serve_realtime_audio_convert(
                              &audio.value, &block, &output, &consumed));
    EXPECT_FALSE(consumed);
    EXPECT_EQ(output.size, 0u);
  }
}

}  // namespace
