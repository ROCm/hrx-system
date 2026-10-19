// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/audio_resampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr size_t kStorageLimit = 32 * 1024 * 1024;
constexpr double kPi = 3.14159265358979323846264338327950288;

class Filter {
 public:
  ~Filter() { loom_serve_audio_resampler_filter_deinitialize(&value); }
  // Stable coefficient owner; streams borrow this value through retirement.
  loom_serve_audio_resampler_filter_t value = {};
};

class Stream {
 public:
  ~Stream() { loom_serve_audio_resampler_deinitialize(&value); }
  // Independent history, destroyed before the borrowed filter.
  loom_serve_audio_resampler_t value = {};
};

void Convert(const loom_serve_audio_resampler_filter_t* filter,
             const std::vector<float>& input, uint32_t channels,
             const std::vector<size_t>& packets,
             const std::vector<size_t>& credits, size_t input_capacity,
             std::vector<float>* result) {
  Stream stream;
  IREE_ASSERT_OK(loom_serve_audio_resampler_initialize(
      filter, channels, input_capacity, kStorageLimit, iree_allocator_system(),
      &stream.value));
  result->clear();
  std::vector<float> output(*std::max_element(credits.begin(), credits.end()) *
                            channels);
  size_t position = 0;
  size_t packet = 0;
  size_t call = 0;
  const size_t frame_count = input.size() / channels;
  while (position < frame_count) {
    const size_t end =
        std::min(frame_count, position + packets[packet++ % packets.size()]);
    while (position < end) {
      const size_t credit = credits[call++ % credits.size()];
      const size_t generated =
          loom_serve_audio_resampler_read(&stream.value, credit, output.data());
      result->insert(result->end(), output.begin(),
                     output.begin() + generated * channels);
      size_t capacity = 0;
      float* target =
          loom_serve_audio_resampler_reserve(&stream.value, &capacity);
      const size_t accepted = credit ? std::min(end - position, capacity) : 0;
      if (accepted) {
        memcpy(target, input.data() + position * channels,
               accepted * channels * sizeof(float));
        loom_serve_audio_resampler_commit(&stream.value, accepted);
        position += accepted;
      }
      ASSERT_TRUE(accepted || generated || !credit);
      EXPECT_FALSE(loom_serve_audio_resampler_is_complete(&stream.value));
    }
  }
  loom_serve_audio_resampler_finish(&stream.value);
  while (!loom_serve_audio_resampler_is_complete(&stream.value)) {
    const size_t credit = credits[call++ % credits.size()];
    const size_t generated =
        loom_serve_audio_resampler_read(&stream.value, credit, output.data());
    result->insert(result->end(), output.begin(),
                   output.begin() + generated * channels);
    ASSERT_TRUE(generated || !credit ||
                loom_serve_audio_resampler_is_complete(&stream.value));
  }
  const uint64_t expected =
      (uint64_t{frame_count} * filter->denominator + filter->numerator - 1) /
      filter->numerator;
  EXPECT_EQ(result->size(), expected * channels);
  loom_serve_audio_resampler_finish(&stream.value);
  EXPECT_EQ(loom_serve_audio_resampler_read(&stream.value, 1, output.data()),
            0u);
  size_t capacity = 1;
  EXPECT_EQ(loom_serve_audio_resampler_reserve(&stream.value, &capacity),
            nullptr);
  EXPECT_EQ(capacity, 0u);
}

TEST(AudioResamplerTest, ClockAndWaveformIgnorePacketAndOutputBoundaries) {
  constexpr uint32_t rates[] = {8000,  16000, 22050,  24000, 44100,
                                48000, 96000, 192000, 16001};
  for (uint32_t input_rate : rates) {
    for (uint32_t output_rate : rates) {
      SCOPED_TRACE(::testing::Message() << input_rate << " -> " << output_rate);
      Filter filter;
      IREE_ASSERT_OK(loom_serve_audio_resampler_filter_initialize(
          input_rate, output_rate, kStorageLimit, iree_allocator_system(),
          &filter.value));
      for (uint32_t channels : {1u, 2u}) {
        std::vector<float> input(1021 * channels);
        for (size_t i = 0; i < input.size(); ++i) {
          input[i] = static_cast<float>(std::sin(i * 0.379) * 0.5);
        }
        std::vector<float> expected;
        ASSERT_NO_FATAL_FAILURE(Convert(&filter.value, input, channels, {1021},
                                        {65536}, 1024, &expected));
        for (const auto& packets :
             {std::vector<size_t>{1}, {7, 83, 960, 17}, {137, 53}}) {
          std::vector<float> actual;
          ASSERT_NO_FATAL_FAILURE(Convert(&filter.value, input, channels,
                                          packets, {0, 1, 0, 7, 67}, 137,
                                          &actual));
          EXPECT_EQ(actual, expected);
        }
        if (input_rate == output_rate) {
          EXPECT_EQ(expected, input);
        }
        std::vector<float> empty;
        ASSERT_NO_FATAL_FAILURE(
            Convert(&filter.value, {}, channels, {1}, {0, 1}, 1, &empty));
        EXPECT_TRUE(empty.empty());
      }
    }
  }
}

TEST(AudioResamplerTest, ExplicitEofPreservesEveryShortFinalFrame) {
  for (const auto& rates : {std::pair<uint32_t, uint32_t>{24000, 16000},
                            {44100, 16000},
                            {16000, 24000},
                            {16001, 16000},
                            {16000, 16000}}) {
    Filter filter;
    IREE_ASSERT_OK(loom_serve_audio_resampler_filter_initialize(
        rates.first, rates.second, kStorageLimit, iree_allocator_system(),
        &filter.value));
    for (size_t length : {1, 2, 3, 17, 63, 64, 65, 319, 320, 321}) {
      SCOPED_TRACE(::testing::Message()
                   << rates.first << " -> " << rates.second << ": " << length);
      const std::vector<float> input(length, 0.25f);
      std::vector<float> expected;
      std::vector<float> actual;
      ASSERT_NO_FATAL_FAILURE(
          Convert(&filter.value, input, 1, {length}, {65536}, 1024, &expected));
      ASSERT_NO_FATAL_FAILURE(Convert(&filter.value, input, 1, {1, 7, 3},
                                      {0, 1, 0, 7}, 1, &actual));
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST(AudioResamplerTest, IndependentStreamsShareOnlyImmutableCoefficients) {
  Filter filter;
  IREE_ASSERT_OK(loom_serve_audio_resampler_filter_initialize(
      24000, 16000, kStorageLimit, iree_allocator_system(), &filter.value));
  Stream stalled;
  IREE_ASSERT_OK(loom_serve_audio_resampler_initialize(
      &filter.value, 2, 17, kStorageLimit, iree_allocator_system(),
      &stalled.value));
  size_t capacity = 0;
  float* target = loom_serve_audio_resampler_reserve(&stalled.value, &capacity);
  std::vector<float> held(capacity * 2, 0.125f);
  memcpy(target, held.data(), held.size() * sizeof(float));
  loom_serve_audio_resampler_commit(&stalled.value, capacity);
  EXPECT_EQ(loom_serve_audio_resampler_read(&stalled.value, 0, nullptr), 0u);
  EXPECT_FALSE(loom_serve_audio_resampler_is_complete(&stalled.value));

  const std::vector<float> peer_input(1021, -0.25f);
  std::vector<float> peer;
  ASSERT_NO_FATAL_FAILURE(
      Convert(&filter.value, peer_input, 1, {7, 83}, {1, 67}, 19, &peer));
  std::vector<float> expected;
  ASSERT_NO_FATAL_FAILURE(
      Convert(&filter.value, held, 2, {capacity}, {4096}, 1024, &expected));
  loom_serve_audio_resampler_finish(&stalled.value);
  std::vector<float> actual(expected.size());
  EXPECT_EQ(loom_serve_audio_resampler_read(&stalled.value, actual.size() / 2,
                                            actual.data()),
            actual.size() / 2);
  EXPECT_TRUE(loom_serve_audio_resampler_is_complete(&stalled.value));
  EXPECT_EQ(actual, expected);
  std::vector<float> fresh_peer;
  ASSERT_NO_FATAL_FAILURE(
      Convert(&filter.value, peer_input, 1, {1021}, {4096}, 1024, &fresh_peer));
  EXPECT_EQ(peer, fresh_peer);
}

TEST(AudioResamplerTest, PreservesAnalyticPassbandAndRejectsAliases) {
  for (const auto& rates : {std::pair<uint32_t, uint32_t>{24000, 16000},
                            {44100, 16000},
                            {48000, 16000},
                            {16000, 24000},
                            {8000, 48000},
                            {192000, 8000},
                            {22050, 24000},
                            {16001, 16000},
                            {16000, 16001},
                            {16000, 16000}}) {
    SCOPED_TRACE(::testing::Message() << rates.first << " -> " << rates.second);
    Filter filter;
    IREE_ASSERT_OK(loom_serve_audio_resampler_filter_initialize(
        rates.first, rates.second, kStorageLimit, iree_allocator_system(),
        &filter.value));
    const double smallest = std::min(rates.first, rates.second);
    const auto signal = [&](double time) {
      return 0.05 + 0.10 * std::sin(2 * kPi * 0.061 * smallest * time) +
             0.11 * std::cos(2 * kPi * 0.233 * smallest * time) +
             0.09 * std::sin(2 * kPi * 0.449 * smallest * time);
    };
    const size_t length = rates.first / 4 + 17;
    std::vector<float> input(length * 2, 0);
    for (size_t i = 0; i < length; ++i) {
      input[i * 2] =
          static_cast<float>(signal(static_cast<double>(i) / rates.first));
    }
    std::vector<float> output;
    ASSERT_NO_FATAL_FAILURE(
        Convert(&filter.value, input, 2, {137}, {1024}, 1024, &output));
    const size_t margin = rates.second / 50;
    double signal_power = 0;
    double error_power = 0;
    double peak_error = 0;
    for (size_t i = margin; i < output.size() / 2 - margin; ++i) {
      const double expected = signal(static_cast<double>(i) / rates.second);
      const double error = output[i * 2] - expected;
      signal_power += expected * expected;
      error_power += error * error;
      peak_error = std::max(peak_error, std::abs(error));
    }
    EXPECT_LT(peak_error, 3e-5);
    EXPECT_LT(error_power, signal_power * 1e-9);
    for (size_t i = 0; i < output.size() / 2; ++i) {
      EXPECT_EQ(output[i * 2 + 1], 0.0f);
    }
    if (rates.first > rates.second) {
      const double frequency =
          std::min(0.5001 * rates.second, 0.499999 * rates.first);
      input.resize(length);
      for (size_t i = 0; i < length; ++i) {
        input[i] =
            static_cast<float>(std::sin(2 * kPi * frequency * i / rates.first));
      }
      ASSERT_NO_FATAL_FAILURE(
          Convert(&filter.value, input, 1, {101}, {1024}, 1024, &output));
      double peak_alias = 0;
      for (size_t i = margin; i < output.size() - margin; ++i) {
        peak_alias =
            std::max(peak_alias, std::abs(static_cast<double>(output[i])));
      }
      EXPECT_LT(peak_alias, 3e-5);
    }
  }
}

TEST(AudioResamplerTest, GeometryAndStorageFailAtConstruction) {
  for (const auto& rates :
       {std::pair<uint32_t, uint32_t>{0, 16000}, {16000, 0}, {0, 0}}) {
    Filter filter;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_audio_resampler_filter_initialize(
                              rates.first, rates.second, kStorageLimit,
                              iree_allocator_system(), &filter.value));
  }
  {
    Filter filter;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          loom_serve_audio_resampler_filter_initialize(
                              UINT32_MAX, 1, kStorageLimit,
                              iree_allocator_system(), &filter.value));
  }
  Filter filter;
  IREE_ASSERT_OK(loom_serve_audio_resampler_filter_initialize(
      24000, 16000, kStorageLimit, iree_allocator_system(), &filter.value));
  for (const auto& geometry : {std::pair<uint32_t, size_t>{0, 1}, {1, 0}}) {
    Stream stream;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_audio_resampler_initialize(
            &filter.value, geometry.first, geometry.second, kStorageLimit,
            iree_allocator_system(), &stream.value));
  }
  for (const auto& geometry : {std::pair<uint32_t, size_t>{1, SIZE_MAX},
                               {UINT32_MAX, SIZE_MAX / 2},
                               {1, 1024}}) {
    Stream stream;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          loom_serve_audio_resampler_initialize(
                              &filter.value, geometry.first, geometry.second,
                              16, iree_allocator_system(), &stream.value));
  }
  Filter identity;
  IREE_ASSERT_OK(loom_serve_audio_resampler_filter_initialize(
      UINT32_MAX, UINT32_MAX, 0, iree_allocator_system(), &identity.value));
  const std::vector<float> samples = {-1.0f, -0.0f, 0.25f, 1.0f};
  std::vector<float> output;
  ASSERT_NO_FATAL_FAILURE(
      Convert(&identity.value, samples, 1, {1}, {1}, 1, &output));
  ASSERT_EQ(output.size(), samples.size());
  EXPECT_EQ(
      memcmp(output.data(), samples.data(), samples.size() * sizeof(float)), 0);
}

}  // namespace
