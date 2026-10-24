// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/media/audio_decoder.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/test_allocator.h"

IREE_FLAG(string, audio_fixture, "", "A file in the audio fixture directory.");

namespace {

using Decoder = std::unique_ptr<loom_serve_audio_decoder_t,
                                decltype(&loom_serve_audio_decoder_free)>;

std::vector<uint8_t> Wave(uint32_t sample_rate, uint32_t channels,
                          const std::vector<float>& samples, bool floating) {
  const uint32_t sample_bytes = floating ? 4 : 2;
  std::vector<uint8_t> data(44 + sample_bytes * samples.size());
  memcpy(data.data(), "RIFF", 4);
  iree_unaligned_store_le_u32(data.data() + 4, data.size() - 8);
  memcpy(data.data() + 8, "WAVEfmt ", 8);
  iree_unaligned_store_le_u32(data.data() + 16, 16);
  iree_unaligned_store_le_u16(data.data() + 20, floating ? 3 : 1);
  iree_unaligned_store_le_u16(data.data() + 22, channels);
  iree_unaligned_store_le_u32(data.data() + 24, sample_rate);
  iree_unaligned_store_le_u32(data.data() + 28,
                              sample_rate * channels * sample_bytes);
  iree_unaligned_store_le_u16(data.data() + 32, channels * sample_bytes);
  iree_unaligned_store_le_u16(data.data() + 34, sample_bytes * 8);
  memcpy(data.data() + 36, "data", 4);
  iree_unaligned_store_le_u32(data.data() + 40, data.size() - 44);
  for (size_t i = 0; i < samples.size(); ++i) {
    if (floating) {
      iree_unaligned_store_le_f32(data.data() + 44 + i * sample_bytes,
                                  samples[i]);
    } else {
      iree_unaligned_store_le_u16(data.data() + 44 + i * sample_bytes,
                                  static_cast<int16_t>(samples[i] * 32768));
    }
  }
  return data;
}

std::vector<uint8_t> File(const char* name) {
  std::ifstream file(
      std::filesystem::path(FLAG_audio_fixture).parent_path() / name,
      std::ios::binary);
  EXPECT_TRUE(file.good());
  return {std::istreambuf_iterator<char>(file), {}};
}

iree_status_t Decode(
    const std::vector<uint8_t>& encoded, uint32_t sample_rate,
    uint32_t channels, size_t block, std::vector<float>* output,
    iree_allocator_t host_allocator = iree_allocator_system()) {
  loom_serve_audio_decoder_t* raw = nullptr;
  IREE_RETURN_IF_ERROR(loom_serve_audio_decoder_allocate(
      iree_make_const_byte_span(encoded.data(), encoded.size()), sample_rate,
      channels, host_allocator, &raw));
  Decoder decoder(raw, loom_serve_audio_decoder_free);
  std::vector<float> buffer(block * channels);
  bool complete = false;
  iree_status_t status = iree_ok_status();
  while (!complete && iree_status_is_ok(status)) {
    iree_host_size_t count = 99;
    status = loom_serve_audio_decoder_read(decoder.get(), 0, nullptr, &count,
                                           &complete);
    EXPECT_EQ(count, 0u);
    EXPECT_FALSE(complete);
    if (iree_status_is_ok(status)) {
      status = loom_serve_audio_decoder_read(decoder.get(), block,
                                             buffer.data(), &count, &complete);
    }
    if (iree_status_is_ok(status)) {
      EXPECT_TRUE(count || complete);
      output->insert(output->end(), buffer.begin(),
                     buffer.begin() + count * channels);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t count = 99;
    status = loom_serve_audio_decoder_read(decoder.get(), block, buffer.data(),
                                           &count, &complete);
    EXPECT_EQ(count, 0u);
    EXPECT_TRUE(complete);
  }
  return status;
}

TEST(AudioDecoderTest, ExactFramesAndIndependentReadCredit) {
  std::vector<float> samples(1607);
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i] = (static_cast<int>(i % 31) - 15) / 32.0f;
  }
  for (bool floating : {false, true}) {
    const auto encoded = Wave(16000, 1, samples, floating);
    for (size_t block : {1, 7, 64, 511, 4096}) {
      std::vector<float> decoded;
      if (LOOM_TEST_AUDIO_WAV) {
        IREE_ASSERT_OK(Decode(encoded, 16000, 1, block, &decoded));
        EXPECT_EQ(decoded, samples);
      } else {
        IREE_EXPECT_NOT_OK(Decode(encoded, 16000, 1, block, &decoded));
      }
    }
  }
}

TEST(AudioDecoderTest, CompressedSamplesMatchIndependentDecoder) {
  for (const auto& fixture : {std::make_pair("flac", LOOM_TEST_AUDIO_FLAC),
                              std::make_pair("mp3", LOOM_TEST_AUDIO_MP3)}) {
    const auto encoded = File((std::string("sample.") + fixture.first).c_str());
    const auto reference = File((std::string(fixture.first) + ".f32").c_str());
    ASSERT_EQ(reference.size() % sizeof(float), 0u);
    for (size_t block : {1, 7, 64, 511, 4096}) {
      std::vector<float> decoded;
      if (fixture.second) {
        IREE_ASSERT_OK(Decode(encoded, 16000, 1, block, &decoded));
        ASSERT_EQ(decoded.size() * sizeof(float), reference.size());
        for (size_t i = 0; i < decoded.size(); ++i) {
          EXPECT_NEAR(decoded[i],
                      iree_unaligned_load_le_f32(reference.data() + i * 4),
                      0.00001f);
        }
      } else {
        IREE_EXPECT_NOT_OK(Decode(encoded, 16000, 1, block, &decoded));
      }
    }
  }
}

TEST(AudioDecoderTest, ResamplingAndChannelMixingSurviveReadBoundaries) {
  for (uint32_t rate : {16000, 24000, 44100, 48000}) {
    std::vector<float> samples(2 * (rate / 10 + 13));
    for (size_t i = 0; i < samples.size() / 2; ++i) {
      samples[i * 2] = 0.5f * std::sin(6.283185307179586 * 731 * i / rate);
      samples[i * 2 + 1] = 0.25f * std::sin(6.283185307179586 * 401 * i / rate);
    }
    const auto encoded = Wave(rate, 2, samples, true);
    std::vector<float> reference;
    if (!LOOM_TEST_AUDIO_WAV) {
      IREE_EXPECT_NOT_OK(Decode(encoded, 16000, 1, 4096, &reference));
      continue;
    }
    IREE_ASSERT_OK(Decode(encoded, 16000, 1, 4096, &reference));
    EXPECT_NEAR(reference.size(), samples.size() / 2.0 * 16000 / rate, 1.0);
    for (size_t block : {1, 7, 64, 511}) {
      std::vector<float> decoded;
      IREE_ASSERT_OK(Decode(encoded, 16000, 1, block, &decoded));
      EXPECT_EQ(decoded, reference);
    }
    if (rate == 16000) {
      for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_FLOAT_EQ(reference[i],
                        (samples[i * 2] + samples[i * 2 + 1]) * 0.5f);
      }
    }
  }
}

TEST(AudioDecoderTest, ZeroGeometryPreservesOriginalSamples) {
  if (!LOOM_TEST_AUDIO_WAV) {
    return;
  }
  for (uint32_t rate : {8000u, 16000u, 22050u, 24000u, 44100u, 48000u}) {
    for (uint32_t channels : {1u, 2u, 6u}) {
      std::vector<float> samples(31 * channels);
      for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = (static_cast<int>(i % 13) - 6) / 16.0f;
      }
      const auto encoded = Wave(rate, channels, samples, true);
      loom_serve_audio_decoder_t* raw = nullptr;
      IREE_ASSERT_OK(loom_serve_audio_decoder_allocate(
          iree_make_const_byte_span(encoded.data(), encoded.size()), 0, 0,
          iree_allocator_system(), &raw));
      Decoder decoder(raw, loom_serve_audio_decoder_free);
      EXPECT_EQ(loom_serve_audio_decoder_sample_rate(raw), rate);
      EXPECT_EQ(loom_serve_audio_decoder_channels(raw), channels);
      std::vector<float> output(samples.size());
      iree_host_size_t count = 0;
      bool complete = false;
      IREE_ASSERT_OK(loom_serve_audio_decoder_read(raw, 31, output.data(),
                                                   &count, &complete));
      EXPECT_EQ(count, 31u);
      EXPECT_EQ(output, samples);
    }
  }
}

TEST(AudioDecoderTest, InvalidInputRetainsNoDecoder) {
  for (const std::string bytes :
       {"", "not audio", "RIFF\x7f\x7f\x7f\x7fWAVE", "fLaC", "ID3"}) {
    loom_serve_audio_decoder_t* decoder = nullptr;
    IREE_EXPECT_NOT_OK(loom_serve_audio_decoder_allocate(
        iree_make_const_byte_span(bytes.data(), bytes.size()), 16000, 1,
        iree_allocator_system(), &decoder));
    EXPECT_EQ(decoder, nullptr);
  }
  loom_serve_audio_decoder_free(nullptr);
}

TEST(AudioDecoderTest, CodecAndResamplerAllocationFailuresPropagate) {
  const std::vector<uint8_t> inputs[] = {
      Wave(16000, 1, std::vector<float>(1607, 0.25f), false),
      File("sample.mp3"),
      File("sample.flac"),
  };
  const bool enabled[] = {LOOM_TEST_AUDIO_WAV, LOOM_TEST_AUDIO_MP3,
                          LOOM_TEST_AUDIO_FLAC};
  for (size_t i = 0; i < IREE_ARRAYSIZE(inputs); ++i) {
    if (!enabled[i]) {
      continue;
    }
    SCOPED_TRACE(i);
    iree::vm::testing::CountingAllocator baseline;
    std::vector<float> output;
    IREE_ASSERT_OK(
        Decode(inputs[i], 24000, 2, 7, &output, baseline.allocator()));
    for (size_t fail_at = 1; fail_at <= baseline.allocation_count();
         ++fail_at) {
      SCOPED_TRACE(fail_at);
      iree::vm::testing::CountingAllocator allocator(fail_at);
      output.clear();
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          Decode(inputs[i], 24000, 2, 7, &output, allocator.allocator()));
    }
  }
}

}  // namespace
