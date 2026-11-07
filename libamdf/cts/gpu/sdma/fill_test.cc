// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

class SdmaFillTest : public GpuCommandTest {
 protected:
  SdmaFillTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER |
                                 AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        }) {}
};

TEST_F(SdmaFillTest, ConstantFillFeedsCopiesAcrossEpochs) {
  constexpr size_t kDataLength = 8192;
  constexpr size_t kControlLength = 4096;
  constexpr size_t kCompletionWord = 16;
  constexpr size_t kEpochCount = 2;
  struct FillSpan {
    // Byte offset of the fill within the target allocation.
    uint32_t target_offset;
    // Byte offset of its dependent copy within the output allocation.
    uint32_t output_offset;
    // Nonzero DWORD-aligned byte length.
    uint32_t length;
    // Full 32-bit patterns, changed between completed uses of the backing.
    std::array<uint32_t, kEpochCount> patterns;
  };
  constexpr std::array<FillSpan, 3> kSpans = {
      {{64, 192, 4, {0x00000000u, 0xffffffffu}},
       {124, 252, 8, {0x80000000u, 0x00000001u}},
       {4092, 4076, 1028, {0x6d2ac491u, 0xb730e85au}}}};
  const bool user_gcr =
      (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
  const size_t words_per_epoch =
      (kSpans.size() * (5 + 7) + 1 + 4 + (user_gcr ? 10 : 0) + 7) & ~size_t{7};
  GpuMemory* target = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* completion = nullptr;
  // The fill destination becomes the copy source without a host observation.
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kDataLength, &target));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kDataLength, &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kControlLength, &completion));
  ASSERT_EQ(target->device_address % 4096, 0u);
  ASSERT_EQ(output->device_address % 4096, 0u);
  ASSERT_EQ(completion->device_address % sizeof(uint32_t), 0u);
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  ASSERT_GT(queue->words().size(), kEpochCount * words_per_epoch);

  // Prebuild both command spans and guard all remaining command backing.
  // Each publication exposes one new span; no command bytes change in flight.
  std::vector<uint32_t> expected_commands(queue->words().size(), 0x6935bdefu);
  std::vector<uint32_t> observed_commands(queue->words().size());
  std::memcpy(queue->words().data(), expected_commands.data(),
              queue->words().size_bytes());
  SdmaCommandWriter commands(queue->words().data(), family_.format_features);
  for (size_t epoch = 0; epoch < kEpochCount; ++epoch) {
    if (user_gcr) {
      commands.AcquireFromSystem();
    }
    for (const auto& span : kSpans) {
      commands.Fill32(target->device_address + span.target_offset,
                      span.patterns[epoch], span.length);
    }
    // Disjoint fills may overlap. Join them before their dependent reads;
    // neither the host nor an intermediate cache operation supplies this edge.
    commands.Noop();
    for (const auto& span : kSpans) {
      commands.CopyLinear(target->device_address + span.target_offset,
                          output->device_address + span.output_offset,
                          span.length);
    }
    if (user_gcr) {
      commands.ReleaseToSystem();
    }
    commands.Fence32(
        completion->device_address + kCompletionWord * sizeof(uint32_t),
        static_cast<uint32_t>(epoch + 1));
    commands.PadToEightWords();
  }
  ASSERT_EQ(commands.word_count(), kEpochCount * words_per_epoch);
  std::memcpy(expected_commands.data(), queue->words().data(),
              queue->words().size_bytes());

  std::array<uint32_t, kDataLength / sizeof(uint32_t)> expected_target;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> expected_output;
  std::array<uint32_t, kControlLength / sizeof(uint32_t)> expected_completion;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> observed_target;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> observed_output;
  std::array<uint32_t, kControlLength / sizeof(uint32_t)> observed_completion;
  auto* target_words = static_cast<uint32_t*>(target->host.pointer);
  auto* output_words = static_cast<uint32_t*>(output->host.pointer);
  auto* completion_words = static_cast<uint32_t*>(completion->host.pointer);
  RecordProperty("sdma_format_features",
                 std::to_string(family_.format_features));
  RecordProperty("fills_per_epoch", static_cast<int>(kSpans.size()));
  RecordProperty("words_per_epoch", static_cast<int>(words_per_epoch));
  RecordProperty("data_checked_bytes_per_epoch", 2 * kDataLength);
  RecordProperty("control_checked_bytes_per_epoch", kControlLength);
  RecordProperty("command_checked_bytes_per_epoch",
                 queue->words().size_bytes());
  RecordProperty("completed_epochs", 0);

  for (size_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    const uint32_t marker = static_cast<uint32_t>(epoch + 1);
    for (size_t i = 0; i < expected_target.size(); ++i) {
      const uint32_t word = static_cast<uint32_t>(i);
      expected_target[i] = 0x25a64bc3u ^ (marker + word * 0x03050709u);
      expected_output[i] = 0x4962d5e7u ^ (marker + word * 0x0507090bu);
    }
    for (size_t i = 0; i < expected_completion.size(); ++i) {
      expected_completion[i] =
          0x6de8912fu ^ (marker + static_cast<uint32_t>(i) * 0x07090b0du);
    }
    std::memcpy(target->host.pointer, expected_target.data(), kDataLength);
    std::memcpy(output->host.pointer, expected_output.data(), kDataLength);
    std::memcpy(completion->host.pointer, expected_completion.data(),
                kControlLength);
    for (const auto& span : kSpans) {
      for (size_t i = 0; i < span.length / sizeof(uint32_t); ++i) {
        // Derive both oracles from the requested pattern, never device output.
        const uint32_t pattern = span.patterns[epoch];
        expected_target[span.target_offset / sizeof(uint32_t) + i] = pattern;
        expected_output[span.output_offset / sizeof(uint32_t) + i] = pattern;
        target_words[span.target_offset / sizeof(uint32_t) + i] =
            pattern ^ 0x55555555u;
        output_words[span.output_offset / sizeof(uint32_t) + i] =
            pattern ^ 0xaaaaaaaau;
      }
    }
    completion_words[kCompletionWord] = marker - 1;
    expected_completion[kCompletionWord] = marker;

    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, (epoch + 1) * words_per_epoch));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(completion->host.pointer) +
            kCompletionWord * sizeof(uint32_t),
        marker));
    if (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR) {
      ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    }

    // Observe the final consumer first. Its bytes establish the fill-to-copy
    // dependency even when native retirement supplies the final release.
    std::memcpy(observed_output.data(), output->host.pointer, kDataLength);
    std::memcpy(observed_target.data(), target->host.pointer, kDataLength);
    std::memcpy(observed_completion.data(), completion->host.pointer,
                kControlLength);
    std::memcpy(observed_commands.data(), queue->words().data(),
                queue->words().size_bytes());
    for (size_t i = 0; i < expected_target.size(); ++i) {
      EXPECT_EQ(observed_output[i], expected_output[i]) << "output word=" << i;
      EXPECT_EQ(observed_target[i], expected_target[i]) << "target word=" << i;
    }
    for (size_t i = 0; i < expected_completion.size(); ++i) {
      EXPECT_EQ(observed_completion[i], expected_completion[i])
          << "completion word=" << i;
    }
    EXPECT_EQ(observed_commands, expected_commands);
    // Oracle failures still retire the accepted commands, but prevent reuse.
    EXPECT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    if (HasFailure()) {
      return;
    }
    RecordProperty("completed_epochs", static_cast<int>(marker));
    RecordProperty(
        "retired_byte_frontier",
        static_cast<int>((epoch + 1) * words_per_epoch * sizeof(uint32_t)));
  }
}

TEST_F(SdmaFillTest, ByteEdgesComposeWithDwordFillsAcrossEpochs) {
  struct Geometry {
    // Natural alignment and period of the repeating pattern, in bytes.
    uint32_t pattern_length;
    // Target displacement modulo a DWORD, aligned to the pattern period.
    uint32_t displacement;
  };
  constexpr std::array<Geometry, 7> kGeometries = {
      {{1, 0}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 2}, {4, 0}}};
  constexpr std::array<uint32_t, 6> kElementCounts = {1, 2, 3, 4, 7, 2051};
  constexpr size_t kSpanCount = kGeometries.size() * kElementCounts.size();
  constexpr size_t kStride = 16384;
  constexpr size_t kDataLength = kSpanCount * kStride;
  constexpr size_t kEpochCount = 2;
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  GpuMemory* patterns = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &patterns));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, kDataLength, &target));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, kDataLength, &output));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &completion));
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(16384, &queue));
  // Each span needs at most two byte copies, one fill, and one consumer copy.
  ASSERT_GE(queue->words().size(), kEpochCount * (kSpanCount * 26 + 32));
  SdmaCommandWriter commands(queue->words().data(), family_.format_features);
  const bool user_gcr =
      (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
  std::array<uint32_t, 1024> expected_patterns;
  std::array<uint32_t, 1024> observed_patterns;
  std::vector<uint8_t> expected_target(kDataLength);
  std::vector<uint8_t> expected_output(kDataLength);
  std::vector<uint8_t> observed_target(kDataLength);
  std::vector<uint8_t> observed_output(kDataLength);
  RecordProperty("fill_spans_per_epoch", kSpanCount);
  RecordProperty("completed_epochs", 0);
  for (size_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    expected_patterns.fill(0xd3916e25u);
    std::fill(expected_target.begin(), expected_target.end(), 0x6d ^ epoch);
    std::fill(expected_output.begin(), expected_output.end(), 0xa7 ^ epoch);
    std::memcpy(target->host.pointer, expected_target.data(), kDataLength);
    std::memcpy(output->host.pointer, expected_output.data(), kDataLength);
    *static_cast<uint32_t*>(completion->host.pointer) = 0;
    if (user_gcr) {
      commands.AcquireFromSystem();
    }
    for (size_t i = 0; i < kSpanCount; ++i) {
      const auto geometry = kGeometries[i / kElementCounts.size()];
      const uint32_t length =
          kElementCounts[i % kElementCounts.size()] * geometry.pattern_length;
      const size_t target_offset = i * kStride + geometry.displacement;
      const uint32_t value =
          (0x6d2ac491u + static_cast<uint32_t>(i) * 0x03050709u) ^
          (epoch ? 0xffffffffu : 0);
      uint32_t pattern = value;
      if (geometry.pattern_length == 1) {
        pattern = (value & 0xffu) * 0x01010101u;
      } else if (geometry.pattern_length == 2) {
        pattern = (value & 0xffffu) * 0x00010001u;
      }
      expected_patterns[i] = pattern;
      const uint64_t pattern_address = patterns->device_address + i * 4;
      const uint64_t target_address = target->device_address + target_offset;
      const uint32_t head =
          std::min(length, static_cast<uint32_t>((-target_address) & 3));
      const uint32_t body = (length - head) & ~uint32_t{3};
      const uint32_t tail = length - head - body;
      if (head) {
        commands.CopyLinear(pattern_address, target_address, head);
      }
      if (body) {
        commands.Fill32(target_address + head, pattern, body);
      }
      if (tail) {
        commands.CopyLinear(pattern_address, target_address + head + body,
                            tail);
      }
      // Natural alignment makes each edge a whole number of pattern periods.
      // The oracle uses original pattern bytes, not the command decomposition.
      for (uint32_t byte = 0; byte < length; ++byte) {
        const uint8_t expected =
            value >> ((byte % geometry.pattern_length) * 8);
        expected_target[target_offset + byte] = expected;
        expected_output[i * kStride + 31 + byte] = expected;
      }
    }
    commands.Noop();
    for (size_t i = 0; i < kSpanCount; ++i) {
      const auto geometry = kGeometries[i / kElementCounts.size()];
      commands.CopyLinear(
          target->device_address + i * kStride + geometry.displacement,
          output->device_address + i * kStride + 31,
          kElementCounts[i % kElementCounts.size()] * geometry.pattern_length);
    }
    if (user_gcr) {
      commands.ReleaseToSystem();
    }
    commands.Fence32(completion->device_address,
                     static_cast<uint32_t>(epoch + 1));
    commands.PadToEightWords();
    std::memcpy(patterns->host.pointer, expected_patterns.data(), 4096);
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(completion->host.pointer), epoch + 1));
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    std::memcpy(observed_output.data(), output->host.pointer, kDataLength);
    std::memcpy(observed_target.data(), target->host.pointer, kDataLength);
    std::memcpy(observed_patterns.data(), patterns->host.pointer, 4096);
    EXPECT_EQ(observed_output, expected_output);
    EXPECT_EQ(observed_target, expected_target);
    EXPECT_EQ(observed_patterns, expected_patterns);
    if (HasFailure()) {
      return;
    }
    RecordProperty("completed_epochs", epoch + 1);
  }
}

}  // namespace
