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

class SdmaTimestampTest : public GpuCommandTest {
 protected:
  SdmaTimestampTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER |
                                 AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        }) {}
};

TEST_F(SdmaTimestampTest, GlobalTimestampsOrderDependentCopies) {
  constexpr uint64_t kByteLength = 65536;
  constexpr uint64_t kWordCount = kByteLength / sizeof(uint32_t);
  constexpr uint64_t kGuard = UINT64_C(0x76543210fedcba98);
  constexpr std::array<size_t, 3> kTimestampIndices = {4, 8, 12};
  constexpr std::array<uint64_t, 3> kInitialTimestamps = {UINT64_MAX, 0,
                                                          UINT64_MAX};
  GpuMemory* source = nullptr;
  GpuMemory* intermediate = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* observations = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kByteLength, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kByteLength, &intermediate));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kByteLength, &target));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &observations));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  auto* input = static_cast<uint32_t*>(source->host.pointer);
  auto* staging = static_cast<uint32_t*>(intermediate->host.pointer);
  auto* output = static_cast<uint32_t*>(target->host.pointer);
  auto* timestamps = static_cast<uint64_t*>(observations->host.pointer);
  for (uint64_t i = 0; i < kWordCount; ++i) {
    input[i] = 0x179b3de1u + static_cast<uint32_t>(i) * 0x01030507u;
    staging[i] = ~input[i];
    output[i] = 0;
  }
  std::fill_n(timestamps, 4096 / sizeof(uint64_t), kGuard);
  // Opposite initial halves expose a partial-width timestamp write through
  // both the replacement checks and ordering of the resulting 64-bit values.
  for (size_t i = 0; i < kTimestampIndices.size(); ++i) {
    timestamps[kTimestampIndices[i]] = kInitialTimestamps[i];
  }
  *static_cast<uint32_t*>(completion->host.pointer) = 0;
  std::vector<uint32_t> observed_staging(kWordCount);
  std::vector<uint32_t> observed_output(kWordCount);
  std::vector<uint32_t> observed_input(kWordCount);
  std::array<uint64_t, 4096 / sizeof(uint64_t)> observed_timestamps;
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  ASSERT_GE(queue->words().size_bytes(), 37u * sizeof(uint32_t));
  SdmaCommandWriter commands(queue->words().data(), family_.format_features);
  const bool user_gcr =
      (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
  if (user_gcr) {
    commands.AcquireFromSystem();
  }
  commands.WriteGlobalTimestamp(observations->device_address + 32);
  commands.CopyLinear(source->device_address, intermediate->device_address,
                      kByteLength);
  // PAL's SDMA timestamp contract completes preceding commands. This marker
  // is the only intervening command before a dependent read of the copy.
  commands.WriteGlobalTimestamp(observations->device_address + 64);
  commands.CopyLinear(intermediate->device_address, target->device_address,
                      kByteLength);
  commands.WriteGlobalTimestamp(observations->device_address + 96);
  if (user_gcr) {
    commands.ReleaseToSystem();
  }
  commands.Fence32(completion->device_address, 1);
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(
      reinterpret_cast<uintptr_t>(completion->host.pointer), 1));
  if (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR) {
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
  }
  // Capture every observation after the queue's advertised release boundary.
  std::memcpy(observed_staging.data(), staging, kByteLength);
  std::memcpy(observed_output.data(), output, kByteLength);
  std::memcpy(observed_input.data(), input, kByteLength);
  std::memcpy(observed_timestamps.data(), timestamps,
              sizeof(observed_timestamps));
  for (uint64_t i = 0; i < kWordCount; ++i) {
    const uint32_t expected =
        0x179b3de1u + static_cast<uint32_t>(i) * 0x01030507u;
    EXPECT_EQ(observed_staging[i], expected) << i;
    EXPECT_EQ(observed_output[i], expected) << i;
    EXPECT_EQ(observed_input[i], expected) << "source word " << i;
  }
  for (size_t i = 0; i < 4096 / sizeof(uint64_t); ++i) {
    const auto timestamp_index =
        std::find(kTimestampIndices.begin(), kTimestampIndices.end(), i);
    if (timestamp_index == kTimestampIndices.end()) {
      EXPECT_EQ(observed_timestamps[i], kGuard) << i;
    } else {
      EXPECT_NE(observed_timestamps[i],
                kInitialTimestamps[timestamp_index - kTimestampIndices.begin()])
          << i;
    }
  }
  EXPECT_LE(observed_timestamps[4], observed_timestamps[8]);
  EXPECT_LE(observed_timestamps[8], observed_timestamps[12]);
  EXPECT_LT(observed_timestamps[4], observed_timestamps[12]);
  // These are raw correctness observations, not elapsed nanoseconds or a
  // performance result. Queue-clock conversion is a separate recipe.
  RecordProperty("sdma_timestamp_first",
                 std::to_string(observed_timestamps[4]));
  RecordProperty("sdma_timestamp_middle",
                 std::to_string(observed_timestamps[8]));
  RecordProperty("sdma_timestamp_last",
                 std::to_string(observed_timestamps[12]));
  // Nonfatal oracle failures still reach normal retirement.
  ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
}

TEST_F(SdmaTimestampTest, CopiesSamplesToEightByteAlignedDestinations) {
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  constexpr uint64_t kGuard = UINT64_C(0xa79d3e516b28c4f0);
  constexpr size_t kSampleIndex = 16;
  constexpr std::array<size_t, 4> kTargetIndices = {1, 2, 3, 4};
  constexpr size_t kEpochCount = 2;
  GpuMemory* observations = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &observations));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &completion));
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  ASSERT_GE(queue->words().size(), kEpochCount * 64);
  SdmaCommandWriter commands(queue->words().data(), family_.format_features);
  const bool user_gcr =
      (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
  std::array<uint64_t, 512> initial;
  std::array<uint64_t, 512> observed;
  uint64_t previous = 0;
  RecordProperty("completed_epochs", 0);
  for (size_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    initial.fill(kGuard);
    for (size_t i : kTargetIndices) {
      initial[i] = (i + epoch) % 2 ? 0 : UINT64_MAX;
      initial[kSampleIndex + i * 4] = UINT64_MAX;
    }
    std::memcpy(observations->host.pointer, initial.data(), sizeof(initial));
    *static_cast<uint32_t*>(completion->host.pointer) = 0;
    if (user_gcr) {
      commands.AcquireFromSystem();
    }
    for (size_t i : kTargetIndices) {
      // Each private aligned sample survives until its delivered copy is
      // checked. The timestamp orders preceding commands; NOP delivers this
      // sample before its dependent copy to an ordinary HAL destination.
      commands.WriteGlobalTimestamp(observations->device_address +
                                    (kSampleIndex + i * 4) * sizeof(uint64_t));
      commands.Noop();
      commands.CopyLinear(observations->device_address +
                              (kSampleIndex + i * 4) * sizeof(uint64_t),
                          observations->device_address + i * sizeof(uint64_t),
                          sizeof(uint64_t));
    }
    if (user_gcr) {
      commands.ReleaseToSystem();
    }
    commands.Fence32(completion->device_address,
                     static_cast<uint32_t>(epoch + 1));
    commands.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(completion->host.pointer), epoch + 1));
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    std::memcpy(observed.data(), observations->host.pointer, sizeof(observed));
    for (size_t i = 0; i < observed.size(); ++i) {
      if (i >= kTargetIndices.front() && i <= kTargetIndices.back()) {
        // Compare each delivery with its own native sample, so duplicated or
        // stale copies cannot pass solely because they remain monotonic.
        // Zero is a valid tick and is never used as a completion condition.
        EXPECT_EQ(observed[i], observed[kSampleIndex + i * 4]);
        EXPECT_NE(observed[i], UINT64_MAX);
        EXPECT_LE(previous, observed[i]);
        previous = observed[i];
      } else if (i >= kSampleIndex + kTargetIndices.front() * 4 &&
                 i <= kSampleIndex + kTargetIndices.back() * 4 && !(i % 4)) {
        EXPECT_NE(observed[i], UINT64_MAX);
      } else {
        EXPECT_EQ(observed[i], kGuard) << i;
      }
    }
    if (HasFailure()) {
      return;
    }
    RecordProperty("completed_epochs", epoch + 1);
  }
}

}  // namespace
