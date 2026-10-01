// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

struct WaitCase {
  // Stable parameter name describing the completion-value relation.
  const char* name;
  // Comparison applied to the full 32-bit memory operand.
  SdmaMemoryComparison comparison;
  // Unsatisfied operand for a consumer started before CPU publication.
  uint32_t initial_value;
  // Reference encoded in the memory poll.
  uint32_t reference;
  // Satisfying operand retained through the consumer's final use.
  uint32_t final_value;
};

// The GE cases stay below bit 31: they exercise exact and advanced completion
// values without assuming signed ordering or a wrap-aware timeline protocol.
constexpr std::array<WaitCase, 4> kWaitCases = {{
    {"EqualZero", SdmaMemoryComparison::kEqual, 1, 0, 0},
    {"EqualHighBit", SdmaMemoryComparison::kEqual, 0x7fffffff, 0x92345678,
     0x92345678},
    {"AtLeastExact", SdmaMemoryComparison::kGreaterOrEqual, 0x12344, 0x12345,
     0x12345},
    {"AtLeastAdvanced", SdmaMemoryComparison::kGreaterOrEqual, 0x12344, 0x12345,
     0x23456},
}};

enum class OperandPublication {
  kBeforeSubmission,
  kAfterArrival,
};

class SdmaWaitTest : public GpuCommandTest,
                     public ::testing::WithParamInterface<WaitCase> {
 protected:
  SdmaWaitTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER |
                                 AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        }) {}

  void RunWait(OperandPublication publication) {
    constexpr size_t kEpochCount = 2;
    constexpr size_t kDataLength = 8192;
    constexpr size_t kControlLength = 4096;
    constexpr size_t kSourceWord = 1020;
    constexpr size_t kTargetWord = 1009;
    constexpr size_t kCopyWordCount = 257;
    constexpr size_t kOperandWord = 16;
    constexpr size_t kArrivalWord = 32;
    constexpr size_t kCompletionWord = 48;
    constexpr size_t kMaximumWordsPerEpoch = 31;
    constexpr size_t kCommandWordCapacity =
        kEpochCount * kMaximumWordsPerEpoch + 1;
    const WaitCase& parameters = GetParam();
    const bool user_gcr =
        (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
    const size_t words_per_epoch = user_gcr ? 31 : 21;

    GpuMemory* source = nullptr;
    GpuMemory* target = nullptr;
    GpuMemory* control = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, kDataLength, &source));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     kDataLength, &target));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     kControlLength, &control));
    GpuCommandQueue* queue = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
    ASSERT_GE(queue->words().size(), kCommandWordCapacity);

    std::array<uint32_t, kCommandWordCapacity> expected_commands;
    expected_commands.fill(0x6935bdef);
    std::memcpy(queue->words().data(), expected_commands.data(),
                sizeof(expected_commands));
    SdmaCommandWriter commands(queue->words().data(), family_.format_features);
    for (uint32_t marker = 1; marker <= kEpochCount; ++marker) {
      commands.Fence32(
          control->device_address + kArrivalWord * sizeof(uint32_t), marker);
      commands.WaitMemory32(
          control->device_address + kOperandWord * sizeof(uint32_t),
          parameters.reference, parameters.comparison);
      if (user_gcr) {
        commands.AcquireFromSystem();
      }
      commands.CopyLinear(
          source->device_address + kSourceWord * sizeof(uint32_t),
          target->device_address + kTargetWord * sizeof(uint32_t),
          kCopyWordCount * sizeof(uint32_t));
      if (user_gcr) {
        commands.ReleaseToSystem();
      }
      commands.Fence32(
          control->device_address + kCompletionWord * sizeof(uint32_t), marker);
    }
    ASSERT_EQ(commands.word_count(), kEpochCount * words_per_epoch);
    std::memcpy(expected_commands.data(), queue->words().data(),
                sizeof(expected_commands));

    std::array<uint32_t, kDataLength / sizeof(uint32_t)> expected_source;
    std::array<uint32_t, kDataLength / sizeof(uint32_t)> expected_target;
    std::array<uint32_t, kControlLength / sizeof(uint32_t)> expected_control;
    std::array<uint32_t, kDataLength / sizeof(uint32_t)> observed_source;
    std::array<uint32_t, kDataLength / sizeof(uint32_t)> observed_target;
    std::array<uint32_t, kControlLength / sizeof(uint32_t)> observed_control;
    std::array<uint32_t, kCommandWordCapacity> observed_commands;
    for (size_t i = 0; i < expected_control.size(); ++i) {
      expected_control[i] =
          0xa1536bc9u ^ (static_cast<uint32_t>(i) * 0x01030709u);
    }
    expected_control[kArrivalWord] = 0;
    expected_control[kCompletionWord] = 0;
    const uintptr_t control_host =
        reinterpret_cast<uintptr_t>(control->host.pointer);
    auto* input = static_cast<uint32_t*>(source->host.pointer);
    auto* output = static_cast<uint32_t*>(target->host.pointer);
    RecordProperty("sdma_format_features",
                   std::to_string(family_.format_features));
    RecordProperty("sdma_publication_mode",
                   queue->publication_mode() == AMDF_QUEUE_PUBLICATION_MODE_USER
                       ? "user"
                       : "kernel");
    RecordProperty("comparison", static_cast<int>(parameters.comparison));
    RecordProperty("reference", std::to_string(parameters.reference));
    RecordProperty("final_value", std::to_string(parameters.final_value));
    RecordProperty("words_per_epoch", static_cast<int>(words_per_epoch));
    RecordProperty("copy_bytes_per_epoch", kCopyWordCount * sizeof(uint32_t));
    RecordProperty("data_checked_bytes_per_epoch", 2 * kDataLength);
    RecordProperty("control_checked_bytes_per_epoch", kControlLength);
    RecordProperty("completed_epochs", 0);

    for (size_t epoch = 0; epoch < kEpochCount; ++epoch) {
      SCOPED_TRACE(epoch);
      const uint32_t marker = static_cast<uint32_t>(epoch + 1);
      for (size_t i = 0; i < expected_source.size(); ++i) {
        const uint32_t word = static_cast<uint32_t>(i);
        expected_source[i] = parameters.final_value ^
                             (marker * 0x17a63d91u + word * 0x01030709u);
        expected_target[i] = 0x296dab41u ^ (marker + word * 0x05070b0du);
        input[i] = publication == OperandPublication::kBeforeSubmission
                       ? expected_source[i]
                       : ~expected_source[i];
        output[i] = expected_target[i];
      }
      for (size_t i = 0; i < kCopyWordCount; ++i) {
        expected_target[kTargetWord + i] = expected_source[kSourceWord + i];
        output[kTargetWord + i] = ~expected_source[kSourceWord + i];
      }
      expected_control[kOperandWord] =
          publication == OperandPublication::kBeforeSubmission
              ? parameters.final_value
              : parameters.initial_value;
      std::memcpy(control->host.pointer, expected_control.data(),
                  kControlLength);

      ASSERT_NO_FATAL_FAILURE(
          queue->Publish(api_, gpu_api_, (epoch + 1) * words_per_epoch));
      if (publication == OperandPublication::kAfterArrival) {
        // The confirmed prefix establishes queue arrival. Only then does the
        // CPU produce this generation's input and release its completion value.
        // No queued producer or host retirement supplies the missing edge.
        GpuWaitEqual<uint32_t>(control_host + kArrivalWord * sizeof(uint32_t),
                               marker);
        EXPECT_EQ(GpuLoadAcquire<uint32_t>(control_host +
                                           kCompletionWord * sizeof(uint32_t)),
                  marker - 1);
        std::memcpy(source->host.pointer, expected_source.data(), kDataLength);
        GpuStoreRelease<uint32_t>(
            control_host + kOperandWord * sizeof(uint32_t),
            parameters.final_value);
      }
      GpuWaitEqual<uint32_t>(control_host + kCompletionWord * sizeof(uint32_t),
                             marker);

      // Snapshot every oracle before native retirement can add synchronization.
      std::memcpy(observed_target.data(), target->host.pointer, kDataLength);
      std::memcpy(observed_source.data(), source->host.pointer, kDataLength);
      std::memcpy(observed_control.data(), control->host.pointer,
                  kControlLength);
      std::memcpy(observed_commands.data(), queue->words().data(),
                  sizeof(observed_commands));
      expected_control[kOperandWord] = parameters.final_value;
      expected_control[kArrivalWord] = marker;
      expected_control[kCompletionWord] = marker;
      for (size_t i = 0; i < expected_source.size(); ++i) {
        EXPECT_EQ(observed_target[i], expected_target[i])
            << "target word=" << i;
        EXPECT_EQ(observed_source[i], expected_source[i])
            << "source word=" << i;
      }
      for (size_t i = 0; i < expected_control.size(); ++i) {
        EXPECT_EQ(observed_control[i], expected_control[i])
            << "control word=" << i;
      }
      EXPECT_EQ(observed_commands, expected_commands);
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
};

TEST_P(SdmaWaitTest, AlreadySatisfiedValueAllowsFollowingCopy) {
  RunWait(OperandPublication::kBeforeSubmission);
}

TEST_P(SdmaWaitTest, CpuPublicationAfterArrivalReleasesFollowingCopy) {
  RunWait(OperandPublication::kAfterArrival);
}

INSTANTIATE_TEST_SUITE_P(Comparison, SdmaWaitTest,
                         ::testing::ValuesIn(kWaitCases),
                         [](const auto& info) { return info.param.name; });

}  // namespace
