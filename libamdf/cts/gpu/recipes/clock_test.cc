// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>

#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"
#include "util/observation.h"

namespace {

class QueueClockRecipeTest
    : public GpuCommandTest,
      public ::testing::WithParamInterface<std::tuple<
          amdf_queue_command_type_t, amdf_queue_publication_modes_t>> {
 protected:
  QueueClockRecipeTest()
      : GpuCommandTest({
            .command_type = std::get<0>(GetParam()),
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
            .publication_modes = std::get<1>(GetParam()),
        }) {}
};

TEST_P(QueueClockRecipeTest, NominalDurationsAndHostBounds) {
  const bool is_pm4 = family_.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
  const Pm4CommandProfile* pm4 = Pm4CommandProfile::Find(gpu_endpoint_info_);
  if (is_pm4 && pm4 == nullptr) {
    GTEST_SKIP() << "no PM4 encoding profile";
  }
  ASSERT_GE(gpu_api_->extension_version, AMDF_GPU_EXTENSION_VERSION_2);
  ObservationHostClock host_clock;
  ASSERT_NO_FATAL_FAILURE(host_clock.Initialize(api_));
  amdf_observation_clock_info_t clock = {};
  const amdf_status_t status =
      gpu_api_->device_query_queue_clock_info(device_, family_.ordinal, &clock);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "native queue-clock metadata unavailable";
  }
  ASSERT_EQ(status, AMDF_STATUS_OK);
  ASSERT_GT(clock.frequency_hz, 0u);
  ASSERT_LE(clock.counter_bit_count, 64u);
  EXPECT_EQ(clock.reserved, 0u);
  RecordProperty("clock_frequency_hz", std::to_string(clock.frequency_hz));
  RecordProperty("clock_counter_bit_count", clock.counter_bit_count);
  RecordProperty("host_frequency_hz",
                 std::to_string(host_clock.frequency_hz()));

  constexpr size_t kWordCount = 4096 / sizeof(uint64_t);
  constexpr size_t kCopyFirst = 8;
  constexpr size_t kCopyCount = 16;
  constexpr uint64_t kGuard = UINT64_C(0xac37cd5913459876);
  constexpr uint32_t kIterationCount = 8;
  GpuMemory* source = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* observations = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &source));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &target));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &observations));
  auto* input = static_cast<uint64_t*>(source->host.pointer);
  auto* output = static_cast<uint64_t*>(target->host.pointer);
  auto* samples = static_cast<uint64_t*>(observations->host.pointer);
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(kIterationCount * 256u * sizeof(uint32_t), &queue));
  ASSERT_GE(queue->words().size(), kIterationCount * 256u);
  size_t published_words = 0;
  uint64_t first_ticks = 0;
  uint64_t previous_end = 0;
  uint64_t host_origin = 0;
  ASSERT_EQ(host_clock.Read(&host_origin), AMDF_STATUS_OK);
  const double nanoseconds_per_tick = 1e9 / clock.frequency_hz;
  double lower_offset = -std::numeric_limits<double>::infinity();
  double upper_offset = std::numeric_limits<double>::infinity();
  for (uint32_t iteration = 0; iteration < kIterationCount; ++iteration) {
    SCOPED_TRACE(iteration);
    for (size_t i = 0; i < kWordCount; ++i) {
      input[i] = kGuard + iteration * UINT64_C(0x102030405) + i;
      output[i] = ~input[i];
    }
    std::fill_n(samples, kWordCount, kGuard);
    samples[0] = UINT64_MAX;
    samples[8] = 0;
    samples[32] = 0;
    const uint64_t source_address = source->device_address + kCopyFirst * 8;
    const uint64_t target_address = target->device_address + kCopyFirst * 8;
    if (is_pm4) {
      Pm4CommandWriter commands(queue->words().data() + published_words, *pm4);
      commands.SystemBarrier();
      commands.CopyGpuClock64(observations->device_address);
      for (size_t i = 0; i < kCopyCount; ++i) {
        commands.CopyData64(source_address + i * 8, target_address + i * 8);
      }
      commands.CopyGpuClock64(observations->device_address + 64);
      commands.SystemBarrier();
      commands.WriteData32(observations->device_address + 256, 1);
      commands.PadToEightWords();
      published_words += commands.word_count();
    } else {
      SdmaCommandWriter commands(queue->words().data() + published_words,
                                 family_.format_features);
      const bool user_gcr =
          family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
      if (user_gcr) {
        commands.AcquireFromSystem();
      }
      commands.WriteGlobalTimestamp(observations->device_address);
      commands.CopyLinear(source_address, target_address, kCopyCount * 8);
      commands.WriteGlobalTimestamp(observations->device_address + 64);
      if (user_gcr) {
        commands.ReleaseToSystem();
      }
      commands.Fence32(observations->device_address + 256, 1);
      commands.PadToEightWords();
      published_words += commands.word_count();
    }
    amdf_observation_clock_sample_t native_before = {};
    ASSERT_EQ(gpu_api_->device_query_clock_sample(device_, family_.ordinal,
                                                  &native_before),
              AMDF_STATUS_OK);
    uint64_t host_before = 0;
    ASSERT_EQ(host_clock.Read(&host_before), AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(queue->Publish(api_, gpu_api_, published_words));
    ASSERT_TRUE(
        GpuWaitEqual<uint32_t>(reinterpret_cast<uintptr_t>(samples + 32), 1));
    if (!is_pm4 && (family_.format_features &
                    AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR)) {
      ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    }
    uint64_t host_after = 0;
    ASSERT_EQ(host_clock.Read(&host_after), AMDF_STATUS_OK);
    const uint64_t begin = samples[0];
    const uint64_t end = samples[8];
    EXPECT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    amdf_observation_clock_sample_t native_after = {};
    ASSERT_EQ(gpu_api_->device_query_clock_sample(device_, family_.ordinal,
                                                  &native_after),
              AMDF_STATUS_OK);
    ASSERT_NE(begin, UINT64_MAX);
    ASSERT_NE(end, 0u);
    ASSERT_LE(previous_end, begin);
    ASSERT_LE(begin, end);
    EXPECT_EQ(native_before.clock.frequency_hz, clock.frequency_hz);
    EXPECT_EQ(native_after.clock.frequency_hz, clock.frequency_hz);
    EXPECT_LE(native_before.host.before, native_before.host.after);
    EXPECT_LE(native_before.host.after, host_before);
    EXPECT_LE(host_after, native_after.host.before);
    EXPECT_LE(native_after.host.before, native_after.host.after);
    EXPECT_LE(native_before.timestamp, native_after.timestamp);
    EXPECT_LE(native_before.host.timestamp, native_after.host.timestamp);
    EXPECT_LE(native_before.host.timestamp, native_before.host.after);
    EXPECT_LE(native_after.host.timestamp, native_after.host.after);
    for (size_t i = 0; i < kWordCount; ++i) {
      const uint64_t expected = kGuard + iteration * UINT64_C(0x102030405) + i;
      EXPECT_EQ(input[i], expected) << "source word " << i;
      EXPECT_EQ(output[i], i >= kCopyFirst && i < kCopyFirst + kCopyCount
                               ? expected
                               : ~expected)
          << "target word " << i;
      if (i != 0 && i != 8) {
        EXPECT_EQ(samples[i], i == 32 ? 1u : kGuard) << "sample guard " << i;
      }
    }
    if (iteration == 0) {
      first_ticks = begin;
    }
    const double nominal_duration_ns = (end - begin) * nanoseconds_per_tick;
    const double before_ns = host_clock.Nanoseconds(host_before - host_origin);
    const double after_ns = host_clock.Nanoseconds(host_after - host_origin);
    // Host enclosure is part of the nominal clock model below. The native
    // rate has no accuracy bound that makes enclosure a timestamp oracle.
    lower_offset = std::max(
        lower_offset, before_ns - (begin - first_ticks) * nanoseconds_per_tick);
    upper_offset = std::min(
        upper_offset, after_ns - (end - first_ticks) * nanoseconds_per_tick);
    const std::string prefix = "clock_sample_" + std::to_string(iteration);
    RecordProperty(prefix + "_nominal_duration_ns",
                   std::to_string(nominal_duration_ns));
    RecordObservationClockSample(prefix + "_native_before", native_before);
    RecordProperty(prefix + "_queue_begin", std::to_string(begin));
    RecordProperty(prefix + "_queue_end", std::to_string(end));
    RecordObservationClockSample(prefix + "_native_after", native_after);
    RecordProperty(prefix + "_host_before_ticks", std::to_string(host_before));
    RecordProperty(prefix + "_host_after_ticks", std::to_string(host_after));
    // Native-query and queue clocks need not share an epoch. Retain the
    // comparison while deriving the host relation from actual queue samples.
    RecordProperty(
        prefix + "_native_encloses_queue",
        native_before.timestamp <= begin && end <= native_after.timestamp);
    RecordProperty(prefix + "_host_bracket_ns",
                   std::to_string(after_ns - before_ns));
    previous_end = end;
    if (HasFailure()) {
      return;
    }
  }
  // Nominal units do not guarantee a fixed offset against the host oscillator.
  // An empty intersection rejects this capture's constant-offset model, not
  // timestamp publication. Consumers own drift estimation and refresh policy.
  RecordProperty("clock_constant_offset_consistent",
                 lower_offset <= upper_offset);
  RecordProperty("clock_offset_lower_ns", std::to_string(lower_offset));
  RecordProperty("clock_offset_upper_ns", std::to_string(upper_offset));
}

INSTANTIATE_TEST_SUITE_P(
    Engine, QueueClockRecipeTest,
    ::testing::Combine(::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                         AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA),
                       ::testing::Values(AMDF_QUEUE_PUBLICATION_MODE_USER,
                                         AMDF_QUEUE_PUBLICATION_MODE_KERNEL)),
    [](const auto& info) {
      return std::string(std::get<0>(info.param) ==
                                 AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                             ? "Pm4"
                             : "Sdma") +
             (std::get<1>(info.param) == AMDF_QUEUE_PUBLICATION_MODE_USER
                  ? "User"
                  : "Kernel");
    });

}  // namespace
