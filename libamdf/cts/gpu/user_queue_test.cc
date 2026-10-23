// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

struct RingCapacityCase {
  // Native packet representation and progress units.
  amdf_queue_command_type_t command_type;
  // Requested primary capacity in bytes, or zero for the provider default.
  uint64_t byte_length;
  // Stable case suffix independent of endpoint identity.
  const char* name;
  // Required primary-ring locality, or the native default.
  amdf_queue_placement_t placement = AMDF_QUEUE_PLACEMENT_DEFAULT;
};

GpuQueueRequirements RingRequirements(amdf_queue_command_type_t command_type) {
  GpuQueueRequirements requirements = {
      .command_type = command_type,
      .roles = AMDF_QUEUE_ROLE_TRANSFER,
      .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
  };
  if (command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {
    requirements.roles |= AMDF_QUEUE_ROLE_CACHE_CONTROL;
    requirements.format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR;
    requirements.cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                    AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
    requirements.cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL;
  }
  return requirements;
}

class GpuUserQueueCapacityTest
    : public GpuCommandTest,
      public ::testing::WithParamInterface<RingCapacityCase> {
 protected:
  GpuUserQueueCapacityTest()
      : GpuCommandTest(RingRequirements(GetParam().command_type)) {}

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    if (GetParam().command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {
      amdf_gpu_endpoint_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
          .structure_size = sizeof(info),
      };
      const auto status = gpu_api_->endpoint_query_info(endpoint, &info);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      pm4_profile_ = Pm4CommandProfile::Find(info);
      if (!pm4_profile_) {
        *out_matches = false;
        return AMDF_STATUS_OK;
      }
    }
    return GpuCommandTest::MatchGpuEndpoint(endpoint, out_matches);
  }

  // Immutable PM4 encoding selected before borrowing the native device.
  const Pm4CommandProfile* pm4_profile_ = nullptr;
};

TEST_P(GpuUserQueueCapacityTest, CopiesAcrossWrapAndRetiredReuse) {
  const auto& test = GetParam();
  if (test.placement != AMDF_QUEUE_PLACEMENT_DEFAULT &&
      !(family_.placement_capabilities & (1u << test.placement))) {
    GTEST_SKIP() << "family does not support requested ring placement";
  }
  if (test.byte_length != 0 &&
      (test.byte_length < family_.minimum_ring_byte_length ||
       test.byte_length > family_.maximum_ring_byte_length ||
       test.byte_length % family_.ring_byte_length_alignment != 0)) {
    GTEST_SKIP() << "family does not admit requested ring capacity";
  }
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(family_, &queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, test.byte_length, 0,
                  nullptr, test.placement));
  const uint64_t capacity = queue->info.ring_byte_length;
  if (test.byte_length != 0) {
    ASSERT_EQ(capacity, test.byte_length);
  }
  ASSERT_GE(capacity, family_.minimum_ring_byte_length);
  ASSERT_LE(capacity, family_.maximum_ring_byte_length);
  ASSERT_EQ(capacity & (capacity - 1), 0u);
  ASSERT_GE(capacity, 4096u);
  RecordProperty("requested_ring_bytes", std::to_string(test.byte_length));
  RecordProperty("ring_bytes", std::to_string(capacity));

  // Each publication occupies exactly half the ring. All command bytes in a
  // half remain immutable until consumption; six halves cross three wraps.
  constexpr uint32_t kPublicationCount = 6;
  const size_t half_word_count = capacity / (2 * sizeof(uint32_t));
  const bool pm4 = test.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
  const bool user_gcr =
      (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
  // PM4: ten-word acquisition, six-word COPY_DATA, eight-word release.
  // SDMA: optional five-word GCRs, seven-word COPY, four-word FENCE.
  const size_t copy_count = pm4 ? (half_word_count - 18) / 6
                                : (half_word_count - (user_gcr ? 14 : 4)) / 7;
  const size_t payload_word_count =
      (32 + 2 * copy_count * kPublicationCount + 1023) & ~size_t{1023};
  GpuMemory* source = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ, payload_word_count * sizeof(uint32_t), &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   payload_word_count * sizeof(uint32_t), &target));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));

  constexpr uint32_t kGuard = 0x51a9c73d;
  constexpr size_t kMarkerWord = 16;
  std::vector<uint32_t> expected_input(payload_word_count, kGuard);
  std::vector<uint32_t> expected_output(payload_word_count, kGuard);
  std::vector<uint32_t> expected_completion(1024, kGuard);
  expected_completion[kMarkerWord] = 0;
  for (size_t i = 0; i < copy_count * kPublicationCount; ++i) {
    // Each copied DWORD has separately owned readable source trailing bytes
    // and an untouched destination guard. Every publication has unique data.
    expected_input[16 + 2 * i] =
        0x712395ab + static_cast<uint32_t>(i) * 0x10301;
  }
  const size_t payload_bytes = payload_word_count * sizeof(uint32_t);
  std::memcpy(source->host.pointer, expected_input.data(), payload_bytes);
  std::memcpy(target->host.pointer, expected_output.data(), payload_bytes);
  std::memcpy(completion->host.pointer, expected_completion.data(), 4096);
  std::vector<uint32_t> expected_ring(capacity / sizeof(uint32_t), 0);
  auto* ring = reinterpret_cast<uint32_t*>(queue->host.ring_address);
  std::memcpy(ring, expected_ring.data(), capacity);
  std::vector<uint32_t> observed_input(payload_word_count);
  std::vector<uint32_t> observed_output(payload_word_count);
  std::vector<uint32_t> observed_completion(1024);
  const uint64_t marker_address =
      completion->device_address + kMarkerWord * sizeof(uint32_t);
  const uint64_t marker_host_address =
      reinterpret_cast<uintptr_t>(completion->host.pointer) +
      kMarkerWord * sizeof(uint32_t);

  for (uint32_t publication = 0; publication < kPublicationCount;
       ++publication) {
    SCOPED_TRACE(publication);
    const size_t ring_offset = (publication % 2) * half_word_count;
    auto* words = expected_ring.data() + ring_offset;
    const size_t first_word = 16 + 2 * publication * copy_count;
    if (pm4) {
      Pm4CommandWriter commands(words, *pm4_profile_);
      commands.SystemBarrier();
      for (size_t i = 0; i < copy_count; ++i) {
        const size_t offset = (first_word + 2 * i) * sizeof(uint32_t);
        commands.CopyData32(source->device_address + offset,
                            target->device_address + offset);
      }
      commands.ReleaseSystem32(marker_address, publication + 1);
      commands.PadToEightWords();
      ASSERT_EQ(commands.word_count(), half_word_count);
    } else {
      SdmaCommandWriter commands(words, family_.format_features);
      if (user_gcr) {
        commands.AcquireFromSystem();
      }
      for (size_t i = 0; i < copy_count; ++i) {
        const size_t offset = (first_word + 2 * i) * sizeof(uint32_t);
        commands.CopyLinear(source->device_address + offset,
                            target->device_address + offset, sizeof(uint32_t));
      }
      if (user_gcr) {
        commands.ReleaseToSystem();
      }
      commands.Fence32(marker_address, publication + 1);
      while (commands.word_count() < half_word_count) {
        commands.Noop();
      }
      ASSERT_EQ(commands.word_count(), half_word_count);
    }
    for (size_t i = 0; i < copy_count; ++i) {
      expected_output[first_word + 2 * i] = expected_input[first_word + 2 * i];
    }
    expected_completion[kMarkerWord] = publication + 1;
    std::memcpy(ring + ring_offset, words, half_word_count * sizeof(uint32_t));
    const uint64_t producer_index =
        (publication + 1) * half_word_count * (pm4 ? 1 : sizeof(uint32_t));
    ASSERT_NO_FATAL_FAILURE(queue->PublishStream(producer_index));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(marker_host_address, publication + 1));
    // Snapshot payload before diagnostics or consumption
    // waits can supply a visibility edge. Retirement precedes any next reuse.
    std::memcpy(observed_output.data(), target->host.pointer, payload_bytes);
    std::memcpy(observed_input.data(), source->host.pointer, payload_bytes);
    std::memcpy(observed_completion.data(), completion->host.pointer, 4096);
    ASSERT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, producer_index));
    EXPECT_EQ(observed_output, expected_output);
    EXPECT_EQ(observed_input, expected_input);
    EXPECT_EQ(observed_completion, expected_completion);
    amdf_user_queue_status_t status = {
        .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS,
        .structure_size = sizeof(status),
    };
    ASSERT_EQ(api_->user_queue_query_status(queue->queue, &status),
              AMDF_STATUS_OK);
    EXPECT_EQ(status.producer_index, producer_index);
    EXPECT_EQ(status.consumed_index, producer_index);
    EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
    EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
  }
  RecordProperty("ring_wrap_count", kPublicationCount / 2);
}

// Small publications repeatedly replace fetched ring lines. Only coherent
// payload completion is read, so no ring readback can mask missing publication.
class GpuUserQueuePublicationTest : public GpuUserQueueCapacityTest {};

TEST_P(GpuUserQueuePublicationTest, IncrementalStoresAcrossRingWrap) {
  const auto& test = GetParam();
  if (!(family_.placement_capabilities & (1u << test.placement))) {
    GTEST_SKIP() << "family does not support requested ring placement";
  }
  const bool pm4 = test.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(family_, &queue,
                                      AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                                      AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
                                      4096, 0, nullptr, test.placement));
  GpuMemory* payload = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &payload));
  auto* host = static_cast<uint32_t*>(payload->host.pointer);
  std::memset(host, 0, 4096);
  host[0] = 0x712395ab;
  auto* ring = reinterpret_cast<uint32_t*>(queue->host.ring_address);
  const uint64_t marker = payload->device_address + 64;
  const uint64_t marker_host = reinterpret_cast<uintptr_t>(host + 16);
  uint64_t position = 0;
  for (uint32_t publication = 0; publication < 4096; ++publication) {
    SCOPED_TRACE(publication);
    size_t offset = position & 1023;
    if (offset + 24 > 1024) {
      if (pm4) {
        Pm4CommandWriter padding(ring + offset, *pm4_profile_);
        padding.Noop(1024 - offset);
      } else {
        std::memset(ring + offset, 0, (1024 - offset) * sizeof(uint32_t));
      }
      position += 1024 - offset;
      offset = 0;
    }
    if (pm4) {
      Pm4CommandWriter commands(ring + offset, *pm4_profile_);
      commands.SystemBarrier();
      commands.CopyData32(payload->device_address,
                          payload->device_address + 128);
      commands.ReleaseSystem32(marker, publication + 1);
      ASSERT_EQ(commands.word_count(), 24u);
    } else {
      SdmaCommandWriter commands(ring + offset, family_.format_features);
      const bool user_gcr = (family_.format_features &
                             AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
      if (user_gcr) {
        commands.AcquireFromSystem();
      }
      commands.CopyLinear(payload->device_address,
                          payload->device_address + 128, 4);
      if (user_gcr) {
        commands.ReleaseToSystem();
      }
      while (commands.word_count() < 20) {
        commands.Noop();
      }
      commands.Fence32(marker, publication + 1);
      ASSERT_EQ(commands.word_count(), 24u);
    }
    position += 24;
    ASSERT_NO_FATAL_FAILURE(
        queue->PublishStream(pm4 ? position : position * 4));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(marker_host, publication + 1));
    ASSERT_EQ(GpuLoadAcquire<uint32_t>(reinterpret_cast<uintptr_t>(host + 32)),
              host[0]);
  }
  ASSERT_NO_FATAL_FAILURE(
      queue->WaitConsumed(api_, pm4 ? position : position * 4));
}

constexpr RingCapacityCase kRingCases[] = {
    {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 4096, "Pm4HostLocal",
     AMDF_QUEUE_PLACEMENT_HOST_LOCAL},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 4096, "Pm4DeviceLocal",
     AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, 4096, "SdmaHostLocal",
     AMDF_QUEUE_PLACEMENT_HOST_LOCAL},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, 4096, "SdmaDeviceLocal",
     AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 0, "Pm4Default"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 4096, "Pm4FourKiB"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 16384, "Pm4SixteenKiB"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, 65536, "Pm4SixtyFourKiB"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, 0, "SdmaDefault"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, 4096, "SdmaFourKiB"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, 16384, "SdmaSixteenKiB"},
    {AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, 65536, "SdmaSixtyFourKiB"},
};

INSTANTIATE_TEST_SUITE_P(
    Placements, GpuUserQueuePublicationTest,
    ::testing::Values(kRingCases[0], kRingCases[1], kRingCases[2],
                      kRingCases[3]),
    [](const ::testing::TestParamInfo<RingCapacityCase>& info) {
      return info.param.name;
    });

INSTANTIATE_TEST_SUITE_P(
    Capacities, GpuUserQueueCapacityTest, ::testing::ValuesIn(kRingCases),
    [](const ::testing::TestParamInfo<RingCapacityCase>& info) {
      return info.param.name;
    });

}  // namespace
