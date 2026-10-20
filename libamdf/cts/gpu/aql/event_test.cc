// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/util/event_fixture.h"

namespace {

enum class CompletionPacket { kDispatch, kBarrierAnd, kBarrierOr };

struct CompletionCase {
  // Firmware packet that produces the consumer's final completion.
  CompletionPacket packet;
  // Whether the producer shares the consumer's event or remains host-polled.
  bool shared_event;
};

class AqlCompletionEventTest
    : public GpuEventTest<AqlDispatchTest>,
      public ::testing::WithParamInterface<CompletionCase> {};

TEST_P(AqlCompletionEventTest, DependentDispatchesRetireBeforeSignalReuse) {
  const auto* kernel = kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr);
  RecordProperty("transform_kernel_target", kernel->target);
  constexpr uint32_t kWordCount = 64;
  constexpr uint32_t kRounds = 131;
  constexpr uint64_t kRingBytes = 4096;
  GpuMemory* payload = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &payload));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  auto* signals = static_cast<aql::Signal*>(completion->host.pointer);
  for (uint32_t i = 0; i < 3; ++i) {
    signals[i] = aql::UserSignal();
  }
  // Single-source cases require the final consumer packet's wake. Shared
  // cases additionally exercise coalesced/early producer hints. The consumer's
  // intermediate dispatch never supplies the barrier's notification.
  for (uint32_t i = GetParam().shared_event ? 0 : 1; i < 2; ++i) {
    signals[i].event.mailbox_address = event_info_.mailbox_address;
    signals[i].event.id = event_info_.interrupt_context;
  }
  std::array<GpuUserQueue*, 2> queues;
  std::array<uint64_t, 2> indices = {};
  std::array<uint64_t, 2> descriptors = {};
  for (uint32_t i = 0; i < queues.size(); ++i) {
    ASSERT_NO_FATAL_FAILURE(
        CreateQueue(family_, &queues[i], AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                    AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, kRingBytes));
    ASSERT_NO_FATAL_FAILURE(PublishKernel(*queues[i], *kernel,
                                          i == 0 ? "producer" : "consumer",
                                          &indices[i], &descriptors[i]));
  }

  const bool barrier_completion =
      GetParam().packet != CompletionPacket::kDispatch;
  const uint64_t producer_address = completion->device_address;
  const uint64_t consumer_address = producer_address + sizeof(aql::Signal);
  const uint64_t intermediate_address = consumer_address + sizeof(aql::Signal);
  const auto dependency =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
                   {producer_address, 0, 0, 0, 0});
  const auto epilogue =
      aql::Barrier(GetParam().packet == CompletionPacket::kBarrierOr
                       ? aql::BarrierType::kOr
                       : aql::BarrierType::kAnd,
                   aql::HeaderBarrier::kDisabled, consumer_address,
                   {intermediate_address, 0, 0, 0, 0});
  auto* words = static_cast<uint32_t*>(payload->host.pointer);
  for (uint32_t round = 0; round < kRounds; ++round) {
    SCOPED_TRACE(round);
    const uint32_t addend = round * 0x01020305u;
    std::array<uint32_t, kWordCount> expected;
    for (uint32_t i = 0; i < kWordCount; ++i) {
      words[i] = round + i * 0x11111111u;
      expected[i] = (words[i] * 3 + addend) * 3 + 7;
      words[kWordCount + i] = ~expected[i];
      words[2 * kWordCount + i] = ~expected[i];
    }
    const std::array<kernels::transform::Arguments, 2> parameters = {{
        {payload->device_address,
         payload->device_address + kWordCount * sizeof(uint32_t), kWordCount,
         addend},
        {payload->device_address + kWordCount * sizeof(uint32_t),
         payload->device_address + 2 * kWordCount * sizeof(uint32_t),
         kWordCount, 7},
    }};
    std::memcpy(arguments->host.pointer, parameters.data(), sizeof(parameters));
    for (uint32_t i = 0; i < 3; ++i) {
      signals[i].value = 1;
    }
    const aql::DispatchGeometry geometry = {
        1,
        {static_cast<uint16_t>(kernel->workgroup_size()), 1, 1},
        {kWordCount, 1, 1}};
    const auto producer = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, geometry,
        kernel->private_segment_byte_length, kernel->group_segment_byte_length,
        descriptors[0], arguments->device_address, producer_address);
    const auto consumer = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, geometry,
        kernel->private_segment_byte_length, kernel->group_segment_byte_length,
        descriptors[1],
        arguments->device_address + sizeof(kernels::transform::Arguments),
        barrier_completion ? intermediate_address : consumer_address);

    // Queue the dependent work first. Firmware, not a host observation of
    // producer completion, releases the consumer.
    GpuStoreRelease(queues[1]->host.write_index_address,
                    indices[1] + (barrier_completion ? 3 : 2));
    ASSERT_TRUE(aql::Publish(*queues[1], indices[1]++, dependency));
    ASSERT_TRUE(aql::Publish(*queues[1], indices[1]++, consumer));
    if (barrier_completion) {
      ASSERT_TRUE(aql::Publish(*queues[1], indices[1]++, epilogue));
    }
    GpuStoreRelease(queues[0]->host.write_index_address, indices[0] + 1);
    ASSERT_TRUE(aql::Publish(*queues[0], indices[0]++, producer));

    // Enter the native event wait before checking completion: host polling
    // cannot mask a missing firmware wake. Coalesced producer hints are safe.
    ASSERT_TRUE(
        WaitForRecord(reinterpret_cast<uintptr_t>(&signals[1].value), 0));
    EXPECT_EQ(
        std::memcmp(words + 2 * kWordCount, expected.data(), sizeof(expected)),
        0);
    EXPECT_EQ(
        GpuLoadAcquire<int64_t>(reinterpret_cast<uintptr_t>(&signals[0].value)),
        0);
    // Consumer completion proves the producer signal's last dependent read
    // finished. Packet consumption additionally protects wrapped ring slots.
    for (uint32_t i = 0; i < queues.size(); ++i) {
      ASSERT_NO_FATAL_FAILURE(queues[i]->WaitConsumed(api_, indices[i]));
    }
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("completion_event_rounds", kRounds);
  RecordProperty("completion_event_ring_packets",
                 static_cast<int>(kRingBytes / sizeof(aql::Packet)));
}

INSTANTIATE_TEST_SUITE_P(
    Completion, AqlCompletionEventTest,
    ::testing::Values(CompletionCase{CompletionPacket::kDispatch, false},
                      CompletionCase{CompletionPacket::kBarrierAnd, false},
                      CompletionCase{CompletionPacket::kBarrierOr, false},
                      CompletionCase{CompletionPacket::kDispatch, true},
                      CompletionCase{CompletionPacket::kBarrierAnd, true},
                      CompletionCase{CompletionPacket::kBarrierOr, true}),
    [](const ::testing::TestParamInfo<CompletionCase>& info) {
      std::string name = info.param.shared_event ? "Shared" : "Single";
      switch (info.param.packet) {
        case CompletionPacket::kDispatch:
          return name + "Dispatch";
        case CompletionPacket::kBarrierAnd:
          return name + "BarrierAnd";
        case CompletionPacket::kBarrierOr:
          return name + "BarrierOr";
      }
      return name;
    });

}  // namespace
