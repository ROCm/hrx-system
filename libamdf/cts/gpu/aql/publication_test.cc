// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/aql/queue/queue_fixture.h"

namespace {

class AqlQueuePlacementTest
    : public AqlQueueTest,
      public ::testing::WithParamInterface<amdf_queue_placement_t> {
 protected:
  void SetUp() override {
    AqlQueueTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    amdf_queue_family_placement_info_t placement = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_PLACEMENT_INFO,
        .structure_size = sizeof(placement),
    };
    ASSERT_EQ(api_->device_query_queue_family_placement_info(
                  device_, family_.ordinal, &placement),
              AMDF_STATUS_OK);
    if (GetParam() != AMDF_QUEUE_PLACEMENT_DEFAULT &&
        !(placement.placement_capabilities & (1u << GetParam()))) {
      GTEST_SKIP() << "family does not support requested ring placement";
    }
  }

  void CreatePlacedQueue(amdf_queue_producer_mode_t producer_mode,
                         GpuUserQueue** out_queue) {
    ASSERT_NO_FATAL_FAILURE(CreateQueue(family_,
                                        GpuUserQueueOptions{
                                            .producer_mode = producer_mode,
                                            .ring_byte_length = 4096,
                                            .placement = GetParam(),
                                        },
                                        out_queue));
  }
};

TEST_P(AqlQueuePlacementTest, CompletesBarriersAndReusesRetiredSlots) {
  GpuMemory* storage = nullptr;
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &storage));
  ASSERT_NO_FATAL_FAILURE(
      CreatePlacedQueue(AMDF_QUEUE_PRODUCER_MODE_SINGLE, &queue));
  std::memset(storage->host.pointer, 0, storage->info.byte_length);
  auto* signals = static_cast<aql::Signal*>(storage->host.pointer);
  signals[0] = aql::UserSignal();
  signals[1] = aql::UserSignal();
  const uint64_t packet_count =
      queue->host.ring_byte_length / sizeof(aql::Packet) + 1;
  uint64_t index = 0;
  for (uint32_t round = 0; round < 64; ++round) {
    const uint32_t completion_index = round % 2;
    const uint32_t dependency_index = 1 - completion_index;
    signals[completion_index].value = packet_count;
    const auto packet = aql::Barrier(
        aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
        storage->device_address + completion_index * sizeof(aql::Signal),
        {storage->device_address + dependency_index * sizeof(aql::Signal)});
    for (uint64_t i = 0; i < packet_count; ++i) {
      GpuStoreRelease(queue->host.write_index_address, index + 1);
      ASSERT_TRUE(aql::Publish(*queue, index++, packet));
    }
    ASSERT_NO_FATAL_FAILURE(
        WaitCompletionAndConsumption(*queue, signals[completion_index], index));
    // The dependency remains zero until every barrier has completed; only
    // completion, not read-index advancement, permits rearming the signal.
  }
}

TEST_P(AqlQueuePlacementTest, MultipleProducersReserveAndPublishIndependently) {
  if ((family_.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI) == 0) {
    GTEST_SKIP() << "family does not admit multiple host producers";
  }
  GpuMemory* storage = nullptr;
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &storage));
  ASSERT_NO_FATAL_FAILURE(
      CreatePlacedQueue(AMDF_QUEUE_PRODUCER_MODE_MULTI, &queue));
  std::memset(storage->host.pointer, 0, storage->info.byte_length);
  auto* completion = static_cast<aql::Signal*>(storage->host.pointer);
  *completion = aql::UserSignal(64);
  const auto packet =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
                   storage->device_address);
  auto publish = [&] {
    auto& write_index =
        *reinterpret_cast<uint64_t*>(queue->host.write_index_address);
    for (uint32_t i = 0; i < 32; ++i) {
      const uint64_t index = std::atomic_ref<uint64_t>(write_index)
                                 .fetch_add(1, std::memory_order_relaxed);
      ASSERT_TRUE(aql::Publish(*queue, index, packet));
    }
  };
  std::thread first(publish);
  std::thread second(publish);
  first.join();
  second.join();
  ASSERT_NO_FATAL_FAILURE(
      WaitCompletionAndConsumption(*queue, *completion, 64));
}

TEST_P(AqlQueuePlacementTest, PublishesReservedPacketsOutOfOrder) {
  if ((family_.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI) == 0) {
    GTEST_SKIP() << "family does not admit multiple host producers";
  }
  GpuMemory* storage = nullptr;
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &storage));
  ASSERT_NO_FATAL_FAILURE(
      CreatePlacedQueue(AMDF_QUEUE_PRODUCER_MODE_MULTI, &queue));
  std::memset(storage->host.pointer, 0, storage->info.byte_length);
  auto* signals = static_cast<aql::Signal*>(storage->host.pointer);
  signals[0] = aql::UserSignal(1);
  signals[1] = aql::UserSignal(1);
  const auto first =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                   storage->device_address);
  const auto second =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                   storage->device_address + sizeof(aql::Signal));
  auto& write_index =
      *reinterpret_cast<uint64_t*>(queue->host.write_index_address);
  const uint64_t index = std::atomic_ref<uint64_t>(write_index)
                             .fetch_add(2, std::memory_order_relaxed);

  // MULTI doorbells may arrive out of order. The earlier INVALID packet holds
  // the launch frontier until its body and valid header have been published.
  ASSERT_TRUE(aql::Publish(*queue, index + 1, second));
  ASSERT_TRUE(aql::Publish(*queue, index, first));
  ASSERT_NO_FATAL_FAILURE(
      WaitCompletionAndConsumption(*queue, signals[1], index + 2));
  EXPECT_EQ(
      GpuLoadAcquire<int64_t>(reinterpret_cast<uintptr_t>(&signals[0].value)),
      0);
}

INSTANTIATE_TEST_SUITE_P(Placements, AqlQueuePlacementTest,
                         ::testing::Values(AMDF_QUEUE_PLACEMENT_DEFAULT,
                                           AMDF_QUEUE_PLACEMENT_HOST_LOCAL,
                                           AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL));

}  // namespace
