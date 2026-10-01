// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/aql/queue_fixture.h"

namespace {

TEST_F(AqlQueueTest, CompletesBarriersAndReusesRetiredSlots) {
  GpuMemory* storage = nullptr;
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &storage));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  std::memset(storage->host.pointer, 0, storage->info.byte_length);
  auto* signals = static_cast<aql::Signal*>(storage->host.pointer);
  signals[0].kind = signals[1].kind = 1;
  const uint64_t packet_count =
      queue->host.ring_byte_length / sizeof(aql::Packet) + 1;
  uint64_t index = 0;
  for (uint32_t round = 0; round < 2; ++round) {
    signals[0].value = packet_count;
    const auto packet =
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
                     storage->device_address,
                     {storage->device_address + sizeof(aql::Signal)});
    for (uint64_t i = 0; i < packet_count; ++i) {
      GpuStoreRelease(queue->host.write_index_address, index + 1);
      aql::Publish(*queue, index++, packet);
    }
    ASSERT_NO_FATAL_FAILURE(
        WaitCompletionAndConsumption(*queue, signals[0], index));
    // The dependency remains zero until every barrier has completed; only
    // completion, not read-index advancement, permits rearming the signal.
  }
}

TEST_F(AqlQueueTest, MultipleProducersReserveAndPublishIndependently) {
  if ((family_.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI) == 0) {
    GTEST_SKIP() << "family does not admit multiple host producers";
  }
  GpuMemory* storage = nullptr;
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &storage));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue, AMDF_QUEUE_PRODUCER_MODE_MULTI));
  std::memset(storage->host.pointer, 0, storage->info.byte_length);
  auto* completion = static_cast<aql::Signal*>(storage->host.pointer);
  completion->kind = 1;
  completion->value = 64;
  const auto packet =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
                   storage->device_address);
  auto publish = [&] {
    auto& write_index =
        *reinterpret_cast<uint64_t*>(queue->host.write_index_address);
    for (uint32_t i = 0; i < 32; ++i) {
      const uint64_t index = std::atomic_ref<uint64_t>(write_index)
                                 .fetch_add(1, std::memory_order_relaxed);
      aql::Publish(*queue, index, packet);
    }
  };
  std::thread first(publish);
  std::thread second(publish);
  first.join();
  second.join();
  ASSERT_NO_FATAL_FAILURE(
      WaitCompletionAndConsumption(*queue, *completion, 64));
}

TEST_F(AqlQueueTest, PublishesReservedPacketsOutOfOrder) {
  if ((family_.producer_modes & AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI) == 0) {
    GTEST_SKIP() << "family does not admit multiple host producers";
  }
  GpuMemory* storage = nullptr;
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &storage));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue, AMDF_QUEUE_PRODUCER_MODE_MULTI));
  std::memset(storage->host.pointer, 0, storage->info.byte_length);
  auto* signals = static_cast<aql::Signal*>(storage->host.pointer);
  signals[0].kind = signals[1].kind = 1;
  signals[0].value = signals[1].value = 1;
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
  aql::Publish(*queue, index + 1, second);
  aql::Publish(*queue, index, first);
  ASSERT_NO_FATAL_FAILURE(
      WaitCompletionAndConsumption(*queue, signals[1], index + 2));
  EXPECT_EQ(
      GpuLoadAcquire<int64_t>(reinterpret_cast<uintptr_t>(&signals[0].value)),
      0);
}

}  // namespace
