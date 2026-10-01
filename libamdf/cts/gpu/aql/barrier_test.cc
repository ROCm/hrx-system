// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/aql/queue_fixture.h"

namespace {

TEST_F(AqlQueueTest, BarrierAndJoinsFiveQueueDependencies) {
  GpuMemory* storage = nullptr;
  GpuUserQueue* producer = nullptr;
  GpuUserQueue* consumer = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &storage));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&producer));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&consumer));
  std::memset(storage->host.pointer, 0, storage->info.byte_length);
  auto* signals = static_cast<aql::Signal*>(storage->host.pointer);
  for (uint32_t i = 0; i < 7; ++i) {
    signals[i].kind = 1;
  }
  std::array<uint64_t, 5> dependencies = {};
  for (uint32_t i = 0; i < dependencies.size(); ++i) {
    dependencies[i] = storage->device_address + i * sizeof(aql::Signal);
  }
  const auto join = aql::Barrier(
      aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
      storage->device_address + 5 * sizeof(aql::Signal), dependencies);
  const auto marker =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                   storage->device_address + 6 * sizeof(aql::Signal));
  uint64_t producer_index = 0;
  uint64_t consumer_index = 0;
  for (uint32_t round = 0; round < 2; ++round) {
    SCOPED_TRACE(round);
    for (uint32_t i = 0; i < 7; ++i) {
      signals[i].value = 1;
    }

    // Publish the complete consumer chain before any producer packet. The
    // dependencies, not cross-queue submission order, connect their execution.
    GpuStoreRelease(consumer->host.write_index_address, consumer_index + 2);
    aql::Publish(*consumer, consumer_index++, join);
    aql::Publish(*consumer, consumer_index++, marker);
    for (uint32_t i = 0; i < dependencies.size(); ++i) {
      const uint32_t slot = round == 0 ? i : 4 - i;
      const auto packet =
          aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                       dependencies[slot]);
      GpuStoreRelease(producer->host.write_index_address, producer_index + 1);
      aql::Publish(*producer, producer_index++, packet);
    }

    // AND/OR completion blocks later launches even with the header barrier
    // bit clear. Every dependency stays zero until this consumer has retired.
    EXPECT_NO_FATAL_FAILURE(
        WaitCompletionAndConsumption(*consumer, signals[6], consumer_index));
    for (uint32_t i = 0; i < 6; ++i) {
      EXPECT_EQ(GpuLoadAcquire<int64_t>(
                    reinterpret_cast<uintptr_t>(&signals[i].value)),
                0)
          << "signal " << i;
    }
    EXPECT_NO_FATAL_FAILURE(producer->WaitConsumed(api_, producer_index));
    if (HasFailure()) {
      return;
    }
  }
}

TEST_F(AqlQueueTest, BarrierOrAcceptsEachSatisfiedSlot) {
  GpuMemory* storage = nullptr;
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &storage));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  std::memset(storage->host.pointer, 0, storage->info.byte_length);
  auto* signals = static_cast<aql::Signal*>(storage->host.pointer);
  for (uint32_t i = 0; i < 7; ++i) {
    signals[i].kind = 1;
  }
  std::array<uint64_t, 5> dependencies = {};
  for (uint32_t i = 0; i < dependencies.size(); ++i) {
    dependencies[i] = storage->device_address + i * sizeof(aql::Signal);
  }
  const auto select = aql::Barrier(
      aql::BarrierType::kOr, aql::HeaderBarrier::kDisabled,
      storage->device_address + 5 * sizeof(aql::Signal), dependencies);
  const auto marker =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                   storage->device_address + 6 * sizeof(aql::Signal));
  uint64_t index = 0;
  for (uint32_t selected = 0; selected < dependencies.size(); ++selected) {
    SCOPED_TRACE(selected);
    std::array<int64_t, 5> values = {INT64_C(1) << 32, -1, 1, -2,
                                     INT64_C(1) << 40};
    values[selected] = 0;
    for (uint32_t i = 0; i < values.size(); ++i) {
      signals[i].value = values[i];
    }
    signals[5].value = signals[6].value = 1;

    // Exactly one live signal is zero. Positive, negative and high-word-only
    // nonzero signals remain unchanged; they must not prevent OR completion.
    GpuStoreRelease(queue->host.write_index_address, index + 2);
    aql::Publish(*queue, index++, select);
    aql::Publish(*queue, index++, marker);
    ASSERT_NO_FATAL_FAILURE(
        WaitCompletionAndConsumption(*queue, signals[6], index));
    ASSERT_EQ(
        GpuLoadAcquire<int64_t>(reinterpret_cast<uintptr_t>(&signals[5].value)),
        0);
    for (uint32_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(GpuLoadAcquire<int64_t>(
                    reinterpret_cast<uintptr_t>(&signals[i].value)),
                values[i])
          << "dependency " << i;
    }
  }
}

TEST_F(AqlQueueTest, BarrierOrCompletesWithPublishedDependencyAndNullSlots) {
  GpuMemory* storage = nullptr;
  GpuUserQueue* producer = nullptr;
  GpuUserQueue* consumer = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &storage));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&producer));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&consumer));
  std::memset(storage->host.pointer, 0, storage->info.byte_length);
  auto* signals = static_cast<aql::Signal*>(storage->host.pointer);
  for (uint32_t i = 0; i < 6; ++i) {
    signals[i].kind = 1;
  }
  const auto marker =
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                   storage->device_address + 5 * sizeof(aql::Signal));
  uint64_t producer_index = 0;
  uint64_t consumer_index = 0;
  for (uint32_t selected = 0; selected < 5; ++selected) {
    SCOPED_TRACE(selected);
    signals[selected].value = signals[5].value = 1;
    std::array<uint64_t, 5> dependencies = {};
    dependencies[selected] =
        storage->device_address + selected * sizeof(aql::Signal);
    const auto select = aql::Barrier(
        aql::BarrierType::kOr, aql::HeaderBarrier::kDisabled, 0, dependencies);
    const auto produce =
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled,
                     dependencies[selected]);

    GpuStoreRelease(consumer->host.write_index_address, consumer_index + 2);
    aql::Publish(*consumer, consumer_index++, select);
    aql::Publish(*consumer, consumer_index++, marker);
    GpuStoreRelease(producer->host.write_index_address, producer_index + 1);
    aql::Publish(*producer, producer_index++, produce);

    // The following marker supplies completion for the sparse OR, whose own
    // completion handle is null. Final values qualify completion of this legal
    // program; they do not independently prove the absence of early release.
    EXPECT_NO_FATAL_FAILURE(
        WaitCompletionAndConsumption(*consumer, signals[5], consumer_index));
    EXPECT_EQ(GpuLoadAcquire<int64_t>(
                  reinterpret_cast<uintptr_t>(&signals[selected].value)),
              0);
    EXPECT_NO_FATAL_FAILURE(producer->WaitConsumed(api_, producer_index));
    if (HasFailure()) {
      return;
    }
  }
}

}  // namespace
