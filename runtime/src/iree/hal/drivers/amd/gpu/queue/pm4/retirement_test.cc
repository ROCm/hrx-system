// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/retirement.h"

#include <array>
#include <cstdint>
#include <semaphore>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

using Record = iree_hal_amd_gpu_pm4_retirement_record_t;
using Ring = iree_hal_amd_gpu_pm4_retirement_ring_t;

// Models only the native producer's already-qualified payload-before-generation
// release. Real GPU execution validates that recipe independently.
void Commit(Record& record, void* action, uint32_t generation) {
  record.action = reinterpret_cast<uintptr_t>(action);
  iree_atomic_store(&record.generation, static_cast<int32_t>(generation),
                    iree_memory_order_release);
}

TEST(RetirementTest, InitialSlotsAreEmptyAndFourFitOneCacheLine) {
  std::array<Record, 4> records;
  Ring ring;
  iree_hal_amd_gpu_pm4_retirement_ring_initialize(records.data(), 0x4000, 4,
                                                  &ring);
  EXPECT_EQ(sizeof(records), 64u);
  EXPECT_EQ(ring.records, records.data());
  EXPECT_EQ(ring.device_address, 0x4000u);
  EXPECT_EQ(ring.capacity_mask, 3u);
  EXPECT_EQ(ring.capacity_shift, 2u);
  for (uint32_t i = 0; i < records.size(); ++i) {
    EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, i), nullptr);
    EXPECT_EQ(records[i].action, 0u);
    EXPECT_EQ(
        iree_atomic_load(&records[i].generation, iree_memory_order_relaxed), 0);
    EXPECT_EQ(records[i].reserved, 0u);
  }
}

TEST(RetirementTest, EmitsPointerThenConfirmedGenerationAtWrappedSlot) {
  std::array<Record, 8> records;
  Ring ring;
  iree_hal_amd_gpu_pm4_retirement_ring_initialize(
      records.data(), UINT64_C(0x00001234ffffffc0), 8, &ring);
  int action = 0;
  const uint64_t pointer = reinterpret_cast<uintptr_t>(&action);
  std::array<uint32_t, 16> words;
  words.fill(0xfed01234);
  auto* end = iree_hal_amd_gpu_pm4_retirement_emit(words.data() + 1, &ring, 13,
                                                   &action, 0x30e);
  EXPECT_EQ(end, words.data() + 15);
  EXPECT_EQ(words.front(), 0xfed01234u);
  EXPECT_EQ(words.back(), 0xfed01234u);
  const std::vector<uint32_t> expected = {
      0xc0043700,
      0x00100200,
      0x00000010,
      0x00001235,
      static_cast<uint32_t>(pointer),
      static_cast<uint32_t>(pointer >> 32),
      0xc0064900,
      0x0030e528,
      0x23010000,
      0x00000018,
      0x00001235,
      2,
      0,
      0,
  };
  EXPECT_EQ(std::vector<uint32_t>(words.data() + 1, end), expected);
  // Emission touches command storage only; device-visible records remain idle.
  EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, 13), nullptr);
}

TEST(RetirementTest, PayloadDoesNotBecomeReadyBeforeGeneration) {
  std::array<Record, 1> records;
  Ring ring;
  iree_hal_amd_gpu_pm4_retirement_ring_initialize(records.data(), 0x1000, 1,
                                                  &ring);
  int action = 0;
  records[0].action = reinterpret_cast<uintptr_t>(&action);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, 0), nullptr);
  iree_atomic_store(&records[0].generation, 1, iree_memory_order_release);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, 0), &action);
  // Returning capacity changes the caller's position, never the native slot.
  EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, 1), nullptr);
  EXPECT_EQ(records[0].action, reinterpret_cast<uintptr_t>(&action));
  EXPECT_EQ(iree_atomic_load(&records[0].generation, iree_memory_order_relaxed),
            1);
}

TEST(RetirementTest, LaterRecordDoesNotProveCompletionOfTheHead) {
  std::array<Record, 4> records;
  std::array<int, 4> actions;
  Ring ring;
  iree_hal_amd_gpu_pm4_retirement_ring_initialize(records.data(), 0x1000, 4,
                                                  &ring);
  Commit(records[3], &actions[3], 1);
  Commit(records[1], &actions[1], 1);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, 0), nullptr);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, 1), &actions[1]);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, 2), nullptr);
  Commit(records[0], &actions[0], 1);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, 0), &actions[0]);
  EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, 2), nullptr);
  Commit(records[2], &actions[2], 1);
  for (uint32_t i = 0; i < records.size(); ++i) {
    EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, i), &actions[i]);
  }
}

TEST(RetirementTest, ReusesSlotsWithoutHostClearing) {
  std::array<Record, 32> records;
  std::array<int, 2> actions;
  for (uint32_t capacity : {1u, 2u, 4u, 32u}) {
    SCOPED_TRACE(capacity);
    Ring ring;
    iree_hal_amd_gpu_pm4_retirement_ring_initialize(records.data(), 0x1000,
                                                    capacity, &ring);
    for (uint32_t position = 0; position < capacity * 100; ++position) {
      EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, position), nullptr);
      void* action = &actions[position % 2];
      auto& record = records[position % capacity];
      record.reserved = 0xa7b8c9d0;
      Commit(record, action, position / capacity + 1);
      EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, position), action);
      EXPECT_EQ(record.reserved, 0xa7b8c9d0u);
    }
  }
}

TEST(RetirementTest, GenerationWrapPreservesPayloadAndReadiness) {
  std::array<Record, 4> records;
  Ring ring;
  iree_hal_amd_gpu_pm4_retirement_ring_initialize(records.data(), 0x1000, 4,
                                                  &ring);
  int action = 0;
  // Establish the valid empty state after 2^32-1 full turns, with the next
  // generation equal to zero. This does not require billions of test packets.
  for (auto& record : records) {
    Commit(record, &action, UINT32_MAX);
  }
  const uint64_t first_position = uint64_t{UINT32_MAX} * 4;
  for (uint32_t i = 0; i < 12; ++i) {
    const uint64_t position = first_position + i;
    EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, position), nullptr);
    std::array<uint32_t, 14> words;
    EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_emit(words.data(), &ring,
                                                   position, &action, 0),
              words.data() + words.size());
    EXPECT_EQ(words[11], i / 4);
    Commit(records[i % 4], &action, i / 4);
    EXPECT_EQ(iree_hal_amd_gpu_pm4_retirement_poll(&ring, position), &action);
  }
}

TEST(RetirementTest, ConcurrentProducerReusesOnlyReturnedSlots) {
  constexpr uint32_t kCapacity = 32;
  constexpr uint32_t kOperationCount = 4096;
  std::array<Record, kCapacity> records;
  std::array<uint64_t, kCapacity> actions;
  Ring ring;
  iree_hal_amd_gpu_pm4_retirement_ring_initialize(records.data(), 0x1000,
                                                  kCapacity, &ring);
  std::counting_semaphore<kCapacity> capacity{kCapacity};
  std::thread producer([&] {
    for (uint32_t position = 0; position < kOperationCount; ++position) {
      capacity.acquire();
      const uint32_t slot = position % kCapacity;
      actions[slot] = position;
      Commit(records[slot], &actions[slot], position / kCapacity + 1);
    }
  });
  for (uint32_t position = 0; position < kOperationCount; ++position) {
    // The generation acquire alone publishes the payload. A producer-to-host
    // test semaphore would hide a missing acquire from the thread sanitizer.
    uint64_t* action = nullptr;
    while (!(action = static_cast<uint64_t*>(
                 iree_hal_amd_gpu_pm4_retirement_poll(&ring, position)))) {
      std::this_thread::yield();
    }
    EXPECT_EQ(action, &actions[position % kCapacity]);
    if (action == &actions[position % kCapacity]) {
      EXPECT_EQ(*action, position);
    }
    capacity.release();
  }
  producer.join();
}

}  // namespace
