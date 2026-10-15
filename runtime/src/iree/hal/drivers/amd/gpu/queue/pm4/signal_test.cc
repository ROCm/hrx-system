// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/signal.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <cstdlib>
#include <semaphore>
#include <thread>

#include "iree/testing/gtest.h"

namespace {

using Signal = iree_hal_amd_gpu_pm4_signal_t;
using Pool = iree_hal_amd_gpu_pm4_signal_pool_t;

template <size_t Capacity>
struct Storage {
  explicit Storage(
      iree_hal_amd_gpu_pm4_signal_pool_notification_t notification = {}) {
    values.fill(UINT64_MAX);
    iree_hal_amd_gpu_pm4_signal_pool_initialize(this, values.data(),
                                                0x1234fffffff8, signals.data(),
                                                Capacity, notification, &pool);
  }
  ~Storage() { iree_hal_amd_gpu_pm4_signal_pool_deinitialize(&pool); }

  // Native backing, accessed by host threads only in these ownership tests.
  std::array<uint64_t, Capacity> values;
  // Ordinary host metadata, independent of the native allocation.
  std::array<Signal, Capacity> signals;
  // Finite bookkeeping borrowing both extents.
  Pool pool;
};

TEST(SignalTest, NullReferences) {
  iree_hal_amd_gpu_pm4_signal_retain(nullptr);
  iree_hal_amd_gpu_pm4_signal_release(nullptr);
}

TEST(SignalTest, EmptyPool) {
  Storage<0> storage;
  EXPECT_EQ(iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool),
            nullptr);
}

TEST(SignalTest, ExhaustionAndIndependentReuse) {
  Storage<5> storage;
  EXPECT_LE(sizeof(Signal), 32u);
  std::array<Signal*, 5> acquired;
  for (size_t i = 0; i < acquired.size(); ++i) {
    auto* signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
    ASSERT_NE(signal, nullptr);
    EXPECT_EQ(signal, &storage.signals[i]);
    EXPECT_EQ(signal->pool, &storage.pool);
    EXPECT_EQ(signal->device_address, 0x1234fffffff8 + i * 8);
    EXPECT_EQ(signal->host_value, &storage.values[i]);
    EXPECT_EQ(*signal->host_value, 0u);
    acquired[i] = signal;
  }
  EXPECT_EQ(iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool),
            nullptr);
  for (size_t i : {3, 1, 4, 0, 2}) {
    iree_hal_amd_gpu_pm4_signal_release(acquired[i]);
    auto* recycled =
        iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
    EXPECT_EQ(recycled, acquired[i]);
  }
  for (auto* signal : acquired) {
    iree_hal_amd_gpu_pm4_signal_release(signal);
  }
}

TEST(SignalTest, EveryOwnerCanBeTheLastUse) {
  // Publication, producer execution, cache, and consumer execution hold four
  // independent references. All release orders preserve the same lifetime.
  Storage<1> storage;
  std::array<int, 4> release_order = {0, 1, 2, 3};
  do {
    auto* signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
    ASSERT_NE(signal, nullptr);
    std::array<Signal*, 4> owners = {signal, signal, signal, signal};
    for (size_t i = 1; i < owners.size(); ++i) {
      iree_hal_amd_gpu_pm4_signal_retain(signal);
    }
    *signal->host_value = 1;
    for (size_t i = 0; i < release_order.size(); ++i) {
      auto*& owner = owners[release_order[i]];
      iree_hal_amd_gpu_pm4_signal_release(owner);
      owner = nullptr;
      if (i + 1 < release_order.size()) {
        EXPECT_EQ(iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool),
                  nullptr);
        EXPECT_EQ(storage.values[0], 1u);
      }
    }
    auto* recycled =
        iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
    ASSERT_NE(recycled, nullptr);
    EXPECT_EQ(*recycled->host_value, 0u);
    iree_hal_amd_gpu_pm4_signal_release(recycled);
  } while (std::next_permutation(release_order.begin(), release_order.end()));
}

TEST(SignalTest, LastReaderDoesNotPinUnrelatedCells) {
  Storage<2> storage;
  auto* held = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
  ASSERT_NE(held, nullptr);
  iree_hal_amd_gpu_pm4_signal_retain(held);
  *held->host_value = 1;
  // Producer retirement leaves only the independent consumer reference.
  iree_hal_amd_gpu_pm4_signal_release(held);
  for (uint32_t i = 0; i < 4096; ++i) {
    auto* other = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
    ASSERT_NE(other, nullptr);
    EXPECT_NE(other, held);
    EXPECT_EQ(*other->host_value, 0u);
    EXPECT_EQ(*held->host_value, 1u);
    *other->host_value = 1;
    iree_hal_amd_gpu_pm4_signal_release(other);
  }
  iree_hal_amd_gpu_pm4_signal_release(held);
}

TEST(SignalTest, NotifyOnlyOnExhaustedToAvailableTransition) {
  uint32_t notification_count = 0;
  Storage<2> storage(
      {[](void* user_data) { ++*static_cast<uint32_t*>(user_data); },
       &notification_count});
  auto* first = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
  auto* second = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  iree_hal_amd_gpu_pm4_signal_retain(first);
  iree_hal_amd_gpu_pm4_signal_release(first);
  EXPECT_EQ(notification_count, 0u);
  iree_hal_amd_gpu_pm4_signal_release(second);
  EXPECT_EQ(notification_count, 1u);
  iree_hal_amd_gpu_pm4_signal_release(first);
  EXPECT_EQ(notification_count, 1u);
}

TEST(SignalTest, NotificationCanAcquireReturnedCapacity) {
  struct Observer {
    // Pool inspected by the capacity notification.
    Pool* pool = nullptr;
    // Reference acquired by the callback and returned by the test.
    Signal* acquired = nullptr;
  } observer;
  Storage<1> storage({[](void* user_data) {
                        auto* observer = static_cast<Observer*>(user_data);
                        if (!observer->acquired) {
                          observer->acquired =
                              iree_hal_amd_gpu_pm4_signal_pool_try_acquire(
                                  observer->pool);
                        }
                      },
                      &observer});
  observer.pool = &storage.pool;
  auto* signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
  ASSERT_NE(signal, nullptr);
  iree_hal_amd_gpu_pm4_signal_release(signal);
  EXPECT_EQ(observer.acquired, signal);
  iree_hal_amd_gpu_pm4_signal_release(observer.acquired);
}

TEST(SignalTest, UnpublishedAndCompletedUsesCanBothReturn) {
  Storage<1> storage;
  for (uint32_t i = 0; i < 4096; ++i) {
    auto* signal = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
    ASSERT_NE(signal, nullptr);
    EXPECT_EQ(*signal->host_value, 0u);
    // Abandoned captures never execute a native write. Clearing during
    // exclusive acquisition makes readiness independent of that history.
    if (i % 3 == 0) {
      *signal->host_value = 1;
    }
    iree_hal_amd_gpu_pm4_signal_release(signal);
  }
}

TEST(SignalTest, SimultaneousLastReleasesJoinBeforeReuse) {
  Storage<1> storage;
  constexpr size_t kReaderCount = 4;
  constexpr uint32_t kIterations = 1024;
  std::barrier<> begin(kReaderCount + 1);
  std::barrier<> end(kReaderCount + 1);
  Signal* current = nullptr;
  std::array<uint32_t, kReaderCount> observed{};
  std::array<std::thread, kReaderCount> readers;
  for (size_t i = 0; i < readers.size(); ++i) {
    readers[i] = std::thread([&, i] {
      for (uint32_t epoch = 1; epoch <= kIterations; ++epoch) {
        begin.arrive_and_wait();
        EXPECT_EQ(*current->host_value, 1u);
        observed[i] = epoch;
        iree_hal_amd_gpu_pm4_signal_release(current);
        end.arrive_and_wait();
      }
    });
  }
  for (uint32_t epoch = 1; epoch <= kIterations; ++epoch) {
    current = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
    if (!current) {
      std::abort();
    }
    for (size_t i = 1; i < readers.size(); ++i) {
      iree_hal_amd_gpu_pm4_signal_retain(current);
    }
    *current->host_value = 1;
    begin.arrive_and_wait();
    end.arrive_and_wait();
    for (uint32_t value : observed) {
      EXPECT_EQ(value, epoch);
    }
  }
  for (auto& reader : readers) {
    reader.join();
  }
}

TEST(SignalTest, ConcurrentProducerAndReaderOwnership) {
  constexpr size_t kLaneCount = 4;
  constexpr uint32_t kIterations = 4096;
  Storage<kLaneCount * 2 + 1> storage;
  // A retained unrelated point stays intact while every lane recycles.
  auto* held = iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
  ASSERT_NE(held, nullptr);
  *held->host_value = 1;
  struct Lane {
    // Producer-to-reader transfer, synchronized by ready.
    Signal* signal = nullptr;
    // Ownership of the one pending handoff slot.
    std::binary_semaphore available{1};
    // Published producer result ready for the consumer.
    std::binary_semaphore ready{0};
  };
  std::array<Lane, kLaneCount> lanes;
  std::array<std::thread, kLaneCount * 2> workers;
  for (size_t i = 0; i < lanes.size(); ++i) {
    workers[i * 2] = std::thread([&, i] {
      auto& lane = lanes[i];
      for (uint32_t j = 0; j < kIterations; ++j) {
        lane.available.acquire();
        auto* signal =
            iree_hal_amd_gpu_pm4_signal_pool_try_acquire(&storage.pool);
        if (!signal) {
          std::abort();
        }
        EXPECT_NE(signal, held);
        EXPECT_EQ(*signal->host_value, 0u);
        *signal->host_value = 1;
        iree_hal_amd_gpu_pm4_signal_retain(signal);
        lane.signal = signal;
        lane.ready.release();
        // The reader can win this race and drop its reference first.
        iree_hal_amd_gpu_pm4_signal_release(signal);
      }
    });
    workers[i * 2 + 1] = std::thread([&, i] {
      auto& lane = lanes[i];
      for (uint32_t j = 0; j < kIterations; ++j) {
        lane.ready.acquire();
        auto* signal = lane.signal;
        EXPECT_EQ(*signal->host_value, 1u);
        iree_hal_amd_gpu_pm4_signal_release(signal);
        lane.available.release();
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  EXPECT_EQ(*held->host_value, 1u);
  iree_hal_amd_gpu_pm4_signal_release(held);
}

}  // namespace
