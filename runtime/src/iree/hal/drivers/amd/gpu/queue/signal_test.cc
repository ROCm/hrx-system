// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/signal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdlib>
#include <semaphore>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using Signal = iree_hal_amd_gpu_signal_t;
using Pool = iree_hal_amd_gpu_signal_pool_t;

template <size_t Capacity, size_t QueueCount = 1>
struct Storage {
  explicit Storage(
      iree_hal_amd_gpu_signal_pool_notification_t notification = {},
      Pool* target = nullptr) {
    values.fill(UINT64_MAX);
    iree_hal_amd_gpu_signal_pool_initialize(notification, &pool);
    iree_hal_amd_gpu_signal_block_initialize(
        target ? target : &pool, this, values.data(), 0x1234fffffff8, 8, 0,
        signals.data(), Capacity, QueueCount, frontiers.data(), &block);
  }
  ~Storage() {
    if (block.pool) {
      iree_hal_amd_gpu_signal_block_deinitialize(&block);
    }
    iree_hal_amd_gpu_signal_frontiers_free(
        iree_hal_amd_gpu_signal_block_frontiers(&block),
        reinterpret_cast<iree_hal_amd_gpu_signal_frontiers_t*>(
            frontiers.data()),
        iree_allocator_system());
    iree_hal_amd_gpu_signal_pool_deinitialize(&pool);
  }

  // Native backing, accessed by host threads only in these ownership tests.
  std::array<uint64_t, Capacity> values;
  // Ordinary host metadata, independent of the native allocation.
  std::array<Signal, Capacity> signals = {};
  // Immutable native-domain vectors belonging to retained signals.
  std::array<uint64_t, 2 + Capacity * QueueCount> frontiers;
  // Publisher-owned free cells and concurrent final returns.
  Pool pool;
  // Finite bookkeeping borrowing all extents.
  iree_hal_amd_gpu_signal_block_t block;
};

TEST(SignalTest, NullReferences) {
  iree_hal_amd_gpu_signal_retain(nullptr);
  iree_hal_amd_gpu_signal_release(nullptr);
}

TEST(SignalTest, EmptyPool) {
  Storage<0> storage;
  EXPECT_EQ(iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0),
            nullptr);
}

TEST(SignalTest, ExhaustionAndIndependentReuse) {
  Storage<5> storage;
  EXPECT_LE(sizeof(Signal), 32u);
  std::array<Signal*, 5> acquired;
  for (size_t i = 0; i < acquired.size(); ++i) {
    auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
    ASSERT_NE(signal, nullptr);
    EXPECT_EQ(signal, &storage.signals[i]);
    EXPECT_EQ(signal->block, &storage.block);
    EXPECT_EQ(signal->device_address, 0x1234fffffff8 + i * 8);
    EXPECT_EQ(signal->host_value, &storage.values[i]);
    EXPECT_EQ(*signal->host_value, 0u);
    acquired[i] = signal;
  }
  EXPECT_EQ(iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0),
            nullptr);
  for (size_t i : {3, 1, 4, 0, 2}) {
    iree_hal_amd_gpu_signal_release(acquired[i]);
    auto* recycled = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
    EXPECT_EQ(recycled, acquired[i]);
  }
  for (auto* signal : acquired) {
    iree_hal_amd_gpu_signal_release(signal);
  }
}

TEST(SignalTest, ReturnedOlderBlockIsImmediatelyAvailable) {
  uint32_t notifications = 0;
  Pool pool;
  iree_hal_amd_gpu_signal_pool_initialize(
      {[](void* data) { ++*static_cast<uint32_t*>(data); }, &notifications},
      &pool);
  {
    Storage<1> older({}, &pool);
    Storage<2> newer({}, &pool);
    std::array<Signal*, 3> held;
    for (auto& signal : held) {
      signal = iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 0);
      ASSERT_NE(signal, nullptr);
      *signal->host_value = 1;
    }
    EXPECT_EQ(held[2]->block, &older.block);
    EXPECT_FALSE(iree_hal_amd_gpu_signal_pool_has_available(&pool));
    iree_hal_amd_gpu_signal_retain(held[2]);
    iree_hal_amd_gpu_signal_release(held[2]);
    EXPECT_FALSE(iree_hal_amd_gpu_signal_pool_has_available(&pool));
    EXPECT_FALSE(iree_hal_amd_gpu_signal_block_try_deinitialize(&older.block));

    iree_hal_amd_gpu_signal_release(held[2]);
    EXPECT_EQ(notifications, 1u);
    auto* reused = iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 0);
    EXPECT_EQ(reused, held[2]);
    EXPECT_EQ(*reused->host_value, 0u);
    EXPECT_EQ(*held[0]->host_value, 1u);
    EXPECT_EQ(*held[1]->host_value, 1u);
    iree_hal_amd_gpu_signal_release(reused);
    EXPECT_TRUE(iree_hal_amd_gpu_signal_block_try_deinitialize(&older.block));
    EXPECT_FALSE(iree_hal_amd_gpu_signal_pool_has_available(&pool));
    for (size_t i = 0; i < 2; ++i) {
      iree_hal_amd_gpu_signal_release(held[i]);
    }
  }
  iree_hal_amd_gpu_signal_pool_deinitialize(&pool);
}

TEST(SignalTest, TrimDetachesAvailableBlocksWithoutChangingLiveCells) {
  Pool pool;
  iree_hal_amd_gpu_signal_pool_initialize({}, &pool);
  {
    Storage<2> first({}, &pool);
    Storage<2> middle({}, &pool);
    Storage<2> last({}, &pool);
    auto* held = iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 0);
    ASSERT_NE(held, nullptr);
    EXPECT_EQ(held->block, &last.block);
    EXPECT_TRUE(iree_hal_amd_gpu_signal_block_try_deinitialize(&middle.block));
    EXPECT_TRUE(iree_hal_amd_gpu_signal_block_try_deinitialize(&first.block));
    EXPECT_FALSE(iree_hal_amd_gpu_signal_block_try_deinitialize(&last.block));
    auto* remaining = iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 0);
    ASSERT_NE(remaining, nullptr);
    EXPECT_EQ(remaining->block, &last.block);
    EXPECT_EQ(iree_hal_amd_gpu_signal_pool_try_acquire(&pool, 0), nullptr);
    iree_hal_amd_gpu_signal_release(held);
    iree_hal_amd_gpu_signal_release(remaining);
  }
  iree_hal_amd_gpu_signal_pool_deinitialize(&pool);
}

TEST(SignalTest, TransitiveFrontierDoesNotReadLaterProducerState) {
  Storage<2, 3> storage;
  auto* first = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
  auto* second = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  uint64_t producer[3] = {4, 0, 0};
  iree_hal_amd_gpu_signal_publish_frontier(first, producer);
  uint64_t intermediate[3] = {0, 7, 0};
  iree_hal_amd_gpu_signal_join_frontier(first, IREE_ARRAYSIZE(intermediate),
                                        intermediate);
  iree_hal_amd_gpu_signal_publish_frontier(second, intermediate);
  producer[0] = 99;
  intermediate[1] = 101;
  uint64_t consumer[3] = {1, 0, 12};
  iree_hal_amd_gpu_signal_join_frontier(second, IREE_ARRAYSIZE(consumer),
                                        consumer);
  EXPECT_EQ(consumer[0], 4u);
  EXPECT_EQ(consumer[1], 7u);
  EXPECT_EQ(consumer[2], 12u);
  iree_hal_amd_gpu_signal_release(first);
  iree_hal_amd_gpu_signal_release(second);
}

TEST(SignalTest, CompleteQueueSetAndRecycledSnapshots) {
  Storage<1, 256> storage;
  std::array<uint64_t, 256> producer;
  std::array<uint64_t, 256> consumer;
  for (size_t i = 0; i < producer.size(); ++i) {
    producer[i] = i + 1;
    consumer[i] = i % 2 ? 1000 : 0;
  }
  auto* first = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
  ASSERT_NE(first, nullptr);
  iree_hal_amd_gpu_signal_publish_frontier(first, producer.data());
  iree_hal_amd_gpu_signal_join_frontier(first, consumer.size(),
                                        consumer.data());
  for (size_t i = 0; i < producer.size(); ++i) {
    EXPECT_EQ(consumer[i], i % 2 ? 1000 : i + 1);
  }
  iree_hal_amd_gpu_signal_release(first);
  auto* second = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
  ASSERT_EQ(first, second);
  producer.fill(0);
  producer[255] = 512;
  iree_hal_amd_gpu_signal_publish_frontier(second, producer.data());
  consumer.fill(0);
  iree_hal_amd_gpu_signal_join_frontier(second, consumer.size(),
                                        consumer.data());
  EXPECT_EQ(producer, consumer);
  iree_hal_amd_gpu_signal_release(second);
}

TEST(SignalTest, FrontierExpansionPreservesNativeCellsAndRetainedSnapshots) {
  Storage<2, 2> storage;
  auto* producer = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 1);
  ASSERT_NE(producer, nullptr);
  uint64_t published[2] = {7, 11};
  iree_hal_amd_gpu_signal_publish_frontier(producer, published);
  iree_hal_amd_gpu_signal_retain(producer);
  const uint64_t address = producer->device_address;
  auto* previous = iree_hal_amd_gpu_signal_block_frontiers(&storage.block);
  const auto* borrowed_row = reinterpret_cast<const uint64_t*>(previous + 1);
  iree_hal_amd_gpu_signal_release(producer);

  IREE_ASSERT_OK(iree_hal_amd_gpu_signal_block_reserve_frontiers(
      &storage.block, 4, iree_allocator_system()));
  auto* expanded = iree_hal_amd_gpu_signal_block_frontiers(&storage.block);
  EXPECT_EQ(expanded->previous, previous);
  EXPECT_EQ(expanded->capacity, 2u);
  EXPECT_EQ(expanded->queue_count, 4u);
  EXPECT_EQ(producer->device_address, address);
  EXPECT_EQ(producer->host_value, storage.values.data());
  EXPECT_EQ(*producer->host_value, 1u);
  EXPECT_EQ(borrowed_row[0], 7u);
  EXPECT_EQ(borrowed_row[1], 11u);

  std::array<uint64_t, 4> consumer = {1, 0, 13, 0};
  iree_hal_amd_gpu_signal_join_frontier(producer, consumer.size(),
                                        consumer.data());
  EXPECT_EQ(consumer, (std::array<uint64_t, 4>{7, 11, 13, 0}));
  auto* next = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
  ASSERT_NE(next, nullptr);
  iree_hal_amd_gpu_signal_publish_frontier(next, consumer.data());
  consumer.fill(0);
  iree_hal_amd_gpu_signal_join_frontier(next, consumer.size(), consumer.data());
  EXPECT_EQ(consumer, (std::array<uint64_t, 4>{7, 11, 13, 0}));

  // An existing peer may still have the old width during cold installation.
  std::array<uint64_t, 3> installing_peer = {0, 0, UINT64_MAX};
  iree_hal_amd_gpu_signal_join_frontier(producer, 2, installing_peer.data());
  EXPECT_EQ(installing_peer, (std::array<uint64_t, 3>{7, 11, UINT64_MAX}));
  iree_hal_amd_gpu_signal_release(next);
  iree_hal_amd_gpu_signal_release(producer);
}

TEST(SignalTest, FailedFrontierExpansionPreservesPublishedOwnership) {
  Storage<1> storage;
  auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 1);
  ASSERT_NE(signal, nullptr);
  uint64_t epoch = 29;
  iree_hal_amd_gpu_signal_publish_frontier(signal, &epoch);
  auto* previous = iree_hal_amd_gpu_signal_block_frontiers(&storage.block);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amd_gpu_signal_block_reserve_frontiers(
                            &storage.block, 2, iree_allocator_null()));
  EXPECT_EQ(iree_hal_amd_gpu_signal_block_frontiers(&storage.block), previous);
  std::array<uint64_t, 2> consumer = {};
  iree_hal_amd_gpu_signal_join_frontier(signal, consumer.size(),
                                        consumer.data());
  EXPECT_EQ(consumer, (std::array<uint64_t, 2>{29, 0}));
  IREE_ASSERT_OK(iree_hal_amd_gpu_signal_block_reserve_frontiers(
      &storage.block, 1, iree_allocator_null()));
  iree_hal_amd_gpu_signal_release(signal);
}

TEST(SignalTest, FrontierExpansionAndIndependentReaderProgress) {
  Storage<2> storage;
  auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 1);
  ASSERT_NE(signal, nullptr);
  uint64_t epoch = 37;
  iree_hal_amd_gpu_signal_publish_frontier(signal, &epoch);
  iree_hal_amd_gpu_signal_retain(signal);
  std::barrier<> begin(2);
  std::thread reader([&] {
    for (uint32_t width = 2; width <= 256; width *= 2) {
      begin.arrive_and_wait();
      for (uint32_t i = 0; i < 128; ++i) {
        std::array<uint64_t, 256> consumer = {};
        iree_hal_amd_gpu_signal_join_frontier(signal, consumer.size(),
                                              consumer.data());
        EXPECT_EQ(consumer[0], epoch);
        for (size_t j = 1; j < consumer.size(); ++j) {
          EXPECT_EQ(consumer[j], 0u);
        }
      }
    }
    iree_hal_amd_gpu_signal_release(signal);
  });
  for (uint32_t width = 2; width <= 256; width *= 2) {
    begin.arrive_and_wait();
    IREE_EXPECT_OK(iree_hal_amd_gpu_signal_block_reserve_frontiers(
        &storage.block, width, iree_allocator_system()));
    // Other cells can immediately publish the new coordinate while readers
    // retain a snapshot of the original producer.
    auto* other = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
    EXPECT_NE(other, nullptr);
    std::array<uint64_t, 256> frontier = {};
    frontier[width - 1] = width;
    iree_hal_amd_gpu_signal_publish_frontier(other, frontier.data());
    iree_hal_amd_gpu_signal_release(other);
  }
  reader.join();
  auto* current = iree_hal_amd_gpu_signal_block_frontiers(&storage.block);
  uint32_t previous_widths = 0;
  for (auto* previous = current->previous; previous;
       previous = previous->previous) {
    previous_widths += previous->queue_count;
  }
  EXPECT_LT(previous_widths, current->queue_count);
  EXPECT_EQ(current->queue_count, 256u);
  iree_hal_amd_gpu_signal_release(signal);
}

TEST(SignalTest, EveryOwnerCanBeTheLastUse) {
  // Publication, producer execution, cache, and consumer execution hold four
  // independent references. All release orders preserve the same lifetime.
  Storage<1> storage;
  std::array<int, 4> release_order = {0, 1, 2, 3};
  do {
    auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
    ASSERT_NE(signal, nullptr);
    std::array<Signal*, 4> owners = {signal, signal, signal, signal};
    for (size_t i = 1; i < owners.size(); ++i) {
      iree_hal_amd_gpu_signal_retain(signal);
    }
    *signal->host_value = 1;
    for (size_t i = 0; i < release_order.size(); ++i) {
      auto*& owner = owners[release_order[i]];
      iree_hal_amd_gpu_signal_release(owner);
      owner = nullptr;
      if (i + 1 < release_order.size()) {
        EXPECT_EQ(iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0),
                  nullptr);
        EXPECT_EQ(storage.values[0], 1u);
      }
    }
    auto* recycled = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
    ASSERT_NE(recycled, nullptr);
    EXPECT_EQ(*recycled->host_value, 0u);
    iree_hal_amd_gpu_signal_release(recycled);
  } while (std::next_permutation(release_order.begin(), release_order.end()));
}

TEST(SignalTest, LastReaderDoesNotPinUnrelatedCells) {
  Storage<2> storage;
  auto* held = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
  ASSERT_NE(held, nullptr);
  iree_hal_amd_gpu_signal_retain(held);
  *held->host_value = 1;
  // Producer retirement leaves only the independent consumer reference.
  iree_hal_amd_gpu_signal_release(held);
  for (uint32_t i = 0; i < 4096; ++i) {
    auto* other = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
    ASSERT_NE(other, nullptr);
    EXPECT_NE(other, held);
    EXPECT_EQ(*other->host_value, 0u);
    EXPECT_EQ(*held->host_value, 1u);
    *other->host_value = 1;
    iree_hal_amd_gpu_signal_release(other);
  }
  iree_hal_amd_gpu_signal_release(held);
}

TEST(SignalTest, NotifyOnlyWhenReturnListBecomesNonempty) {
  uint32_t notification_count = 0;
  Storage<2> storage(
      {[](void* user_data) { ++*static_cast<uint32_t*>(user_data); },
       &notification_count});
  auto* first = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
  auto* second = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  iree_hal_amd_gpu_signal_retain(first);
  iree_hal_amd_gpu_signal_release(first);
  EXPECT_EQ(notification_count, 0u);
  iree_hal_amd_gpu_signal_release(second);
  EXPECT_EQ(notification_count, 1u);
  iree_hal_amd_gpu_signal_release(first);
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
                              iree_hal_amd_gpu_signal_pool_try_acquire(
                                  observer->pool, 0);
                        }
                      },
                      &observer});
  observer.pool = &storage.pool;
  auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
  ASSERT_NE(signal, nullptr);
  iree_hal_amd_gpu_signal_release(signal);
  EXPECT_EQ(observer.acquired, signal);
  iree_hal_amd_gpu_signal_release(observer.acquired);
}

TEST(SignalTest, InactiveBackingCanRetireWhileTheLastReleaseReturns) {
  constexpr uint32_t kIterations = 256;
  std::binary_semaphore ready{0};
  std::binary_semaphore returned{0};
  std::atomic<uint32_t> notifications{0};
  Signal* signal = nullptr;
  std::thread reader([&] {
    for (uint32_t i = 0; i < kIterations; ++i) {
      ready.acquire();
      iree_hal_amd_gpu_signal_release(signal);
      returned.release();
    }
  });
  for (uint32_t i = 0; i < kIterations; ++i) {
    auto* storage = new Storage<1>(
        {[](void* user_data) {
           static_cast<std::atomic<uint32_t>*>(user_data)->fetch_add(1);
         },
         &notifications});
    signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage->pool, 0);
    ready.release();
    // Detachment joins the final native reader on another queue. The copied
    // notification target remains live even after block metadata is reclaimed.
    bool unused = false;
    while (!unused) {
      unused = iree_hal_amd_gpu_signal_block_try_deinitialize(&storage->block);
      if (!unused) {
        std::this_thread::yield();
      }
    }
    delete storage;
    returned.acquire();
  }
  reader.join();
  EXPECT_EQ(notifications.load(), kIterations);
}

TEST(SignalTest, UnpublishedAndCompletedUsesCanBothReturn) {
  Storage<1> storage;
  for (uint32_t i = 0; i < 4096; ++i) {
    auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
    ASSERT_NE(signal, nullptr);
    EXPECT_EQ(*signal->host_value, 0u);
    // Abandoned captures never execute a native write. Clearing during
    // exclusive acquisition makes readiness independent of that history.
    if (i % 3 == 0) {
      *signal->host_value = 1;
    }
    iree_hal_amd_gpu_signal_release(signal);
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
        iree_hal_amd_gpu_signal_release(current);
        end.arrive_and_wait();
      }
    });
  }
  for (uint32_t epoch = 1; epoch <= kIterations; ++epoch) {
    current = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
    if (!current) {
      std::abort();
    }
    for (size_t i = 1; i < readers.size(); ++i) {
      iree_hal_amd_gpu_signal_retain(current);
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

TEST(SignalTest, SinglePublisherAndConcurrentReaderOwnership) {
  constexpr size_t kLaneCount = 4;
  constexpr uint32_t kIterations = 4096;
  Storage<kLaneCount * 2 + 1> storage;
  // A retained unrelated point stays intact while every lane recycles.
  auto* held = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
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
  std::array<std::thread, kLaneCount> readers;
  for (size_t i = 0; i < lanes.size(); ++i) {
    readers[i] = std::thread([&, i] {
      auto& lane = lanes[i];
      for (uint32_t j = 0; j < kIterations; ++j) {
        lane.ready.acquire();
        auto* signal = lane.signal;
        EXPECT_EQ(*signal->host_value, 1u);
        iree_hal_amd_gpu_signal_release(signal);
        lane.available.release();
      }
    });
  }
  // The real queue has one exclusive publisher even when callers arrive on
  // many threads. Independent readers return capacity while it keeps issuing.
  for (uint32_t j = 0; j < kIterations; ++j) {
    for (auto& lane : lanes) {
      lane.available.acquire();
      auto* signal = iree_hal_amd_gpu_signal_pool_try_acquire(&storage.pool, 0);
      if (!signal) {
        std::abort();
      }
      EXPECT_NE(signal, held);
      EXPECT_EQ(*signal->host_value, 0u);
      *signal->host_value = 1;
      iree_hal_amd_gpu_signal_retain(signal);
      lane.signal = signal;
      lane.ready.release();
      // The reader can win this race and drop its reference first.
      iree_hal_amd_gpu_signal_release(signal);
    }
  }
  for (auto& reader : readers) {
    reader.join();
  }
  EXPECT_EQ(*held->host_value, 1u);
  iree_hal_amd_gpu_signal_release(held);
}

}  // namespace
