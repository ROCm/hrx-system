// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/publisher.h"

#include <array>
#include <atomic>
#include <barrier>
#include <semaphore>
#include <thread>

#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"

namespace {

using Entry = iree_hal_amd_gpu_publication_entry_t;

class PublisherTest : public ::testing::Test {
 protected:
  void SetUp() override { iree_hal_amd_gpu_publisher_initialize(&publisher_); }
  void TearDown() override {
    iree_hal_amd_gpu_publisher_deinitialize(&publisher_);
  }

  bool Enqueue(Entry* entry) {
    return iree_hal_amd_gpu_publisher_enqueue(&publisher_, entry);
  }
  bool Handoff(Entry* remainder = nullptr) {
    return iree_hal_amd_gpu_publisher_handoff(&publisher_, remainder);
  }
  bool AcquireService() {
    return iree_hal_amd_gpu_publisher_acquire_service(&publisher_);
  }
  Entry* Next() { return iree_hal_amd_gpu_publisher_next(&publisher_); }

  // Exact production owner shared by calling producers and one service.
  iree_hal_amd_gpu_publisher_t publisher_;
};

TEST_F(PublisherTest, IdleCallerPublishesBeforeServiceTurn) {
  Entry first;
  EXPECT_TRUE(iree_hal_amd_gpu_publisher_is_idle(&publisher_));
  ASSERT_TRUE(Enqueue(&first));
  EXPECT_FALSE(iree_hal_amd_gpu_publisher_is_idle(&publisher_));
  EXPECT_FALSE(AcquireService());
  EXPECT_EQ(Next(), &first);
  EXPECT_FALSE(Handoff());
  EXPECT_FALSE(AcquireService());
  EXPECT_TRUE(iree_hal_amd_gpu_publisher_is_idle(&publisher_));
}

TEST_F(PublisherTest, FinalEnqueueBeforeHandoffTransfersToService) {
  Entry first;
  Entry last;
  ASSERT_TRUE(Enqueue(&first));
  EXPECT_EQ(Next(), &first);
  EXPECT_FALSE(Enqueue(&last));
  EXPECT_FALSE(AcquireService());
  EXPECT_TRUE(Handoff());
  ASSERT_TRUE(AcquireService());
  EXPECT_EQ(Next(), &last);
  EXPECT_EQ(Next(), nullptr);
  EXPECT_FALSE(Handoff());
}

TEST_F(PublisherTest, FinalEnqueueAfterHandoffClaimsCaller) {
  Entry first;
  Entry last;
  ASSERT_TRUE(Enqueue(&first));
  EXPECT_EQ(Next(), &first);
  ASSERT_FALSE(Handoff());
  ASSERT_TRUE(Enqueue(&last));
  EXPECT_EQ(Next(), &last);
  EXPECT_FALSE(AcquireService());
  EXPECT_FALSE(Handoff());
}

TEST_F(PublisherTest, PendingServiceOwnershipRemainsReserved) {
  std::array<Entry, 3> entries;
  ASSERT_TRUE(Enqueue(&entries[0]));
  EXPECT_EQ(Next(), &entries[0]);
  ASSERT_FALSE(Enqueue(&entries[1]));
  ASSERT_TRUE(Handoff());
  ASSERT_FALSE(Enqueue(&entries[2]));
  ASSERT_TRUE(AcquireService());
  EXPECT_EQ(Next(), &entries[1]);
  EXPECT_EQ(Next(), &entries[2]);
  EXPECT_FALSE(Handoff());
}

TEST_F(PublisherTest, BoundedServiceTurnPreservesUnconsumedEntries) {
  std::array<Entry, 6> entries;
  ASSERT_TRUE(Enqueue(&entries[0]));
  EXPECT_EQ(Next(), &entries[0]);
  for (size_t i = 1; i < 4; ++i) {
    ASSERT_FALSE(Enqueue(&entries[i]));
  }
  ASSERT_TRUE(Handoff());
  ASSERT_TRUE(AcquireService());
  EXPECT_EQ(Next(), &entries[1]);
  EXPECT_FALSE(Enqueue(&entries[4]));
  EXPECT_FALSE(Enqueue(&entries[5]));
  ASSERT_TRUE(Handoff());
  ASSERT_TRUE(AcquireService());
  for (size_t i = 2; i < entries.size(); ++i) {
    EXPECT_EQ(Next(), &entries[i]);
  }
  EXPECT_EQ(Next(), nullptr);
  EXPECT_FALSE(Handoff());
}

TEST_F(PublisherTest, CapacityDeferralPreservesDetachedEntry) {
  std::array<Entry, 3> entries;
  ASSERT_TRUE(Enqueue(&entries[0]));
  Entry* current = Next();
  ASSERT_EQ(current, &entries[0]);
  ASSERT_FALSE(Enqueue(&entries[1]));
  ASSERT_TRUE(Handoff(current));
  ASSERT_TRUE(AcquireService());
  EXPECT_EQ(Next(), current);
  ASSERT_FALSE(Enqueue(&entries[2]));
  ASSERT_TRUE(Handoff(current));
  ASSERT_TRUE(AcquireService());
  for (Entry& entry : entries) {
    EXPECT_EQ(Next(), &entry);
  }
  EXPECT_FALSE(Handoff());
}

TEST_F(PublisherTest, CompletedEntriesMayBeFreedBeforeHandoff) {
  Entry* first = new Entry;
  Entry* second = new Entry;
  ASSERT_TRUE(Enqueue(first));
  ASSERT_EQ(Next(), first);
  ASSERT_FALSE(Enqueue(second));
  delete first;
  ASSERT_TRUE(Handoff());
  ASSERT_TRUE(AcquireService());
  ASSERT_EQ(Next(), second);
  delete second;
  // Detaching the last real node leaves only the permanent stub reachable.
  EXPECT_EQ(Next(), nullptr);
  EXPECT_FALSE(Handoff());
}

TEST_F(PublisherTest, UnfinishedProducerLinkLeavesEntriesOwned) {
  Entry first;
  ASSERT_TRUE(Enqueue(&first));
  ASSERT_EQ(Next(), &first);
  Entry delayed;
  Entry later;
  // Represent a producer paused after reserving the tail but before linking
  // its predecessor. Another complete enqueue can follow this legal state.
  iree_atomic_store(&delayed.next, 0, iree_memory_order_relaxed);
  auto* predecessor = reinterpret_cast<Entry*>(iree_atomic_exchange(
      &publisher_.tail, reinterpret_cast<intptr_t>(&delayed),
      iree_memory_order_acq_rel));
  ASSERT_FALSE(Enqueue(&later));
  EXPECT_EQ(Next(), nullptr);
  EXPECT_FALSE(Handoff());
  EXPECT_FALSE(AcquireService());
  // Finish the delayed link, then let an actual new caller claim progress.
  // The consumer never detached or mutated either pending entry across the gap.
  iree_atomic_store(&predecessor->next, reinterpret_cast<intptr_t>(&delayed),
                    iree_memory_order_release);
  Entry last;
  ASSERT_TRUE(Enqueue(&last));
  EXPECT_EQ(Next(), &last);
  ASSERT_TRUE(Handoff());
  ASSERT_TRUE(AcquireService());
  EXPECT_EQ(Next(), &delayed);
  EXPECT_EQ(Next(), &later);
  EXPECT_EQ(Next(), nullptr);
  EXPECT_FALSE(Handoff());
}

TEST_F(PublisherTest, EventPostedBetweenPredicateAndParkIsRemembered) {
  std::array<Entry, 2> entries;
  ASSERT_TRUE(Enqueue(&entries[0]));
  EXPECT_EQ(Next(), &entries[0]);
  ASSERT_FALSE(Enqueue(&entries[1]));
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::barrier<> rendezvous(2);
  std::thread service([&] {
    const auto token = iree_notification_prepare_wait(&notification);
    EXPECT_FALSE(AcquireService());
    rendezvous.arrive_and_wait();
    rendezvous.arrive_and_wait();
    EXPECT_TRUE(iree_notification_commit_wait(&notification, token, 0,
                                              IREE_TIME_INFINITE_FUTURE));
    ASSERT_TRUE(AcquireService());
    EXPECT_EQ(Next(), &entries[1]);
    EXPECT_FALSE(Handoff());
  });
  rendezvous.arrive_and_wait();
  EXPECT_TRUE(Handoff());
  iree_notification_post(&notification, 1);
  rendezvous.arrive_and_wait();
  service.join();
  iree_notification_deinitialize(&notification);
}

TEST_F(PublisherTest, EventBeforeEnrollmentIsFoundByPredicate) {
  std::array<Entry, 2> entries;
  ASSERT_TRUE(Enqueue(&entries[0]));
  EXPECT_EQ(Next(), &entries[0]);
  ASSERT_FALSE(Enqueue(&entries[1]));
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  ASSERT_TRUE(Handoff());
  iree_notification_post(&notification, 1);
  iree_notification_prepare_wait(&notification);
  ASSERT_TRUE(AcquireService());
  EXPECT_EQ(Next(), &entries[1]);
  iree_notification_cancel_wait(&notification);
  EXPECT_FALSE(Handoff());
  iree_notification_deinitialize(&notification);
}

TEST_F(PublisherTest, FinalEnqueueRacesIdleHandoffWithoutLosingOwnership) {
  constexpr size_t kEpochCount = 4096;
  Entry first;
  Entry last;
  std::array<uint32_t, kEpochCount> visits = {};
  std::barrier<> rendezvous(2);
  std::thread producer([&] {
    for (size_t i = 0; i < kEpochCount; ++i) {
      rendezvous.arrive_and_wait();
      if (Enqueue(&last)) {
        EXPECT_EQ(Next(), &last);
        ++visits[i];
        EXPECT_FALSE(Handoff());
      }
      rendezvous.arrive_and_wait();
    }
  });
  for (size_t i = 0; i < kEpochCount; ++i) {
    EXPECT_TRUE(Enqueue(&first));
    EXPECT_EQ(Next(), &first);
    rendezvous.arrive_and_wait();
    Handoff();
    rendezvous.arrive_and_wait();
    if (AcquireService()) {
      EXPECT_EQ(Next(), &last);
      ++visits[i];
      EXPECT_FALSE(Handoff());
    }
    EXPECT_EQ(visits[i], 1u) << i;
  }
  producer.join();
}

TEST_F(PublisherTest, ConcurrentProducersReuseRetiredStorage) {
  constexpr size_t kProducerCount = 4;
  constexpr size_t kSlotCount = 8;
  constexpr size_t kEpochCount = 128;
  struct Operation : Entry {
    // Exact ordinal for this use of the captured storage.
    size_t ordinal;
    // A completion permits the submitting thread to reuse this slot.
    std::binary_semaphore reusable{1};
  };
  std::array<std::array<Operation, kSlotCount>, kProducerCount> operations;
  std::array<uint32_t, kProducerCount * kSlotCount * kEpochCount> visits = {};
  std::atomic<bool> stopping = false;
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  auto publish = [&](Entry* entry) {
    auto* operation = static_cast<Operation*>(entry);
    // The publisher is the only writer, independent of the calling thread.
    ++visits[operation->ordinal];
    operation->reusable.release();
  };
  std::thread service([&] {
    while (true) {
      const auto token = iree_notification_prepare_wait(&notification);
      if (!AcquireService()) {
        if (stopping.load(std::memory_order_acquire)) {
          iree_notification_cancel_wait(&notification);
          return;
        }
        iree_notification_commit_wait(&notification, token, 0,
                                      IREE_TIME_INFINITE_FUTURE);
        continue;
      }
      iree_notification_cancel_wait(&notification);
      // Deliberately cross the per-producer slot boundary across service turns.
      for (size_t i = 0; i < 7; ++i) {
        Entry* entry = Next();
        if (!entry) {
          break;
        }
        publish(entry);
      }
      Handoff();
    }
  });
  std::barrier<> start(kProducerCount);
  std::array<std::thread, kProducerCount> producers;
  for (size_t i = 0; i < kProducerCount; ++i) {
    producers[i] = std::thread([&, i] {
      start.arrive_and_wait();
      for (size_t epoch = 0; epoch < kEpochCount; ++epoch) {
        for (size_t slot = 0; slot < kSlotCount; ++slot) {
          auto& operation = operations[i][slot];
          operation.reusable.acquire();
          operation.ordinal = (epoch * kProducerCount + i) * kSlotCount + slot;
          if (Enqueue(&operation)) {
            Entry* entry = Next();
            if (entry) {
              publish(entry);
            }
            if (Handoff()) {
              iree_notification_post(&notification, 1);
            }
          }
        }
      }
      for (auto& operation : operations[i]) {
        operation.reusable.acquire();
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  stopping.store(true, std::memory_order_release);
  iree_notification_post(&notification, 1);
  service.join();
  for (size_t i = 0; i < visits.size(); ++i) {
    EXPECT_EQ(visits[i], 1u) << i;
  }
  iree_notification_deinitialize(&notification);
}

}  // namespace
