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
using List = iree_hal_amd_gpu_publication_list_t;

class PublisherTest : public ::testing::Test {
 protected:
  void SetUp() override { iree_hal_amd_gpu_publisher_initialize(&publisher_); }
  void TearDown() override {
    iree_hal_amd_gpu_publisher_deinitialize(&publisher_);
  }

  bool Enqueue(Entry* entry) {
    return iree_hal_amd_gpu_publisher_enqueue(&publisher_, entry);
  }
  bool Handoff(List remainder = {}) {
    return iree_hal_amd_gpu_publisher_handoff(&publisher_, remainder);
  }
  List Take() { return iree_hal_amd_gpu_publisher_take(&publisher_); }

  // Exact production owner shared by calling producers and one service.
  iree_hal_amd_gpu_publisher_t publisher_;
};

TEST_F(PublisherTest, IdleCallerPublishesBeforeServiceTurn) {
  Entry first;
  EXPECT_TRUE(iree_hal_amd_gpu_publisher_is_idle(&publisher_));
  EXPECT_TRUE(Enqueue(&first));
  EXPECT_FALSE(iree_hal_amd_gpu_publisher_is_idle(&publisher_));
  EXPECT_EQ(Take().head, nullptr);
  EXPECT_FALSE(Handoff());
  EXPECT_EQ(Take().head, nullptr);
  EXPECT_TRUE(iree_hal_amd_gpu_publisher_is_idle(&publisher_));
}

TEST_F(PublisherTest, FinalEnqueueBeforeHandoffTransfersToService) {
  Entry first;
  Entry last;
  ASSERT_TRUE(Enqueue(&first));
  EXPECT_FALSE(Enqueue(&last));
  EXPECT_EQ(Take().head, nullptr);
  EXPECT_TRUE(Handoff());
  const List work = Take();
  EXPECT_EQ(work.head, &last);
  EXPECT_EQ(work.tail, &last);
  EXPECT_EQ(last.next, nullptr);
  EXPECT_FALSE(Handoff());
}

TEST_F(PublisherTest, FinalEnqueueAfterHandoffClaimsCaller) {
  Entry first;
  Entry last;
  ASSERT_TRUE(Enqueue(&first));
  ASSERT_FALSE(Handoff());
  EXPECT_TRUE(Enqueue(&last));
  EXPECT_EQ(Take().head, nullptr);
  EXPECT_FALSE(Handoff());
}

TEST_F(PublisherTest, DetachedSuffixPrecedesConcurrentAppends) {
  std::array<Entry, 6> entries;
  ASSERT_TRUE(Enqueue(&entries[0]));
  for (size_t i = 1; i < 4; ++i) {
    ASSERT_FALSE(Enqueue(&entries[i]));
  }
  ASSERT_TRUE(Handoff());
  List work = Take();
  ASSERT_EQ(work.head, &entries[1]);
  ASSERT_EQ(work.tail, &entries[3]);
  // One bounded publication finishes while two more callers append.
  work.head = work.head->next;
  EXPECT_FALSE(Enqueue(&entries[4]));
  EXPECT_FALSE(Enqueue(&entries[5]));
  ASSERT_TRUE(Handoff(work));
  work = Take();
  EXPECT_EQ(work.tail, &entries[5]);
  for (size_t i = 2; i < entries.size(); ++i) {
    ASSERT_EQ(work.head, &entries[i]);
    work.head = work.head->next;
  }
  EXPECT_EQ(work.head, nullptr);
  EXPECT_FALSE(Handoff(work));
}

TEST_F(PublisherTest, EnqueueDuringDetachedFinalItemIsNotLost) {
  std::array<Entry, 3> entries;
  ASSERT_TRUE(Enqueue(&entries[0]));
  ASSERT_FALSE(Enqueue(&entries[1]));
  ASSERT_TRUE(Handoff());
  List work = Take();
  ASSERT_EQ(work.head, &entries[1]);
  work.head = work.head->next;
  ASSERT_EQ(work.head, nullptr);
  EXPECT_FALSE(Enqueue(&entries[2]));
  EXPECT_TRUE(Handoff(work));
  work = Take();
  EXPECT_EQ(work.head, &entries[2]);
  EXPECT_EQ(work.tail, &entries[2]);
  EXPECT_FALSE(Handoff());
}

TEST_F(PublisherTest, CapacityDeferralPreservesUnpublishedCallerEntry) {
  std::array<Entry, 3> entries;
  ASSERT_TRUE(Enqueue(&entries[0]));
  ASSERT_FALSE(Enqueue(&entries[1]));
  // The inline caller found no native capacity. It gives its own unpublished
  // entry to the service before every entry that arrived while it owned state.
  ASSERT_TRUE(Handoff({&entries[0], &entries[0]}));
  List work = Take();
  EXPECT_EQ(work.head, &entries[0]);
  EXPECT_EQ(work.tail, &entries[1]);
  ASSERT_FALSE(Enqueue(&entries[2]));
  // A failed capacity check does not consume the detached prefix.
  ASSERT_TRUE(Handoff(work));
  work = Take();
  for (Entry& entry : entries) {
    ASSERT_EQ(work.head, &entry);
    work.head = work.head->next;
  }
  EXPECT_FALSE(Handoff(work));
}

TEST_F(PublisherTest, CompletedEntriesMayBeFreedBeforeHandoff) {
  Entry* first = new Entry;
  Entry* second = new Entry;
  ASSERT_TRUE(Enqueue(first));
  ASSERT_FALSE(Enqueue(second));
  delete first;
  ASSERT_TRUE(Handoff());
  List work = Take();
  ASSERT_EQ(work.head, second);
  work.head = second->next;
  delete second;
  // The empty head means the already-retired tail is never dereferenced.
  EXPECT_FALSE(Handoff(work));
}

TEST_F(PublisherTest, EventPostedBetweenPredicateAndParkIsRemembered) {
  std::array<Entry, 2> entries;
  ASSERT_TRUE(Enqueue(&entries[0]));
  ASSERT_FALSE(Enqueue(&entries[1]));
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::barrier<> rendezvous(2);
  std::thread service([&] {
    const auto token = iree_notification_prepare_wait(&notification);
    EXPECT_EQ(Take().head, nullptr);
    rendezvous.arrive_and_wait();
    rendezvous.arrive_and_wait();
    EXPECT_TRUE(iree_notification_commit_wait(&notification, token, 0,
                                              IREE_TIME_INFINITE_FUTURE));
    const List work = Take();
    EXPECT_EQ(work.head, &entries[1]);
    EXPECT_EQ(work.tail, &entries[1]);
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
  ASSERT_FALSE(Enqueue(&entries[1]));
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  ASSERT_TRUE(Handoff());
  iree_notification_post(&notification, 1);
  iree_notification_prepare_wait(&notification);
  const List work = Take();
  EXPECT_EQ(work.head, &entries[1]);
  iree_notification_cancel_wait(&notification);
  EXPECT_FALSE(Handoff());
  iree_notification_deinitialize(&notification);
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
      List work = Take();
      if (!work.head) {
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
      for (size_t i = 0; work.head && i < 7; ++i) {
        Entry* entry = work.head;
        work.head = entry->next;
        publish(entry);
      }
      Handoff(work);
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
            publish(&operation);
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
