// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/notification.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

//===----------------------------------------------------------------------===//
// Native-independent event-count state
//===----------------------------------------------------------------------===//

TEST(NotificationStateTest, PublicationAndEnrollmentShareOrdering) {
  iree_notification_state_t state;
  iree_notification_state_initialize(&state);

  // Publication before enrollment is observed by the enrolling caller, even
  // though there was no native wake obligation at publication time.
  iree_wait_token_t earlier_token = iree_notification_state_query_epoch(&state);
  EXPECT_EQ(iree_notification_state_post(&state), 0u);
  iree_wait_token_t token = iree_notification_state_prepare_wait(&state);
  EXPECT_NE(token, earlier_token);

  // Publication after enrollment carries the wake obligation and changes the
  // token observed by a caller that has not entered its native wait yet.
  EXPECT_EQ(iree_notification_state_post(&state), 1u);
  EXPECT_NE(iree_notification_state_query_epoch(&state), token);
  iree_notification_state_cancel_wait(&state);
  EXPECT_EQ(iree_notification_state_post(&state), 0u);
}

TEST(NotificationStateTest, AcquiresPublishedData) {
  iree_notification_state_t state;
  iree_notification_state_initialize(&state);
  iree_wait_token_t token = iree_notification_state_prepare_wait(&state);
  int payload = 0;
  std::thread producer([&] {
    payload = 42;
    EXPECT_EQ(iree_notification_state_post(&state), 1u);
  });

  while (iree_notification_state_query_epoch(&state) == token) {
    std::this_thread::yield();
  }
  // Read before joining: visibility must come from the state's acquire, not
  // the thread join or a second application synchronization channel.
  EXPECT_EQ(payload, 42);
  iree_notification_state_cancel_wait(&state);
  producer.join();
  EXPECT_EQ(iree_notification_state_post(&state), 0u);
}

TEST(NotificationStateTest, ConcurrentEnrollmentAndPublication) {
  iree_notification_state_t state;
  iree_notification_state_initialize(&state);
  constexpr uint32_t kThreadCount = 8;
  constexpr uint32_t kPublicationsPerThread = 1000;
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (uint32_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (uint32_t j = 0; j < kPublicationsPerThread; ++j) {
        iree_wait_token_t token = iree_notification_state_prepare_wait(&state);
        uint32_t waiter_count = iree_notification_state_post(&state);
        // Our own enrollment must remain visible until we withdraw it, even
        // as other callers enroll, publish, and withdraw concurrently.
        EXPECT_GE(waiter_count, 1u);
        EXPECT_LE(waiter_count, kThreadCount);
        EXPECT_NE(iree_notification_state_query_epoch(&state), token);
        iree_notification_state_cancel_wait(&state);
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(iree_notification_state_query_epoch(&state),
            kThreadCount * kPublicationsPerThread);
  EXPECT_EQ(iree_notification_state_post(&state), 0u);
}

//==============================================================================
// Basic notification tests
//==============================================================================

TEST(NotificationTest, Lifetime) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, PostNoWaiters) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  // Should not block or crash when there are no waiters.
  iree_notification_post(&notification, 1);
  iree_notification_post(&notification, IREE_ALL_WAITERS);

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, TimeoutImmediate) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  EXPECT_FALSE(iree_notification_await(
      &notification,
      +[](void* entry_arg) -> bool {
        return false;  // Condition is never true.
      },
      NULL, iree_immediate_timeout()));

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, Timeout) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  iree_time_t start_ns = iree_time_now();

  EXPECT_FALSE(iree_notification_await(
      &notification,
      +[](void* entry_arg) -> bool {
        return false;  // Condition is never true.
      },
      NULL, iree_make_timeout_ms(100)));

  iree_duration_t delta_ns = iree_time_now() - start_ns;
  iree_duration_t delta_ms = delta_ns / 1000000;
  EXPECT_GE(delta_ms, 50);  // Should wait at least some time.

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, AwaitConditionAlreadyTrue) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  // Condition already true - should return immediately.
  bool result = iree_notification_await(
      &notification, +[](void*) { return true; }, nullptr,
      iree_immediate_timeout());
  EXPECT_TRUE(result);

  iree_notification_deinitialize(&notification);
}

//==============================================================================
// Post and await tests
//==============================================================================

TEST(NotificationTest, PostAndAwait) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<bool> ready{false};
  std::atomic<bool> waiter_started{false};

  std::thread waiter([&]() {
    waiter_started.store(true, std::memory_order_release);
    iree_notification_await(
        &notification,
        +[](void* arg) {
          return static_cast<std::atomic<bool>*>(arg)->load(
              std::memory_order_acquire);
        },
        &ready, iree_infinite_timeout());
  });

  // Wait for waiter thread to start (not necessarily in kernel wait, but that's
  // fine - the condition check in await handles the race).
  while (!waiter_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  ready.store(true, std::memory_order_release);
  iree_notification_post(&notification, IREE_ALL_WAITERS);
  waiter.join();

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, MultipleWaiters) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<bool> ready{false};
  std::atomic<int> started_count{0};
  std::atomic<int> woken_count{0};
  constexpr int kNumWaiters = 4;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      started_count.fetch_add(1, std::memory_order_release);
      iree_notification_await(
          &notification,
          +[](void* arg) {
            return static_cast<std::atomic<bool>*>(arg)->load(
                std::memory_order_acquire);
          },
          &ready, iree_infinite_timeout());
      woken_count.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to start.
  while (started_count.load(std::memory_order_acquire) < kNumWaiters) {
    std::this_thread::yield();
  }

  ready.store(true, std::memory_order_release);
  iree_notification_post(&notification, IREE_ALL_WAITERS);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(kNumWaiters, woken_count.load(std::memory_order_acquire));

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, PostSingleWaiter) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<int> phase{0};
  std::atomic<int> started_count{0};
  std::atomic<int> woken_count{0};
  constexpr int kNumWaiters = 4;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      started_count.fetch_add(1, std::memory_order_release);
      // Wait until phase >= 1.
      iree_notification_await(
          &notification,
          +[](void* arg) {
            return static_cast<std::atomic<int>*>(arg)->load(
                       std::memory_order_acquire) >= 1;
          },
          &phase, iree_infinite_timeout());
      woken_count.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to start.
  while (started_count.load(std::memory_order_acquire) < kNumWaiters) {
    std::this_thread::yield();
  }

  // Post to wake just one waiter.
  phase.store(1, std::memory_order_release);
  iree_notification_post(&notification, 1);

  // Wait for at least one waiter to wake.
  while (woken_count.load(std::memory_order_acquire) < 1) {
    std::this_thread::yield();
  }

  int initial_woken = woken_count.load(std::memory_order_acquire);
  // At least 1 should have woken (could be more due to spurious wakeups).
  EXPECT_GE(initial_woken, 1);

  // Wake remaining waiters.
  iree_notification_post(&notification, IREE_ALL_WAITERS);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(kNumWaiters, woken_count.load(std::memory_order_acquire));

  iree_notification_deinitialize(&notification);
}

//==============================================================================
// Low-level prepare/commit/cancel API tests
//==============================================================================

TEST(NotificationTest, PrepareAndCancelWait) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  iree_wait_token_t token = iree_notification_prepare_wait(&notification);
  (void)token;  // We only care that prepare doesn't crash.
  iree_notification_cancel_wait(&notification);
  // Should not deadlock or crash.

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, PrepareCommitWaitAlreadySignaled) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  // Prepare wait (captures epoch).
  iree_wait_token_t token = iree_notification_prepare_wait(&notification);

  // Post before commit - should cause commit to return quickly.
  iree_notification_post(&notification, IREE_ALL_WAITERS);

  // Another waiter may enroll before the first thread reaches commit_wait.
  // That enrollment must not erase the post observed by the earlier token.
  const iree_wait_token_t next_token =
      iree_notification_prepare_wait(&notification);
  EXPECT_NE(next_token, token);

  // Commit should return true (already signaled since epoch changed).
  bool result =
      iree_notification_commit_wait(&notification, token,
                                    /*spin_ns=*/0, IREE_TIME_INFINITE_FUTURE);
  EXPECT_TRUE(result);

  iree_notification_post(&notification, IREE_ALL_WAITERS);
  EXPECT_TRUE(iree_notification_commit_wait(&notification, next_token,
                                            IREE_DURATION_ZERO,
                                            IREE_TIME_INFINITE_FUTURE));
  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, PrepareCommitWaitTimeout) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);

  // Prepare wait.
  iree_wait_token_t token = iree_notification_prepare_wait(&notification);

  iree_time_t start = iree_time_now();

  // Commit with short timeout - should timeout since no post.
  bool result = iree_notification_commit_wait(&notification, token,
                                              /*spin_ns=*/0,
                                              iree_time_now() + 50 * 1000000);

  iree_time_t elapsed = iree_time_now() - start;

  EXPECT_FALSE(result);
  EXPECT_GE(elapsed, 40 * 1000000);  // Should have waited close to 50ms.

  iree_notification_deinitialize(&notification);
}

TEST(NotificationTest, PrepareCommitSpinWaitExpiredDeadline) {
  const iree_duration_t spin_durations[] = {500, IREE_DURATION_INFINITE - 1,
                                            IREE_DURATION_INFINITE};
  for (iree_duration_t spin_ns : spin_durations) {
    SCOPED_TRACE(spin_ns);
    iree_notification_t notification;
    iree_notification_initialize(&notification);
    iree_wait_token_t token = iree_notification_prepare_wait(&notification);

    // Even an unbounded active wait must respect a finite caller deadline.
    // An expired deadline exercises the arithmetic without delaying the test.
    EXPECT_FALSE(iree_notification_commit_wait(&notification, token, spin_ns,
                                               iree_time_now()));

    iree_notification_deinitialize(&notification);
  }
}

TEST(NotificationTest, PrepareCommitSpinWaitFiniteTimeout) {
  const iree_duration_t spin_durations[] = {500, IREE_DURATION_INFINITE - 1,
                                            IREE_DURATION_INFINITE};
  for (iree_duration_t spin_ns : spin_durations) {
    SCOPED_TRACE(spin_ns);
    iree_notification_t notification;
    iree_notification_initialize(&notification);
    iree_wait_token_t token = iree_notification_prepare_wait(&notification);
    const iree_time_t deadline_ns = iree_time_now() + iree_make_duration_ms(1);

    // Saturating the spin budget must neither return before the deadline nor
    // turn a finite wait into an infinite one.
    EXPECT_FALSE(iree_notification_commit_wait(&notification, token, spin_ns,
                                               deadline_ns));
    EXPECT_GE(iree_time_now(), deadline_ns);

    iree_notification_deinitialize(&notification);
  }
}

TEST(NotificationTest, PrepareCommitWaitWoken) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<bool> waiter_prepared{false};
  std::atomic<bool> waiter_result{false};

  std::thread waiter([&]() {
    iree_wait_token_t token = iree_notification_prepare_wait(&notification);
    waiter_prepared.store(true, std::memory_order_release);

    bool result =
        iree_notification_commit_wait(&notification, token,
                                      /*spin_ns=*/0, IREE_TIME_INFINITE_FUTURE);
    waiter_result.store(result, std::memory_order_release);
  });

  // Wait for waiter to prepare.
  while (!waiter_prepared.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  // Post to wake the waiter. Even if commit_wait hasn't entered the kernel yet,
  // the epoch will have advanced and commit_wait will return immediately.
  iree_notification_post(&notification, IREE_ALL_WAITERS);

  waiter.join();

  EXPECT_TRUE(waiter_result.load(std::memory_order_acquire));

  iree_notification_deinitialize(&notification);
}

//==============================================================================
// Stress tests
//==============================================================================

TEST(NotificationTest, RapidPostAwait) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<int> counter{0};
  constexpr int kIterations = 100;

  std::thread producer([&]() {
    for (int i = 0; i < kIterations; ++i) {
      counter.fetch_add(1, std::memory_order_release);
      iree_notification_post(&notification, IREE_ALL_WAITERS);
      std::this_thread::yield();
    }
  });

  std::thread consumer([&]() {
    int last_seen = 0;
    while (last_seen < kIterations) {
      // Use a struct on the stack instead of heap allocation.
      struct await_state_t {
        std::atomic<int>* counter;
        int last_seen;
      } state = {&counter, last_seen};
      EXPECT_TRUE(iree_notification_await(
          &notification,
          +[](void* arg) {
            auto* state = static_cast<await_state_t*>(arg);
            return state->counter->load(std::memory_order_acquire) >
                   state->last_seen;
          },
          &state, iree_infinite_timeout()));
      last_seen = counter.load(std::memory_order_acquire);
    }
  });

  producer.join();
  consumer.join();

  EXPECT_EQ(kIterations, counter.load(std::memory_order_acquire));

  iree_notification_deinitialize(&notification);
}

}  // namespace
