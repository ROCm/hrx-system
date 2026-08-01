// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/futex.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/threading/thread.h"
#include "iree/testing/gtest.h"

namespace {

#if defined(IREE_RUNTIME_USE_FUTEX)

// Tests that waking an address with no waiters returns immediately without
// blocking or error.
TEST(FutexTest, WakeNoWaiters) {
  uint32_t futex_word = 0;
  // Should complete immediately - no waiters.
  iree_futex_wake(&futex_word, 1);
  iree_futex_wake(&futex_word, IREE_ALL_WAITERS);
}

// Tests that iree_futex_wait returns IREE_STATUS_OK immediately when the value
// at the address doesn't match the expected value (spurious wakeup handling).
TEST(FutexTest, WaitValueMismatch) {
  uint32_t futex_word = 42;
  // Expected value doesn't match - should return immediately.
  // Note: Linux futex returns EAGAIN which maps to OK (retry expected).
  // Windows returns "value changed" which also completes successfully.
  iree_status_code_t status =
      iree_futex_wait(&futex_word, 0, IREE_TIME_INFINITE_FUTURE);
  // Both OK (spurious) and UNAVAILABLE (value mismatch) are acceptable.
  EXPECT_TRUE(status == IREE_STATUS_OK || status == IREE_STATUS_UNAVAILABLE);
}

// Tests that a background thread can be woken by the main thread.
TEST(FutexTest, WakeWakesWaiter) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<bool> waiter_started{false};
  std::atomic<bool> waiter_completed{false};

  std::thread waiter([&]() {
    waiter_started.store(true, std::memory_order_release);

    // Wait for the value to change from 0.
    while (futex_word.load(std::memory_order_acquire) == 0) {
      iree_futex_wait(const_cast<std::atomic<uint32_t>*>(&futex_word), 0,
                      IREE_TIME_INFINITE_FUTURE);
    }

    waiter_completed.store(true, std::memory_order_release);
  });

  // Wait for waiter thread to start.
  while (!waiter_started.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  // Change the value and wake.
  futex_word.store(1, std::memory_order_release);
  iree_futex_wake(const_cast<std::atomic<uint32_t>*>(&futex_word), 1);

  waiter.join();

  EXPECT_TRUE(waiter_completed.load(std::memory_order_acquire));
}

// Tests that waking all waiters releases every thread waiting on the address.
TEST(FutexTest, WakeAllWakesMultipleWaiters) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<int> waiters_started{0};
  std::atomic<int> waiters_woken{0};
  constexpr int kNumWaiters = 3;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      waiters_started.fetch_add(1, std::memory_order_acq_rel);

      // Wait until the value changes from 0.
      while (futex_word.load(std::memory_order_acquire) == 0) {
        iree_futex_wait(const_cast<std::atomic<uint32_t>*>(&futex_word), 0,
                        IREE_TIME_INFINITE_FUTURE);
      }

      waiters_woken.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to start.
  while (waiters_started.load(std::memory_order_acquire) < kNumWaiters) {
    iree_thread_yield();
  }

  // Publishing the value before waking makes this safe whether each waiter is
  // already blocked in the kernel or is about to check the predicate.
  futex_word.store(1, std::memory_order_release);
  iree_futex_wake(const_cast<std::atomic<uint32_t>*>(&futex_word),
                  IREE_ALL_WAITERS);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(waiters_woken.load(std::memory_order_acquire), kNumWaiters);
}

// Tests that a future deadline eventually terminates an unchanged wait.
TEST(FutexTest, WaitFutureDeadlineExpires) {
  uint32_t futex_word = 0;

  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(1);
  iree_status_code_t status = IREE_STATUS_OK;
  while (status == IREE_STATUS_OK) {
    status = iree_futex_wait(&futex_word, 0, deadline);
  }

  EXPECT_EQ(status, IREE_STATUS_DEADLINE_EXCEEDED);
}

// Tests that iree_futex_wait with immediate deadline returns immediately.
TEST(FutexTest, WaitImmediateDeadline) {
  uint32_t futex_word = 0;

  // IREE_TIME_INFINITE_PAST should cause a polling wait that reports the
  // deadline was already exceeded.
  iree_status_code_t status =
      iree_futex_wait(&futex_word, 0, IREE_TIME_INFINITE_PAST);

  EXPECT_EQ(status, IREE_STATUS_DEADLINE_EXCEEDED);
}

#else

// Placeholder test when futex is not available.
TEST(FutexTest, NotAvailable) {
  GTEST_SKIP() << "Futex not available on this platform/configuration";
}

#endif  // IREE_RUNTIME_USE_FUTEX

}  // namespace
