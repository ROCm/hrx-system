// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/futex.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/threading/thread.h"
#include "iree/testing/coordinated_test.h"
#include "iree/testing/gtest.h"

#if defined(IREE_PLATFORM_LINUX) && defined(IREE_PLATFORM_HAS_FUTEX)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_LINUX && IREE_PLATFORM_HAS_FUTEX

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
// at the address doesn't match the expected value. Notification commit_wait
// relies on this when publication precedes entry into the native wait.
TEST(FutexTest, WaitValueMismatch) {
  uint32_t futex_word = 42;
  // Expected value doesn't match - should return immediately.
  iree_status_code_t status =
      iree_futex_wait(&futex_word, 0, IREE_TIME_INFINITE_FUTURE);
  EXPECT_EQ(status, IREE_STATUS_OK);
  EXPECT_EQ(iree_futex_wait_shared(&futex_word, 0, IREE_TIME_INFINITE_FUTURE),
            IREE_STATUS_OK);
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

#if defined(IREE_PLATFORM_LINUX) && defined(IREE_PLATFORM_HAS_FUTEX)

// The futex is only a blocking mechanism. The release/acquire epoch publishes
// the payload, including when raw shared waits run under ThreadSanitizer.
struct SharedMessage {
  // Even values acknowledge receipt; odd values publish the next payload.
  iree_atomic_int32_t epoch;
  // Data protected by the epoch handoff, not by the futex syscall.
  uint32_t payload;
  // Publisher mapping address used to ensure the reader has a different alias.
  uint64_t publisher_address;
};

static void WaitForEpoch(SharedMessage* message, int32_t expected_epoch) {
  int32_t observed_epoch =
      iree_atomic_load(&message->epoch, iree_memory_order_acquire);
  while (observed_epoch != expected_epoch) {
    auto result = iree_futex_wait_shared(&message->epoch, observed_epoch,
                                         IREE_TIME_INFINITE_FUTURE);
    ASSERT_TRUE(result == IREE_STATUS_OK || result == IREE_STATUS_UNAVAILABLE);
    observed_epoch =
        iree_atomic_load(&message->epoch, iree_memory_order_acquire);
  }
}

static void PublishMessages(SharedMessage* message) {
  for (int32_t i = 0; i < 200; ++i) {
    message->payload = 0xC0FFEE00u + i;
    iree_atomic_store(&message->epoch, i * 2 + 1, iree_memory_order_release);
    iree_futex_wake_shared(&message->epoch, 1);
    WaitForEpoch(message, i * 2 + 2);
  }
}

static void ReceiveMessages(SharedMessage* message) {
  for (int32_t i = 0; i < 200; ++i) {
    WaitForEpoch(message, i * 2 + 1);
    EXPECT_EQ(message->payload, 0xC0FFEE00u + i);
    iree_atomic_store(&message->epoch, i * 2 + 2, iree_memory_order_release);
    iree_futex_wake_shared(&message->epoch, 1);
  }
}

TEST(FutexTest, SharedWaitPublishesPayloadAcrossThreads) {
  SharedMessage message = {};
  std::thread receiver([&] { ReceiveMessages(&message); });
  PublishMessages(&message);
  receiver.join();
}

static int PublisherRole(int argc, char** argv, const char* temp_directory) {
  std::string path = std::string(temp_directory) + "/futex.bin";
  int fd = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
  if (fd < 0) {
    return 1;
  }
  int resize_result = ftruncate(fd, sizeof(SharedMessage));
  if (resize_result != 0) {
    close(fd);
    return 1;
  }
  void* address = mmap(nullptr, sizeof(SharedMessage), PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
  close(fd);
  if (address == MAP_FAILED) {
    return 1;
  }
  auto* message = static_cast<SharedMessage*>(address);
  message->publisher_address = reinterpret_cast<uintptr_t>(address);
  iree_atomic_store(&message->epoch, 0, iree_memory_order_release);
  iree_coordinated_test_signal_ready(temp_directory);
  PublishMessages(message);
  EXPECT_EQ(munmap(address, sizeof(SharedMessage)), 0);
  return ::testing::Test::HasFailure() ? 1 : 0;
}

static int ReceiverRole(int argc, char** argv, const char* temp_directory) {
  std::string path = std::string(temp_directory) + "/futex.bin";
  int fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    return 1;
  }
  void* address = mmap(nullptr, sizeof(SharedMessage), PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
  if (address == MAP_FAILED) {
    close(fd);
    return 1;
  }
  auto* message = static_cast<SharedMessage*>(address);
  if (message->publisher_address == reinterpret_cast<uintptr_t>(address)) {
    // Keep the first alias occupied until the distinct second alias exists.
    void* alternate = mmap(nullptr, sizeof(SharedMessage),
                           PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    EXPECT_EQ(munmap(address, sizeof(SharedMessage)), 0);
    address = alternate;
    message = static_cast<SharedMessage*>(address);
  }
  close(fd);
  if (address == MAP_FAILED) {
    return 1;
  }
  EXPECT_NE(message->publisher_address, reinterpret_cast<uintptr_t>(address));
  ReceiveMessages(message);
  EXPECT_EQ(munmap(address, sizeof(SharedMessage)), 0);
  return ::testing::Test::HasFailure() ? 1 : 0;
}

static const iree_test_role_t kSharedFutexRoles[] = {
    {"publisher", PublisherRole, /*signals_ready=*/true},
    {"receiver", ReceiverRole, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kSharedFutexConfig = {
    .roles = kSharedFutexRoles,
    .role_count = IREE_ARRAYSIZE(kSharedFutexRoles),
};
IREE_COORDINATED_TEST_REGISTER(kSharedFutexConfig);

TEST(FutexTest, SharedWaitUsesBackingPageAcrossProcesses) {
  EXPECT_EQ(iree_coordinated_test_run(iree_coordinated_test_argc(),
                                      iree_coordinated_test_argv(),
                                      &kSharedFutexConfig),
            0);
}

#endif  // IREE_PLATFORM_LINUX && IREE_PLATFORM_HAS_FUTEX

}  // namespace
