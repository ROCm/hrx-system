// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async futex operations (FUTEX_WAIT/FUTEX_WAKE).
//
// These tests verify the io_uring futex2 integration available on Linux 6.7+.
// Unlike the base futex syscall wrappers in iree/base/futex.h, these tests
// cover the async operation types that can be submitted to a proactor:
//   - IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT
//   - IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE
//
// The key pattern: submit FUTEX_WAIT to the proactor, wake it from a background
// thread using direct syscalls, verify completion, then optionally have the
// completion callback wake the main thread back for round-trip verification.
//
// Tests requiring kernel 6.7+ check
// IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS and skip gracefully on older
// kernels.

#include "iree/async/operations/futex.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/futex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

namespace iree::async::cts {

class FutexTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (!iree_any_bit_set(capabilities_,
                          IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS)) {
      GTEST_SKIP() << "backend lacks futex operations capability";
    }
  }

  // Initializes a FUTEX_WAIT operation.
  static void InitFutexWaitOp(iree_async_futex_wait_operation_t* operation,
                              void* address, uint32_t expected,
                              iree_async_completion_fn_t callback,
                              void* user_data) {
    memset(operation, 0, sizeof(*operation));
    operation->base.type = IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT;
    operation->base.completion_fn = callback;
    operation->base.user_data = user_data;
    operation->futex_address = address;
    operation->expected_value = expected;
    operation->futex_flags =
        IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;
  }

  // Initializes a FUTEX_WAKE operation.
  static void InitFutexWakeOp(iree_async_futex_wake_operation_t* operation,
                              void* address, int32_t count,
                              iree_async_completion_fn_t callback,
                              void* user_data) {
    memset(operation, 0, sizeof(*operation));
    operation->base.type = IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE;
    operation->base.completion_fn = callback;
    operation->base.user_data = user_data;
    operation->futex_address = address;
    operation->wake_count = count;
    operation->futex_flags =
        IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;
  }
};

#if defined(IREE_RUNTIME_USE_FUTEX)

static bool atomic_flag_is_set(void* user_data) {
  return static_cast<std::atomic<bool>*>(user_data)->load(
      std::memory_order_acquire);
}

// Owns a thread that performs a futex state transition when triggered.
//
// The worker is created and parked before construction returns. This keeps
// process thread creation outside the lifetime of any subsequently submitted
// private io_uring futex wait. Linux kernels that migrate an mm to a private
// futex hash during thread creation have shipped with bugs that can strand an
// already-armed io_uring waiter during that migration.
class GatedFutexWaker {
 public:
  explicit GatedFutexWaker(std::atomic<uint32_t>* futex_word)
      : futex_word_(futex_word) {
    iree_notification_initialize(&gate_);
    thread_ = std::thread([this]() {
      worker_ready_.store(true, std::memory_order_release);
      iree_notification_post(&gate_, IREE_ALL_WAITERS);

      bool was_triggered =
          iree_notification_await(&gate_, atomic_flag_is_set, &wake_requested_,
                                  iree_infinite_timeout());
      IREE_ASSERT(was_triggered);
      futex_word_->store(1, std::memory_order_release);
      iree_futex_wake(futex_word_, 1);
    });

    bool is_ready = iree_notification_await(
        &gate_, atomic_flag_is_set, &worker_ready_, iree_infinite_timeout());
    IREE_ASSERT(is_ready);
  }

  ~GatedFutexWaker() {
    Trigger();
    Join();
    iree_notification_deinitialize(&gate_);
  }

  void Trigger() {
    if (!wake_requested_.exchange(true, std::memory_order_acq_rel)) {
      iree_notification_post(&gate_, IREE_ALL_WAITERS);
    }
  }

  void Join() {
    if (thread_.joinable()) thread_.join();
  }

 private:
  std::atomic<uint32_t>* const futex_word_;
  iree_notification_t gate_;
  std::atomic<bool> worker_ready_{false};
  std::atomic<bool> wake_requested_{false};
  std::thread thread_;
};

// Submit FUTEX_WAIT to the proactor and complete the state transition from a
// pre-created background thread.
TEST_P(FutexTest, BasicFutexWaitWake) {
  std::atomic<uint32_t> futex_word{0};
  GatedFutexWaker waker(&futex_word);

  CompletionTracker wait_tracker;
  iree_async_futex_wait_operation_t wait_op;
  InitFutexWaitOp(&wait_op, &futex_word, 0, CompletionTracker::Callback,
                  &wait_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));
  waker.Trigger();
  PollUntil(/*min_completions=*/1);
  waker.Join();

  EXPECT_EQ(wait_tracker.call_count, 1);
  IREE_EXPECT_OK(wait_tracker.ConsumeStatus());
}

// Submit FUTEX_WAIT with wrong expected value. The operation should complete
// immediately with OK status since the value has already "changed" from the
// perspective of the wait (it never matched).
TEST_P(FutexTest, FutexWaitValueMismatch) {
  std::atomic<uint32_t> futex_word{42};  // Value is 42, not 0.

  CompletionTracker tracker;
  iree_async_futex_wait_operation_t wait_op;
  InitFutexWaitOp(&wait_op, &futex_word, 0,  // Expected 0, but actual is 42.
                  CompletionTracker::Callback, &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Should complete immediately since value doesn't match expected.
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Submit FUTEX_WAKE operation to wake a background thread doing a syscall wait.
TEST_P(FutexTest, FutexWakeWakesWaiter) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<bool> waiter_started{false};
  std::atomic<bool> waiter_woken{false};

  // Background thread: wait on the futex using direct syscall.
  std::thread waiter([&]() {
    waiter_started.store(true, std::memory_order_release);

    // Wait on the futex (direct syscall, not async).
    while (futex_word.load(std::memory_order_acquire) == 0) {
      iree_futex_wait(&futex_word, 0, IREE_TIME_INFINITE_FUTURE);
    }

    waiter_woken.store(true, std::memory_order_release);
  });

  // Wait for waiter to start.
  while (!waiter_started.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  // Change the futex word.
  futex_word.store(1, std::memory_order_release);

  // Submit async FUTEX_WAKE operation.
  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, 1, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  // Poll for completion.
  PollUntil(/*min_completions=*/1);

  waiter.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_GE(wake_op.woken_count, 0);  // May be 0 if waiter saw value change.
  EXPECT_TRUE(waiter_woken.load(std::memory_order_acquire));
}

// Submit FUTEX_WAKE(2) and verify the kernel wake count respects the bound.
TEST_P(FutexTest, FutexWakeCountIsBounded) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<int> waiters_ready{0};
  std::atomic<int> waiters_woken{0};
  constexpr int kNumWaiters = 3;
  constexpr int kWakeCount = 2;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      waiters_ready.fetch_add(1, std::memory_order_acq_rel);

      // Wait on the futex (direct syscall).
      while (futex_word.load(std::memory_order_acquire) == 0) {
        iree_futex_wait(&futex_word, 0, IREE_TIME_INFINITE_FUTURE);
      }

      waiters_woken.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to be ready.
  while (waiters_ready.load(std::memory_order_acquire) < kNumWaiters) {
    iree_thread_yield();
  }

  // Change the futex word.
  futex_word.store(1, std::memory_order_release);

  // Submit async FUTEX_WAKE for exactly 2 waiters.
  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, kWakeCount,
                  CompletionTracker::Callback, &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  // Poll for completion of the wake operation.
  PollUntil(/*min_completions=*/1);

  // Wake remaining waiters so we can join.
  iree_futex_wake(&futex_word, IREE_ALL_WAITERS);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_GE(wake_op.woken_count, 0);
  EXPECT_LE(wake_op.woken_count, kWakeCount);
}

// Submit FUTEX_WAKE(INT32_MAX) and verify all waiters wake.
TEST_P(FutexTest, FutexWakeAll) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<int> waiters_ready{0};
  std::atomic<int> waiters_woken{0};
  constexpr int kNumWaiters = 3;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      waiters_ready.fetch_add(1, std::memory_order_acq_rel);

      // Wait on the futex (direct syscall).
      while (futex_word.load(std::memory_order_acquire) == 0) {
        iree_futex_wait(&futex_word, 0, IREE_TIME_INFINITE_FUTURE);
      }

      waiters_woken.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to be ready.
  while (waiters_ready.load(std::memory_order_acquire) < kNumWaiters) {
    iree_thread_yield();
  }

  // Change the futex word.
  futex_word.store(1, std::memory_order_release);

  // Submit async FUTEX_WAKE for all waiters.
  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, INT32_MAX, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  // Poll for completion.
  PollUntil(/*min_completions=*/1);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(waiters_woken.load(std::memory_order_acquire), kNumWaiters);
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Submit FUTEX_WAKE with no waiters. Verify woken_count == 0.
TEST_P(FutexTest, FutexWakeNoWaiters) {
  std::atomic<uint32_t> futex_word{0};

  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, INT32_MAX, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(wake_op.woken_count, 0);
}

// Full async round-trip: main submits WAIT, a background thread wakes it, and
// the completion callback publishes a second futex state change.
TEST_P(FutexTest, FutexRoundTrip) {
  std::atomic<uint32_t> futex_a{0};
  std::atomic<uint32_t> futex_b{0};
  GatedFutexWaker waker(&futex_a);

  struct RoundTripContext {
    // Futex state published by the completion callback.
    std::atomic<uint32_t>* futex_b;
    // Completion status owned until the test consumes it.
    iree_status_t status = iree_ok_status();
    // Set after the completion callback publishes |futex_b|.
    bool callback_fired = false;
  };
  RoundTripContext context = {&futex_b};

  iree_async_futex_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT;
  wait_op.base.completion_fn = [](void* user_data, iree_async_operation_t* op,
                                  iree_status_t status,
                                  iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<RoundTripContext*>(user_data);
    ctx->callback_fired = true;
    ctx->status = status;
    // Wake the main thread by setting futex_b.
    ctx->futex_b->store(1, std::memory_order_release);
    iree_futex_wake(ctx->futex_b, 1);
  };
  wait_op.base.user_data = &context;
  wait_op.futex_address = &futex_a;
  wait_op.expected_value = 0;
  wait_op.futex_flags =
      IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;

  // Submit the wait operation.
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));
  waker.Trigger();

  PollUntilCondition(
      [&] { return futex_b.load(std::memory_order_acquire) != 0; },
      "futex round trip");

  waker.Join();

  EXPECT_TRUE(context.callback_fired);
  IREE_EXPECT_OK(context.status);
}

// Submit FUTEX_WAIT + LINK + NOP, verify NOP only runs after wait completes.
TEST_P(FutexTest, FutexChainedOperations) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
    GTEST_SKIP() << "backend lacks linked operations capability";
  }

  std::atomic<uint32_t> futex_word{0};
  GatedFutexWaker waker(&futex_word);

  struct OrderTracker {
    // Callback order observed by the polling thread.
    std::vector<int> order;
    // Wait operation status owned until the test consumes it.
    iree_status_t wait_status = iree_ok_status();
    // NOP operation status owned until the test consumes it.
    iree_status_t nop_status = iree_ok_status();
    ~OrderTracker() {
      iree_status_free(wait_status);
      iree_status_free(nop_status);
    }
    static void WaitCallback(void* u, iree_async_operation_t* o,
                             iree_status_t s, iree_async_completion_flags_t f) {
      auto* tracker = static_cast<OrderTracker*>(u);
      tracker->order.push_back(0);
      tracker->wait_status = s;
    }
    static void NopCallback(void* u, iree_async_operation_t* o, iree_status_t s,
                            iree_async_completion_flags_t f) {
      auto* tracker = static_cast<OrderTracker*>(u);
      tracker->order.push_back(1);
      tracker->nop_status = s;
    }
  };

  OrderTracker tracker;

  iree_async_futex_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT;
  wait_op.base.completion_fn = OrderTracker::WaitCallback;
  wait_op.base.user_data = &tracker;
  wait_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;  // Link to next op.
  wait_op.futex_address = &futex_word;
  wait_op.expected_value = 0;
  wait_op.futex_flags =
      IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;

  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop.base.completion_fn = OrderTracker::NopCallback;
  nop.base.user_data = &tracker;

  iree_async_operation_t* ops[] = {&wait_op.base, &nop.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));
  waker.Trigger();

  // Poll until both complete.
  PollUntil(/*min_completions=*/2);

  waker.Join();

  ASSERT_EQ(tracker.order.size(), 2u);
  // Wait must complete before NOP (enforced by LINK).
  EXPECT_EQ(tracker.order[0], 0);
  EXPECT_EQ(tracker.order[1], 1);
  IREE_EXPECT_OK(tracker.wait_status);
  tracker.wait_status = iree_ok_status();
  IREE_EXPECT_OK(tracker.nop_status);
  tracker.nop_status = iree_ok_status();
}

// Submit FUTEX_WAIT, cancel it, verify callback fires with CANCELLED status.
TEST_P(FutexTest, FutexCancellation) {
  std::atomic<uint32_t> futex_word{0};

  CompletionTracker tracker;
  iree_async_futex_wait_operation_t wait_op;
  InitFutexWaitOp(&wait_op, &futex_word, 0, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Cancel the wait operation immediately.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  // Poll to receive the cancellation callback.
  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
}

// Tests that a background thread waiting on a futex can be woken from a
// completion callback. This is a common pattern: async I/O completes, callback
// wakes a worker thread that was waiting for data.
TEST_P(FutexTest, WakeFromCompletionCallback) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<bool> waiter_started{false};
  std::atomic<bool> waiter_completed{false};

  // Background thread: wait for futex_word to become non-zero.
  std::thread waiter([&]() {
    waiter_started.store(true, std::memory_order_release);

    while (futex_word.load(std::memory_order_acquire) == 0) {
      iree_futex_wait(&futex_word, 0, IREE_TIME_INFINITE_FUTURE);
    }

    waiter_completed.store(true, std::memory_order_release);
  });

  // Wait for waiter to start.
  while (!waiter_started.load(std::memory_order_acquire)) {
    iree_thread_yield();
  }

  // Use an async FUTEX_WAKE operation (submitted to proactor) to wake the
  // waiter.
  futex_word.store(1, std::memory_order_release);

  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, 1, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  // Poll until the wake completes.
  PollUntil(/*min_completions=*/1);

  // Wait for the waiter thread to finish.
  waiter.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_TRUE(waiter_completed.load(std::memory_order_acquire));
}

// Tests that multiple waiters can be woken with IREE_ALL_WAITERS.
TEST_P(FutexTest, WakeAllFromCallback) {
  std::atomic<uint32_t> futex_word{0};
  std::atomic<int> waiters_started{0};
  std::atomic<int> waiters_completed{0};
  constexpr int kNumWaiters = 3;

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiters.emplace_back([&]() {
      waiters_started.fetch_add(1, std::memory_order_acq_rel);

      while (futex_word.load(std::memory_order_acquire) == 0) {
        iree_futex_wait(&futex_word, 0, IREE_TIME_INFINITE_FUTURE);
      }

      waiters_completed.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  // Wait for all waiters to start.
  while (waiters_started.load(std::memory_order_acquire) < kNumWaiters) {
    iree_thread_yield();
  }

  // Use async FUTEX_WAKE to wake all waiters.
  futex_word.store(1, std::memory_order_release);

  CompletionTracker tracker;
  iree_async_futex_wake_operation_t wake_op;
  InitFutexWakeOp(&wake_op, &futex_word, IREE_ALL_WAITERS,
                  CompletionTracker::Callback, &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wake_op.base));

  PollUntil(/*min_completions=*/1);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(waiters_completed.load(std::memory_order_acquire), kNumWaiters);
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
}

// Double-cancel a FUTEX_WAIT operation. Second cancel should be harmless.
TEST_P(FutexTest, FutexDoubleCancellation) {
  std::atomic<uint32_t> futex_word{0};

  CompletionTracker tracker;
  iree_async_futex_wait_operation_t wait_op;
  InitFutexWaitOp(&wait_op, &futex_word, 0, CompletionTracker::Callback,
                  &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // First cancel.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  // Second cancel should be harmless.
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));

  // Poll to receive the cancellation callback.
  PollUntil(/*min_completions=*/1);

  // Exactly one callback should have fired.
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, tracker.ConsumeStatus());
}

// Cancel a FUTEX_WAIT that races with a wake. Either outcome is valid but the
// callback must fire exactly once.
TEST_P(FutexTest, CancelRacesWithWake) {
  static constexpr int kIterations = 10;

  for (int iter = 0; iter < kIterations; ++iter) {
    std::atomic<uint32_t> futex_word{0};
    GatedFutexWaker waker(&futex_word);

    CompletionTracker tracker;
    iree_async_futex_wait_operation_t wait_op;
    InitFutexWaitOp(&wait_op, &futex_word, 0, CompletionTracker::Callback,
                    &tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

    // Concurrently cancel and wake to create a race.
    waker.Trigger();
    IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait_op.base));
    waker.Join();

    // Poll for the callback.
    PollUntil(/*min_completions=*/1);

    // Must have exactly one callback.
    EXPECT_EQ(tracker.call_count, 1) << "Iteration " << iter;

    // Either CANCELLED or OK is valid.
    iree_status_t status = tracker.ConsumeStatus();
    if (!iree_status_is_ok(status) && !iree_status_is_cancelled(status)) {
      IREE_EXPECT_OK(status) << "Iteration " << iter;
    } else if (iree_status_is_cancelled(status)) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status)
          << "Iteration " << iter;
    } else {
      IREE_EXPECT_OK(status) << "Iteration " << iter;
    }
  }
}

#else

// Placeholder when futex is not available (e.g., macOS, Windows without
// appropriate support).
TEST_P(FutexTest, NotAvailable) {
  GTEST_SKIP() << "Futex not available on this platform/configuration";
}

#endif  // IREE_RUNTIME_USE_FUTEX

CTS_REGISTER_TEST_SUITE(FutexTest);

}  // namespace iree::async::cts
