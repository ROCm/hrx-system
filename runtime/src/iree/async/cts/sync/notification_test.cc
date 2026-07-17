// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async notification operations.
//
// Notifications are lightweight synchronization primitives for proactor-
// integrated thread wakeup. Unlike events (edge-triggered, one signal per
// wait), notifications use an epoch counter that allows multiple signals
// to coalesce before a wait.
//
// Implementation varies by platform and capability:
//   - io_uring 6.7+: Uses futex word with FUTEX_WAIT/WAKE operations
//   - io_uring <6.7: Uses eventfd with linked POLL_ADD+READ pattern
//   - Other platforms: Platform-specific implementations

#include "iree/async/notification.h"

#include <atomic>
#include <future>
#include <thread>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/scheduling.h"

namespace iree::async::cts {

class NotificationTest : public CtsTestBase<> {
 protected:
  // Initializes a NOTIFICATION_WAIT operation.
  static void InitNotificationWaitOp(
      iree_async_notification_wait_operation_t* operation,
      iree_async_notification_t* notification,
      iree_async_completion_fn_t callback, void* user_data) {
    memset(operation, 0, sizeof(*operation));
    operation->base.type = IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT;
    operation->base.completion_fn = callback;
    operation->base.user_data = user_data;
    operation->notification = notification;
    // wait_token is set by the proactor at submit time by default.
  }

  // Initializes a NOTIFICATION_SIGNAL operation.
  static void InitNotificationSignalOp(
      iree_async_notification_signal_operation_t* operation,
      iree_async_notification_t* notification, int32_t wake_count,
      iree_async_completion_fn_t callback, void* user_data) {
    memset(operation, 0, sizeof(*operation));
    operation->base.type = IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL;
    operation->base.completion_fn = callback;
    operation->base.user_data = user_data;
    operation->notification = notification;
    operation->wake_count = wake_count;
  }
};

// Create notification, retain/release, verify proper lifecycle.
TEST_P(NotificationTest, RetainRelease) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  // Initial ref count is 1.
  iree_async_notification_retain(notification);
  // Now ref count is 2.

  iree_async_notification_release(notification);
  // Now ref count is 1.

  iree_async_notification_release(notification);
  // Now ref count is 0, notification is destroyed.
}

// Signal notification with no waiters - should complete without error.
TEST_P(NotificationTest, SignalNoWaiters) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  // Signal with no waiters - this should be a no-op that succeeds.
  iree_async_notification_signal(notification, 1);

  // Verify notification is still usable.
  iree_async_notification_signal(notification, INT32_MAX);

  iree_async_notification_release(notification);
}

TEST_P(NotificationTest, SignalIfObservedNoWaitersSkipsWake) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  const uint32_t epoch_before =
      iree_async_notification_query_epoch(notification);
  EXPECT_FALSE(iree_async_notification_signal_if_observed(notification, 1));
  EXPECT_NE(iree_async_notification_query_epoch(notification), epoch_before);

  iree_async_notification_release(notification);
}

TEST_P(NotificationTest, SignalIfObservedExplicitObservationAdvancesEpoch) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  const uint32_t wait_token =
      iree_async_notification_begin_observe(notification);
  EXPECT_TRUE(iree_async_notification_signal_if_observed(notification, 1));
  EXPECT_TRUE(iree_async_notification_wait_for_token(notification, wait_token,
                                                     iree_make_timeout_ms(0)));
  iree_async_notification_end_observe(notification);

  iree_async_notification_release(notification);
}

// Synchronous wait with signal from another thread.
//
// The waiter publishes readiness only after beginning an observation scope, so
// the signal cannot race ahead of its epoch token.
TEST_P(NotificationTest, SyncWaitCrossThread) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<bool> wait_completed{false};
  std::promise<void> waiter_ready_promise;
  auto waiter_ready = waiter_ready_promise.get_future();

  std::thread waiter([&]() {
    uint32_t wait_token = iree_async_notification_begin_observe(notification);
    waiter_ready_promise.set_value();
    bool result = iree_async_notification_wait_for_token(
        notification, wait_token, iree_infinite_timeout());
    iree_async_notification_end_observe(notification);
    wait_completed.store(result, std::memory_order_release);
  });

  waiter_ready.wait();
  iree_async_notification_signal(notification, 1);

  waiter.join();

  EXPECT_TRUE(wait_completed.load(std::memory_order_acquire));

  iree_async_notification_release(notification);
}

// Synchronous wait timeout.
TEST_P(NotificationTest, SyncWaitTimeout) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  // Wait with short timeout - should return false.
  iree_time_t start = iree_time_now();
  bool result =
      iree_async_notification_wait(notification, iree_make_timeout_ms(50));
  iree_time_t elapsed = iree_time_now() - start;

  EXPECT_FALSE(result);
  // Should have waited at least 10ms (generous slack for system scheduling).
  EXPECT_GE(elapsed, iree_make_duration_ms(10));

  iree_async_notification_release(notification);
}

// Multiple signals coalesce - waiter wakes exactly once regardless of how
// many signals arrive during a single wait.
TEST_P(NotificationTest, MultipleSignalsWhileWaiting) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<bool> wait_completed{false};
  std::promise<void> waiter_ready_promise;
  auto waiter_ready = waiter_ready_promise.get_future();

  std::thread waiter([&]() {
    uint32_t wait_token = iree_async_notification_begin_observe(notification);
    waiter_ready_promise.set_value();
    bool result = iree_async_notification_wait_for_token(
        notification, wait_token, iree_infinite_timeout());
    iree_async_notification_end_observe(notification);
    wait_completed.store(result, std::memory_order_release);
  });

  waiter_ready.wait();
  iree_async_notification_signal(notification, 1);
  iree_async_notification_signal(notification, 1);
  iree_async_notification_signal(notification, 1);

  waiter.join();
  EXPECT_TRUE(wait_completed.load(std::memory_order_acquire));

  // The epoch advanced once per signal, but one wait returned only once.
  EXPECT_EQ(iree_async_notification_query_epoch(notification), 3u);

  iree_async_notification_release(notification);
}

// Repeated wait/signal cycles work correctly.
//
// Each cycle publishes readiness after capturing its wait token, allowing one
// signal to complete exactly one wait without scheduler-dependent retries.
TEST_P(NotificationTest, RepeatedWaitSignalCycles) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<int> cycles_completed{0};
  constexpr int kCycles = 3;
  std::promise<void> cycle_ready_promises[kCycles];
  std::future<void> cycle_ready_futures[kCycles];
  for (int i = 0; i < kCycles; ++i) {
    cycle_ready_futures[i] = cycle_ready_promises[i].get_future();
  }

  // Worker thread waits for signals in a loop.
  std::thread worker([&]() {
    for (int i = 0; i < kCycles; ++i) {
      uint32_t wait_token = iree_async_notification_begin_observe(notification);
      cycle_ready_promises[i].set_value();
      bool result = iree_async_notification_wait_for_token(
          notification, wait_token, iree_infinite_timeout());
      iree_async_notification_end_observe(notification);
      if (result) {
        cycles_completed.fetch_add(1, std::memory_order_acq_rel);
      }
    }
  });

  for (int i = 0; i < kCycles; ++i) {
    cycle_ready_futures[i].wait();
    iree_async_notification_signal(notification, 1);
  }

  worker.join();

  EXPECT_EQ(cycles_completed.load(std::memory_order_acquire), kCycles);

  iree_async_notification_release(notification);
}

// Multiple waiters all wake when signaled.
TEST_P(NotificationTest, SingleWakeMakesProgressWithMultipleWaiters) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<int> waiters_woken{0};
  constexpr int kNumWaiters = 3;
  std::promise<void> waiter_ready_promises[kNumWaiters];
  std::future<void> waiter_ready_futures[kNumWaiters];
  std::promise<void> first_waiter_woken_promise;
  auto first_waiter_woken = first_waiter_woken_promise.get_future();

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiter_ready_futures[i] = waiter_ready_promises[i].get_future();
    waiters.emplace_back([&, i]() {
      uint32_t wait_token = iree_async_notification_begin_observe(notification);
      waiter_ready_promises[i].set_value();
      bool result = iree_async_notification_wait_for_token(
          notification, wait_token, iree_infinite_timeout());
      iree_async_notification_end_observe(notification);
      if (result) {
        int old_count = waiters_woken.fetch_add(1, std::memory_order_acq_rel);
        if (old_count == 0) first_waiter_woken_promise.set_value();
      }
    });
  }

  for (auto& waiter_ready : waiter_ready_futures) waiter_ready.wait();

  // A single platform wake guarantees progress. More than one observer may
  // complete if it captured the old epoch but had not entered its kernel wait
  // before the signal advanced the epoch.
  iree_async_notification_signal(notification, 1);
  first_waiter_woken.wait();
  EXPECT_GE(waiters_woken.load(std::memory_order_acquire), 1);

  // Release any observers that did reach the platform wait.
  iree_async_notification_signal(notification, INT32_MAX);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(waiters_woken.load(std::memory_order_acquire), kNumWaiters);

  iree_async_notification_release(notification);
}

// Broadcast wake (INT32_MAX) wakes all waiters.
TEST_P(NotificationTest, BroadcastWake) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<int> waiters_woken{0};
  constexpr int kNumWaiters = 3;
  std::promise<void> waiter_ready_promises[kNumWaiters];
  std::future<void> waiter_ready_futures[kNumWaiters];

  std::vector<std::thread> waiters;
  for (int i = 0; i < kNumWaiters; ++i) {
    waiter_ready_futures[i] = waiter_ready_promises[i].get_future();
    waiters.emplace_back([&, i]() {
      uint32_t wait_token = iree_async_notification_begin_observe(notification);
      waiter_ready_promises[i].set_value();
      bool result = iree_async_notification_wait_for_token(
          notification, wait_token, iree_infinite_timeout());
      iree_async_notification_end_observe(notification);
      if (result) {
        waiters_woken.fetch_add(1, std::memory_order_acq_rel);
      }
    });
  }

  for (auto& waiter_ready : waiter_ready_futures) waiter_ready.wait();
  iree_async_notification_signal(notification, INT32_MAX);

  for (auto& t : waiters) {
    t.join();
  }

  EXPECT_EQ(waiters_woken.load(std::memory_order_acquire), kNumWaiters);

  iree_async_notification_release(notification);
}

// Async NOTIFICATION_WAIT operation via proactor.
TEST_P(NotificationTest, AsyncWait) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  CompletionTracker tracker;
  iree_async_notification_wait_operation_t wait_op;
  InitNotificationWaitOp(&wait_op, notification, CompletionTracker::Callback,
                         &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Complete the backend registration before signaling from another thread.
  iree_async_proactor_wake(proactor_);
  PollOneProgressEvent();
  EXPECT_EQ(tracker.call_count, 0);

  // Signal the notification from another thread.
  std::thread signaler(
      [notification]() { iree_async_notification_signal(notification, 1); });

  // Poll until the wait completes.
  PollUntil(/*min_completions=*/1);

  signaler.join();

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_notification_release(notification);
}

TEST_P(NotificationTest, SignalIfObservedSubmittedAsyncWaitCompletes) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  CompletionTracker tracker;
  iree_async_notification_wait_operation_t wait_op;
  InitNotificationWaitOp(&wait_op, notification, CompletionTracker::Callback,
                         &tracker);
  wait_op.wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
  wait_op.wait_token = iree_async_notification_begin_observe(notification);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));
  iree_async_notification_end_observe(notification);
  EXPECT_TRUE(iree_async_notification_signal_if_observed(notification, 1));

  while (tracker.call_count < 1) {
    iree_host_size_t completed = 0;
    IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                            &completed));
  }

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_notification_release(notification);
}

// Async NOTIFICATION_WAIT with a caller-provided wait token. This validates the
// observe-check-wait protocol: a caller can observe the notification epoch,
// check a protected condition, and then arm a wait that still completes if a
// signal races before submission.
TEST_P(NotificationTest, AsyncWaitWithCallerWaitToken) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  const uint32_t wait_token = iree_async_notification_query_epoch(notification);
  iree_async_notification_signal(notification, 1);

  CompletionTracker tracker;
  iree_async_notification_wait_operation_t wait_op;
  InitNotificationWaitOp(&wait_op, notification, CompletionTracker::Callback,
                         &tracker);
  wait_op.wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
  wait_op.wait_token = wait_token;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  PollUntil(/*min_completions=*/1);

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  iree_async_notification_release(notification);
}

// Async NOTIFICATION_SIGNAL operation via proactor.
TEST_P(NotificationTest, AsyncSignal) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  std::atomic<bool> waiter_completed{false};
  std::promise<void> waiter_ready_promise;
  auto waiter_ready = waiter_ready_promise.get_future();

  // Background thread waits synchronously.
  std::thread waiter([&]() {
    uint32_t wait_token = iree_async_notification_begin_observe(notification);
    waiter_ready_promise.set_value();
    bool result = iree_async_notification_wait_for_token(
        notification, wait_token, iree_infinite_timeout());
    iree_async_notification_end_observe(notification);
    waiter_completed.store(result, std::memory_order_release);
  });

  waiter_ready.wait();
  CompletionTracker tracker;
  iree_async_notification_signal_operation_t signal_op;
  InitNotificationSignalOp(&signal_op, notification, 1,
                           CompletionTracker::Callback, &tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &signal_op.base));

  PollUntil(/*min_completions=*/1);

  IREE_EXPECT_OK(tracker.ConsumeStatus());

  waiter.join();

  EXPECT_TRUE(waiter_completed.load(std::memory_order_acquire));

  iree_async_notification_release(notification);
}

// Chain: NOTIFICATION_WAIT -> NOP, verify order.
TEST_P(NotificationTest, ChainedWaitNop) {
  if (!iree_any_bit_set(capabilities_,
                        IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS)) {
    GTEST_SKIP() << "backend lacks linked operations capability";
  }

  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  struct OrderTracker {
    std::vector<int> order;
    static void WaitCallback(void* u, iree_async_operation_t* o,
                             iree_status_t s, iree_async_completion_flags_t f) {
      static_cast<OrderTracker*>(u)->order.push_back(0);
      iree_status_ignore(s);
    }
    static void NopCallback(void* u, iree_async_operation_t* o, iree_status_t s,
                            iree_async_completion_flags_t f) {
      static_cast<OrderTracker*>(u)->order.push_back(1);
      iree_status_ignore(s);
    }
  };

  OrderTracker tracker;

  iree_async_notification_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT;
  wait_op.base.completion_fn = OrderTracker::WaitCallback;
  wait_op.base.user_data = &tracker;
  wait_op.base.flags = IREE_ASYNC_OPERATION_FLAG_LINKED;
  wait_op.notification = notification;

  iree_async_nop_operation_t nop;
  memset(&nop, 0, sizeof(nop));
  nop.base.type = IREE_ASYNC_OPERATION_TYPE_NOP;
  nop.base.completion_fn = OrderTracker::NopCallback;
  nop.base.user_data = &tracker;

  iree_async_operation_t* ops[] = {&wait_op.base, &nop.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  // Complete the backend wait registration before signaling.
  iree_async_proactor_wake(proactor_);
  PollOneProgressEvent();
  EXPECT_TRUE(tracker.order.empty());

  // Signal the notification from another thread.
  std::thread signaler(
      [notification]() { iree_async_notification_signal(notification, 1); });

  // Poll until both complete.
  PollUntil(/*min_completions=*/2);

  signaler.join();

  ASSERT_EQ(tracker.order.size(), 2u);
  EXPECT_EQ(tracker.order[0], 0);  // Wait first.
  EXPECT_EQ(tracker.order[1], 1);  // NOP second.

  iree_async_notification_release(notification);
}

// Round-trip: a cross-thread signal completes an async wait callback.
TEST_P(NotificationTest, RoundTrip) {
  iree_async_notification_t* async_notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &async_notification));

  struct RoundTripContext {
    bool callback_fired = false;
  };
  RoundTripContext context = {false};

  iree_async_notification_wait_operation_t wait_op;
  memset(&wait_op, 0, sizeof(wait_op));
  wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT;
  wait_op.base.completion_fn = [](void* user_data, iree_async_operation_t* op,
                                  iree_status_t status,
                                  iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<RoundTripContext*>(user_data);
    ctx->callback_fired = true;
    iree_status_ignore(status);
  };
  wait_op.base.user_data = &context;
  wait_op.notification = async_notification;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &wait_op.base));

  // Complete the backend wait registration before releasing the signaler.
  iree_async_proactor_wake(proactor_);
  PollOneProgressEvent();
  EXPECT_FALSE(context.callback_fired);

  // Background thread signals the async notification.
  std::thread signaler(
      [&]() { iree_async_notification_signal(async_notification, 1); });

  PollUntilCondition([&] { return context.callback_fired; },
                     "notification round trip");

  signaler.join();

  EXPECT_TRUE(context.callback_fired);

  iree_async_notification_release(async_notification);
}

CTS_REGISTER_TEST_SUITE(NotificationTest);

}  // namespace iree::async::cts
