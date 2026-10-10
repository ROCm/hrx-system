// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/signals.h"

#include <signal.h>
#include <unistd.h>

#include <atomic>

#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct SignalObserver {
  SignalObserver() { iree_notification_initialize(&notification); }
  ~SignalObserver() { iree_notification_deinitialize(&notification); }

  iree_async_signal_callback_t Callback() {
    return {[](void* self, iree_async_signal_t signal) {
              auto* observer = static_cast<SignalObserver*>(self);
              observer->last.store(signal);
              observer->count.fetch_add(1);
              iree_notification_post(&observer->notification, IREE_ALL_WAITERS);
            },
            this};
  }

  void Await() {
    iree_notification_await(
        &notification,
        [](void* self) {
          auto* observer = static_cast<SignalObserver*>(self);
          return observer->count.load() == observer->expected;
        },
        this, iree_infinite_timeout());
  }

  // Delivery count published on the shared poll owner.
  std::atomic<uint32_t> count{0};
  // Last delivered kind, observed only after the corresponding count.
  std::atomic<iree_async_signal_t> last{IREE_ASYNC_SIGNAL_INTERRUPT};
  // Count expected by this outer application wait.
  uint32_t expected = 0;
  // Advisory readiness for explicit signal delivery.
  iree_notification_t notification;
};

TEST(ServingSignalsTest, SubscriptionsRetireWithoutStoppingSharedProgress) {
  IREE_ASSERT_OK(iree_async_signal_block_default());
  iree_async_proactor_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_create(
      1, nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &pool));
  SignalObserver first;
  SignalObserver second;
  loom_serve_signals_t* first_signals = nullptr;
  loom_serve_signals_t* second_signals = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_signals_create(pool, {}, &first_signals,
                                                  iree_allocator_system()));
  EXPECT_EQ(first_signals, nullptr);
  IREE_ASSERT_OK(loom_serve_signals_create(
      pool, first.Callback(), &first_signals, iree_allocator_system()));
  IREE_ASSERT_OK(loom_serve_signals_create(
      pool, second.Callback(), &second_signals, iree_allocator_system()));
  first.expected = 1;
  second.expected = 1;
  ASSERT_EQ(kill(getpid(), SIGINT), 0);
  first.Await();
  second.Await();
  EXPECT_EQ(first.last.load(), IREE_ASYNC_SIGNAL_INTERRUPT);
  EXPECT_EQ(second.last.load(), IREE_ASYNC_SIGNAL_INTERRUPT);

  loom_serve_signals_destroy(first_signals);
  second.expected = 2;
  ASSERT_EQ(kill(getpid(), SIGTERM), 0);
  second.Await();
  EXPECT_EQ(first.count.load(), 1u);
  EXPECT_EQ(second.last.load(), IREE_ASYNC_SIGNAL_TERMINATE);
  loom_serve_signals_destroy(second_signals);
  iree_async_proactor_pool_release(pool);
}

}  // namespace
