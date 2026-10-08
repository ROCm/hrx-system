// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/timer.h"

#include <atomic>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class HttpTimerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    Initialize();
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    IREE_ASSERT_OK(iree_async_proactor_pool_get(pool_, 0, &proactor_));
  }

  void TearDown() override {
    IREE_EXPECT_OK(Stop());
    loom_serve_http_timer_deinitialize(&timer_);
    iree_async_proactor_pool_release(pool_);
    iree_notification_deinitialize(&notification_);
  }

  static void Wake(void* user_data) {
    auto* self = static_cast<HttpTimerTest*>(user_data);
    self->wake_count_.fetch_add(1, std::memory_order_release);
    iree_notification_post(&self->notification_, IREE_ALL_WAITERS);
  }

  void Initialize() {
    loom_serve_http_timer_initialize(&timer_);
    loom_serve_http_timer_set_observer(&timer_, {Wake, this});
  }

  void Configure(iree_duration_t interval) {
    loom_serve_http_timer_configure(&timer_, proactor_, interval);
  }

  void AwaitWake(uint32_t previous_count) {
    for (;;) {
      const auto token = iree_notification_prepare_wait(&notification_);
      if (wake_count_.load(std::memory_order_acquire) != previous_count) {
        iree_notification_cancel_wait(&notification_);
        return;
      }
      iree_notification_commit_wait(&notification_, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }

  iree_status_t Stop() {
    iree_status_t status = iree_ok_status();
    for (;;) {
      const auto token = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      bool tick = false;
      status = iree_status_join(status, loom_serve_http_timer_advance(
                                            &timer_, true, &progress, &tick));
      EXPECT_FALSE(tick);
      if (!loom_serve_http_timer_is_pending(&timer_)) {
        iree_notification_cancel_wait(&notification_);
        return status;
      }
      if (progress) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, token, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }

  // Production timer with real intrusive proactor operations.
  loom_serve_http_timer_t timer_ = {};
  // Shared I/O pool, released only after reporter callback retirement.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Borrowed proactor driven by the pool's normal poll owner.
  iree_async_proactor_t* proactor_ = nullptr;
  // Outer test-owner wake, never waited on by a task worker or callback.
  iree_notification_t notification_;
  // Callback count for explicit readiness, independent of wall-clock sleeps.
  std::atomic<uint32_t> wake_count_{0};
};

TEST_F(HttpTimerTest, DisabledReportingAcceptsNoWork) {
  Configure(IREE_DURATION_ZERO);
  bool progress = false;
  bool tick = false;
  IREE_ASSERT_OK(
      loom_serve_http_timer_advance(&timer_, false, &progress, &tick));
  EXPECT_FALSE(progress);
  EXPECT_FALSE(tick);
  EXPECT_FALSE(loom_serve_http_timer_is_pending(&timer_));
  EXPECT_EQ(wake_count_.load(), 0u);
}

TEST_F(HttpTimerTest, DeadlineProducesPeriodicTicks) {
  Configure(iree_make_duration_ms(1));
  const uint32_t previous_count = wake_count_.load(std::memory_order_acquire);
  bool progress = false;
  bool tick = false;
  IREE_ASSERT_OK(
      loom_serve_http_timer_advance(&timer_, false, &progress, &tick));
  EXPECT_TRUE(progress);
  EXPECT_TRUE(tick);
  AwaitWake(previous_count);
  IREE_ASSERT_OK(
      loom_serve_http_timer_advance(&timer_, false, &progress, &tick));
  EXPECT_TRUE(progress);
  EXPECT_TRUE(tick);
}

TEST_F(HttpTimerTest, FarFutureTimerStopsWithoutWaitingForItsDeadline) {
  Configure(iree_make_duration_ms(3600000));
  bool progress = false;
  bool tick = false;
  IREE_ASSERT_OK(
      loom_serve_http_timer_advance(&timer_, false, &progress, &tick));
  EXPECT_TRUE(loom_serve_http_timer_is_pending(&timer_));
  IREE_ASSERT_OK(Stop());
  EXPECT_FALSE(loom_serve_http_timer_is_pending(&timer_));
}

TEST_F(HttpTimerTest, ExpiredTimerStopsWithoutRearming) {
  Configure(iree_make_duration_ms(1));
  const uint32_t previous_count = wake_count_.load(std::memory_order_acquire);
  bool progress = false;
  bool tick = false;
  IREE_ASSERT_OK(
      loom_serve_http_timer_advance(&timer_, false, &progress, &tick));
  AwaitWake(previous_count);
  IREE_ASSERT_OK(Stop());
  EXPECT_FALSE(loom_serve_http_timer_is_pending(&timer_));
}

TEST_F(HttpTimerTest, CancellationReceiptPermitsAddressReuse) {
  for (int iteration = 0; iteration < 32; ++iteration) {
    SCOPED_TRACE(iteration);
    Configure(iteration % 2 ? 1 : iree_make_duration_ms(3600000));
    bool progress = false;
    bool tick = false;
    IREE_ASSERT_OK(
        loom_serve_http_timer_advance(&timer_, false, &progress, &tick));
    IREE_ASSERT_OK(Stop());
    loom_serve_http_timer_deinitialize(&timer_);
    Initialize();
  }
}

}  // namespace
