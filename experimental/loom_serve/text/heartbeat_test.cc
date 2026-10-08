// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/heartbeat.h"

#include <atomic>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TextHeartbeatTest : public ::testing::Test {
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
    loom_serve_text_heartbeat_deinitialize(&heartbeat_);
    iree_async_proactor_pool_release(pool_);
    iree_notification_deinitialize(&notification_);
  }

  static void Wake(void* user_data) {
    auto* self = static_cast<TextHeartbeatTest*>(user_data);
    self->wake_count_.fetch_add(1, std::memory_order_release);
    iree_notification_post(&self->notification_, IREE_ALL_WAITERS);
  }

  void Initialize() {
    loom_serve_text_heartbeat_initialize(&heartbeat_);
    heartbeat_.observer = {Wake, this};
  }

  void Configure(iree_duration_t interval) {
    loom_serve_text_heartbeat_configure(&heartbeat_, proactor_,
                                        IREE_SV("\"test\""), interval);
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
      status = iree_status_join(status, loom_serve_text_heartbeat_advance(
                                            &heartbeat_, true, &progress));
      if (!loom_serve_text_heartbeat_is_pending(&heartbeat_)) {
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

  // Production reporter with real intrusive proactor operations.
  loom_serve_text_heartbeat_t heartbeat_ = {};
  // Shared I/O pool, released only after reporter callback retirement.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Borrowed proactor driven by the pool's normal poll owner.
  iree_async_proactor_t* proactor_ = nullptr;
  // Outer test-owner wake, never waited on by a task worker or callback.
  iree_notification_t notification_;
  // Callback count for explicit readiness, independent of wall-clock sleeps.
  std::atomic<uint32_t> wake_count_{0};
};

TEST_F(TextHeartbeatTest, DisabledReportingAcceptsNoWork) {
  Configure(IREE_DURATION_ZERO);
  bool progress = false;
  IREE_ASSERT_OK(
      loom_serve_text_heartbeat_advance(&heartbeat_, false, &progress));
  EXPECT_FALSE(progress);
  EXPECT_FALSE(loom_serve_text_heartbeat_is_pending(&heartbeat_));
  EXPECT_EQ(wake_count_.load(), 0u);
}

TEST_F(TextHeartbeatTest, TimerReadinessPublishesCopiedCounters) {
  Configure(iree_make_duration_ms(1));
  heartbeat_.snapshot.prefill_tokens = 42;
  const uint32_t previous_count = wake_count_.load(std::memory_order_acquire);
  bool progress = false;
  IREE_ASSERT_OK(
      loom_serve_text_heartbeat_advance(&heartbeat_, false, &progress));
  EXPECT_TRUE(progress);
  EXPECT_EQ(heartbeat_.previous.prefill, 42u);
  AwaitWake(previous_count);
  iree_slim_mutex_lock(&heartbeat_.mutex);
  heartbeat_.snapshot.prefill_tokens = 49;
  iree_slim_mutex_unlock(&heartbeat_.mutex);
  IREE_ASSERT_OK(
      loom_serve_text_heartbeat_advance(&heartbeat_, false, &progress));
  EXPECT_TRUE(progress);
  EXPECT_EQ(heartbeat_.previous.prefill, 49u);
}

TEST_F(TextHeartbeatTest, FarFutureTimerStopsWithoutWaitingForItsDeadline) {
  Configure(iree_make_duration_ms(3600000));
  bool progress = false;
  IREE_ASSERT_OK(
      loom_serve_text_heartbeat_advance(&heartbeat_, false, &progress));
  EXPECT_TRUE(loom_serve_text_heartbeat_is_pending(&heartbeat_));
  IREE_ASSERT_OK(Stop());
  EXPECT_FALSE(loom_serve_text_heartbeat_is_pending(&heartbeat_));
}

TEST_F(TextHeartbeatTest, ExpiredTimerStopsWithoutRearming) {
  Configure(iree_make_duration_ms(1));
  const uint32_t previous_count = wake_count_.load(std::memory_order_acquire);
  bool progress = false;
  IREE_ASSERT_OK(
      loom_serve_text_heartbeat_advance(&heartbeat_, false, &progress));
  AwaitWake(previous_count);
  IREE_ASSERT_OK(Stop());
  EXPECT_FALSE(loom_serve_text_heartbeat_is_pending(&heartbeat_));
}

TEST_F(TextHeartbeatTest, CancellationReceiptPermitsAddressReuse) {
  for (int iteration = 0; iteration < 32; ++iteration) {
    SCOPED_TRACE(iteration);
    Configure(iteration % 2 ? 1 : iree_make_duration_ms(3600000));
    bool progress = false;
    IREE_ASSERT_OK(
        loom_serve_text_heartbeat_advance(&heartbeat_, false, &progress));
    IREE_ASSERT_OK(Stop());
    loom_serve_text_heartbeat_deinitialize(&heartbeat_);
    Initialize();
  }
}

}  // namespace
