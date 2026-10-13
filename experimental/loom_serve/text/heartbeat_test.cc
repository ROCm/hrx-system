// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/heartbeat.h"

#include <atomic>
#include <string>

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
    IREE_ASSERT_OK(loom_serve_telemetry_stream_create(8, &stream_,
                                                      iree_allocator_system()));
  }

  void TearDown() override {
    IREE_EXPECT_OK(Stop());
    loom_serve_text_heartbeat_deinitialize(&heartbeat_);
    loom_serve_telemetry_stream_destroy(stream_);
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
    loom_serve_timer_set_observer(&heartbeat_.timer, {Wake, this});
  }

  void Configure(iree_duration_t interval) {
    loom_serve_text_heartbeat_configure(&heartbeat_, proactor_, stream_,
                                        &counters_, interval);
  }

  std::string ReadRecords(uint64_t* cursor) {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    loom_serve_telemetry_record_t record;
    while (loom_serve_telemetry_stream_read(stream_, cursor, &record) ==
           LOOM_SERVE_TELEMETRY_READ_RECORD) {
      IREE_EXPECT_OK(
          loom_serve_telemetry_record_format(*cursor - 1, &record, &output));
    }
    const auto view = iree_string_builder_view(&output);
    const std::string result(view.data, view.size);
    iree_string_builder_deinitialize(&output);
    return result;
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
  // Bounded production transport; its readers never hold timer readiness.
  loom_serve_telemetry_stream_t* stream_ = nullptr;
  // Serialized model-owner counters borrowed by heartbeat publication.
  loom_serve_telemetry_counters_t counters_ = {};
};

TEST_F(TextHeartbeatTest, TimerReadinessPublishesCopiedCounters) {
  Configure(iree_make_duration_ms(1));
  heartbeat_.snapshot.prefill_tokens = 42;
  heartbeat_.snapshot.session_cache = {24, 17};
  counters_.input_tokens = 42;
  heartbeat_.snapshot.weight_activity = {2, 1, 64, 32, 4};
  heartbeat_.snapshot.device_memory.workspace.bytes_committed = 1234;
  const uint32_t previous_count = wake_count_.load(std::memory_order_acquire);
  bool progress = false;
  IREE_ASSERT_OK(
      loom_serve_text_heartbeat_advance(&heartbeat_, false, &progress));
  EXPECT_TRUE(progress);
  uint64_t cursor = 0;
  const std::string first = ReadRecords(&cursor);
  EXPECT_NE(first.find("\"prefill_tokens\":42"), std::string::npos);
  EXPECT_NE(first.find("\"input_tokens\":42"), std::string::npos);
  EXPECT_NE(first.find("\"session_cache\":{\"capacity\":24,\"endpoints\":17}"),
            std::string::npos);
  EXPECT_NE(
      first.find("\"activations\":2,\"deactivations\":1,\"read_bytes\":64"),
      std::string::npos);
  EXPECT_NE(first.find("\"workspace_committed_bytes\":1234"),
            std::string::npos);
  EXPECT_EQ(cursor, 6u);
  AwaitWake(previous_count);
  iree_slim_mutex_lock(&heartbeat_.mutex);
  heartbeat_.snapshot.prefill_tokens = 49;
  counters_.input_tokens = 49;
  iree_slim_mutex_unlock(&heartbeat_.mutex);
  IREE_ASSERT_OK(
      loom_serve_text_heartbeat_advance(&heartbeat_, false, &progress));
  EXPECT_TRUE(progress);
  const std::string second = ReadRecords(&cursor);
  EXPECT_NE(second.find("\"prefill_tokens\":49"), std::string::npos);
  EXPECT_EQ(cursor, 12u);
}

}  // namespace
