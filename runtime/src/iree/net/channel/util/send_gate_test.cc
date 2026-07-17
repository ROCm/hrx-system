// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/send_gate.h"

#include <atomic>
#include <thread>

#include "iree/testing/gtest.h"

namespace iree::net {
namespace {

static bool AtomicFlagSet(void* user_data) {
  return static_cast<std::atomic<bool>*>(user_data)->load(
      std::memory_order_acquire);
}

class SendGateTest : public ::testing::Test {
 protected:
  void SetUp() override { iree_net_channel_send_gate_initialize(&gate_); }

  void TearDown() override { iree_net_channel_send_gate_deinitialize(&gate_); }

  iree_net_channel_send_gate_t gate_;
};

TEST_F(SendGateTest, OpenGateTracksAdmissions) {
  EXPECT_EQ(iree_net_channel_send_gate_pending_count(&gate_), 0u);
  ASSERT_TRUE(iree_net_channel_send_gate_try_enter(&gate_));
  ASSERT_TRUE(iree_net_channel_send_gate_try_enter(&gate_));
  EXPECT_EQ(iree_net_channel_send_gate_pending_count(&gate_), 2u);

  iree_net_channel_send_gate_leave(&gate_);
  iree_net_channel_send_gate_leave(&gate_);
  EXPECT_EQ(iree_net_channel_send_gate_pending_count(&gate_), 0u);
}

TEST_F(SendGateTest, CloseWaitsForExactQuiescence) {
  ASSERT_TRUE(iree_net_channel_send_gate_try_enter(&gate_));

  iree_notification_t close_started_notification;
  iree_notification_initialize(&close_started_notification);
  std::atomic<bool> close_started{false};
  std::atomic<bool> close_owned{false};
  std::thread close_thread([&]() {
    close_owned.store(iree_net_channel_send_gate_begin_close(&gate_),
                      std::memory_order_release);
    close_started.store(true, std::memory_order_release);
    iree_notification_post(&close_started_notification, IREE_ALL_WAITERS);
    iree_net_channel_send_gate_await_quiescence(&gate_);
    iree_net_channel_send_gate_finish_close(&gate_);
  });

  ASSERT_TRUE(iree_notification_await(&close_started_notification,
                                      AtomicFlagSet, &close_started,
                                      iree_infinite_timeout()));
  EXPECT_TRUE(close_owned.load(std::memory_order_acquire));
  EXPECT_FALSE(iree_net_channel_send_gate_try_enter(&gate_));

  iree_net_channel_send_gate_leave(&gate_);
  close_thread.join();
  EXPECT_FALSE(iree_net_channel_send_gate_try_enter(&gate_));
  iree_notification_deinitialize(&close_started_notification);
}

TEST_F(SendGateTest, ConcurrentCloseJoinsOwner) {
  ASSERT_TRUE(iree_net_channel_send_gate_begin_close(&gate_));

  std::atomic<bool> follower_returned{false};
  std::thread follower([&]() {
    EXPECT_FALSE(iree_net_channel_send_gate_begin_close(&gate_));
    iree_net_channel_send_gate_await_closed(&gate_);
    follower_returned.store(true, std::memory_order_release);
  });

  iree_net_channel_send_gate_await_quiescence(&gate_);
  iree_net_channel_send_gate_finish_close(&gate_);
  follower.join();
  EXPECT_TRUE(follower_returned.load(std::memory_order_acquire));
}

}  // namespace
}  // namespace iree::net
