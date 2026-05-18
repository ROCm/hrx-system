// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/bulk_session.h"

#include <memory>

#include "iree/hal/remote/server/bulk_test_util.h"
#include "iree/hal/remote/server/server.h"
#include "iree/hal/remote/server/session.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::hal::remote::server::testing::MockCarrier;
using iree::hal::remote::server::testing::MockEndpoint;
using iree::hal::remote::server::testing::TestBufferPool;

class BulkSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_atomic_ref_count_init(&server_.ref_count);
    server_.host_allocator = iree_allocator_system();
    iree_slim_mutex_initialize(&server_.session_mutex);
    session_.server = &server_;
    session_.session_id = 1;
    session_.session = reinterpret_cast<iree_net_session_t*>(this);
    IREE_ASSERT_OK(iree_hal_remote_server_bulk_session_create(
        &session_, /*options=*/nullptr, iree_allocator_system(),
        &session_.bulk_session));
  }

  void TearDown() override {
    iree_hal_remote_server_bulk_session_free(session_.bulk_session);
    session_.bulk_session = nullptr;
    iree_slim_mutex_deinitialize(&server_.session_mutex);
  }

  iree_status_t CreateBulkChannel(iree_net_bulk_channel_t** out_channel) {
    *out_channel = nullptr;
    carrier_ = MockCarrier::Create();
    endpoint_.carrier = carrier_.get();
    TestBufferPool buffer_pool;
    iree_status_t status =
        buffer_pool.Initialize(/*buffer_count=*/16, /*buffer_size=*/1024);
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_channel_create(
          endpoint_.as_endpoint(), /*options=*/nullptr, buffer_pool.release(),
          iree_hal_remote_server_bulk_session_channel_callbacks(&session_),
          iree_allocator_system(), out_channel);
    }
    if (iree_status_is_ok(status)) {
      status = iree_net_bulk_channel_activate(*out_channel);
    }
    return status;
  }

  static void CountStopped(void* user_data) {
    int* stopped_count = static_cast<int*>(user_data);
    ++*stopped_count;
  }

  iree_hal_remote_server_t server_ = {};
  iree_hal_remote_server_session_t session_ = {};
  std::unique_ptr<MockCarrier> carrier_;
  MockEndpoint endpoint_;
};

TEST_F(BulkSessionTest, AttachTakeOwnsChannelReference) {
  iree_net_bulk_channel_t* bulk_channel = nullptr;
  IREE_ASSERT_OK(CreateBulkChannel(&bulk_channel));

  IREE_ASSERT_OK(iree_hal_remote_server_bulk_session_attach_channel(
      &session_, bulk_channel));
  EXPECT_EQ(iree_hal_remote_server_bulk_session_channel(&session_),
            bulk_channel);

  iree_net_bulk_channel_t* taken_channel =
      iree_hal_remote_server_bulk_session_take_channel(&session_);
  EXPECT_EQ(taken_channel, bulk_channel);
  EXPECT_EQ(iree_hal_remote_server_bulk_session_channel(&session_), nullptr);

  iree_net_bulk_channel_release(taken_channel);
  iree_net_bulk_channel_release(bulk_channel);
}

TEST_F(BulkSessionTest, AttachRejectsInactiveSession) {
  iree_net_bulk_channel_t* bulk_channel = nullptr;
  IREE_ASSERT_OK(CreateBulkChannel(&bulk_channel));

  session_.session = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_remote_server_bulk_session_attach_channel(
                            &session_, bulk_channel));
  EXPECT_EQ(iree_hal_remote_server_bulk_session_channel(&session_), nullptr);

  iree_net_bulk_channel_release(bulk_channel);
}

TEST_F(BulkSessionTest, DrainCompletionReleasesSlotState) {
  int stopped_count = 0;
  server_.state = IREE_HAL_REMOTE_SERVER_STATE_STOPPING;
  server_.active_session_count = 1;
  server_.stopped_callback = {CountStopped, &stopped_count};
  session_.session = nullptr;
  session_.flags |= IREE_HAL_REMOTE_SERVER_SESSION_FLAG_BULK_DRAIN_PENDING;

  iree_hal_remote_server_bulk_session_deinitialize_transfers(&session_);

  EXPECT_EQ(server_.active_session_count, 0u);
  EXPECT_EQ(stopped_count, 1);
  EXPECT_EQ(session_.flags, 0u);
  EXPECT_EQ(session_.bulk_session, nullptr);
  EXPECT_TRUE(
      iree_hal_remote_server_bulk_session_is_empty(session_.bulk_session));
}

}  // namespace
