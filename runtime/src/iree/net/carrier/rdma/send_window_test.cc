// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/send_window.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(RdmaSendWindowTest, TracksLocalSendSlots) {
  iree_net_rdma_send_window_t window;
  IREE_ASSERT_OK(iree_net_rdma_send_window_initialize(
      /*send_queue_depth=*/2, /*initial_remote_recv_credits=*/0, &window));

  EXPECT_EQ(2u, iree_net_rdma_send_window_available(
                    &window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE));

  IREE_ASSERT_OK(iree_net_rdma_send_window_acquire(
      &window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE));
  EXPECT_EQ(1u, iree_net_rdma_send_window_available_send_slots(&window));

  IREE_ASSERT_OK(iree_net_rdma_send_window_acquire(
      &window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE));
  EXPECT_EQ(0u, iree_net_rdma_send_window_available_send_slots(&window));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_rdma_send_window_acquire(
          &window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE));

  IREE_ASSERT_OK(iree_net_rdma_send_window_complete(&window));
  EXPECT_EQ(1u, iree_net_rdma_send_window_available_send_slots(&window));
}

TEST(RdmaSendWindowTest, TracksRemoteReceiveCredits) {
  iree_net_rdma_send_window_t window;
  IREE_ASSERT_OK(iree_net_rdma_send_window_initialize(
      /*send_queue_depth=*/4, /*initial_remote_recv_credits=*/1, &window));

  EXPECT_EQ(1u, iree_net_rdma_send_window_available(
                    &window,
                    IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT));

  IREE_ASSERT_OK(iree_net_rdma_send_window_acquire(
      &window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT));
  EXPECT_EQ(0u, iree_net_rdma_send_window_available_recv_credits(&window));
  EXPECT_EQ(3u, iree_net_rdma_send_window_available(
                    &window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_rdma_send_window_acquire(
          &window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT));

  iree_net_rdma_send_window_refresh_remote_credits(
      &window, /*remote_recv_credit_limit=*/3);
  EXPECT_EQ(2u, iree_net_rdma_send_window_available_recv_credits(&window));
  EXPECT_EQ(2u, iree_net_rdma_send_window_available(
                    &window,
                    IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT));

  IREE_ASSERT_OK(iree_net_rdma_send_window_complete(&window));
  EXPECT_EQ(4u, iree_net_rdma_send_window_available_send_slots(&window));
  EXPECT_EQ(2u, iree_net_rdma_send_window_available_recv_credits(&window));
}

TEST(RdmaSendWindowTest, AbortRestoresAdmission) {
  iree_net_rdma_send_window_t window;
  IREE_ASSERT_OK(iree_net_rdma_send_window_initialize(
      /*send_queue_depth=*/2, /*initial_remote_recv_credits=*/2, &window));

  IREE_ASSERT_OK(iree_net_rdma_send_window_acquire(
      &window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT));
  EXPECT_EQ(1u, iree_net_rdma_send_window_available_send_slots(&window));
  EXPECT_EQ(1u, iree_net_rdma_send_window_available_recv_credits(&window));

  IREE_ASSERT_OK(iree_net_rdma_send_window_abort(
      &window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT));
  EXPECT_EQ(2u, iree_net_rdma_send_window_available_send_slots(&window));
  EXPECT_EQ(2u, iree_net_rdma_send_window_available_recv_credits(&window));
}

TEST(RdmaSendWindowTest, IgnoresStaleCreditRefresh) {
  iree_net_rdma_send_window_t window;
  IREE_ASSERT_OK(iree_net_rdma_send_window_initialize(
      /*send_queue_depth=*/8, /*initial_remote_recv_credits=*/4, &window));

  iree_net_rdma_send_window_refresh_remote_credits(
      &window, /*remote_recv_credit_limit=*/6);
  EXPECT_EQ(6u, iree_net_rdma_send_window_available_recv_credits(&window));

  iree_net_rdma_send_window_refresh_remote_credits(
      &window, /*remote_recv_credit_limit=*/3);
  EXPECT_EQ(6u, iree_net_rdma_send_window_available_recv_credits(&window));
}

TEST(RdmaSendWindowTest, HandlesCreditSequenceWrap) {
  iree_net_rdma_send_window_t window;
  IREE_ASSERT_OK(iree_net_rdma_send_window_initialize(
      /*send_queue_depth=*/8,
      /*initial_remote_recv_credits=*/UINT32_MAX - 1, &window));

  iree_net_rdma_send_window_refresh_remote_credits(
      &window, /*remote_recv_credit_limit=*/2);
  EXPECT_EQ(2u, window.remote_recv_credit_limit);
}

TEST(RdmaSendWindowTest, RejectsInvalidArguments) {
  iree_net_rdma_send_window_t window;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_window_initialize(
                            /*send_queue_depth=*/0,
                            /*initial_remote_recv_credits=*/0, &window));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_rdma_send_window_initialize(
                            /*send_queue_depth=*/1,
                            /*initial_remote_recv_credits=*/0, nullptr));

  IREE_ASSERT_OK(iree_net_rdma_send_window_initialize(
      /*send_queue_depth=*/1, /*initial_remote_recv_credits=*/0, &window));
  EXPECT_EQ(0u, iree_net_rdma_send_window_available(
                    &window, (iree_net_rdma_send_window_acquire_flags_t)0x80u));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_send_window_acquire(
          &window, (iree_net_rdma_send_window_acquire_flags_t)0x80u));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_send_window_abort(
          &window, (iree_net_rdma_send_window_acquire_flags_t)0x80u));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_send_window_complete(&window));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_rdma_send_window_abort(
          &window, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE));
}

TEST(RdmaSendWindowStandaloneTest, NullQueriesReturnZero) {
  EXPECT_EQ(0u, iree_net_rdma_send_window_available_send_slots(nullptr));
  EXPECT_EQ(0u, iree_net_rdma_send_window_available_recv_credits(nullptr));
  EXPECT_EQ(0u, iree_net_rdma_send_window_available(
                    nullptr, IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE));
}

}  // namespace
