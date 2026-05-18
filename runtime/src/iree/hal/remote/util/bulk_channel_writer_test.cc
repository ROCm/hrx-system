// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/util/bulk_channel_writer.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(BulkChannelWriterTest, OkStatusIsAccepted) {
  iree_status_t failure_status = iree_ok_status();
  EXPECT_EQ(iree_hal_remote_bulk_channel_send_result_from_status(
                iree_ok_status(), &failure_status),
            IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_ACCEPTED);
  IREE_EXPECT_OK(failure_status);
}

TEST(BulkChannelWriterTest, ResourceExhaustedStatusIsBlocked) {
  iree_status_t failure_status = iree_ok_status();
  EXPECT_EQ(iree_hal_remote_bulk_channel_send_result_from_status(
                iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                 "send contexts exhausted"),
                &failure_status),
            IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_BLOCKED);
  IREE_EXPECT_OK(failure_status);
}

TEST(BulkChannelWriterTest, OtherFailureStatusIsPreserved) {
  iree_status_t failure_status = iree_ok_status();
  EXPECT_EQ(
      iree_hal_remote_bulk_channel_send_result_from_status(
          iree_make_status(IREE_STATUS_UNAVAILABLE, "bulk channel detached"),
          &failure_status),
      IREE_HAL_REMOTE_BULK_CHANNEL_SEND_RESULT_FAILED);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, failure_status);
}

}  // namespace
