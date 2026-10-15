// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/event.h"

#include <semaphore>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::async {
namespace {

class NativeEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_event_native_initialize(&event_));
  }
  void TearDown() override { iree_async_event_native_deinitialize(&event_); }

  // Owned event, joined by each test before fixture teardown.
  iree_async_event_native_t event_ = {};
};

TEST_F(NativeEventTest, SignalBeforeWaitIsConsumedOnce) {
  iree_async_event_native_set(&event_);
  IREE_ASSERT_OK(
      iree_async_event_native_wait(&event_, iree_immediate_timeout()));
  EXPECT_EQ(
      Status(iree_async_event_native_wait(&event_, iree_immediate_timeout()))
          .code(),
      StatusCode::kDeadlineExceeded);
}

TEST_F(NativeEventTest, UnsignaledFiniteWaitExpires) {
  EXPECT_EQ(
      Status(iree_async_event_native_wait(&event_, iree_make_timeout_ms(1)))
          .code(),
      StatusCode::kDeadlineExceeded);
}

TEST_F(NativeEventTest, CrossThreadSignalsSurviveRepeatedRearm) {
  std::binary_semaphore consumed{0};
  std::thread producer([&] {
    for (int i = 0; i < 128; ++i) {
      iree_async_event_native_set(&event_);
      consumed.acquire();
    }
  });
  for (int i = 0; i < 128; ++i) {
    IREE_EXPECT_OK(
        iree_async_event_native_wait(&event_, iree_infinite_timeout()));
    consumed.release();
  }
  producer.join();
}

TEST_F(NativeEventTest, CoalescedSignalsStillWake) {
  iree_async_event_native_set(&event_);
  iree_async_event_native_set(&event_);
  IREE_ASSERT_OK(
      iree_async_event_native_wait(&event_, iree_infinite_timeout()));
  EXPECT_EQ(
      Status(iree_async_event_native_wait(&event_, iree_immediate_timeout()))
          .code(),
      StatusCode::kDeadlineExceeded);
}

}  // namespace
}  // namespace iree::async
