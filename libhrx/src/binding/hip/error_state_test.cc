// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/error_state.h"

#include <atomic>
#include <thread>

#include "iree/testing/gtest.h"

namespace {

class HipErrorStateTest : public testing::Test {
 protected:
  void SetUp() override { iree_hip_error_state_reset(); }
  void TearDown() override { iree_hip_error_state_reset(); }
};

TEST_F(HipErrorStateTest, NonFatalErrorsRemainThreadLocalAndClearable) {
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_error_state_publish(hipErrorInvalidValue));
  EXPECT_EQ(hipErrorInvalidValue, iree_hip_error_state_peek());

  std::atomic<hipError_t> other_thread_error{hipErrorUnknown};
  std::thread other_thread([&] {
    other_thread_error.store(iree_hip_error_state_peek(),
                             std::memory_order_release);
  });
  other_thread.join();

  EXPECT_EQ(hipSuccess, other_thread_error.load(std::memory_order_acquire));
  EXPECT_EQ(hipErrorInvalidValue, iree_hip_error_state_get_and_clear());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_peek());
}

TEST_F(HipErrorStateTest, IllegalAddressIsProcessSharedAndSticky) {
  std::thread observer([] {
    EXPECT_EQ(hipErrorIllegalAddress,
              iree_hip_error_state_publish(hipErrorIllegalAddress));
  });
  observer.join();

  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_fatal_result());
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_peek());
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_publish(hipSuccess));
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_get_and_clear());
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_peek());
}

}  // namespace
