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

static hipError_t TestPublicQuery(int* output) {
  HIP_API_BEGIN();
  *output = 42;
  HIP_RETURN_ERROR(hipSuccess);
}

static int TestPublicValueQuery(int* output) {
  HIP_API_BEGIN_OR_RETURN(-1);
  *output = 42;
  return 42;
}

TEST_F(HipErrorStateTest, NonFatalErrorsRemainThreadLocalAndClearable) {
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_error_state_publish(hipErrorInvalidValue));
  EXPECT_EQ(hipErrorInvalidValue, iree_hip_error_state_peek_last_error());

  std::atomic<hipError_t> other_thread_error{hipErrorUnknown};
  std::thread other_thread([&] {
    other_thread_error.store(iree_hip_error_state_peek_last_error(),
                             std::memory_order_release);
  });
  other_thread.join();

  EXPECT_EQ(hipSuccess, other_thread_error.load(std::memory_order_acquire));
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_error_state_get_and_clear_last_error());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_peek_last_error());
}

TEST_F(HipErrorStateTest, CommandAndOrdinaryResultsHaveDistinctSemantics) {
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_error_state_publish(hipErrorInvalidValue));
  EXPECT_EQ(hipSuccess, iree_hip_error_state_publish(hipSuccess));

  EXPECT_EQ(hipSuccess, iree_hip_error_state_get_and_clear_command_error());
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_error_state_get_and_clear_last_error());

  EXPECT_EQ(hipErrorNotReady, iree_hip_error_state_publish(hipErrorNotReady));
  EXPECT_EQ(hipErrorNotReady,
            iree_hip_error_state_get_and_clear_command_error());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_peek_last_error());
}

TEST_F(HipErrorStateTest, IllegalAddressIsProcessSharedAndSticky) {
  std::thread observer([] {
    EXPECT_EQ(hipErrorIllegalAddress,
              iree_hip_error_state_publish(hipErrorIllegalAddress));
  });
  observer.join();

  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_fatal_result());
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_peek_last_error());
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_publish(hipSuccess));
  EXPECT_EQ(hipErrorIllegalAddress,
            iree_hip_error_state_get_and_clear_last_error());
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_peek_last_error());
}

TEST_F(HipErrorStateTest, FatalEntryBoundaryPrecedesApiSideEffects) {
  EXPECT_EQ(hipErrorIllegalAddress,
            iree_hip_error_state_publish(hipErrorIllegalAddress));

  int output = 7;
  EXPECT_EQ(hipErrorIllegalAddress, TestPublicQuery(&output));
  EXPECT_EQ(7, output);
  EXPECT_EQ(-1, TestPublicValueQuery(&output));
  EXPECT_EQ(7, output);
  EXPECT_EQ(hipErrorIllegalAddress,
            iree_hip_error_state_get_and_clear_command_error());
}

TEST_F(HipErrorStateTest, FatalEntryDoesNotRelatchAcrossReset) {
  EXPECT_EQ(hipErrorIllegalAddress,
            iree_hip_error_state_publish(hipErrorIllegalAddress));
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_begin());

  iree_hip_error_state_reset();
  EXPECT_EQ(hipSuccess, iree_hip_error_state_fatal_result());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_begin());
}

TEST_F(HipErrorStateTest, ResetInvalidatesOtherThreadsFatalState) {
  std::atomic<int> phase{0};
  std::atomic<hipError_t> after_reset{hipErrorUnknown};
  std::thread observer([&] {
    EXPECT_EQ(hipErrorIllegalAddress,
              iree_hip_error_state_publish(hipErrorIllegalAddress));
    phase.store(1, std::memory_order_release);
    while (phase.load(std::memory_order_acquire) != 2) {
      std::this_thread::yield();
    }
    after_reset.store(iree_hip_error_state_peek_last_error(),
                      std::memory_order_release);
  });

  while (phase.load(std::memory_order_acquire) != 1) {
    std::this_thread::yield();
  }
  iree_hip_error_state_reset();
  phase.store(2, std::memory_order_release);
  observer.join();

  EXPECT_EQ(hipSuccess, after_reset.load(std::memory_order_acquire));
}

}  // namespace
