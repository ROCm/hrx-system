// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/error_state.h"

#include <atomic>
#include <thread>
#include <vector>

#include "binding/hip/error_state_test_util.h"
#include "iree/testing/gtest.h"

namespace {

struct CompareExchangeBarrier {
  std::atomic<int> call_count{0};
  std::atomic<bool> reached{false};
  std::atomic<bool> proceed{false};
};

static void WaitAtFirstCompareExchange(void* user_data) {
  auto* barrier = static_cast<CompareExchangeBarrier*>(user_data);
  if (barrier->call_count.fetch_add(1, std::memory_order_acq_rel) != 0) {
    return;
  }
  barrier->reached.store(true, std::memory_order_release);
  while (!barrier->proceed.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

class HipErrorStateTest : public testing::Test {
 protected:
  void SetUp() override {
    iree_hip_error_state_test_set_publish_hook(nullptr, nullptr);
    iree_hip_error_state_test_set_compare_exchange_hook(nullptr, nullptr);
    iree_hip_error_state_test_reset_compare_exchange_count();
    iree_hip_error_state_reset();
  }

  void TearDown() override {
    iree_hip_error_state_test_set_publish_hook(nullptr, nullptr);
    iree_hip_error_state_test_set_compare_exchange_hook(nullptr, nullptr);
    iree_hip_error_state_reset();
  }

  static hipError_t Publish(hipError_t result) {
    return iree_hip_error_state_publish(iree_hip_error_state_capture(), result);
  }
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

static hipError_t TestRuntimeDeinit() {
  HIP_API_CAPTURE_ERROR_STATE();
  iree_hip_error_state_reset();
  HIP_RETURN_ERROR(hipSuccess);
}

static hipError_t TestRuntimeInit() {
  HIP_API_BEGIN();
  HIP_RETURN_ERROR(hipSuccess);
}

static hipError_t TestNestedPublicResult(hipError_t result) {
  HIP_API_BEGIN();
  HIP_RETURN_ERROR(result);
}

static hipError_t TestOuterPublicCallAcrossReset(hipError_t* nested_result) {
  HIP_API_BEGIN();
  iree_hip_error_state_reset();
  *nested_result = TestNestedPublicResult(hipErrorInvalidValue);
  HIP_RETURN_ERROR(hipSuccess);
}

TEST_F(HipErrorStateTest, NonFatalErrorsRemainThreadLocalAndClearable) {
  EXPECT_EQ(hipErrorInvalidValue, Publish(hipErrorInvalidValue));
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
  EXPECT_EQ(hipErrorInvalidValue, Publish(hipErrorInvalidValue));
  EXPECT_EQ(hipSuccess, Publish(hipSuccess));

  EXPECT_EQ(hipSuccess, iree_hip_error_state_get_and_clear_command_error());
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_error_state_get_and_clear_last_error());

  EXPECT_EQ(hipErrorNotReady, Publish(hipErrorNotReady));
  EXPECT_EQ(hipErrorNotReady,
            iree_hip_error_state_get_and_clear_command_error());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_peek_last_error());
}

TEST_F(HipErrorStateTest, IllegalAddressIsProcessSharedAndSticky) {
  std::thread observer([] {
    EXPECT_EQ(hipErrorIllegalAddress, Publish(hipErrorIllegalAddress));
  });
  observer.join();

  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_fatal_result());
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_peek_last_error());
  EXPECT_EQ(hipErrorIllegalAddress, Publish(hipSuccess));
  EXPECT_EQ(hipErrorIllegalAddress,
            iree_hip_error_state_get_and_clear_last_error());
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_peek_last_error());
}

TEST_F(HipErrorStateTest, FatalEntryBoundaryPrecedesApiSideEffects) {
  EXPECT_EQ(hipErrorIllegalAddress, Publish(hipErrorIllegalAddress));
  iree_hip_error_state_test_reset_compare_exchange_count();

  int output = 7;
  EXPECT_EQ(hipErrorIllegalAddress, TestPublicQuery(&output));
  EXPECT_EQ(7, output);
  EXPECT_EQ(-1, TestPublicValueQuery(&output));
  EXPECT_EQ(7, output);
  EXPECT_EQ(hipErrorIllegalAddress,
            iree_hip_error_state_get_and_clear_command_error());
  EXPECT_EQ(0u, iree_hip_error_state_test_compare_exchange_count());
}

TEST_F(HipErrorStateTest, FatalEntryDoesNotRelatchAcrossReset) {
  EXPECT_EQ(hipErrorIllegalAddress, Publish(hipErrorIllegalAddress));
  EXPECT_EQ(hipErrorIllegalAddress, iree_hip_error_state_begin().fatal_result);

  iree_hip_error_state_reset();
  EXPECT_EQ(hipSuccess, iree_hip_error_state_fatal_result());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_begin().fatal_result);
}

TEST_F(HipErrorStateTest, ResetInvalidatesOtherThreadsFatalState) {
  std::atomic<int> phase{0};
  std::atomic<hipError_t> after_reset{hipErrorUnknown};
  std::thread observer([&] {
    EXPECT_EQ(hipErrorIllegalAddress, Publish(hipErrorIllegalAddress));
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

TEST_F(HipErrorStateTest,
       OldGenerationPublishAfterDeinitReinitDoesNotRetryIntoNewGeneration) {
  const iree_hip_error_state_token_t old_token = iree_hip_error_state_capture();
  CompareExchangeBarrier barrier;
  iree_hip_error_state_test_reset_compare_exchange_count();
  iree_hip_error_state_test_set_compare_exchange_hook(
      WaitAtFirstCompareExchange, &barrier);

  std::atomic<hipError_t> publish_result{hipErrorUnknown};
  std::atomic<hipError_t> thread_command_result{hipErrorUnknown};
  std::atomic<hipError_t> thread_ordinary_result{hipErrorUnknown};
  std::atomic<hipError_t> thread_public_result{hipErrorUnknown};
  std::atomic<int> thread_public_output{0};
  std::thread publisher([&] {
    publish_result.store(
        iree_hip_error_state_publish(old_token, hipErrorIllegalAddress),
        std::memory_order_release);
    thread_command_result.store(
        iree_hip_error_state_get_and_clear_command_error(),
        std::memory_order_release);
    thread_ordinary_result.store(
        iree_hip_error_state_get_and_clear_last_error(),
        std::memory_order_release);
    int output = 0;
    thread_public_result.store(TestPublicQuery(&output),
                               std::memory_order_release);
    thread_public_output.store(output, std::memory_order_release);
  });

  while (!barrier.reached.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  const hipError_t deinit_result = TestRuntimeDeinit();
  const hipError_t init_result = TestRuntimeInit();
  int main_thread_output = 0;
  const hipError_t main_thread_result = TestPublicQuery(&main_thread_output);
  barrier.proceed.store(true, std::memory_order_release);
  publisher.join();
  iree_hip_error_state_test_set_compare_exchange_hook(nullptr, nullptr);

  ASSERT_EQ(hipSuccess, deinit_result);
  ASSERT_EQ(hipSuccess, init_result);
  ASSERT_EQ(hipSuccess, main_thread_result);
  ASSERT_EQ(42, main_thread_output);
  EXPECT_EQ(hipErrorIllegalAddress,
            publish_result.load(std::memory_order_acquire));
  EXPECT_EQ(hipSuccess, thread_command_result.load(std::memory_order_acquire));
  EXPECT_EQ(hipSuccess, thread_ordinary_result.load(std::memory_order_acquire));
  EXPECT_EQ(hipSuccess, thread_public_result.load(std::memory_order_acquire));
  EXPECT_EQ(42, thread_public_output.load(std::memory_order_acquire));
  EXPECT_EQ(hipSuccess, iree_hip_error_state_fatal_result());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_peek_last_error());
  EXPECT_EQ(1u, iree_hip_error_state_test_compare_exchange_count());
}

TEST_F(HipErrorStateTest, ExistingFatalLatchPerformsNoFurtherStrongRmw) {
  EXPECT_EQ(hipErrorIllegalAddress, Publish(hipErrorIllegalAddress));
  iree_hip_error_state_test_reset_compare_exchange_count();

  constexpr int kThreadCount = 16;
  std::vector<std::thread> publishers;
  publishers.reserve(kThreadCount);
  std::atomic<int> unexpected_results{0};
  for (int i = 0; i < kThreadCount; ++i) {
    publishers.emplace_back([&] {
      if (Publish(hipErrorIllegalAddress) != hipErrorIllegalAddress) {
        unexpected_results.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& publisher : publishers) {
    publisher.join();
  }

  EXPECT_EQ(0, unexpected_results.load(std::memory_order_acquire));
  EXPECT_EQ(0u, iree_hip_error_state_test_compare_exchange_count());
}

TEST_F(HipErrorStateTest, NestedPublicCallRetainsEachFrameToken) {
  hipError_t nested_result = hipSuccess;
  EXPECT_EQ(hipSuccess, TestOuterPublicCallAcrossReset(&nested_result));
  EXPECT_EQ(hipErrorInvalidValue, nested_result);
  EXPECT_EQ(hipSuccess, iree_hip_error_state_fatal_result());
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_error_state_get_and_clear_command_error());
  EXPECT_EQ(hipErrorInvalidValue,
            iree_hip_error_state_get_and_clear_last_error());
  EXPECT_EQ(hipSuccess, iree_hip_error_state_peek_last_error());
}

}  // namespace
