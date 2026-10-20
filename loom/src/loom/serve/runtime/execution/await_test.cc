// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/await.h"

#include "iree/async/proactor_platform.h"
#include "iree/async/semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ExecutionAwaitTest : public ::testing::TestWithParam<size_t> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    for (auto*& semaphore : semaphores_) {
      IREE_ASSERT_OK(iree_async_semaphore_create(
          proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
          iree_allocator_system(), &semaphore));
    }
  }

  void TearDown() override {
    if (pending_) {
      iree_async_semaphore_fail(semaphores_[0],
                                iree_status_from_code(IREE_STATUS_CANCELLED));
      Drain();
    }
    for (auto* semaphore : semaphores_) {
      iree_async_semaphore_release(semaphore);
    }
    iree_async_proactor_release(proactor_);
  }

  void Submit(uint64_t first, uint64_t second) {
    loom_serve_execution_frontier_t frontier = {
        .semaphores =
            {
                reinterpret_cast<iree_hal_semaphore_t*>(semaphores_[0]),
                reinterpret_cast<iree_hal_semaphore_t*>(semaphores_[1]),
            },
        .values = {first, second},
    };
    IREE_ASSERT_OK(loom_serve_execution_await(
        proactor_, &frontier,
        [](void* self, iree_async_operation_t*, iree_status_t status,
           iree_async_completion_flags_t) {
          auto* test = static_cast<ExecutionAwaitTest*>(self);
          test->code_ = iree_status_code(status);
          iree_status_free(status);
          test->pending_ = false;
          ++test->completions_;
        },
        this, &wait_));
    pending_ = true;
  }

  void Signal(size_t branch, uint64_t value) {
    IREE_ASSERT_OK(
        iree_async_semaphore_signal(semaphores_[branch], value, nullptr));
  }

  void Drain() {
    while (pending_) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
  }

  // The test owns polling, so callback return joins every borrowed access.
  iree_async_proactor_t* proactor_ = nullptr;
  // Toll-free readiness handles for independent work and feedback branches.
  iree_async_semaphore_t* semaphores_[2] = {};
  // One record reused only after its completion has returned.
  loom_serve_execution_await_t wait_ = {};
  // Accepted wait whose callback has not returned to the poll owner.
  bool pending_ = false;
  // Number of complete waits, observed by the same poll owner.
  size_t completions_ = 0;
  // Last callback outcome, independent of the semaphore storage.
  iree_status_code_t code_ = IREE_STATUS_OK;
};

TEST_P(ExecutionAwaitTest, CapturesBothBranchesAndReusesStorage) {
  ASSERT_NO_FATAL_FAILURE(Submit(1, 2));
  const size_t first = GetParam();
  ASSERT_NO_FATAL_FAILURE(Signal(first, first + 1));
  iree_status_t status =
      iree_async_proactor_poll(proactor_, iree_immediate_timeout(), nullptr);
  if (iree_status_is_deadline_exceeded(status)) {
    iree_status_ignore(status);
  } else {
    IREE_ASSERT_OK(status);
  }
  EXPECT_EQ(completions_, 0u);
  ASSERT_NO_FATAL_FAILURE(Signal(1 - first, 2 - first));
  ASSERT_NO_FATAL_FAILURE(Drain());
  EXPECT_EQ(completions_, 1u);
  EXPECT_EQ(code_, IREE_STATUS_OK);

  ASSERT_NO_FATAL_FAILURE(Submit(3, 4));
  ASSERT_NO_FATAL_FAILURE(Signal(0, 3));
  ASSERT_NO_FATAL_FAILURE(Signal(1, 4));
  ASSERT_NO_FATAL_FAILURE(Drain());
  EXPECT_EQ(completions_, 2u);
  EXPECT_EQ(code_, IREE_STATUS_OK);
}

TEST_P(ExecutionAwaitTest, FailureCompletesWithoutTheOtherBranch) {
  ASSERT_NO_FATAL_FAILURE(Submit(1, 2));
  iree_async_semaphore_fail(semaphores_[GetParam()],
                            iree_status_from_code(IREE_STATUS_DATA_LOSS));
  ASSERT_NO_FATAL_FAILURE(Drain());
  EXPECT_EQ(completions_, 1u);
  EXPECT_EQ(code_, IREE_STATUS_DATA_LOSS);
}

INSTANTIATE_TEST_SUITE_P(Branches, ExecutionAwaitTest, ::testing::Values(0, 1));

}  // namespace
