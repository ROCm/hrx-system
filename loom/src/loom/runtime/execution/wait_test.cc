// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/execution/wait.h"

#include <array>

#include "iree/async/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/host/host.h"

namespace {

class ExecutionWaitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    IREE_ASSERT_OK(
        loom_runtime_host_create(1, nullptr, iree_allocator_system(), &host_));
    iree_async_proactor_t* proactor = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_pool_get(
        loom_runtime_host_proactor_pool(host_), 0, &proactor));
    loom_runtime_readiness_initialize(loom_runtime_host_executor(host_),
                                      proactor, &readiness_);
    loom_runtime_readiness_set_observer(&readiness_, {Wake, this});
    for (size_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_async_semaphore_create(
          proactor, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
          iree_allocator_system(), &semaphores_[i]));
      frontier_.semaphores[i] =
          reinterpret_cast<iree_hal_semaphore_t*>(semaphores_[i]);
      frontier_.values[i] = 1;
    }
  }

  void TearDown() override {
    loom_runtime_readiness_set_observer(&readiness_, {});
    loom_runtime_readiness_deinitialize(&readiness_);
    for (auto* semaphore : semaphores_) {
      iree_async_semaphore_release(semaphore);
    }
    loom_runtime_host_release(host_);
    iree_notification_deinitialize(&notification_);
  }

  static void Wake(void* user_data) {
    auto* self = static_cast<ExecutionWaitTest*>(user_data);
    iree_notification_post(&self->notification_, IREE_ALL_WAITERS);
  }

  iree_status_t Await(loom_runtime_readiness_result_t* result) {
    for (;;) {
      const auto token = iree_notification_prepare_wait(&notification_);
      iree_status_t status = iree_ok_status();
      const bool ready =
          loom_runtime_readiness_take(&readiness_, result, &status);
      if (ready) {
        iree_notification_cancel_wait(&notification_);
        return status;
      }
      iree_notification_commit_wait(&notification_, token, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }

  // Production task/proactor owner, outliving both publication channels.
  loom_runtime_host_t* host_ = nullptr;
  // Callback storage reused only after owner consumption.
  loom_runtime_readiness_t readiness_ = {};
  // Independent CPU borrower whose final release publishes its result.
  loom_runtime_readiness_task_t task_ = {};
  // Captured work/feedback wait independent of the CPU operation.
  loom_runtime_execution_wait_t wait_ = {};
  // Independently signaled model-work and feedback frontiers.
  iree_async_semaphore_t* semaphores_[2] = {};
  // Submission capture borrowed only during await registration.
  loom_runtime_execution_frontier_t frontier_ = {};
  // Owner notification with prepare/check/commit lost-wakeup protection.
  iree_notification_t notification_;
};

TEST_F(ExecutionWaitTest, CpuReleaseAndDeviceFrontierAreIndependent) {
  IREE_ASSERT_OK(loom_runtime_execution_await(&readiness_, &frontier_, &wait_));
  // Later submissions cannot change the accepted ALL wait's values.
  frontier_.values[0] = frontier_.values[1] = 2;
  int work_count = 0;
  auto run = +[](void* user_data) -> iree_status_t {
    ++*static_cast<int*>(user_data);
    return iree_ok_status();
  };
  loom_runtime_readiness_schedule(&readiness_, run, &work_count, &task_);
  IREE_ASSERT_OK(Await(&task_.result));
  EXPECT_EQ(work_count, 1);
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphores_[0], 1, nullptr));
  iree_status_t status = iree_ok_status();
  EXPECT_FALSE(
      loom_runtime_readiness_take(&readiness_, &wait_.wait.result, &status));
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphores_[1], 1, nullptr));
  IREE_ASSERT_OK(Await(&wait_.wait.result));
  EXPECT_FALSE(
      loom_runtime_readiness_take(&readiness_, &task_.result, &status));
  loom_runtime_readiness_schedule(&readiness_, run, &work_count, &task_);
  IREE_ASSERT_OK(Await(&task_.result));
  EXPECT_EQ(work_count, 2);
  IREE_ASSERT_OK(loom_runtime_execution_await(&readiness_, &frontier_, &wait_));
  for (auto* semaphore : semaphores_) {
    IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 2, nullptr));
  }
  IREE_EXPECT_OK(Await(&wait_.wait.result));
}

TEST_F(ExecutionWaitTest, FailuresTransferToOwnerAndCpuChannelCanBeReused) {
  IREE_ASSERT_OK(loom_runtime_execution_await(&readiness_, &frontier_, &wait_));
  loom_runtime_readiness_schedule(
      &readiness_,
      +[](void* user_data) -> iree_status_t {
        return iree_make_status(IREE_STATUS_DATA_LOSS, "translation failed");
      },
      nullptr, &task_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Await(&task_.result));
  iree_async_semaphore_fail(
      semaphores_[0],
      iree_make_status(IREE_STATUS_ABORTED, "model frontier failed"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, Await(&wait_.wait.result));
  loom_runtime_readiness_schedule(
      &readiness_,
      +[](void* user_data) -> iree_status_t { return iree_ok_status(); },
      nullptr, &task_);
  IREE_EXPECT_OK(Await(&task_.result));
}

TEST_F(ExecutionWaitTest, IndependentOperationsCompleteOutOfSubmissionOrder) {
  std::array<loom_runtime_execution_wait_t, 3> waits = {};
  const uint64_t values[] = {3, 1, 2};
  for (size_t i = 0; i < waits.size(); ++i) {
    frontier_.values[0] = frontier_.values[1] = values[i];
    IREE_ASSERT_OK(
        loom_runtime_execution_await(&readiness_, &frontier_, &waits[i]));
  }
  std::array<loom_runtime_readiness_task_t, 2> tasks = {};
  std::array<int, 2> counts = {};
  for (size_t i = 0; i < tasks.size(); ++i) {
    loom_runtime_readiness_schedule(
        &readiness_,
        +[](void* self) -> iree_status_t {
          ++*static_cast<int*>(self);
          return iree_ok_status();
        },
        &counts[i], &tasks[i]);
  }
  for (size_t ordinal : {1u, 2u, 0u}) {
    for (auto* semaphore : semaphores_) {
      IREE_ASSERT_OK(
          iree_async_semaphore_signal(semaphore, values[ordinal], nullptr));
    }
    IREE_ASSERT_OK(Await(&waits[ordinal].wait.result));
    iree_status_t status = iree_ok_status();
    EXPECT_FALSE(loom_runtime_readiness_take(
        &readiness_, &waits[ordinal].wait.result, &status));
    if (ordinal != 0) {
      EXPECT_FALSE(loom_runtime_readiness_take(&readiness_,
                                               &waits[0].wait.result, &status));
    }
  }
  for (size_t i = 0; i < tasks.size(); ++i) {
    IREE_ASSERT_OK(Await(&tasks[i].result));
    EXPECT_EQ(counts[i], 1);
  }
}

TEST_F(ExecutionWaitTest, ReadinessWaitRetainsEveryProducerDependency) {
  iree_async_semaphore_t* third = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      readiness_.proactor, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &third));
  iree_async_semaphore_t* semaphores[] = {semaphores_[0], third,
                                          semaphores_[1]};
  uint64_t values[] = {1, 2, 3};
  loom_runtime_readiness_wait_t wait = {};
  IREE_ASSERT_OK(
      loom_runtime_readiness_await(&readiness_, 3, semaphores, values, &wait));
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphores_[0], 1, nullptr));
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphores_[1], 3, nullptr));
  iree_status_t status = iree_ok_status();
  EXPECT_FALSE(loom_runtime_readiness_take(&readiness_, &wait.result, &status));
  IREE_ASSERT_OK(iree_async_semaphore_signal(third, 2, nullptr));
  IREE_EXPECT_OK(Await(&wait.result));
  iree_async_semaphore_release(third);
}

}  // namespace
