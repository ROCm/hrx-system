// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/readiness.h"

#include "iree/async/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/host/host.h"

namespace {

class TextReadinessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    IREE_ASSERT_OK(
        loom_runtime_host_create(1, nullptr, iree_allocator_system(), &host_));
    iree_async_proactor_t* proactor = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_pool_get(
        loom_runtime_host_proactor_pool(host_), 0, &proactor));
    loom_serve_text_readiness_initialize(loom_runtime_host_executor(host_),
                                         proactor, &readiness_);
    loom_serve_text_readiness_set_observer(&readiness_, {Wake, this});
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
    loom_serve_text_readiness_set_observer(&readiness_, {});
    loom_serve_text_readiness_deinitialize(&readiness_);
    for (auto* semaphore : semaphores_) {
      iree_async_semaphore_release(semaphore);
    }
    loom_runtime_host_release(host_);
    iree_notification_deinitialize(&notification_);
  }

  static void Wake(void* user_data) {
    auto* self = static_cast<TextReadinessTest*>(user_data);
    iree_notification_post(&self->notification_, IREE_ALL_WAITERS);
  }

  iree_status_t Await(bool cpu) {
    for (;;) {
      const auto token = iree_notification_prepare_wait(&notification_);
      iree_status_t status = iree_ok_status();
      const bool ready =
          cpu ? loom_serve_text_readiness_take_cpu(&readiness_, &status)
              : loom_serve_text_readiness_take_device(&readiness_, &status);
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
  loom_serve_text_readiness_t readiness_ = {};
  // Independently signaled model-work and feedback frontiers.
  iree_async_semaphore_t* semaphores_[2] = {};
  // Submission capture borrowed only during await registration.
  loom_runtime_execution_frontier_t frontier_ = {};
  // Owner notification with prepare/check/commit lost-wakeup protection.
  iree_notification_t notification_;
};

TEST_F(TextReadinessTest, CpuReleaseAndDeviceFrontierAreIndependent) {
  IREE_ASSERT_OK(loom_serve_text_readiness_await(&readiness_, &frontier_));
  // Later submissions cannot change the accepted ALL wait's values.
  frontier_.values[0] = frontier_.values[1] = 2;
  int work_count = 0;
  auto run = +[](void* user_data) -> iree_status_t {
    ++*static_cast<int*>(user_data);
    return iree_ok_status();
  };
  loom_serve_text_readiness_schedule(&readiness_, run, &work_count);
  IREE_ASSERT_OK(Await(true));
  EXPECT_EQ(work_count, 1);
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphores_[0], 1, nullptr));
  iree_status_t status = iree_ok_status();
  EXPECT_FALSE(loom_serve_text_readiness_take_device(&readiness_, &status));
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphores_[1], 1, nullptr));
  IREE_ASSERT_OK(Await(false));
  EXPECT_FALSE(loom_serve_text_readiness_take_cpu(&readiness_, &status));
  loom_serve_text_readiness_schedule(&readiness_, run, &work_count);
  IREE_ASSERT_OK(Await(true));
  EXPECT_EQ(work_count, 2);
}

TEST_F(TextReadinessTest, FailuresTransferToOwnerAndCpuChannelCanBeReused) {
  IREE_ASSERT_OK(loom_serve_text_readiness_await(&readiness_, &frontier_));
  loom_serve_text_readiness_schedule(
      &readiness_,
      +[](void* user_data) -> iree_status_t {
        return iree_make_status(IREE_STATUS_DATA_LOSS, "translation failed");
      },
      nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Await(true));
  iree_async_semaphore_fail(
      semaphores_[0],
      iree_make_status(IREE_STATUS_ABORTED, "model frontier failed"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, Await(false));
  loom_serve_text_readiness_schedule(
      &readiness_,
      +[](void* user_data) -> iree_status_t { return iree_ok_status(); },
      nullptr);
  IREE_EXPECT_OK(Await(true));
}

}  // namespace
