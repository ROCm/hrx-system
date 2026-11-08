// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/host/reclamation.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ReclamationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    const iree_allocator_t allocator = {
        this,
        [](void* user_data, iree_allocator_command_t command,
           const void* parameters, void** pointer) -> iree_status_t {
          auto* self = static_cast<ReclamationTest*>(user_data);
          if (command == IREE_ALLOCATOR_COMMAND_TRIM) {
            ++self->trims_;
            if (self->failure_ != IREE_STATUS_OK) {
              return iree_status_from_code(self->failure_);
            }
          }
          const auto backing = iree_allocator_system();
          return backing.ctl(backing.self, command, parameters, pointer);
        }};
    IREE_ASSERT_OK(loom_runtime_host_create(1, nullptr, allocator, &host_));
  }

  void TearDown() override {
    for (size_t i = 0; i < initialized_; ++i) {
      IREE_EXPECT_OK(Drain(owners_[i], true));
      loom_runtime_reclamation_set_observer(&owners_[i], {});
      loom_runtime_reclamation_deinitialize(&owners_[i]);
    }
    loom_runtime_host_release(host_);
    iree_notification_deinitialize(&notification_);
  }

  static void Wake(void* user_data) {
    iree_notification_post(static_cast<iree_notification_t*>(user_data),
                           IREE_ALL_WAITERS);
  }

  void Initialize(iree_duration_t timeout = 0) {
    auto& owner = owners_[initialized_++];
    IREE_ASSERT_OK(loom_runtime_reclamation_initialize(host_, timeout, &owner));
    loom_runtime_reclamation_set_observer(&owner, {Wake, &notification_});
  }

  iree_status_t Drain(loom_runtime_reclamation_t& owner,
                      bool stopping = false) {
    iree_status_t status = iree_ok_status();
    for (;;) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      bool progress = false;
      iree_status_t advance_status = loom_runtime_reclamation_advance(
          &owner, stopping || !iree_status_is_ok(status), &progress);
      status = iree_status_join(status, advance_status);
      if (!loom_runtime_reclamation_is_pending(&owner)) {
        iree_notification_cancel_wait(&notification_);
        return status;
      }
      if (progress) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
  }

  void Prepare() {
    loom_runtime_host_preparation_begin(host_);
    loom_runtime_host_preparation_end(host_);
  }

  // Shared host with exactly one worker, making queued work deterministic.
  loom_runtime_host_t* host_ = nullptr;
  // Independent maintenance owners sharing the host's reclamation claim.
  loom_runtime_reclamation_t owners_[2] = {};
  // Initialized owner synchronization requiring teardown.
  size_t initialized_ = 0;
  // Outer-owner notification; no runtime worker waits on this condition.
  iree_notification_t notification_;
  // Actual allocator controls, independent of timer observations.
  std::atomic<unsigned> trims_{0};
  // External allocator control outcome, changed only when no task is pending.
  iree_status_code_t failure_ = IREE_STATUS_OK;
};

TEST(Reclamation, DisabledAndInvalidConfigurationsAcceptNoWork) {
  for (const auto timeout : {iree_duration_t{-1}, iree_duration_t{0}}) {
    loom_runtime_reclamation_t owner;
    auto status = loom_runtime_reclamation_initialize(nullptr, timeout, &owner);
    if (timeout < 0) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
    } else {
      IREE_EXPECT_OK(status);
      bool progress = true;
      IREE_EXPECT_OK(
          loom_runtime_reclamation_advance(&owner, false, &progress));
      EXPECT_FALSE(progress);
    }
    EXPECT_FALSE(loom_runtime_reclamation_is_pending(&owner));
    loom_runtime_reclamation_deinitialize(&owner);
  }
}

TEST_F(ReclamationTest, QuietHostCreatesNoIoServiceOrWork) {
  ASSERT_NO_FATAL_FAILURE(Initialize(iree_make_duration_ms(1000)));
  IREE_ASSERT_OK(Drain(owners_[0]));
  EXPECT_EQ(owners_[0].timer.proactor, nullptr);
  EXPECT_FALSE(owners_[0].task_pending);
  EXPECT_EQ(trims_.load(), 0u);
  bool progress = true;
  IREE_ASSERT_OK(
      loom_runtime_reclamation_advance(&owners_[0], false, &progress));
  EXPECT_FALSE(progress);
}

TEST_F(ReclamationTest, FourPreparationMembersShareOneReclamation) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  loom_runtime_host_preparation_begin(host_);
  for (int i = 0; i < 4; ++i) {
    Prepare();
    IREE_ASSERT_OK(Drain(owners_[0]));
    EXPECT_EQ(trims_.load(), 0u);
    EXPECT_FALSE(loom_runtime_reclamation_is_pending(&owners_[0]));
  }
  loom_runtime_host_preparation_end(host_);
  IREE_ASSERT_OK(Drain(owners_[0]));
  EXPECT_EQ(trims_.load(), 1u);
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(Drain(owners_[0]));
    EXPECT_EQ(trims_.load(), 1u);
  }
}

TEST_F(ReclamationTest, IdleAlarmRetiresWithoutPeriodicPolling) {
  ASSERT_NO_FATAL_FAILURE(Initialize(iree_make_duration_ms(1)));
  Prepare();
  IREE_ASSERT_OK(Drain(owners_[0]));
  EXPECT_EQ(trims_.load(), 1u);
  EXPECT_FALSE(loom_runtime_reclamation_is_pending(&owners_[0]));
  Prepare();
  IREE_ASSERT_OK(Drain(owners_[0]));
  EXPECT_EQ(trims_.load(), 2u);
}

TEST_F(ReclamationTest, InfiniteQuietIntervalAcceptsNoIdleWork) {
  ASSERT_NO_FATAL_FAILURE(Initialize(IREE_DURATION_INFINITE));
  Prepare();
  IREE_ASSERT_OK(Drain(owners_[0]));
  EXPECT_EQ(owners_[0].timer.proactor, nullptr);
  EXPECT_FALSE(loom_runtime_reclamation_is_pending(&owners_[0]));
  EXPECT_EQ(trims_.load(), 0u);
}

TEST_F(ReclamationTest, StoppingJoinsFarFutureAlarm) {
  ASSERT_NO_FATAL_FAILURE(Initialize(iree_make_duration_ms(3600000)));
  Prepare();
  bool progress = false;
  IREE_ASSERT_OK(
      loom_runtime_reclamation_advance(&owners_[0], false, &progress));
  EXPECT_TRUE(loom_runtime_reclamation_is_pending(&owners_[0]));
  IREE_ASSERT_OK(Drain(owners_[0], true));
  EXPECT_EQ(trims_.load(), 0u);
  Prepare();
  IREE_ASSERT_OK(Drain(owners_[0], true));
  EXPECT_EQ(trims_.load(), 0u);
}

TEST_F(ReclamationTest, IndependentOwnersClaimOneSharedTrim) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  ASSERT_NO_FATAL_FAILURE(Initialize());
  Prepare();
  for (auto& owner : owners_) {
    bool progress = false;
    IREE_ASSERT_OK(loom_runtime_reclamation_advance(&owner, false, &progress));
  }
  IREE_ASSERT_OK(Drain(owners_[0]));
  IREE_ASSERT_OK(Drain(owners_[1]));
  EXPECT_EQ(trims_.load(), 1u);
}

TEST_F(ReclamationTest,
       AllocatorFailureBelongsToMaintenanceAndHostRemainsUsable) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  failure_ = IREE_STATUS_UNAVAILABLE;
  Prepare();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, Drain(owners_[0]));
  EXPECT_EQ(trims_.load(), 1u);
  EXPECT_FALSE(owners_[0].observing);
  IREE_ASSERT_OK(Drain(owners_[0], true));
  failure_ = IREE_STATUS_OK;
  ASSERT_NO_FATAL_FAILURE(Initialize());
  Prepare();
  IREE_ASSERT_OK(Drain(owners_[1]));
  EXPECT_EQ(trims_.load(), 2u);
}

// An independent borrower occupies the sole real worker until explicitly
// released. This controls scheduling without replacing the reclamation code.
class WorkerGate {
 public:
  explicit WorkerGate(loom_runtime_host_t* host) {
    iree_notification_initialize(&notification_);
    loom_runtime_readiness_initialize(loom_runtime_host_executor(host), nullptr,
                                      &readiness_);
    loom_runtime_readiness_set_observer(
        &readiness_, {[](void* user_data) {
                        iree_notification_post(
                            static_cast<iree_notification_t*>(user_data),
                            IREE_ALL_WAITERS);
                      },
                      &notification_});
    loom_runtime_readiness_schedule(
        &readiness_,
        [](void* payload) -> iree_status_t {
          auto* self = static_cast<WorkerGate*>(payload);
          std::unique_lock<std::mutex> lock(self->mutex_);
          self->entered_ = true;
          self->condition_.notify_one();
          self->condition_.wait(lock, [&] { return self->released_; });
          return iree_ok_status();
        },
        this, &task_);
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }

  ~WorkerGate() {
    Release();
    iree_status_t status = iree_ok_status();
    for (;;) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      if (loom_runtime_readiness_take(&readiness_, &task_.result, &status)) {
        iree_notification_cancel_wait(&notification_);
        break;
      }
      iree_notification_commit_wait(&notification_, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
    IREE_EXPECT_OK(status);
    loom_runtime_readiness_deinitialize(&readiness_);
    iree_notification_deinitialize(&notification_);
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    condition_.notify_one();
  }

 private:
  // Synchronizes the test dependency's arrival and release.
  std::mutex mutex_;
  // Explicit worker-entry and release events.
  std::condition_variable condition_;
  // Worker has begun executing the dependency.
  bool entered_ = false;
  // Outer owner has released the dependency.
  bool released_ = false;
  // Final CPU process retirement notification.
  iree_notification_t notification_;
  // Real shared-worker publication domain.
  loom_runtime_readiness_t readiness_;
  // Borrowed CPU work, kept alive through final release.
  loom_runtime_readiness_task_t task_ = {};
};

TEST_F(ReclamationTest, NewPreparationInvalidatesQueuedTrim) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  WorkerGate gate(host_);
  Prepare();
  bool progress = false;
  IREE_ASSERT_OK(
      loom_runtime_reclamation_advance(&owners_[0], false, &progress));
  EXPECT_TRUE(owners_[0].task_pending);
  loom_runtime_host_preparation_begin(host_);
  gate.Release();
  IREE_ASSERT_OK(Drain(owners_[0]));
  EXPECT_EQ(trims_.load(), 0u);
  loom_runtime_host_preparation_end(host_);
  IREE_ASSERT_OK(Drain(owners_[0]));
  EXPECT_EQ(trims_.load(), 1u);
}

TEST_F(ReclamationTest, StoppingRetiresQueuedTrimWithoutReclaiming) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  WorkerGate gate(host_);
  Prepare();
  bool progress = false;
  IREE_ASSERT_OK(
      loom_runtime_reclamation_advance(&owners_[0], false, &progress));
  EXPECT_TRUE(owners_[0].task_pending);
  IREE_ASSERT_OK(
      loom_runtime_reclamation_advance(&owners_[0], true, &progress));
  gate.Release();
  IREE_ASSERT_OK(Drain(owners_[0], true));
  EXPECT_EQ(trims_.load(), 0u);
}

}  // namespace
