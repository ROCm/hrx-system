// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/activation.h"

#include <condition_variable>
#include <mutex>

#include "experimental/loom_serve/runtime/device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Uses the real task HAL and proactor so CPU sanitizers observe the actual
// readiness publication. Weighted GPU output and pin pressure are exercised by
// weights_test; this parameter-free group isolates callback storage lifetime.
class ActivationTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  static void Wake(void* user_data) {
    auto* self = static_cast<ActivationTest*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex_);
    self->ready_ = true;
    self->condition_.notify_one();
  }

  void AwaitWake() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return ready_; });
  }

  void SetUp() override {
    IREE_ASSERT_OK(
        loom_serve_host_create(GetParam(), &host_, iree_allocator_system()));
    const loom_serve_device_options_t options = {.uri = IREE_SV("task")};
    IREE_ASSERT_OK(loom_serve_device_group_create(host_, 1, &options, &group_,
                                                  iree_allocator_system()));
    auto* device = loom_serve_device_group_at(group_, 0);
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution_));
    IREE_ASSERT_OK(loom_serve_residency_create(
        loom_serve_device_residency_cache(device), execution_, 0, nullptr,
        &residency_, iree_allocator_system()));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
    uint64_t value = 1;
    IREE_ASSERT_OK(
        loom_serve_execution_barrier(execution_, {1, &gate_, &value}));
    IREE_ASSERT_OK(loom_serve_activation_create(
        residency_, loom_serve_host_proactor_pool(host_), Wake, this,
        &activation_, iree_allocator_system()));
  }

  void TearDown() override {
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 2, nullptr));
    }
    if (activation_) {
      if (phase_ != LOOM_SERVE_ACTIVATION_COMPLETE) {
        IREE_EXPECT_OK(loom_serve_activation_advance(
            activation_, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase_));
        if (phase_ == LOOM_SERVE_ACTIVATION_LOADING) {
          AwaitWake();
          IREE_EXPECT_OK(loom_serve_activation_advance(
              activation_, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase_));
        }
      }
      loom_serve_activation_destroy(activation_);
    }
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    loom_serve_residency_destroy(residency_);
    loom_serve_execution_release(execution_);
    iree_hal_semaphore_release(gate_);
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
    loom_serve_host_release(host_);
  }

  // Shared task workers and proactor pool, outliving every model borrower.
  loom_serve_host_t* host_ = nullptr;
  // Native device and shared residency cache owner.
  loom_serve_device_group_t* group_ = nullptr;
  // Model timeline containing an explicit unresolved predecessor.
  loom_serve_execution_t* execution_ = nullptr;
  // Parameter-free native group used by the real activation owner.
  loom_serve_residency_t* residency_ = nullptr;
  // Explicit dependency; only the outer owner publishes its completion.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Warmup operation whose final callback must precede destruction.
  loom_serve_activation_t* activation_ = nullptr;
  // Last owner-observed phase, also bounding partial-test cleanup.
  loom_serve_activation_phase_t phase_ = LOOM_SERVE_ACTIVATION_WAITING;
  // Synchronizes the outer test with the production callback's wake.
  std::mutex mutex_;
  // Outer test wait, never executed on the task pool.
  std::condition_variable condition_;
  // The warmup operation has requested an owner turn.
  bool ready_ = false;
};

TEST_P(ActivationTest, UnstartedCancellationHasNoCallbackOrActivation) {
  IREE_ASSERT_OK(loom_serve_activation_advance(
      activation_, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase_));
  EXPECT_EQ(phase_, LOOM_SERVE_ACTIVATION_COMPLETE);
  EXPECT_FALSE(loom_serve_residency_snapshot(residency_).active);
  std::lock_guard<std::mutex> lock(mutex_);
  EXPECT_FALSE(ready_);
}

TEST_P(ActivationTest, SubmittedCancellationJoinsRealReadinessPublication) {
  IREE_ASSERT_OK(loom_serve_activation_advance(activation_, 0, &phase_));
  EXPECT_EQ(phase_, LOOM_SERVE_ACTIVATION_LOADING);
  IREE_ASSERT_OK(loom_serve_activation_advance(
      activation_, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase_));
  EXPECT_EQ(phase_, LOOM_SERVE_ACTIVATION_LOADING);
  EXPECT_TRUE(loom_serve_residency_snapshot(residency_).active);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    EXPECT_FALSE(ready_);
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  AwaitWake();
  IREE_ASSERT_OK(loom_serve_activation_advance(
      activation_, LOOM_SERVE_ACTIVATION_FLAG_CANCEL, &phase_));
  EXPECT_EQ(phase_, LOOM_SERVE_ACTIVATION_COMPLETE);
}

INSTANTIATE_TEST_SUITE_P(Workers, ActivationTest, ::testing::Values(1u, 4u));

}  // namespace
