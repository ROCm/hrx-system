// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/runtime.h"

#include "iree/async/proactor_platform.h"
#include "iree/hal/utils/host_semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

struct RuntimeProgressCapture {
  loom_run_hal_runtime_progress_phase_t
      phases[LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_COUNT];
  iree_host_size_t count = 0;
  loom_run_hal_runtime_progress_phase_t rejected_phase =
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_COUNT;
};

static iree_status_t CaptureRuntimeProgress(
    void* user_data, loom_run_hal_runtime_progress_phase_t phase) {
  auto* capture = static_cast<RuntimeProgressCapture*>(user_data);
  if (phase == capture->rejected_phase) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "runtime progress rejected by test");
  }
  capture->phases[capture->count++] = phase;
  return iree_ok_status();
}

class HalRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    IREE_ASSERT_OK(iree_hal_host_semaphore_create(
        proactor_, 0, iree_allocator_system(), &semaphore_));
  }

  void TearDown() override {
    iree_hal_semaphore_release(semaphore_);
    iree_async_proactor_release(proactor_);
  }

  // Owned proactor that outlives the semaphore.
  iree_async_proactor_t* proactor_ = nullptr;
  // Owned host semaphore using the shared code-only wait contract.
  iree_hal_semaphore_t* semaphore_ = nullptr;
};

TEST_F(HalRuntimeTest, PublishesRuntimeConstructionProgress) {
  RuntimeProgressCapture capture;
  loom_run_hal_runtime_options_t options;
  loom_run_hal_runtime_options_initialize(IREE_SV("task"), &options);
  options.progress_sink = {
      .fn = CaptureRuntimeProgress,
      .user_data = &capture,
  };
  loom_run_hal_runtime_t runtime = {};
  IREE_ASSERT_OK(loom_run_hal_runtime_initialize(
      &options, iree_allocator_system(), &runtime));

  const loom_run_hal_runtime_progress_phase_t expected[] = {
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_PROACTOR_POOL_CREATE,
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_FRONTIER_TRACKER_CREATE,
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_DEVICE_CREATE,
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_PROACTOR_POOL_RELEASE,
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_DEVICE_GROUP_CREATE,
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_FRONTIER_TRACKER_RELEASE,
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_DISPATCH_QUEUE_SELECT,
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_TRANSFER_QUEUE_SELECT,
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_DEVICE_SOURCE_POOL_CREATE,
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_DEVICE_POOL_CREATE,
      LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_STAGING_POOL_CREATE,
  };
  ASSERT_EQ(capture.count, IREE_ARRAYSIZE(expected));
  for (iree_host_size_t i = 0; i < capture.count; ++i) {
    EXPECT_EQ(capture.phases[i], expected[i]);
  }

  loom_run_hal_runtime_deinitialize(&runtime);
}

TEST_F(HalRuntimeTest, ProgressFailureReleasesConstructedRuntime) {
  RuntimeProgressCapture capture = {
      .rejected_phase =
          LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_PROACTOR_POOL_RELEASE,
  };
  loom_run_hal_runtime_options_t options;
  loom_run_hal_runtime_options_initialize(IREE_SV("task"), &options);
  options.progress_sink = {
      .fn = CaptureRuntimeProgress,
      .user_data = &capture,
  };
  loom_run_hal_runtime_t runtime = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_run_hal_runtime_initialize(
                            &options, iree_allocator_system(), &runtime));
  EXPECT_EQ(runtime.device, nullptr);
  EXPECT_EQ(runtime.device_group, nullptr);
  EXPECT_EQ(runtime.device_pool, nullptr);
  EXPECT_EQ(runtime.staging_pool, nullptr);
  ASSERT_GT(capture.count, 0u);
  EXPECT_EQ(capture.phases[capture.count - 1],
            LOOM_RUN_HAL_RUNTIME_PROGRESS_PHASE_FAILURE_DEINITIALIZE);
}

TEST_F(HalRuntimeTest, FailedWaitQueriesDiagnosticBeforeSemaphoreRelease) {
  iree_hal_semaphore_fail(
      semaphore_,
      iree_make_status(IREE_STATUS_DATA_LOSS, "dispatch feedback failure"));
  iree::Status status = loom_run_hal_semaphore_wait(
      semaphore_, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  iree_hal_semaphore_release(semaphore_);
  semaphore_ = nullptr;

  EXPECT_THAT(status,
              iree::testing::status::StatusIs(iree::StatusCode::kDataLoss));
#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS) != 0
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("dispatch feedback failure"));
#endif  // has IREE_STATUS_FEATURE_ANNOTATIONS
}

TEST_F(HalRuntimeTest, SuccessfulWaitReturnsOk) {
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore_, 1, nullptr));
  IREE_EXPECT_OK(loom_run_hal_semaphore_wait(
      semaphore_, 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_F(HalRuntimeTest, TimeoutPreservesWaitFailure) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      loom_run_hal_semaphore_wait(semaphore_, 1, iree_immediate_timeout(),
                                  IREE_ASYNC_WAIT_FLAG_NONE));
}

}  // namespace
}  // namespace loom
