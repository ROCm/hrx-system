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

class HalRuntimePoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_run_hal_runtime_options_t options;
    loom_run_hal_runtime_options_initialize(IREE_SV("task"), &options);
    IREE_ASSERT_OK(loom_run_hal_runtime_initialize(
        &options, iree_allocator_system(), &runtime_));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        runtime_.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore_));
  }

  void TearDown() override {
    for (auto* buffer : buffers_) {
      iree_hal_buffer_release(buffer);
    }
    iree_hal_semaphore_release(semaphore_);
    loom_run_hal_runtime_deinitialize(&runtime_);
  }

  // One real execution runtime shared by all allocations in a test.
  loom_run_hal_runtime_t runtime_ = {};
  // Completion timeline for allocation and retirement operations.
  iree_hal_semaphore_t* semaphore_ = nullptr;
  // Owned materialized wrappers, retained through queue retirement.
  iree_hal_buffer_t* buffers_[4] = {};
};

TEST_F(HalRuntimePoolTest,
       SmallBuffersShareBackingAndLargeBuffersStayDedicated) {
  const iree_hal_pool_reservation_request_t small_requests[] = {
      iree_hal_make_pool_reservation_request(13, 0),
      iree_hal_make_pool_reservation_request(257, 0),
      iree_hal_make_pool_reservation_request(4096, 0),
  };
  uint64_t timeline = 1;
  const iree_hal_semaphore_list_t completion = {
      .count = 1,
      .semaphores = &semaphore_,
      .payload_values = &timeline,
  };
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      runtime_.transfer_queue, iree_hal_semaphore_list_empty(), completion,
      runtime_.device_pool, IREE_ARRAYSIZE(small_requests), small_requests,
      buffers_));
  IREE_ASSERT_OK(loom_run_hal_semaphore_wait(semaphore_, timeline,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(runtime_.device_pool, &stats);
  EXPECT_EQ(stats.reservation_count, 3u);
  EXPECT_EQ(stats.slab_count, 1u);
  for (size_t i = 0; i < IREE_ARRAYSIZE(small_requests); ++i) {
    EXPECT_EQ(iree_hal_buffer_byte_length(buffers_[i]),
              small_requests[i].allocation_size);
  }

  const iree_hal_pool_reservation_request_t large_request =
      iree_hal_make_pool_reservation_request(2u * 1024u * 1024u, 0);
  ++timeline;
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      runtime_.transfer_queue, iree_hal_semaphore_list_empty(), completion,
      runtime_.device_pool, 1, &large_request, &buffers_[3]));
  IREE_ASSERT_OK(loom_run_hal_semaphore_wait(semaphore_, timeline,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_pool_query_stats(runtime_.device_pool, &stats);
  EXPECT_EQ(stats.reservation_count, 4u);
  EXPECT_EQ(stats.slab_count, 2u);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffers_[3]),
            large_request.allocation_size);

  ++timeline;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(
      runtime_.transfer_queue, iree_hal_semaphore_list_empty(), completion,
      IREE_ARRAYSIZE(buffers_), buffers_));
  IREE_ASSERT_OK(loom_run_hal_semaphore_wait(semaphore_, timeline,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
  iree_hal_pool_query_stats(runtime_.device_pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
}

}  // namespace
}  // namespace loom
