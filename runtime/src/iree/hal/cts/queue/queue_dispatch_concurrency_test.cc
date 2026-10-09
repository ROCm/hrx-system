// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for exact-queue dispatch concurrency queries.

#include <cstdint>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class QueueDispatchConcurrencyTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }

    dispatch_queue_ =
        QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
    if (!dispatch_queue_) {
      GTEST_SKIP() << "device has no provisioned dispatch-capable queue";
    }
    LoadExecutableOrSkipUnsupported("command_buffer_dispatch_test.bin",
                                    &executable_);
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_hal_queue_t* dispatch_queue_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(QueueDispatchConcurrencyTest, ReportsExactQueueResidency) {
  iree_hal_queue_dispatch_concurrency_params_t params = {
      .workgroup_size = {1, 1, 1},
      .dynamic_workgroup_local_memory = 0,
  };
  iree_hal_queue_dispatch_concurrency_t concurrency;
  iree_status_t status = iree_hal_queue_query_dispatch_concurrency(
      dispatch_queue_, executable_, iree_hal_executable_function_from_index(0),
      params, IREE_HAL_QUEUE_DISPATCH_CONCURRENCY_FLAG_NONE, &concurrency);
  if (iree_status_code(status) == IREE_STATUS_UNIMPLEMENTED) {
    iree_status_free(status);
    GTEST_SKIP() << "queue cannot calculate exact dispatch concurrency";
  }
  IREE_ASSERT_OK(status);

  EXPECT_GT(concurrency.scheduling_domain_count, 0u);
  EXPECT_GT(concurrency.maximum_concurrent_workgroup_count_per_domain, 0u);
  EXPECT_GT(
      iree_hal_queue_dispatch_concurrency_total_workgroup_count(concurrency),
      0u);

  params.dynamic_workgroup_local_memory = UINT32_MAX;
  IREE_ASSERT_OK(iree_hal_queue_query_dispatch_concurrency(
      dispatch_queue_, executable_, iree_hal_executable_function_from_index(0),
      params, IREE_HAL_QUEUE_DISPATCH_CONCURRENCY_FLAG_NONE, &concurrency));
  EXPECT_EQ(concurrency.maximum_concurrent_workgroup_count_per_domain, 0u);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(QueueDispatchConcurrencyTest);

}  // namespace iree::hal::cts
