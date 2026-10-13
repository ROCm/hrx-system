// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"

namespace {

class CpuDeviceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        loom_serve_host_create(GetParam(), &host_, iree_allocator_system()));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group_));
    loom_serve_host_release(host_);
  }

  // Shared workers retained through all device lifetimes.
  loom_serve_host_t* host_ = nullptr;
  // Nullable group also covering partial initialization failure.
  loom_serve_device_group_t* group_ = nullptr;
};

TEST_P(CpuDeviceTest, DevicesUseApplicationWorkerBudgetAndOneGroup) {
  const loom_serve_device_options_t options[] = {{.uri = IREE_SV("task")},
                                                 {.uri = IREE_SV("task")}};
  IREE_ASSERT_OK(loom_serve_device_group_create(host_, IREE_ARRAYSIZE(options),
                                                options, &group_,
                                                iree_allocator_system()));
  EXPECT_EQ(loom_serve_device_group_count(group_), 2u);
  auto* first = loom_serve_device_group_at(group_, 0);
  auto* second = loom_serve_device_group_at(group_, 1);
  EXPECT_EQ(loom_serve_device_group(first), loom_serve_device_group(second));
  EXPECT_NE(loom_serve_device_handle(first), loom_serve_device_handle(second));
  EXPECT_NE(loom_serve_device_dispatch_queue(first),
            loom_serve_device_dispatch_queue(second));
  for (auto* device : {first, second}) {
    EXPECT_EQ(loom_serve_device_host(device), host_);
    const auto* dispatch = iree_hal_device_spec_dispatch(
        iree_hal_device_spec(loom_serve_device_handle(device)));
    EXPECT_EQ(dispatch->execution.unit_count,
              iree_task_executor_worker_count(loom_serve_host_executor(host_)));
  }
}

TEST_P(CpuDeviceTest, LaterUriFailureReleasesBorrowedWorkers) {
  const loom_serve_device_options_t options[] = {
      {.uri = IREE_SV("task")}, {.uri = IREE_SV("task://missing")}};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_device_group_create(host_, IREE_ARRAYSIZE(options), options,
                                     &group_, iree_allocator_system()));
  EXPECT_EQ(group_, nullptr);
  IREE_ASSERT_OK(loom_serve_device_group_create(host_, 1, options, &group_,
                                                iree_allocator_system()));
  EXPECT_EQ(loom_serve_device_group_count(group_), 1u);
}

INSTANTIATE_TEST_SUITE_P(Workers, CpuDeviceTest, ::testing::Values(1, 4));

}  // namespace
