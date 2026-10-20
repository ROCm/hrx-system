// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/scheduler.h"

namespace {

class CpuDeviceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(loom_serve_host_create(GetParam(), nullptr,
                                          iree_allocator_system(), &host_));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
    loom_serve_host_release(host_);
  }

  // Shared workers retained through all device lifetimes.
  loom_serve_host_t* host_ = nullptr;
  // Nullable group also covering partial initialization failure.
  loom_serve_device_group_t* group_ = nullptr;
};

TEST_P(CpuDeviceTest, DevicesUseApplicationWorkerBudgetAndOneGroup) {
  const loom_serve_device_options_t options[] = {
      {.driver = loom_serve_host_task_driver(host_), .uri = IREE_SV("task")},
      {.driver = loom_serve_host_task_driver(host_), .uri = IREE_SV("task")}};
  IREE_ASSERT_OK(loom_serve_device_group_allocate(
      host_, IREE_ARRAYSIZE(options), options, nullptr, iree_allocator_system(),
      &group_));
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
      {.driver = loom_serve_host_task_driver(host_), .uri = IREE_SV("task")},
      {
          .driver = loom_serve_host_task_driver(host_),
          .uri = IREE_SV("task://missing"),
      }};
  iree::Status status = loom_serve_device_group_allocate(
      host_, IREE_ARRAYSIZE(options), options, nullptr, iree_allocator_system(),
      &group_);
  EXPECT_EQ(status.code(), iree::StatusCode::kNotFound);
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("creating serving device 1 (task://missing)"));
  EXPECT_EQ(group_, nullptr);
  IREE_ASSERT_OK(loom_serve_device_group_allocate(
      host_, 1, options, nullptr, iree_allocator_system(), &group_));
  EXPECT_EQ(loom_serve_device_group_count(group_), 1u);
}

TEST_P(CpuDeviceTest, DistinctTaskDevicesClaimTheirActualSharedExecutor) {
  const loom_serve_device_options_t options[] = {
      {.driver = loom_serve_host_task_driver(host_), .uri = IREE_SV("task")},
      {.driver = loom_serve_host_task_driver(host_), .uri = IREE_SV("task")}};
  const uint32_t limits[] = {1, 0, 0};
  const loom_serve_execution_scope_t scopes[] = {
      {1, LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE},
      {2, LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE},
      {4, LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE}};
  const loom_serve_execution_policy_t policy = {
      .max_active = limits,
      .scope_count = IREE_ARRAYSIZE(scopes),
      .scopes = scopes,
  };
  IREE_ASSERT_OK(loom_serve_device_group_allocate(
      host_, IREE_ARRAYSIZE(options), options, &policy, iree_allocator_system(),
      &group_));
  auto* first = loom_serve_device_group_at(group_, 0);
  auto* second = loom_serve_device_group_at(group_, 1);
  EXPECT_EQ(loom_serve_device_scheduler(first),
            loom_serve_device_scheduler(second));
  EXPECT_EQ(loom_serve_device_execution_resources(first), 3u);
  EXPECT_EQ(loom_serve_device_execution_resources(second), 5u);
  loom_serve_execution_permit_t first_work, second_work;
  loom_serve_execution_permit_initialize(
      loom_serve_device_scheduler(first),
      loom_serve_device_execution_resources(first), &first_work);
  loom_serve_execution_permit_initialize(
      loom_serve_device_scheduler(second),
      loom_serve_device_execution_resources(second), &second_work);
  loom_serve_execution_permit_request(&first_work,
                                      LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
  loom_serve_execution_permit_request(&second_work,
                                      LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(&first_work));
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(&second_work));
  loom_serve_execution_permit_complete(&first_work);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(&second_work));
  loom_serve_execution_permit_complete(&second_work);
  loom_serve_execution_permit_deinitialize(&first_work);
  loom_serve_execution_permit_deinitialize(&second_work);
}

INSTANTIATE_TEST_SUITE_P(Workers, CpuDeviceTest, ::testing::Values(1, 4));

}  // namespace
