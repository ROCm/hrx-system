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
  char first_name[] = "first";
  const loom_serve_device_options_t options[] = {
      {
          .driver = loom_serve_host_task_driver(host_),
          .uri = IREE_SV("task"),
          .name = iree_make_cstring_view(first_name),
          .completion_wait_flags = IREE_ASYNC_WAIT_FLAG_ACTIVE,
      },
      {
          .driver = loom_serve_host_task_driver(host_),
          .uri = IREE_SV("task"),
          .name = IREE_SV("second"),
          .completion_wait_flags = IREE_ASYNC_WAIT_FLAG_NONE,
      }};
  IREE_ASSERT_OK(
      loom_serve_device_group_allocate(host_, IREE_ARRAYSIZE(options), options,
                                       iree_allocator_system(), &group_));
  EXPECT_EQ(loom_serve_device_group_count(group_), 2u);
  EXPECT_EQ(loom_serve_device_group_pool_count(group_), 0u);
  first_name[0] = 'x';
  const auto names = loom_serve_device_group_names(group_);
  ASSERT_EQ(names.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(names.values[0], IREE_SV("first")));
  EXPECT_TRUE(iree_string_view_equal(names.values[1], IREE_SV("second")));
  auto* first = loom_serve_device_group_at(group_, 0);
  auto* second = loom_serve_device_group_at(group_, 1);
  EXPECT_TRUE(
      iree_string_view_equal(loom_serve_device_name(first), names.values[0]));
  EXPECT_TRUE(
      iree_string_view_equal(loom_serve_device_name(second), names.values[1]));
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

TEST_P(CpuDeviceTest, CompletionPolicyReachesHalCreationAndFailureCleanup) {
  const loom_serve_device_options_t options[] = {
      {
          .driver = loom_serve_host_task_driver(host_),
          .uri = IREE_SV("task"),
          .name = IREE_SV("first"),
          .completion_wait_flags = IREE_ASYNC_WAIT_FLAG_ACTIVE,
      },
      {
          .driver = loom_serve_host_task_driver(host_),
          .uri = IREE_SV("task"),
          .name = IREE_SV("second"),
          .completion_wait_flags = 1u << 31,
      }};
  iree::Status status =
      loom_serve_device_group_allocate(host_, IREE_ARRAYSIZE(options), options,
                                       iree_allocator_system(), &group_);
  EXPECT_EQ(status.code(), iree::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("unknown completion wait flags"));
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("creating serving device 1"));
  EXPECT_EQ(group_, nullptr);
  IREE_ASSERT_OK(loom_serve_device_group_allocate(
      host_, 1, options, iree_allocator_system(), &group_));
}

TEST_P(CpuDeviceTest, LaterUriFailureReleasesBorrowedWorkers) {
  const loom_serve_device_options_t options[] = {
      {
          .driver = loom_serve_host_task_driver(host_),
          .uri = IREE_SV("task"),
          .name = IREE_SV("first"),
      },
      {
          .driver = loom_serve_host_task_driver(host_),
          .uri = IREE_SV("task://missing"),
          .name = IREE_SV("second"),
      }};
  iree::Status status =
      loom_serve_device_group_allocate(host_, IREE_ARRAYSIZE(options), options,
                                       iree_allocator_system(), &group_);
  EXPECT_EQ(status.code(), iree::StatusCode::kNotFound);
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr("creating serving device 1 (task://missing)"));
  EXPECT_EQ(group_, nullptr);
  IREE_ASSERT_OK(loom_serve_device_group_allocate(
      host_, 1, options, iree_allocator_system(), &group_));
  EXPECT_EQ(loom_serve_device_group_count(group_), 1u);
}

TEST_P(CpuDeviceTest, PoolIdentityAndMembershipAreIndependentOfDevices) {
  const loom_serve_device_options_t options = {
      .driver = loom_serve_host_task_driver(host_),
      .uri = IREE_SV("task"),
  };
  IREE_ASSERT_OK(loom_serve_device_group_allocate(
      host_, 1, &options, iree_allocator_system(), &group_));
  loom_serve_pool_options_t pool_options = {};
  const loom_serve_pool_binding_t invalid[] = {{IREE_SV(""), 1},
                                               {IREE_SV("bad name"), 1},
                                               {IREE_SV("pool"), 0},
                                               {IREE_SV("pool"), 2}};
  loom_serve_pool_t* pool = nullptr;
  for (auto binding : invalid) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_device_group_create_pool(
                              group_, binding, &pool_options, &pool));
    EXPECT_EQ(pool, nullptr);
    EXPECT_EQ(loom_serve_device_group_pool_count(group_), 0u);
  }
  pool_options.memory_limit = 4096;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_device_group_create_pool(group_, {IREE_SV("pool"), 1},
                                          &pool_options, &pool));
  EXPECT_EQ(pool, nullptr);
  pool_options.memory_limit = 0;
  char name[] = "agents";
  IREE_ASSERT_OK(loom_serve_device_group_create_pool(
      group_, {iree_make_cstring_view(name), 1}, &pool_options, &pool));
  name[0] = 'x';
  const auto* bindings = loom_serve_device_group_pool_bindings(group_);
  EXPECT_TRUE(iree_string_view_equal(bindings[0].name, IREE_SV("agents")));
  EXPECT_EQ(bindings[0].device_mask, 1u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_serve_device_group_create_pool(group_, {IREE_SV("agents"), 1},
                                          &pool_options, &pool));
  IREE_ASSERT_OK(loom_serve_device_group_create_pool(
      group_, {IREE_SV("images"), 1}, &pool_options, &pool));
  EXPECT_EQ(loom_serve_device_group_pool_count(group_), 2u);
  EXPECT_NE(loom_serve_device_group_pool_at(group_, 0), pool);
}

TEST_P(CpuDeviceTest, RequiredPlacementCannotFallBackToUnplacedHostAllocation) {
  const loom_serve_device_options_t options = {
      .driver = loom_serve_host_task_driver(host_),
      .uri = IREE_SV("task"),
  };
  IREE_ASSERT_OK(loom_serve_device_group_allocate(
      host_, 1, &options, iree_allocator_system(), &group_));
  loom_serve_pool_options_t pool_options = {};
  loom_serve_pool_t* pool = nullptr;
  pool_options.placement = {LOOM_SERVE_POOL_PLACEMENT_DEVICE, 1};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_device_group_create_pool(group_, {IREE_SV("outside"), 1},
                                          &pool_options, &pool));
  pool_options.placement.ordinal = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_serve_device_group_create_pool(group_, {IREE_SV("required"), 1},
                                          &pool_options, &pool));
  pool_options.placement = {LOOM_SERVE_POOL_PLACEMENT_HOST_NUMA, UINT32_MAX};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_serve_device_group_create_pool(group_, {IREE_SV("missing-node"), 1},
                                          &pool_options, &pool));
  EXPECT_EQ(pool, nullptr);
  EXPECT_EQ(loom_serve_device_group_pool_count(group_), 0u);
  pool_options.placement = {};
  IREE_ASSERT_OK(loom_serve_device_group_create_pool(
      group_, {IREE_SV("automatic"), 1}, &pool_options, &pool));
  EXPECT_NE(pool, nullptr);
}

INSTANTIATE_TEST_SUITE_P(Workers, CpuDeviceTest, ::testing::Values(1, 4));

}  // namespace
