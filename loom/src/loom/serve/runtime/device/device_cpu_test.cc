// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"

namespace {

class CpuDeviceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override {
    iree_notification_initialize(&notification_);
    IREE_ASSERT_OK(loom_serve_host_create(GetParam(), nullptr,
                                          iree_allocator_system(), &host_));
  }

  void TearDown() override {
    for (auto& work : work_) {
      iree_hal_buffer_release(work.buffer);
      work.buffer = nullptr;
    }
    AdvanceUntilRetired();
    for (auto& work : work_) {
      if (work.execution) {
        IREE_EXPECT_OK(loom_serve_execution_drain(work.execution));
        loom_serve_execution_set_retirement_observer(work.execution, nullptr,
                                                     nullptr);
        loom_serve_execution_release(work.execution);
      }
    }
    loom_serve_device_group_set_observer(group_, {});
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
    loom_serve_host_release(host_);
    iree_notification_deinitialize(&notification_);
  }

  static void Wake(void* self) {
    auto* test = static_cast<CpuDeviceTest*>(self);
    iree_notification_post(&test->notification_, IREE_ALL_WAITERS);
  }

  void AdvanceUntilRetired() {
    while (true) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      auto status = loom_serve_device_group_advance(group_);
      bool pending = loom_serve_device_group_is_pending(group_);
      for (auto& work : work_) {
        pending |=
            work.execution && !loom_serve_execution_is_retired(work.execution);
      }
      if (!pending || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
      if (!pending) {
        break;
      }
    }
  }

  // Shared workers retained through all device lifetimes.
  loom_serve_host_t* host_ = nullptr;
  // Nullable group also covering partial initialization failure.
  loom_serve_device_group_t* group_ = nullptr;
  // All native producers wake this outer owner; no HTTP notification is used.
  iree_notification_t notification_;
  // Independent work sharing the group but not physical pool admission.
  struct Work {
    // Owned execution, released after its accepted consumers retire.
    loom_serve_execution_t* execution = nullptr;
    // Caller allocation view, relinquished after the download captures it.
    iree_hal_buffer_t* buffer = nullptr;
    // Host download destination, alive through accepted queue retirement.
    std::array<uint32_t, 4> output = {};
  } work_[2];
};

TEST_P(CpuDeviceTest, GroupProgressRetiresIndependentElasticPools) {
  const loom_serve_device_options_t options = {
      .driver = loom_serve_host_task_driver(host_),
      .uri = IREE_SV("task"),
  };
  IREE_ASSERT_OK(loom_serve_device_group_allocate(
      host_, 1, &options, iree_allocator_system(), &group_));
  auto* device = loom_serve_device_group_at(group_, 0);
  constexpr uint64_t kSlabSize = 2 * 1024 * 1024;
  const loom_serve_pool_options_t pool_options = {
      .elastic = true,
      .weight_slab_size = kSlabSize,
      .state_slab_size = kSlabSize / 4,
      .memory_limit = 2 * kSlabSize,
  };
  for (const auto name : {IREE_SV("first"), IREE_SV("second")}) {
    loom_serve_pool_t* pool = nullptr;
    IREE_ASSERT_OK(loom_serve_device_group_create_pool(group_, {name, 1},
                                                       &pool_options, &pool));
  }
  loom_serve_device_group_set_observer(group_, {Wake, this});
  for (size_t i = 0; i < IREE_ARRAYSIZE(work_); ++i) {
    auto& work = work_[i];
    IREE_ASSERT_OK(loom_serve_device_create_execution(
        device, loom_serve_device_group_pool_at(group_, i), &work.execution));
    loom_serve_execution_set_retirement_observer(work.execution, Wake, this);
    const auto request =
        iree_hal_make_pool_reservation_request(sizeof(work.output), 16);
    IREE_ASSERT_OK(
        loom_serve_execution_alloca(work.execution, 1, &request, &work.buffer));
    const uint32_t pattern = 0x12340000u + i;
    const iree_hal_transfer_operation_t fill = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill =
            {
                .target_buffer = work.buffer,
                .length = sizeof(work.output),
                .pattern = &pattern,
                .pattern_length = sizeof(pattern),
            },
    };
    uint64_t value = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(work.execution, 1, &fill, &value));
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = work.buffer,
                .target = work.output.data(),
                .length = sizeof(work.output),
            },
    };
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(work.execution, 1, &download, &value));
    iree_hal_buffer_release(work.buffer);
    work.buffer = nullptr;
  }

  ASSERT_NO_FATAL_FAILURE(AdvanceUntilRetired());
  for (size_t i = 0; i < IREE_ARRAYSIZE(work_); ++i) {
    IREE_ASSERT_OK(loom_serve_execution_drain(work_[i].execution));
    for (const auto value : work_[i].output) {
      EXPECT_EQ(value, 0x12340000u + i);
    }
  }
}

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
