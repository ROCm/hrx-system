// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <utility>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class DriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(context_.Initialize(iree_allocator_system()));
    iree_hal_amdgpu_driver_options_t options;
    iree_hal_amdgpu_driver_options_initialize(&options);
    IREE_ASSERT_OK(iree_hal_amdgpu_driver_create(
        IREE_SV("legacy-amdgpu"), &options, iree_allocator_system(), &driver_));
  }

  void TearDown() override {
    iree_hal_device_release(device_);
    iree_hal_device_group_release(group_);
    iree_hal_driver_release(driver_);
  }

  void CheckQueues(uint32_t expected_count) {
    IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
        device_, context_.frontier_tracker(), iree_allocator_system(),
        &group_));
    const auto* queues =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    ASSERT_GT(queues->family_count, 0u);
    for (iree_host_size_t family = 0; family < queues->family_count; ++family) {
      ASSERT_EQ(queues->families[family].provisioned_queue_count,
                expected_count);
      EXPECT_EQ(iree_hal_device_queue(device_, family, expected_count),
                nullptr);
      for (uint32_t ordinal = 0; ordinal < expected_count; ++ordinal) {
        auto* queue = iree_hal_device_queue(device_, family, ordinal);
        ASSERT_NE(queue, nullptr);
        iree::hal::cts::SemaphoreList signals(device_, {0}, {1});
        IREE_ASSERT_OK(iree_hal_queue_barrier(
            queue, iree_hal_semaphore_list_empty(), signals, nullptr,
            IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
        IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
            signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      }
    }
  }

  // Async services and frontier namespace outliving every created device.
  iree::hal::cts::DeviceCreateContext context_;
  // Driver shared by the test's construction calls.
  iree_hal_driver_t* driver_ = nullptr;
  // Created device, released before its topology-owning group.
  iree_hal_device_t* device_ = nullptr;
  // Sealed topology required for queue submissions.
  iree_hal_device_group_t* group_ = nullptr;
};

TEST_F(DriverTest, DefaultConstructionKeepsOneQueue) {
  IREE_ASSERT_OK(iree_hal_driver_create_default_device(
      driver_, context_.params(), iree_allocator_system(), &device_));
  ASSERT_NO_FATAL_FAILURE(CheckQueues(1));
}

TEST_F(DriverTest, UriProvisionsRequestedQueues) {
  IREE_ASSERT_OK(iree_hal_driver_create_device_by_uri(
      driver_, IREE_SV("legacy-amdgpu://0?queue_count=0002"), context_.params(),
      iree_allocator_system(), &device_));
  ASSERT_NO_FATAL_FAILURE(CheckQueues(2));
}

TEST_F(DriverTest, DeviceIdProvisionsRequestedQueues) {
  const auto parameter = iree_make_cstring_pair("queue_count", "3");
  IREE_ASSERT_OK(iree_hal_driver_create_device_by_id(
      driver_, IREE_HAL_DEVICE_ID_DEFAULT, 1, &parameter, context_.params(),
      iree_allocator_system(), &device_));
  ASSERT_NO_FATAL_FAILURE(CheckQueues(3));
}

TEST_F(DriverTest, InvalidCountsLeaveNoDevice) {
  const std::pair<const char*, iree_status_code_t> cases[] = {
      {"", IREE_STATUS_OUT_OF_RANGE},
      {"0", IREE_STATUS_OUT_OF_RANGE},
      {"257", IREE_STATUS_OUT_OF_RANGE},
      {"4294967297", IREE_STATUS_OUT_OF_RANGE},
      {"18446744073709551617", IREE_STATUS_OUT_OF_RANGE},
      {"-1", IREE_STATUS_INVALID_ARGUMENT},
      {"2x", IREE_STATUS_INVALID_ARGUMENT},
      {"2.5", IREE_STATUS_INVALID_ARGUMENT},
  };
  for (const auto& [value, expected_status] : cases) {
    SCOPED_TRACE(value);
    std::string uri = "legacy-amdgpu://0?queue_count=";
    uri.append(value);
    IREE_EXPECT_STATUS_IS(
        expected_status,
        iree_hal_driver_create_device_by_uri(
            driver_, iree_make_cstring_view(uri.c_str()), context_.params(),
            iree_allocator_system(), &device_));
    EXPECT_EQ(device_, nullptr);
  }
}

TEST_F(DriverTest, QueueCountDoesNotHideUnknownParameters) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_driver_create_device_by_uri(
          driver_, IREE_SV("legacy-amdgpu://0?queue_count=2&unknown=value"),
          context_.params(), iree_allocator_system(), &device_));
  EXPECT_EQ(device_, nullptr);
}

}  // namespace
}  // namespace iree::hal::amdgpu
