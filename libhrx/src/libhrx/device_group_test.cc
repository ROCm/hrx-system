// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"

namespace {

testing::AssertionResult Check(hrx_status_t status) {
  auto result = hrx_status_is_ok(status) ? testing::AssertionSuccess()
                                         : testing::AssertionFailure();
  if (!hrx_status_is_ok(status)) {
    result << hrx::runtime::format_status(status);
  }
  hrx_status_ignore(status);
  return result;
}

TEST(DeviceGroupTest, RetainedDeviceKeepsGroupAndServicesUsable) {
  hrx::runtime::runtime_ptr runtime;
  ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                       runtime.for_output())));
  ASSERT_TRUE(runtime);
  const hrx_device_create_params_t params[] = {{.uri = "task://"},
                                               {.uri = "task://"}};
  hrx::runtime::device_group_ptr group;
  ASSERT_TRUE(
      Check(hrx_device_group_create(runtime, 2, params, group.for_output())));
  ASSERT_TRUE(group);
  EXPECT_EQ(hrx_device_group_device_count(group), 2u);
  EXPECT_EQ(hrx_device_group_get_device(group, 2), nullptr);
  auto first =
      hrx::runtime::device_ptr::borrow(hrx_device_group_get_device(group, 0));
  auto second =
      hrx::runtime::device_ptr::borrow(hrx_device_group_get_device(group, 1));
  ASSERT_NE(first.get(), second.get());
  runtime.reset();
  group.reset();

  hrx::runtime::semaphore_ptr semaphore;
  ASSERT_TRUE(Check(hrx_semaphore_create(first, 0, semaphore.for_output())));
  ASSERT_TRUE(semaphore);
  hrx::runtime::buffer_ptr buffer;
  const hrx_buffer_params_t buffer_params = {
      .type = HRX_MEMORY_TYPE_HOST_LOCAL,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_DEFAULT,
  };
  ASSERT_TRUE(Check(hrx_allocator_allocate_buffer(
      hrx_device_allocator(second), buffer_params, 256, buffer.for_output())));
  ASSERT_TRUE(buffer);
  uint64_t value = 1;
  hrx_semaphore_t signal = semaphore;
  const hrx_semaphore_list_t signals = {&signal, &value, 1};
  const uint32_t pattern = 0x12345678;
  ASSERT_TRUE(Check(hrx_queue_fill(hrx_device_get_queue(second, 0, 0), nullptr,
                                   &signals, buffer, 0, 256, &pattern,
                                   sizeof(pattern))));
  ASSERT_TRUE(Check(hrx_semaphore_wait(semaphore, value, UINT64_MAX)));
  second.reset();
  first.reset();

  void* data = nullptr;
  ASSERT_TRUE(Check(hrx_buffer_map(buffer, HRX_MAP_READ, 0, 256, &data)));
  ASSERT_NE(data, nullptr);
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(static_cast<uint32_t*>(data)[i], pattern);
  }
  ASSERT_TRUE(Check(hrx_buffer_unmap(buffer)));
}

TEST(DeviceGroupTest, FailedConstructionPreservesExistingGroup) {
  hrx::runtime::runtime_ptr runtime;
  ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                       runtime.for_output())));
  ASSERT_TRUE(runtime);
  hrx::runtime::device_group_ptr kept;
  const hrx_device_create_params_t task = {.uri = "task://"};
  ASSERT_TRUE(
      Check(hrx_device_group_create(runtime, 1, &task, kept.for_output())));
  ASSERT_TRUE(kept);

  const hrx_device_create_params_t invalid[] = {task,
                                                {.uri = "missing-driver://"}};
  hrx_device_group_t rejected = nullptr;
  hrx_status_t status = hrx_device_group_create(runtime, 2, invalid, &rejected);
  EXPECT_FALSE(hrx_status_is_ok(status));
  hrx_status_ignore(status);
  EXPECT_EQ(rejected, nullptr);

  hrx::runtime::semaphore_ptr semaphore;
  ASSERT_TRUE(Check(hrx_semaphore_create(hrx_device_group_get_device(kept, 0),
                                         0, semaphore.for_output())));
  ASSERT_TRUE(semaphore);
  ASSERT_TRUE(Check(hrx_semaphore_signal(semaphore, 17)));
  ASSERT_TRUE(Check(hrx_semaphore_wait(semaphore, 17, UINT64_MAX)));
}

TEST(DeviceGroupTest, RejectsInvalidInputsBeforeActivation) {
  hrx::runtime::runtime_ptr runtime;
  ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                       runtime.for_output())));
  ASSERT_TRUE(runtime);
  hrx_device_group_t group = nullptr;
  const hrx_device_create_params_t invalid = {
      .uri = "task://",
      .completion_wait_flags = UINT32_MAX,
  };
  hrx_status_t status = hrx_device_group_create(runtime, 1, &invalid, &group);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
  hrx_status_ignore(status);
  EXPECT_EQ(group, nullptr);
  status = hrx_device_group_create(runtime, 0, nullptr, &group);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
  hrx_status_ignore(status);
  EXPECT_EQ(group, nullptr);
}

}  // namespace
