// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "runtime.h"

#include <memory>

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

TEST(RuntimeTest, DiscoveryOwnsMetadataWithoutActivatingExecutionServices) {
  hrx::runtime::runtime_ptr runtime;
  ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                       runtime.for_output())));
  const auto free_infos = [](hrx_device_info_t* infos) {
    hrx_host_allocator_free(hrx_host_allocator_system(), infos);
  };
  size_t count = 0;
  hrx_device_info_t* raw_infos = nullptr;
  ASSERT_TRUE(Check(hrx_runtime_query_available_devices(
      runtime, "task", hrx_host_allocator_system(), &count, &raw_infos)));
  std::unique_ptr<hrx_device_info_t, decltype(free_infos)> infos(raw_infos,
                                                                 free_infos);
  ASSERT_EQ(count, 1u);
  ASSERT_NE(infos, nullptr);
  EXPECT_EQ(runtime.get()->task_driver, nullptr);
  runtime.reset();
  EXPECT_STREQ(infos->uri, "task://");
  EXPECT_STREQ(infos->name, "default");

  // The returned URI survives the discovery owner and constructs a real
  // device in a new runtime. The group does not borrow the inventory strings.
  ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                       runtime.for_output())));
  const hrx_device_create_params_t params = {.uri = infos->uri};
  hrx::runtime::device_group_ptr group;
  ASSERT_TRUE(
      Check(hrx_device_group_create(runtime, 1, &params, group.for_output())));
  infos.reset();
  runtime.reset();
  hrx_device_t device = hrx_device_group_get_device(group, 0);
  ASSERT_NE(device, nullptr);
  hrx::runtime::semaphore_ptr completed;
  ASSERT_TRUE(Check(hrx_semaphore_create(device, 0, completed.for_output())));
  hrx_semaphore_t signal = completed;
  uint64_t value = 1;
  const hrx_semaphore_list_t signals = {&signal, &value, 1};
  ASSERT_TRUE(Check(hrx_queue_barrier(hrx_device_get_queue(device, 0, 0),
                                      nullptr, &signals)));
  ASSERT_TRUE(Check(hrx_semaphore_wait(completed, value, UINT64_MAX)));
}

TEST(RuntimeTest, DiscoveryRejectsInvalidInputsWithoutReturningMetadata) {
  hrx::runtime::runtime_ptr runtime;
  ASSERT_TRUE(Check(hrx_runtime_create(nullptr, hrx_host_allocator_system(),
                                       runtime.for_output())));
  const char* names[] = {nullptr, ""};
  for (const char* name : names) {
    hrx_device_info_t placeholder = {};
    hrx_device_info_t* infos = &placeholder;
    size_t count = 1;
    const hrx_status_t status = hrx_runtime_query_available_devices(
        runtime, name, hrx_host_allocator_system(), &count, &infos);
    EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
    hrx_status_ignore(status);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(infos, nullptr);
  }
  size_t count = 0;
  hrx_device_info_t* infos = nullptr;
  hrx_status_t status = hrx_runtime_query_available_devices(
      nullptr, "task", hrx_host_allocator_system(), &count, &infos);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
  hrx_status_ignore(status);
  status =
      hrx_runtime_query_available_devices(runtime, "task", {}, &count, &infos);
  EXPECT_EQ(hrx_status_code(status), HRX_STATUS_INVALID_ARGUMENT);
  hrx_status_ignore(status);
  EXPECT_EQ(count, 0u);
  EXPECT_EQ(infos, nullptr);
}

}  // namespace
