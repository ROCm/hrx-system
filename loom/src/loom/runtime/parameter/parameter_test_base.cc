// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/parameter/parameter_test_base.h"

#include <array>

#include "iree/testing/status_matchers.h"
#include "loom/runtime/testing/device.h"

namespace loom_runtime::testing {
namespace {

class ParameterTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    for (auto& application : applications_) {
      loom_runtime_host_t* host = nullptr;
      iree_hal_driver_t* driver = nullptr;
      auto status = loom_runtime_host_create(application.worker_count, nullptr,
                                             iree_allocator_system(), &host);
      if (iree_status_is_ok(status)) {
        status = loom_runtime_test_driver_acquire(host, &driver);
      }
      if (iree_status_is_ok(status)) {
        const loom_runtime_device_options_t options = {
            .driver = driver,
            .uri = loom_runtime_test_device()->device_uri,
        };
        status = loom_runtime_device_group_allocate(
            host, 1, &options, iree_allocator_system(), &application.group);
      }
      iree_hal_driver_release(driver);
      loom_runtime_host_release(host);
      IREE_ASSERT_OK(status);
      const loom_runtime_pool_options_t options = {};
      loom_runtime_pool_t* pool = nullptr;
      IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
          application.group, {IREE_SV("pool"), 1}, &options, &pool));
    }
  }

  void TearDown() override {
    for (auto& application : applications_) {
      IREE_CHECK_OK(loom_runtime_device_group_free(application.group));
      application.group = nullptr;
    }
  }

  loom_runtime_device_group_t* group(uint32_t worker_count) {
    for (auto& application : applications_) {
      if (application.worker_count == worker_count) {
        return application.group;
      }
    }
    return nullptr;
  }

 private:
  struct Application {
    // Executor size under test, independent of file and transfer parameters.
    uint32_t worker_count;
    // Sole physical owner for this executor configuration.
    loom_runtime_device_group_t* group = nullptr;
  };
  // Both deliberate executor configurations remain alive across all cases.
  std::array<Application, 2> applications_ = {Application{1}, Application{4}};
};

auto* const environment = static_cast<ParameterTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new ParameterTestEnvironment));

}  // namespace

void ParameterTest::SetUp() {
  auto* group = environment->group(GetParam());
  ASSERT_NE(group, nullptr);
  owner_ = loom_runtime_device_group_at(group, 0);
  storage_owner_ = loom_runtime_device_group_pool_at(group, 0);
  device_ = loom_runtime_device_handle(owner_);
  queue_ = loom_runtime_device_transfer_queue(owner_);
}

void ParameterTest::TearDown() {
  if (!storage_owner_) {
    return;
  }
  IREE_ASSERT_OK(loom_runtime_test_pool_trim_and_drain(storage_owner_, 0));
  EXPECT_EQ(loom_runtime_pool_statistics(storage_owner_).storage.bytes_reserved,
            0u);
  EXPECT_FALSE(loom_runtime_pool_is_in_use(storage_owner_));
}

}  // namespace loom_runtime::testing
