// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/testing/application.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/testing/compiler.h"
#include "loom/runtime/testing/device.h"
#include "loom/runtime/testing/native_allocator.h"

struct loom_runtime_test_application_t {
  // Exact configuration selected by the first fixture.
  loom_runtime_test_application_options_t options;
  // Shared task, compiler, and I/O services.
  loom_runtime_host_t* host = nullptr;
  // Native allocator dependency retained beyond complete device teardown.
  std::unique_ptr<loom_runtime::testing::NativeAllocator> native_allocator;
  // Single physical application owner for the entire test process.
  loom_runtime_device_group_t* group = nullptr;
};

namespace {

loom_runtime_test_application_t& GetTestApplication() {
  static loom_runtime_test_application_t application;
  return application;
}

bool OptionsEqual(const loom_runtime_test_application_options_t& lhs,
                  const loom_runtime_test_application_options_t& rhs) {
  return lhs.worker_count == rhs.worker_count &&
         lhs.device_count == rhs.device_count && lhs.backing == rhs.backing &&
         lhs.slab_size == rhs.slab_size && lhs.memory_limit == rhs.memory_limit;
}

class ServiceTestEnvironment : public ::testing::Environment {
 public:
  void TearDown() override {
    auto& application = GetTestApplication();
    iree_status_t status = loom_runtime_device_group_free(application.group);
    if (application.native_allocator) {
      application.native_allocator->AwaitReleased();
    }
    loom_runtime_host_release(application.host);
    application = {};
    IREE_EXPECT_OK(status);
  }
};

::testing::Environment* const service_test_environment =
    ::testing::AddGlobalTestEnvironment(new ServiceTestEnvironment);

}  // namespace

loom_runtime_test_application_options_t
loom_runtime_test_application_options_default(void) {
  return {
      .worker_count = 1,
      .device_count = 1,
      .backing = loom_runtime_test_device()->service_backing,
      .slab_size = 2 * 1024 * 1024,
  };
}

iree_status_t loom_runtime_test_application_acquire(
    const loom_runtime_test_application_options_t* options,
    iree_allocator_t host_allocator,
    loom_runtime_test_application_t** out_application) {
  *out_application = nullptr;
  if (options->worker_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "service test application needs a worker");
  }
  if (options->device_count == 0 ||
      options->device_count > IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "service test application needs between 1 and %d "
                            "devices",
                            IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT);
  }

  auto& application = GetTestApplication();
  if (application.group) {
    if (!OptionsEqual(application.options, *options)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "source service test process requested a second application "
          "configuration");
    }
    *out_application = &application;
    return iree_ok_status();
  }

  loom_runtime_host_t* host = nullptr;
  IREE_RETURN_IF_ERROR(loom_runtime_test_host_create(options->worker_count,
                                                     host_allocator, &host));
  iree_hal_driver_t* driver = nullptr;
  iree_status_t status = loom_runtime_test_driver_acquire(host, &driver);
  loom_runtime_device_group_t* group = nullptr;
  const iree_string_view_t device_uri = loom_runtime_test_device()->device_uri;
  if (iree_status_is_ok(status)) {
    application.native_allocator =
        std::make_unique<loom_runtime::testing::NativeAllocator>(
            host_allocator);
    std::vector<loom_runtime_device_options_t> device_options(
        options->device_count);
    std::vector<std::string> device_names(options->device_count);
    for (size_t i = 0; i < device_options.size(); ++i) {
      device_names[i] = "device" + std::to_string(i);
      auto& device_option = device_options[i];
      device_option = {
          .driver = driver,
          .uri = device_uri,
          .name = iree_make_cstring_view(device_names[i].c_str()),
      };
    }
    status = loom_runtime_device_group_allocate(
        host, options->device_count, device_options.data(),
        application.native_allocator->allocator(), &group);
    const loom_runtime_pool_options_t pool_options = {
        .elastic = options->backing == LOOM_RUNTIME_POOL_BACKING_ELASTIC,
        .weight_slab_size = options->slab_size,
        .state_slab_size = options->slab_size,
        .memory_limit = options->memory_limit,
    };
    for (iree_host_size_t i = 0;
         i < options->device_count && iree_status_is_ok(status); ++i) {
      const std::string name = "pool" + std::to_string(i);
      const loom_runtime_pool_binding_t binding = {
          .name = iree_make_cstring_view(name.c_str()),
          .device_mask = UINT64_C(1) << i,
      };
      loom_runtime_pool_t* pool = nullptr;
      status = loom_runtime_device_group_create_pool(group, binding,
                                                     &pool_options, &pool);
    }
    if (iree_status_is_ok(status) &&
        options->backing == LOOM_RUNTIME_POOL_BACKING_ELASTIC) {
      for (iree_host_size_t i = 0;
           i < options->device_count && iree_status_is_ok(status); ++i) {
        status = application.native_allocator->Observe(
            loom_runtime_device_group_at(group, i),
            loom_runtime_device_group_pool_at(group, i));
      }
    }
  }
  iree_hal_driver_release(driver);
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(status, loom_runtime_device_group_free(group));
    application.native_allocator.reset();
    loom_runtime_host_release(host);
    return status;
  }

  application.options = *options;
  application.host = host;
  application.group = group;
  *out_application = &application;
  return iree_ok_status();
}

loom_runtime_host_t* loom_runtime_test_application_host(
    const loom_runtime_test_application_t* application) {
  return application->host;
}

loom_runtime_device_group_t* loom_runtime_test_application_device_group(
    const loom_runtime_test_application_t* application) {
  return application->group;
}

loom_runtime_device_t* loom_runtime_test_application_device_at(
    const loom_runtime_test_application_t* application,
    iree_host_size_t device_index) {
  return loom_runtime_device_group_at(application->group, device_index);
}
