// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for Vulkan devices using native replay caching.

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/vulkan/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateVulkanReplayCacheDevice(
    iree_string_view_t uri,
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  iree_status_t status = iree_hal_vulkan_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("vulkan"),
        iree_allocator_system(), &driver);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_device_by_uri(
        driver, uri, create_params, iree_allocator_system(), &device);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

static iree_status_t CreateVulkanReplayCacheDeviceWithCache(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  return CreateVulkanReplayCacheDevice(
      iree_make_cstring_view("vulkan://?cached_bda_replay_instances=2&"
                             "retained_cached_bda_replay_instances=0"),
      create_params, out_driver, out_device);
}

static bool vulkan_replay_cache_registered_ =
    (CtsRegistry::RegisterBackend({
         "vulkan_replay_cache",
         {"vulkan_replay_cache", CreateVulkanReplayCacheDeviceWithCache,
          /*executable_format=*/nullptr,
          /*executable_data=*/nullptr, RecordingMode::kDirect,
          /*unsupported_tests=*/{},
          /*expected_failures=*/{}},
         {"async_queue", "file_io", "vulkan", "vulkan_replay_cache"},
     }),
     true);

}  // namespace iree::hal::cts
