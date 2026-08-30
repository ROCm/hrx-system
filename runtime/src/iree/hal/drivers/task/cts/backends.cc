// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the task HAL driver.
//
// Registers a single "task" backend that creates a multithreaded
// task-system-based HAL device using the default driver configuration.
// The factory uses the driver registry to create drivers identically to how
// applications create them (via iree_hal_register_all_available_drivers).

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/task/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateTaskDevice(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  // Register the driver module with the global registry. Subsequent calls
  // return ALREADY_EXISTS; only true errors propagate.
  iree_status_t status =
      iree_hal_task_driver_module_register(iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_free(status);
    status = iree_ok_status();
  }

  // Create the driver. This sets up the task executor pool, executable loaders,
  // and heap allocator via the driver module's flag-based configuration.
  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("task"),
        iree_allocator_system(), &driver);
  }

  // Create the default device from the driver.
  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_default_device(
        driver, create_params, iree_allocator_system(), &device);
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

// Registration at static init time. The comma operator evaluates
// RegisterBackend() for its side effect and yields true for the bool.
static bool task_registered_ =
    (CtsRegistry::RegisterBackend({
         "task",
         {"task", CreateTaskDevice,
          /*executable_target_family=*/nullptr,
          /*executable_target_key=*/nullptr,
          /*executable_data=*/nullptr, RecordingMode::kDirect,
          /*unsupported_tests=*/{},
          /*expected_failures=*/{}},
         {"async_queue", "events", "file_io", "host_calls", "mapping",
          "indirect"},
     }),
     true);

}  // namespace iree::hal::cts
