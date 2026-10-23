// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/amd/driver.h"

namespace iree::hal::cts {

// Both architecture views share one process-native owner. The CTS backend cache
// releases its devices and retained driver references before this owner dies.
class ProcessDriver {
 public:
  ProcessDriver()
      : status_(iree_hal_amd_driver_create(
            AMDF_NATIVE_LIFETIME_PROCESS, iree_allocator_system(), &driver_)) {}
  ~ProcessDriver() {
    iree_hal_driver_release(driver_);
    iree_status_free(status_);
  }

  ProcessDriver(const ProcessDriver&) = delete;
  ProcessDriver& operator=(const ProcessDriver&) = delete;

  iree_status_t Acquire(iree_hal_driver_t** out_driver) const {
    IREE_RETURN_IF_ERROR(iree_status_clone(status_));
    iree_hal_driver_retain(driver_);
    *out_driver = driver_;
    return iree_ok_status();
  }

 private:
  // Native fabric owner shared by all registered GPU architecture views.
  iree_hal_driver_t* driver_ = nullptr;
  // Construction result retained for every backend that requests this driver.
  iree_status_t status_;
};

static iree_status_t CreateGpuDevice(
    iree_string_view_t scheme,
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  static ProcessDriver process_driver;
  iree_hal_driver_t* driver = nullptr;
  IREE_RETURN_IF_ERROR(process_driver.Acquire(&driver));
  iree_hal_device_t* device = nullptr;
  const iree_string_pair_t options[] = {
      {.key = IREE_SV("queue_count"), .value = IREE_SV("2")}};
  iree_status_t status = iree_hal_driver_create_device_by_path(
      driver, scheme, iree_string_view_empty(), IREE_ARRAYSIZE(options),
      options, create_params, iree_allocator_system(), &device);
  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_driver_release(driver);
    if (iree_status_is_unimplemented(status) ||
        iree_status_is_not_found(status)) {
      // Missing native execution services are unavailable backend instances;
      // once creation succeeds, operation failures remain test failures.
      iree_string_view_t message = iree_status_message(status);
      iree_status_t unavailable_status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "native %.*s execution services are unavailable: %s; %.*s",
          (int)scheme.size, scheme.data,
          iree_status_code_string(iree_status_code(status)), (int)message.size,
          message.data);
      iree_status_free(status);
      status = unavailable_status;
    }
  }
  return status;
}

static bool registered_ = []() {
  for (const char* scheme : {"rdna", "cdna"}) {
    CtsRegistry::RegisterBackend({
        scheme,
        {
            .name = scheme,
            .factory =
                [scheme](const iree_hal_device_create_params_t* create_params,
                         iree_hal_driver_t** out_driver,
                         iree_hal_device_t** out_device) {
                  return CreateGpuDevice(iree_make_cstring_view(scheme),
                                         create_params, out_driver, out_device);
                },
        },
        {"async_queue", "file_io", "indirect"},
    });
  }
  return true;
}();

}  // namespace iree::hal::cts
