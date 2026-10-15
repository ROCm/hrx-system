// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/amd/driver.h"

namespace iree::hal::cts {

static iree_status_t CreateRdnaDevice(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  iree_hal_driver_t* driver = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amd_driver_create(
      AMDF_NATIVE_LIFETIME_PROCESS, iree_allocator_system(), &driver));
  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_driver_create_device_by_uri(
      driver, IREE_SV("rdna"), create_params, iree_allocator_system(), &device);
  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_driver_release(driver);
    if (iree_status_is_unimplemented(status) ||
        iree_status_is_not_found(status)) {
      // Missing native execution services are unavailable backend instances;
      // once creation succeeds, operation failures remain test failures.
      status = iree_status_join(
          iree_make_status(IREE_STATUS_UNAVAILABLE,
                           "native RDNA execution services are unavailable"),
          status);
    }
  }
  return status;
}

static bool rdna_registered_ =
    (CtsRegistry::RegisterBackend({
         "rdna",
         {.name = "rdna", .factory = CreateRdnaDevice},
         {"async_queue", "file_io"},
     }),
     true);

}  // namespace iree::hal::cts
