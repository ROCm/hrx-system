// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the AMD XDNA HAL driver.

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/amd/xdna/driver.h"

namespace iree::hal::cts {

static iree_status_t CreateXdnaDevice(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  iree_hal_driver_t* driver = nullptr;
  iree_status_t status =
      iree_hal_amd_xdna_driver_create(iree_allocator_system(), &driver);

  iree_host_size_t device_count = 0;
  iree_hal_device_info_t* device_infos = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_query_available_devices(
        driver, iree_allocator_system(), &device_count, &device_infos);
  }
  iree_allocator_free(iree_allocator_system(), device_infos);
  if (iree_status_is_ok(status) && device_count == 0) {
    status =
        iree_make_status(IREE_STATUS_UNAVAILABLE, "no native XDNA endpoint");
  }

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

static bool xdna_registered_ =
    (CtsRegistry::RegisterBackend({
         "xdna",
         {"xdna",
          CreateXdnaDevice,
          /*executable_target_family=*/nullptr,
          /*executable_target_key=*/nullptr,
          /*executable_data=*/nullptr,
          RecordingMode::kDirect,
          /*unsupported_tests=*/{},
          /*expected_failures=*/
          {
              {"CommandBufferBasicTest.*",
               "reusable command buffers are outside the XDNA v1 surface"},
              {"CommandBufferCopyBufferTest.*",
               "reusable command buffers are outside the XDNA v1 surface"},
              {"CommandBufferFillBufferTest.*",
               "reusable command buffers are outside the XDNA v1 surface"},
              {"CommandBufferStressTest.*",
               "reusable command buffers are outside the XDNA v1 surface"},
              {"TransientBufferTest.*",
               "reusable command buffers are outside the XDNA v1 surface"},
              {"AsyncTransientBufferTest.*",
               "reusable command buffers are outside the XDNA v1 surface"},
              {"CommandBufferUpdateBufferTest.*",
               "reusable command buffers are outside the XDNA v1 surface"},
              {"QueueAllocaTest.*",
               "queue pool backends are outside the XDNA v1 surface"},
              {"QueueHostCallTest.*",
               "queue host calls are outside the XDNA v1 surface"},
              {"AsyncQueueHostCallLifetimeTest.*",
               "queue host calls are outside the XDNA v1 surface"},
              {"SemaphoreSubmissionTest."
               "TransferCommandBufferExecutesOnEveryTransferQueue",
               "reusable command buffers are outside the XDNA v1 surface"},
              {"SemaphoreSubmissionTest."
               "IndirectCommandBufferBindingTableRetainedUntilSignal",
               "reusable command buffers are outside the XDNA v1 surface"},
          }},
         {"async_queue"},
     }),
     true);

}  // namespace iree::hal::cts
