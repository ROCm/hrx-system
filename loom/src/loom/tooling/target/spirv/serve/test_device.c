// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/vulkan/registration/driver_module.h"
#include "loom/runtime/testing/device.h"

const loom_runtime_test_device_t* loom_runtime_test_device(void) {
  static const loom_runtime_test_device_t device = {
      .driver_register = iree_hal_vulkan_driver_module_register,
      .driver_name = IREE_SVL("vulkan"),
      .device_uri = IREE_SVL("vulkan"),
      .service_backing = LOOM_RUNTIME_POOL_BACKING_FIXED,
  };
  return &device;
}
