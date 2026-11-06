// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/registration/driver_module.h"
#include "loom/runtime/testing/device.h"

const loom_runtime_test_device_t* loom_runtime_test_device(void) {
  static const loom_runtime_test_device_t device = {
      .driver_register = iree_hal_amd_driver_module_register,
      .driver_name = IREE_SVL("amdgpu"),
      .device_uri = IREE_SVL("amdgpu"),
      .service_backing = LOOM_RUNTIME_POOL_BACKING_ELASTIC,
  };
  return &device;
}
