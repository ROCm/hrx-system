// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/registration/driver_module.h"
#include "loom/serve/testing/device.h"

const loom_serve_test_device_t* loom_serve_test_device(void) {
  static const loom_serve_test_device_t device = {
      .driver_register = iree_hal_amd_driver_module_register,
      .driver_name = IREE_SVL("rdna"),
      .device_uri = IREE_SVL("rdna"),
      .service_backing = LOOM_SERVE_DEVICE_BACKING_FIXED,
  };
  return &device;
}
