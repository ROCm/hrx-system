// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/testing/device.h"

const loom_runtime_test_device_t* loom_runtime_test_device(void) {
  static const loom_runtime_test_device_t device = {
      .driver_name = IREE_SVL("task"),
      .device_uri = IREE_SVL("task"),
      .service_backing = LOOM_RUNTIME_POOL_BACKING_ELASTIC,
  };
  return &device;
}
