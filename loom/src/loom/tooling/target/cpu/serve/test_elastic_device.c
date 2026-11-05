// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/testing/device.h"

const loom_serve_test_device_t* loom_serve_test_device(void) {
  static const loom_serve_test_device_t device = {
      .driver_name = IREE_SVL("task"),
      .device_uri = IREE_SVL("task"),
      .service_backing = LOOM_SERVE_POOL_BACKING_ELASTIC,
  };
  return &device;
}
