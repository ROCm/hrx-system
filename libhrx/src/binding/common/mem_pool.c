// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Memory pool management. Pools are now backed by hrx_mem_pool_t from
// libhrx. This file only contains the device-level accessors.

#include "common/internal.h"

//===----------------------------------------------------------------------===//
// Device pool accessors
//===----------------------------------------------------------------------===//

hrx_mem_pool_t iree_hal_streaming_device_default_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return device->default_mem_pool;
}

hrx_mem_pool_t iree_hal_streaming_device_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return device->current_mem_pool;
}

void iree_hal_streaming_device_set_mem_pool(iree_hal_streaming_device_t* device,
                                            hrx_mem_pool_t pool) {
  IREE_ASSERT_ARGUMENT(device);

  if (device->current_mem_pool) {
    hrx_mem_pool_release(device->current_mem_pool);
  }

  device->current_mem_pool = pool;
  if (pool) {
    hrx_mem_pool_retain(pool);
  }
}
