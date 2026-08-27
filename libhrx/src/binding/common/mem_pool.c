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
  iree_slim_mutex_lock(&device->primary_context_mutex);
  hrx_mem_pool_t pool = device->default_mem_pool;
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  return pool;
}

hrx_mem_pool_t iree_hal_streaming_device_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  iree_slim_mutex_lock(&device->primary_context_mutex);
  hrx_mem_pool_t pool = device->current_mem_pool;
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  return pool;
}

hrx_mem_pool_t iree_hal_streaming_device_retain_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  iree_slim_mutex_lock(&device->primary_context_mutex);
  hrx_mem_pool_t pool = device->current_mem_pool;
  if (!pool) {
    pool = device->default_mem_pool;
  }
  hrx_mem_pool_retain(pool);
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  return pool;
}

hrx_mem_pool_t iree_hal_streaming_device_retain_default_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  iree_slim_mutex_lock(&device->primary_context_mutex);
  hrx_mem_pool_t pool = device->default_mem_pool;
  hrx_mem_pool_retain(pool);
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  return pool;
}

void iree_hal_streaming_device_set_mem_pool(iree_hal_streaming_device_t* device,
                                            hrx_mem_pool_t pool) {
  IREE_ASSERT_ARGUMENT(device);

  // Retain before publishing so the new selection remains valid even if its
  // caller releases its handle immediately after this API returns.
  hrx_mem_pool_retain(pool);
  iree_slim_mutex_lock(&device->primary_context_mutex);
  hrx_mem_pool_t previous_pool = device->current_mem_pool;
  device->current_mem_pool = pool;
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  hrx_mem_pool_release(previous_pool);
}

void iree_hal_streaming_device_reset_mem_pool_if_current(
    iree_hal_streaming_device_t* device, hrx_mem_pool_t pool) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(pool);

  iree_slim_mutex_lock(&device->primary_context_mutex);
  hrx_mem_pool_t previous_pool = device->current_mem_pool;
  if (previous_pool != pool || previous_pool == device->default_mem_pool) {
    iree_slim_mutex_unlock(&device->primary_context_mutex);
    return;
  }

  hrx_mem_pool_t default_pool = device->default_mem_pool;
  hrx_mem_pool_retain(default_pool);
  device->current_mem_pool = default_pool;
  iree_slim_mutex_unlock(&device->primary_context_mutex);
  hrx_mem_pool_release(previous_pool);
}
