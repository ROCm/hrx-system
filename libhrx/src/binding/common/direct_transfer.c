// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/direct_transfer.h"

#include "common/internal.h"
#include "iree/hal/buffer_transfer.h"

// Direct HAL transfers may use backend-owned staging resources that are not
// safe to drive concurrently. Keep serialization and chunking in one place so
// callers cannot accidentally bypass the context-wide transfer contract.
#define IREE_HAL_STREAMING_DIRECT_H2D_CHUNK_SIZE (63 * 1024)
#define IREE_HAL_STREAMING_DIRECT_D2H_CHUNK_SIZE (4 * 1024 * 1024)

iree_status_t iree_hal_streaming_direct_transfer_h2d(
    iree_hal_streaming_context_t* context, const void* source,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(source || length == 0);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (IREE_UNLIKELY(length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "direct H2D transfer exceeds host address range");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  const uint8_t* source_bytes = (const uint8_t*)source;
  iree_device_size_t transferred = 0;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&context->direct_transfer_mutex);
  while (transferred < length && iree_status_is_ok(status)) {
    const iree_device_size_t chunk_length =
        iree_min(length - transferred,
                 (iree_device_size_t)IREE_HAL_STREAMING_DIRECT_H2D_CHUNK_SIZE);
    status = iree_hal_device_transfer_h2d(
        context->device, source_bytes + (iree_host_size_t)transferred,
        target_buffer, target_offset + transferred, chunk_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    transferred += chunk_length;
  }
  iree_slim_mutex_unlock(&context->direct_transfer_mutex);
  return status;
}

iree_status_t iree_hal_streaming_direct_transfer_d2h(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, void* target, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target || length == 0);
  if (IREE_UNLIKELY(length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "direct D2H transfer exceeds host address range");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(source_buffer, source_offset, length));

  uint8_t* target_bytes = (uint8_t*)target;
  iree_device_size_t transferred = 0;
  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&context->direct_transfer_mutex);
  while (transferred < length && iree_status_is_ok(status)) {
    const iree_device_size_t chunk_length =
        iree_min(length - transferred,
                 (iree_device_size_t)IREE_HAL_STREAMING_DIRECT_D2H_CHUNK_SIZE);
    status = iree_hal_device_transfer_d2h(
        context->device, source_buffer, source_offset + transferred,
        target_bytes + (iree_host_size_t)transferred, chunk_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    transferred += chunk_length;
  }
  iree_slim_mutex_unlock(&context->direct_transfer_mutex);
  return status;
}
