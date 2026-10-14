// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/channel_transfer.h"

#include "iree/hal/detail.h"

IREE_HAL_API_RETAIN_RELEASE(channel_transfer);

IREE_API_EXPORT iree_status_t iree_hal_channel_prepare_transfer(
    iree_hal_channel_t* channel,
    const iree_hal_channel_transfer_params_t* params,
    iree_hal_channel_transfer_t** out_transfer) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_transfer);
  *out_transfer = NULL;
  if (params->mode != IREE_HAL_CHANNEL_TRANSFER_MODE_SEND &&
      params->mode != IREE_HAL_CHANNEL_TRANSFER_MODE_RECEIVE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid channel transfer mode");
  }
  int32_t rank = 0;
  int32_t count = 0;
  iree_hal_channel_query_rank_and_count(channel, &rank, &count);
  if (params->peer_rank < 0 || params->peer_rank >= count ||
      params->peer_rank == rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "channel transfer requires a distinct peer rank");
  }
  if (!params->pool || !params->pool->memory_contract || !params->capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "channel transfer requires a qualified pool and nonzero capacity");
  }
  const iree_hal_channel_vtable_t* vtable =
      (const iree_hal_channel_vtable_t*)((iree_hal_resource_t*)channel)->vtable;
  if (!vtable->prepare_transfer) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "channel does not implement prepared transfers");
  }
  return vtable->prepare_transfer(channel, params, out_transfer);
}

IREE_API_EXPORT void iree_hal_channel_transfer_initialize(
    const iree_hal_channel_transfer_vtable_t* vtable,
    iree_hal_channel_t* channel, iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    const iree_hal_channel_transfer_params_t* params,
    iree_hal_channel_transfer_t* out_transfer) {
  iree_hal_resource_initialize(vtable, &out_transfer->resource);
  iree_hal_channel_retain(channel);
  out_transfer->channel = channel;
  out_transfer->device = device;
  out_transfer->queue_family_affinity = queue_family_affinity;
  out_transfer->params = *params;
  iree_hal_pool_retain(params->pool);
}

IREE_API_EXPORT void iree_hal_channel_transfer_deinitialize(
    iree_hal_channel_transfer_t* transfer) {
  iree_hal_pool_release(transfer->params.pool);
  iree_hal_channel_release(transfer->channel);
}

IREE_API_EXPORT iree_status_t iree_hal_channel_transfer_validate_buffer(
    const iree_hal_channel_transfer_t* transfer, iree_hal_buffer_ref_t buffer) {
  if (!buffer.buffer || !buffer.length ||
      buffer.length > transfer->params.capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "channel range must be nonempty and fit capacity");
  }
  const iree_device_size_t buffer_length =
      iree_hal_buffer_byte_length(buffer.buffer);
  if (buffer.offset > buffer_length ||
      buffer.length > buffer_length - buffer.offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "channel range exceeds its buffer");
  }
  if (buffer.buffer->memory.contract !=
      transfer->params.pool->memory_contract) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "channel range has a different pool contract");
  }
  return iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer.buffer),
      transfer->params.mode == IREE_HAL_CHANNEL_TRANSFER_MODE_SEND
          ? IREE_HAL_MEMORY_ACCESS_READ
          : IREE_HAL_MEMORY_ACCESS_WRITE);
}
