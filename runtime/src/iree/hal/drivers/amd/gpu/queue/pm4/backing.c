// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/backing.h"

iree_status_t iree_hal_amd_gpu_pm4_signal_storage_initialize(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_signal_pool_t* pool,
    iree_hal_amd_gpu_signal_storage_t* storage) {
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_backing_query_address(
      context, &storage->memory, &address));
  iree_hal_amd_gpu_signal_block_initialize(
      pool, context, (uint64_t*)storage->memory.contents.data, address, 8, 0,
      storage->signals, storage->capacity, storage->queue_count,
      storage->frontiers, &storage->block);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_signal_storage_allocate(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_signal_pool_t* pool,
    const iree_hal_amd_fabric_memory_options_t* options, uint32_t capacity,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_signal_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_signal_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_signal_storage_prepare(
      capacity, queue_count, host_allocator, &storage));
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      context->api, options, (uint64_t)capacity * sizeof(uint64_t), 8,
      context->event_sink, &storage->memory, NULL);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amd_gpu_pm4_signal_storage_initialize(context, pool, storage);
  }
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    iree_hal_amd_gpu_signal_storage_free(context, storage);
  }
  return status;
}
