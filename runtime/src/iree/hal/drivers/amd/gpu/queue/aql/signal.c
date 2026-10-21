// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/signal.h"

void iree_hal_amd_gpu_aql_signal_block_initialize(
    iree_hal_amd_gpu_signal_pool_t* pool, const void* domain,
    const amdf_gpu_event_info_t* event,
    iree_hal_amd_gpu_aql_signal_t* native_signals, uint64_t device_address,
    iree_hal_amd_gpu_signal_t* signals, uint32_t capacity, uint32_t queue_count,
    void* frontier_storage, iree_hal_amd_gpu_signal_block_t* out_block) {
  for (uint32_t i = 0; i < capacity; ++i) {
    native_signals[i] = (iree_hal_amd_gpu_aql_signal_t){
        .kind = 1,
        .event_mailbox = event->mailbox_address,
        .event_id = event->interrupt_context,
    };
  }
  iree_hal_amd_gpu_signal_block_initialize(
      pool, domain, native_signals, device_address,
      sizeof(iree_hal_amd_gpu_aql_signal_t),
      offsetof(iree_hal_amd_gpu_aql_signal_t, value), signals, capacity,
      queue_count, frontier_storage, out_block);
}

iree_status_t iree_hal_amd_gpu_aql_signal_storage_initialize(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_signal_pool_t* pool,
    iree_hal_amd_gpu_signal_storage_t* storage) {
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_backing_query_address(
      context, &storage->memory, &address));
  iree_hal_amd_gpu_aql_signal_block_initialize(
      pool, context, &context->event_info, (void*)storage->memory.contents.data,
      address, storage->signals, storage->capacity, storage->queue_count,
      storage->frontiers, &storage->block);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_aql_signal_storage_allocate(
    iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_signal_pool_t* pool,
    const iree_hal_amd_fabric_memory_options_t* options, uint32_t capacity,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_signal_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_signal_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_signal_storage_prepare(
      capacity, queue_count, host_allocator, &storage));
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      context->api, options,
      (uint64_t)capacity * sizeof(iree_hal_amd_gpu_aql_signal_t), 64,
      context->event_sink, &storage->memory, NULL);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amd_gpu_aql_signal_storage_initialize(context, pool, storage);
  }
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    iree_hal_amd_gpu_signal_storage_free(context, storage);
  }
  return status;
}
