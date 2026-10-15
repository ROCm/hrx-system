// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/backing.h"

#include "iree/hal/drivers/amd/status.h"

iree_status_t iree_hal_amd_gpu_pm4_backing_allocate(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t length,
    uint64_t alignment, iree_hal_amd_fabric_memory_t* memory,
    uint64_t* out_address) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, options, length, alignment, context->event_sink, memory));
  if (memory->host_cacheability != IREE_HAL_HOST_CACHEABILITY_WRITE_BACK) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 backing requires a coherent write-back mapping");
  }
  return out_address
             ? IREE_HAL_AMD_STATUS_FROM_AMDF(
                   context->api->memory_query_address(
                       memory->handle, 0, AMDF_MEMORY_ADDRESS_GPU, out_address),
                   "memory_query_address")
             : iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_argument_storage_allocate(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t capacity,
    uint64_t begin, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_argument_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_pm4_argument_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*storage),
                                             (void**)&storage));
  storage->capacity = capacity;
  storage->begin = begin;
  storage->end = begin;
  iree_status_t status = iree_hal_amd_gpu_pm4_backing_allocate(
      context, options, capacity, capacity, &storage->memory,
      &storage->address);
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    iree_hal_amd_gpu_pm4_argument_storage_free(context, host_allocator,
                                               storage);
  }
  return status;
}

void iree_hal_amd_gpu_pm4_argument_storage_free(
    const iree_hal_amd_gpu_context_t* context, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_argument_storage_t* storage) {
  iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                          &storage->memory);
  iree_allocator_free(host_allocator, storage);
}

iree_status_t iree_hal_amd_gpu_pm4_signal_storage_allocate(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint32_t capacity,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_signal_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_pm4_signal_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator,
      sizeof(*storage) + capacity * sizeof(iree_hal_amd_gpu_pm4_signal_t),
      (void**)&storage));
  uint64_t address = 0;
  iree_status_t status = iree_hal_amd_gpu_pm4_backing_allocate(
      context, options, capacity * sizeof(uint64_t), 8, &storage->memory,
      &address);
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_signal_pool_initialize(
        context, (uint64_t*)storage->memory.contents.data, address,
        (iree_hal_amd_gpu_pm4_signal_t*)(storage + 1), capacity,
        (iree_hal_amd_gpu_pm4_signal_pool_notification_t){0}, &storage->pool);
    *out_storage = storage;
  } else {
    iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                            &storage->memory);
    iree_allocator_free(host_allocator, storage);
  }
  return status;
}

void iree_hal_amd_gpu_pm4_signal_storage_free(
    const iree_hal_amd_gpu_context_t* context, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_signal_storage_t* storage) {
  iree_hal_amd_gpu_pm4_signal_pool_deinitialize(&storage->pool);
  iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                          &storage->memory);
  iree_allocator_free(host_allocator, storage);
}
