// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/backing.h"

#include "iree/hal/drivers/amd/status.h"

static_assert(sizeof(iree_hal_amd_gpu_signal_cache_t) <=
                  3 * iree_hardware_destructive_interference_size,
              "signal capacity and stable wake ownership fit in three lines");

iree_status_t iree_hal_amd_gpu_backing_query_address(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_t* memory, uint64_t* out_address) {
  if (memory->host_cacheability != IREE_HAL_HOST_CACHEABILITY_WRITE_BACK) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "GPU backing requires a coherent write-back mapping");
  }
  return out_address
             ? IREE_HAL_AMD_STATUS_FROM_AMDF(
                   context->api->memory_query_address(
                       memory->handle, 0, AMDF_MEMORY_ADDRESS_GPU, out_address),
                   "memory_query_address")
             : iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_backing_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t length,
    uint64_t alignment, iree_hal_amd_fabric_memory_t* memory,
    uint64_t* out_address) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, options, length, alignment, context->event_sink, memory,
      NULL));
  return iree_hal_amd_gpu_backing_query_address(context, memory, out_address);
}

iree_status_t iree_hal_amd_gpu_argument_storage_prepare(
    uint64_t capacity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_argument_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_argument_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*storage),
                                             (void**)&storage));
  storage->host_allocator = host_allocator;
  storage->capacity = capacity;
  *out_storage = storage;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_argument_storage_initialize(
    iree_hal_amd_gpu_context_t* context, uint64_t begin,
    iree_hal_amd_gpu_argument_storage_t* storage) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_backing_query_address(
      context, &storage->memory, &storage->address));
  storage->begin = begin;
  storage->end = begin;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_argument_storage_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* options, uint64_t capacity,
    uint64_t begin, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_argument_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_argument_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_argument_storage_prepare(
      capacity, host_allocator, &storage));
  iree_status_t status = iree_hal_amd_fabric_memory_allocate(
      context->api, options, capacity, capacity, context->event_sink,
      &storage->memory, NULL);
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amd_gpu_argument_storage_initialize(context, begin, storage);
  }
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    iree_hal_amd_gpu_argument_storage_free(context, storage);
  }
  return status;
}

static void iree_hal_amd_gpu_argument_storage_released(void* user_data) {
  iree_hal_amd_gpu_argument_storage_t* storage = user_data;
  iree_allocator_free(storage->host_allocator, storage);
}

void iree_hal_amd_gpu_argument_storage_free(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_argument_storage_t* storage) {
  iree_hal_amd_gpu_context_release_memory(
      context, 1, &storage->memory,
      (iree_hal_amd_fabric_memory_release_callback_t){
          .fn = iree_hal_amd_gpu_argument_storage_released,
          .user_data = storage},
      &storage->release);
}

iree_status_t iree_hal_amd_gpu_signal_storage_prepare(
    uint32_t capacity, uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_signal_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_signal_storage_t* storage = NULL;
  iree_host_size_t total_size = 0, signals_offset = 0, frontiers_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*storage), &total_size,
      IREE_STRUCT_FIELD(capacity, iree_hal_amd_gpu_signal_t, &signals_offset),
      IREE_STRUCT_FIELD(1, iree_hal_amd_gpu_signal_frontiers_t,
                        &frontiers_offset),
      IREE_STRUCT_ARRAY_FIELD(capacity, queue_count, uint64_t, NULL)));
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&storage));
  storage->host_allocator = host_allocator;
  storage->capacity = capacity;
  storage->queue_count = queue_count;
  storage->signals = (void*)((uint8_t*)storage + signals_offset);
  storage->frontiers = (void*)((uint8_t*)storage + frontiers_offset);
  *out_storage = storage;
  return iree_ok_status();
}

static void iree_hal_amd_gpu_signal_storage_released(void* user_data) {
  iree_hal_amd_gpu_signal_storage_t* storage = user_data;
  iree_hal_amd_gpu_signal_frontiers_t* frontiers =
      iree_hal_amd_gpu_signal_block_frontiers(&storage->block);
  if (frontiers) {
    iree_hal_amd_gpu_signal_frontiers_free(frontiers, storage->frontiers,
                                           storage->host_allocator);
  }
  iree_allocator_free(storage->host_allocator, storage);
}

void iree_hal_amd_gpu_signal_storage_free(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_signal_storage_t* storage) {
  if (storage->block.pool) {
    iree_hal_amd_gpu_signal_block_deinitialize(&storage->block);
  }
  iree_hal_amd_gpu_context_release_memory(
      context, 1, &storage->memory,
      (iree_hal_amd_fabric_memory_release_callback_t){
          .fn = iree_hal_amd_gpu_signal_storage_released, .user_data = storage},
      &storage->release);
}

static void iree_hal_amd_gpu_signal_cache_available(void* user_data) {
  iree_hal_amd_gpu_signal_cache_t* cache = user_data;
  if (iree_atomic_load(&cache->waiting, iree_memory_order_acquire) &&
      !iree_atomic_exchange(&cache->ready, 1, iree_memory_order_acq_rel)) {
    iree_hal_amd_gpu_publication_notify(cache->publication,
                                        cache->source_ordinal);
  }
}

void iree_hal_amd_gpu_signal_cache_initialize(
    iree_hal_amd_gpu_publication_t* publication, uint32_t source_ordinal,
    iree_hal_amd_gpu_signal_cache_t* out_cache) {
  *out_cache = (iree_hal_amd_gpu_signal_cache_t){
      .publication = publication, .source_ordinal = source_ordinal};
  iree_hal_amd_gpu_signal_pool_initialize(
      (iree_hal_amd_gpu_signal_pool_notification_t){
          .fn = iree_hal_amd_gpu_signal_cache_available,
          .user_data = out_cache},
      &out_cache->pool);
}

void iree_hal_amd_gpu_signal_cache_trim(
    iree_hal_amd_gpu_context_t* context, uint32_t minimum_block_count,
    iree_hal_amd_gpu_signal_cache_t* cache) {
  iree_hal_amd_gpu_signal_storage_t** link = &cache->storage;
  while (*link && cache->pool.block_count > minimum_block_count) {
    iree_hal_amd_gpu_signal_storage_t* storage = *link;
    if (iree_hal_amd_gpu_signal_block_try_deinitialize(&storage->block)) {
      *link = storage->next;
      iree_hal_amd_gpu_signal_storage_free(context, storage);
    } else {
      link = &storage->next;
    }
  }
}

void iree_hal_amd_gpu_signal_cache_deinitialize(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_signal_cache_t* cache) {
  iree_atomic_store(&cache->waiting, 0, iree_memory_order_release);
  iree_hal_amd_gpu_signal_cache_trim(context, 0, cache);
  iree_hal_amd_gpu_signal_pool_deinitialize(&cache->pool);
}
