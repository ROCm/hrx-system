// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"

#include <string.h>

// Only unused storage contains a link. Acquisition transfers the entire block
// to the capture, including this first word and the system's alignment.
typedef struct iree_hal_amd_gpu_capture_block_t {
  // Next unowned block, protected by the list containing this block.
  struct iree_hal_amd_gpu_capture_block_t* next;
} iree_hal_amd_gpu_capture_block_t;

void iree_hal_amd_gpu_capture_pool_initialize(
    iree_host_size_t block_size, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_capture_pool_t* out_pool) {
  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->block_size = block_size;
  out_pool->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&out_pool->available.mutex);
  iree_slim_mutex_initialize(&out_pool->returned.mutex);
}

void iree_hal_amd_gpu_capture_pool_deinitialize(
    iree_hal_amd_gpu_capture_pool_t* pool) {
  iree_hal_amd_gpu_capture_pool_trim(pool);
  iree_slim_mutex_deinitialize(&pool->returned.mutex);
  iree_slim_mutex_deinitialize(&pool->available.mutex);
}

static void iree_hal_amd_gpu_capture_pool_free_blocks(
    iree_allocator_t host_allocator, iree_hal_amd_gpu_capture_block_t* head) {
  while (head) {
    iree_hal_amd_gpu_capture_block_t* next = head->next;
    iree_allocator_free(host_allocator, head);
    head = next;
  }
}

iree_status_t iree_hal_amd_gpu_capture_pool_preallocate(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_host_size_t count) {
  iree_hal_amd_gpu_capture_block_t* head = NULL;
  iree_hal_amd_gpu_capture_block_t* tail = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_gpu_capture_block_t* block = NULL;
    status = iree_allocator_malloc_uninitialized(
        pool->host_allocator, pool->block_size, (void**)&block);
    if (iree_status_is_ok(status)) {
      block->next = head;
      head = block;
      if (!tail) {
        tail = block;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    if (head) {
      iree_slim_mutex_lock(&pool->returned.mutex);
      tail->next = pool->returned.head;
      pool->returned.head = head;
      iree_slim_mutex_unlock(&pool->returned.mutex);
    }
  } else {
    iree_hal_amd_gpu_capture_pool_free_blocks(pool->host_allocator, head);
  }
  return status;
}

void iree_hal_amd_gpu_capture_pool_trim(iree_hal_amd_gpu_capture_pool_t* pool) {
  iree_slim_mutex_lock(&pool->available.mutex);
  iree_slim_mutex_lock(&pool->returned.mutex);
  iree_hal_amd_gpu_capture_block_t* available = pool->available.head;
  iree_hal_amd_gpu_capture_block_t* returned = pool->returned.head;
  pool->available.head = NULL;
  pool->returned.head = NULL;
  iree_slim_mutex_unlock(&pool->returned.mutex);
  iree_slim_mutex_unlock(&pool->available.mutex);
  iree_hal_amd_gpu_capture_pool_free_blocks(pool->host_allocator, available);
  iree_hal_amd_gpu_capture_pool_free_blocks(pool->host_allocator, returned);
}

iree_status_t iree_hal_amd_gpu_capture_pool_acquire(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_host_size_t minimum_size,
    iree_byte_span_t* out_storage) {
  *out_storage = iree_byte_span_empty();
  iree_host_size_t allocation_size = minimum_size;
  if (minimum_size <= pool->block_size) {
    allocation_size = pool->block_size;
    iree_slim_mutex_lock(&pool->available.mutex);
    iree_hal_amd_gpu_capture_block_t* block = pool->available.head;
    if (!block) {
      iree_slim_mutex_lock(&pool->returned.mutex);
      block = pool->returned.head;
      pool->returned.head = NULL;
      iree_slim_mutex_unlock(&pool->returned.mutex);
    }
    pool->available.head = block ? block->next : NULL;
    iree_slim_mutex_unlock(&pool->available.mutex);
    if (block) {
      *out_storage = iree_make_byte_span(block, allocation_size);
      return iree_ok_status();
    }
  }
  void* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      pool->host_allocator, allocation_size, &data));
  *out_storage = iree_make_byte_span(data, allocation_size);
  return iree_ok_status();
}

void iree_hal_amd_gpu_capture_pool_release(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_byte_span_t storage) {
  if (storage.data_length != pool->block_size) {
    iree_allocator_free(pool->host_allocator, storage.data);
    return;
  }
  iree_hal_amd_gpu_capture_block_t* block = (void*)storage.data;
  iree_slim_mutex_lock(&pool->returned.mutex);
  block->next = pool->returned.head;
  pool->returned.head = block;
  iree_slim_mutex_unlock(&pool->returned.mutex);
}
