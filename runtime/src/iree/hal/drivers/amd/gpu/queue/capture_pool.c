// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/capture_pool.h"

#include <string.h>

#include "iree/base/internal/debugging.h"

// Only unused storage contains a link. Acquisition transfers the entire block
// to the capture, including this first word and the system's alignment.
typedef struct iree_hal_amd_gpu_capture_block_t {
  // Next unowned block, protected by the list containing this block.
  struct iree_hal_amd_gpu_capture_block_t* next;
} iree_hal_amd_gpu_capture_block_t;

// Slab metadata is cold: acquire and release touch only the free-list links.
// A block's immutable owner pointer follows its aligned usable payload.
typedef struct iree_hal_amd_gpu_capture_slab_t {
  // Private construction rollback or detached-trim list, never a live registry.
  struct iree_hal_amd_gpu_capture_slab_t* next;
  // Immutable number of independently owned blocks in this allocation.
  iree_host_size_t block_count;
  // Free blocks detached by the active trim, protected by trim_mutex.
  iree_host_size_t trim_count;
} iree_hal_amd_gpu_capture_slab_t;

static iree_hal_amd_gpu_capture_block_t* iree_hal_amd_gpu_capture_slab_head(
    iree_hal_amd_gpu_capture_slab_t* slab) {
  return (void*)((uint8_t*)slab +
                 iree_host_align(sizeof(*slab), iree_max_align_t));
}

static iree_hal_amd_gpu_capture_slab_t** iree_hal_amd_gpu_capture_block_owner(
    const iree_hal_amd_gpu_capture_pool_t* pool,
    iree_hal_amd_gpu_capture_block_t* block) {
  return (void*)((uint8_t*)block +
                 iree_host_align(pool->block_size, iree_max_align_t));
}

// Capture size is a caller-selected capacity. Check its allocation geometry
// before reserving the payload and owner trailer, including very large blocks.
static iree_status_t iree_hal_amd_gpu_capture_slab_geometry(
    const iree_hal_amd_gpu_capture_pool_t* pool, iree_host_size_t* out_stride,
    iree_host_size_t* out_capacity) {
  const iree_host_size_t header_size = iree_host_align(
      sizeof(iree_hal_amd_gpu_capture_slab_t), iree_max_align_t);
  iree_host_size_t stride = 0;
  iree_host_size_t minimum_size = 0;
  if (!iree_host_size_checked_align(pool->block_size, iree_max_align_t,
                                    &stride) ||
      !iree_host_size_checked_add(stride, iree_max_align_t, &stride) ||
      !iree_host_size_checked_add(header_size, stride, &minimum_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "capture slab allocation size overflows");
  }
  *out_stride = stride;
  *out_capacity = iree_max(1, (64 * 1024 - header_size) / stride);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_capture_slab_allocate(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_host_size_t stride,
    iree_host_size_t count, iree_hal_amd_gpu_capture_slab_t** out_slab) {
  const iree_host_size_t header_size = iree_host_align(
      sizeof(iree_hal_amd_gpu_capture_slab_t), iree_max_align_t);
  iree_hal_amd_gpu_capture_slab_t* slab = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      pool->host_allocator, header_size + count * stride, (void**)&slab));
  *slab = (iree_hal_amd_gpu_capture_slab_t){.block_count = count};
  uint8_t* data = (void*)iree_hal_amd_gpu_capture_slab_head(slab);
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_amd_gpu_capture_block_t* block = (void*)(data + i * stride);
    block->next = i + 1 < count ? (void*)(data + (i + 1) * stride) : NULL;
    *iree_hal_amd_gpu_capture_block_owner(pool, block) = slab;
    IREE_ASAN_POISON_MEMORY_REGION((uint8_t*)block + sizeof(*block),
                                   pool->block_size - sizeof(*block));
  }
  *out_slab = slab;
  return iree_ok_status();
}

static void iree_hal_amd_gpu_capture_pool_return_blocks(
    iree_hal_amd_gpu_capture_pool_t* pool,
    iree_hal_amd_gpu_capture_block_t* head,
    iree_hal_amd_gpu_capture_block_t* tail) {
  if (!head) {
    return;
  }
  iree_slim_mutex_lock(&pool->returned.mutex);
  tail->next = pool->returned.head;
  pool->returned.head = head;
  iree_slim_mutex_unlock(&pool->returned.mutex);
}

void iree_hal_amd_gpu_capture_pool_initialize(
    iree_host_size_t block_size, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_capture_pool_t* out_pool) {
  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->block_size = block_size;
  out_pool->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&out_pool->available.mutex);
  iree_slim_mutex_initialize(&out_pool->returned.mutex);
  iree_slim_mutex_initialize(&out_pool->trim_mutex);
}

void iree_hal_amd_gpu_capture_pool_deinitialize(
    iree_hal_amd_gpu_capture_pool_t* pool) {
  iree_hal_amd_gpu_capture_pool_trim(pool);
  iree_slim_mutex_deinitialize(&pool->trim_mutex);
  iree_slim_mutex_deinitialize(&pool->returned.mutex);
  iree_slim_mutex_deinitialize(&pool->available.mutex);
}

iree_status_t iree_hal_amd_gpu_capture_pool_preallocate(
    iree_hal_amd_gpu_capture_pool_t* pool, iree_host_size_t count) {
  if (!count) {
    return iree_ok_status();
  }
  iree_host_size_t stride = 0, capacity = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_capture_slab_geometry(pool, &stride, &capacity));
  iree_hal_amd_gpu_capture_slab_t* slabs = NULL;
  iree_hal_amd_gpu_capture_block_t* head = NULL;
  iree_hal_amd_gpu_capture_block_t* tail = NULL;
  iree_status_t status = iree_ok_status();
  while (count && iree_status_is_ok(status)) {
    const iree_host_size_t batch_count = iree_min(count, capacity);
    iree_hal_amd_gpu_capture_slab_t* slab = NULL;
    status = iree_hal_amd_gpu_capture_slab_allocate(pool, stride, batch_count,
                                                    &slab);
    if (iree_status_is_ok(status)) {
      slab->next = slabs;
      slabs = slab;
      iree_hal_amd_gpu_capture_block_t* first =
          iree_hal_amd_gpu_capture_slab_head(slab);
      iree_hal_amd_gpu_capture_block_t* last =
          (void*)((uint8_t*)first + (batch_count - 1) * stride);
      last->next = head;
      head = first;
      if (!tail) {
        tail = last;
      }
      count -= batch_count;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_capture_pool_return_blocks(pool, head, tail);
  } else {
    while (slabs) {
      iree_hal_amd_gpu_capture_slab_t* next = slabs->next;
      iree_allocator_free(pool->host_allocator, slabs);
      slabs = next;
    }
  }
  return status;
}

void iree_hal_amd_gpu_capture_pool_trim(iree_hal_amd_gpu_capture_pool_t* pool) {
  iree_slim_mutex_lock(&pool->trim_mutex);
  iree_slim_mutex_lock(&pool->available.mutex);
  iree_slim_mutex_lock(&pool->returned.mutex);
  iree_hal_amd_gpu_capture_block_t* available = pool->available.head;
  iree_hal_amd_gpu_capture_block_t* returned = pool->returned.head;
  pool->available.head = NULL;
  pool->returned.head = NULL;
  iree_slim_mutex_unlock(&pool->returned.mutex);
  iree_slim_mutex_unlock(&pool->available.mutex);
  iree_hal_amd_gpu_capture_slab_t* slabs = NULL;
  iree_hal_amd_gpu_capture_block_t* lists[] = {available, returned};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(lists); ++i) {
    for (iree_hal_amd_gpu_capture_block_t* block = lists[i]; block;
         block = block->next) {
      iree_hal_amd_gpu_capture_slab_t* slab =
          *iree_hal_amd_gpu_capture_block_owner(pool, block);
      if (!slab->trim_count++) {
        slab->next = slabs;
        slabs = slab;
      }
    }
  }
  // Only a detached set containing every block proves that the slab has no
  // live borrower. Concurrent returns into the pool belong to the next trim.
  iree_hal_amd_gpu_capture_block_t* retained_head = NULL;
  iree_hal_amd_gpu_capture_block_t* retained_tail = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(lists); ++i) {
    for (iree_hal_amd_gpu_capture_block_t* block = lists[i]; block;) {
      iree_hal_amd_gpu_capture_block_t* next = block->next;
      const iree_hal_amd_gpu_capture_slab_t* slab =
          *iree_hal_amd_gpu_capture_block_owner(pool, block);
      if (slab->trim_count != slab->block_count) {
        block->next = retained_head;
        retained_head = block;
        if (!retained_tail) {
          retained_tail = block;
        }
      }
      block = next;
    }
  }
  while (slabs) {
    iree_hal_amd_gpu_capture_slab_t* next = slabs->next;
    if (slabs->trim_count == slabs->block_count) {
      iree_allocator_free(pool->host_allocator, slabs);
    } else {
      slabs->trim_count = 0;
      slabs->next = NULL;
    }
    slabs = next;
  }
  iree_hal_amd_gpu_capture_pool_return_blocks(pool, retained_head,
                                              retained_tail);
  iree_slim_mutex_unlock(&pool->trim_mutex);
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
      IREE_ASAN_UNPOISON_MEMORY_REGION(block, allocation_size);
      *out_storage = iree_make_byte_span(block, allocation_size);
      return iree_ok_status();
    }
    iree_host_size_t stride = 0, capacity = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_capture_slab_geometry(pool, &stride, &capacity));
    iree_hal_amd_gpu_capture_slab_t* slab = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_capture_slab_allocate(pool, stride, capacity, &slab));
    block = iree_hal_amd_gpu_capture_slab_head(slab);
    iree_hal_amd_gpu_capture_pool_return_blocks(
        pool, block->next, (void*)((uint8_t*)block + (capacity - 1) * stride));
    IREE_ASAN_UNPOISON_MEMORY_REGION(block, allocation_size);
    *out_storage = iree_make_byte_span(block, allocation_size);
    return iree_ok_status();
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
  IREE_ASAN_POISON_MEMORY_REGION((uint8_t*)block + sizeof(*block),
                                 pool->block_size - sizeof(*block));
  iree_hal_amd_gpu_capture_pool_return_blocks(pool, block, block);
}
