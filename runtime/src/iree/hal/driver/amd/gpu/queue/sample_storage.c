// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sample_storage.h"

iree_status_t iree_hal_amd_gpu_sample_storage_allocate(
    iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_fabric_memory_options_t* memory, uint64_t capacity,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_sample_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_sample_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*storage),
                                             (void**)&storage));
  iree_status_t status = iree_hal_amd_gpu_byte_storage_allocate(
      context, memory, capacity, 0, host_allocator, &storage->current);
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    iree_allocator_free(host_allocator, storage);
  }
  return status;
}

void iree_hal_amd_gpu_sample_storage_free(
    iree_hal_amd_gpu_context_t* context, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_sample_storage_t* storage) {
  if (!storage) {
    return;
  }
  storage->current->next = storage->retired;
  for (iree_hal_amd_gpu_byte_storage_t* generation = storage->current;
       generation;) {
    iree_hal_amd_gpu_byte_storage_t* next = generation->next;
    iree_hal_amd_gpu_byte_storage_free(context, generation);
    generation = next;
  }
  iree_allocator_free(host_allocator, storage);
}

iree_hal_amd_gpu_sample_reservation_t iree_hal_amd_gpu_sample_storage_reserve(
    iree_hal_amd_gpu_sample_storage_t* storage, uint64_t length,
    uint64_t alignment, uint64_t* out_offset, uint64_t* out_end) {
  const iree_hal_amd_gpu_byte_storage_t* generation = storage->current;
  if (length > generation->capacity || alignment > generation->capacity) {
    return IREE_HAL_AMD_GPU_SAMPLE_GROW;
  }
  uint64_t relative =
      iree_align_uint64(generation->end - generation->begin, alignment);
  uint64_t offset = relative & (generation->capacity - 1);
  if (length > generation->capacity - offset) {
    relative += generation->capacity - offset;
    offset = 0;
  }
  const uint64_t end = generation->begin + relative + length;
  const uint64_t consumed =
      (uint64_t)iree_atomic_load(&storage->consumed, iree_memory_order_acquire);
  const uint64_t reusable = consumed >= generation->end
                                ? generation->begin + relative
                                : iree_max(consumed, generation->begin);
  if (end - reusable > generation->capacity) {
    return IREE_HAL_AMD_GPU_SAMPLE_CAPACITY;
  }
  *out_offset = offset;
  *out_end = end;
  return IREE_HAL_AMD_GPU_SAMPLE_RESERVED;
}

void iree_hal_amd_gpu_sample_storage_commit(
    iree_hal_amd_gpu_sample_storage_t* storage, uint64_t end, bool pending) {
  if (pending) {
    storage->pending = storage->current;
    storage->pending_begin = storage->current->end;
  }
  storage->current->end = end;
}

void iree_hal_amd_gpu_sample_storage_accept(
    iree_hal_amd_gpu_sample_storage_t* storage) {
  if (storage) {
    storage->pending = NULL;
  }
}

void iree_hal_amd_gpu_sample_storage_rollback(
    iree_hal_amd_gpu_sample_storage_t* storage) {
  if (storage && storage->pending) {
    storage->pending->end = storage->pending_begin;
    storage->pending = NULL;
  }
}

void iree_hal_amd_gpu_sample_storage_install(
    iree_hal_amd_gpu_sample_storage_t* storage,
    iree_hal_amd_gpu_byte_storage_t* generation) {
  storage->current->next = storage->retired;
  storage->retired = storage->current;
  storage->current = generation;
}

void iree_hal_amd_gpu_sample_storage_trim(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_sample_storage_t* storage) {
  if (!storage) {
    return;
  }
  const uint64_t consumed =
      (uint64_t)iree_atomic_load(&storage->consumed, iree_memory_order_acquire);
  iree_hal_amd_gpu_byte_storage_t** link = &storage->retired;
  while (*link) {
    iree_hal_amd_gpu_byte_storage_t* generation = *link;
    if (generation->end <= consumed) {
      *link = generation->next;
      iree_hal_amd_gpu_byte_storage_free(context, generation);
    } else {
      link = &generation->next;
    }
  }
}
