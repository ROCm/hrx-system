// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/fabric/transient_buffer_pool.h"

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/debugging.h"
#include "iree/base/internal/math.h"

enum {
  IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_CHUNK_SIZE = 2 * 1024 * 1024,
  IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_ALIGNMENT = 64,
};

struct iree_hal_amd_fabric_transient_buffer_chunk_t {
  // Pool dominating all occupied slots and concurrent release callers.
  iree_hal_amd_fabric_transient_buffer_pool_t* pool;
  // Original host allocation, before cache-line alignment padding.
  void* allocation;
  // Next available chunk of the same slot size; protected by the pool lock.
  iree_hal_amd_fabric_transient_buffer_chunk_t* next;
  // Log2 of slot capacity; pointer subtraction and this shift recover its ID.
  uint32_t slot_shift;
  // Byte offset from this header to the first slot.
  uint32_t data_offset;
  // Number of complete slots in this chunk.
  uint32_t slot_count;
  // Number of bitmap words covering the slots.
  uint32_t word_count;
  // Next bitmap word examined by the serialized acquisition owner.
  uint32_t word_cursor;
  // Available slots, counted after their free bits have been published.
  iree_atomic_int32_t free_count;
  // Set bits identify free slots. Release publishes its bit before its count.
  iree_atomic_int64_t words[];
};

void iree_hal_amd_fabric_transient_buffer_pool_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amd_fabric_transient_buffer_pool_t* out_pool) {
  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&out_pool->mutex);
  iree_slim_mutex_initialize(&out_pool->growth_mutex);
}

void iree_hal_amd_fabric_transient_buffer_pool_trim(
    iree_hal_amd_fabric_transient_buffer_pool_t* pool) {
  iree_hal_amd_fabric_transient_buffer_chunk_t* retired = NULL;
  iree_slim_mutex_lock(&pool->mutex);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(pool->available); ++i) {
    iree_hal_amd_fabric_transient_buffer_chunk_t** link = &pool->available[i];
    while (*link) {
      iree_hal_amd_fabric_transient_buffer_chunk_t* chunk = *link;
      if ((uint32_t)iree_atomic_load(&chunk->free_count,
                                     iree_memory_order_acquire) ==
          chunk->slot_count) {
        *link = chunk->next;
        chunk->next = retired;
        retired = chunk;
        --pool->chunk_count;
      } else {
        link = &chunk->next;
      }
    }
  }
  iree_slim_mutex_unlock(&pool->mutex);
  while (retired) {
    iree_hal_amd_fabric_transient_buffer_chunk_t* next = retired->next;
    iree_allocator_free(pool->host_allocator, retired->allocation);
    retired = next;
  }
}

void iree_hal_amd_fabric_transient_buffer_pool_deinitialize(
    iree_hal_amd_fabric_transient_buffer_pool_t* pool) {
  iree_hal_amd_fabric_transient_buffer_pool_trim(pool);
  IREE_ASSERT(pool->chunk_count == 0,
              "transient buffer storage still has live slots");
  iree_slim_mutex_deinitialize(&pool->growth_mutex);
  iree_slim_mutex_deinitialize(&pool->mutex);
}

// Initializes private storage without excluding cached acquisition or release.
// Publication and the first slot acquisition happen together under pool->mutex.
static iree_status_t iree_hal_amd_fabric_transient_buffer_chunk_allocate(
    iree_hal_amd_fabric_transient_buffer_pool_t* pool, uint32_t slot_shift,
    iree_hal_amd_fabric_transient_buffer_chunk_t** out_chunk) {
  const iree_host_size_t slot_size = (iree_host_size_t)1 << slot_shift;
  const iree_host_size_t alignment =
      IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_ALIGNMENT;
  // A large native table still fits one slot in a correspondingly larger
  // chunk; ordinary classes share the same 2 MiB growth unit.
  iree_host_size_t chunk_size = 0;
  if (!iree_host_size_checked_add(
          slot_size,
          iree_host_align(sizeof(**out_chunk) + sizeof(iree_atomic_int64_t),
                          alignment),
          &chunk_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "transient metadata chunk size overflow");
  }
  chunk_size =
      iree_max(chunk_size, IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_CHUNK_SIZE);
  const uint32_t maximum_slots = (uint32_t)(chunk_size >> slot_shift);
  const uint32_t maximum_words = (maximum_slots + 63) / 64;
  const uint32_t data_offset = (uint32_t)iree_host_align(
      sizeof(**out_chunk) + maximum_words * sizeof(iree_atomic_int64_t),
      alignment);
  const uint32_t slot_count =
      (uint32_t)((chunk_size - data_offset) >> slot_shift);
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(chunk_size, alignment - 1,
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "transient metadata alignment size overflow");
  }
  void* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      pool->host_allocator, allocation_size, &allocation));
  iree_hal_amd_fabric_transient_buffer_chunk_t* chunk =
      (void*)iree_host_align((uintptr_t)allocation, alignment);
  *chunk = (iree_hal_amd_fabric_transient_buffer_chunk_t){
      .pool = pool,
      .allocation = allocation,
      .slot_shift = slot_shift,
      .data_offset = data_offset,
      .slot_count = slot_count,
      .word_count = (slot_count + 63) / 64,
  };
  iree_atomic_store(&chunk->free_count, slot_count, iree_memory_order_relaxed);
  for (uint32_t i = 0; i < chunk->word_count; ++i) {
    const uint32_t bits = iree_min(slot_count - i * 64, 64);
    const uint64_t mask = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
    iree_atomic_store(&chunk->words[i], (int64_t)mask,
                      iree_memory_order_relaxed);
  }
  IREE_ASAN_POISON_MEMORY_REGION((uint8_t*)chunk + data_offset,
                                 slot_size * slot_count);
  *out_chunk = chunk;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_fabric_transient_buffer_pool_acquire(
    iree_hal_amd_fabric_transient_buffer_pool_t* pool,
    iree_host_size_t byte_length, iree_host_size_t max_count,
    iree_hal_amd_fabric_transient_buffer_slots_t* out_slots) {
  memset(out_slots, 0, sizeof(*out_slots));
  const uint32_t slot_shift =
      iree_max(IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_MIN_SHIFT,
               64u - iree_math_count_leading_zeros_u64(byte_length - 1));
  if (slot_shift >= sizeof(iree_host_size_t) * 8) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "transient metadata slot size overflow");
  }
  const uint32_t index =
      slot_shift - IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_MIN_SHIFT;
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_amd_fabric_transient_buffer_chunk_t* chunk = pool->available[index];
  iree_status_t status = iree_ok_status();
  if (!chunk) {
    iree_slim_mutex_unlock(&pool->mutex);
    iree_slim_mutex_lock(&pool->growth_mutex);
    iree_slim_mutex_lock(&pool->mutex);
    // A prior grower or a returning slot may already have supplied this class.
    chunk = pool->available[index];
    if (!chunk) {
      iree_slim_mutex_unlock(&pool->mutex);
      status = iree_hal_amd_fabric_transient_buffer_chunk_allocate(
          pool, slot_shift, &chunk);
      iree_slim_mutex_lock(&pool->mutex);
      if (iree_status_is_ok(status)) {
        // Preserve chunks returned while allocation was in progress. Keep
        // exclusion through the first claim so trim cannot reclaim this chunk.
        chunk->next = pool->available[index];
        pool->available[index] = chunk;
        ++pool->chunk_count;
      }
    }
    iree_slim_mutex_unlock(&pool->growth_mutex);
  }
  if (iree_status_is_ok(status)) {
    // A free bit can precede its count publication. Limiting acquisition to
    // the published count keeps that in-flight release independently owned.
    const uint32_t available = (uint32_t)iree_atomic_load(
        &chunk->free_count, iree_memory_order_acquire);
    uint32_t word_index = chunk->word_cursor;
    uint64_t word = (uint64_t)iree_atomic_load(&chunk->words[word_index],
                                               iree_memory_order_acquire);
    while (!word) {
      word_index = word_index + 1 == chunk->word_count ? 0 : word_index + 1;
      word = (uint64_t)iree_atomic_load(&chunk->words[word_index],
                                        iree_memory_order_acquire);
    }
    chunk->word_cursor = word_index;
    const uint32_t first = iree_math_count_trailing_zeros_u64(word);
    const uint32_t run =
        word == UINT64_MAX
            ? 64
            : iree_math_count_trailing_zeros_u64(~(word >> first));
    const uint32_t count = (uint32_t)iree_min(
        max_count, iree_min(available, iree_min(run, 64 - first)));
    const uint64_t mask =
        (count == 64 ? UINT64_MAX : (UINT64_C(1) << count) - 1) << first;
    iree_atomic_fetch_and(&chunk->words[word_index], (int64_t)~mask,
                          iree_memory_order_acq_rel);
    if ((uint32_t)iree_atomic_fetch_sub(&chunk->free_count, count,
                                        iree_memory_order_acq_rel) == count) {
      pool->available[index] = chunk->next;
      chunk->next = NULL;
    }
    const iree_host_size_t stride = (iree_host_size_t)1 << slot_shift;
    uint8_t* data = (uint8_t*)chunk + chunk->data_offset +
                    (word_index * 64 + first) * stride;
    IREE_ASAN_UNPOISON_MEMORY_REGION(data, count * stride);
    *out_slots = (iree_hal_amd_fabric_transient_buffer_slots_t){
        .chunk = chunk, .data = data, .stride = stride, .count = count};
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

void iree_hal_amd_fabric_transient_buffer_pool_release(
    iree_hal_amd_fabric_transient_buffer_chunk_t* chunk, void* data) {
  iree_hal_amd_fabric_transient_buffer_pool_t* pool = chunk->pool;
  const uint32_t slot_shift = chunk->slot_shift;
  const uint32_t slot =
      (uint32_t)(((uint8_t*)data - (uint8_t*)chunk - chunk->data_offset) >>
                 slot_shift);
  IREE_ASAN_POISON_MEMORY_REGION(data, (iree_host_size_t)1 << slot_shift);
  iree_atomic_fetch_or(&chunk->words[slot / 64],
                       (int64_t)(UINT64_C(1) << (slot % 64)),
                       iree_memory_order_release);
  // This is the final chunk access unless we own its unavailable->available
  // transition. In that case trim cannot see it until we publish under lock.
  if (iree_atomic_fetch_add(&chunk->free_count, 1, iree_memory_order_acq_rel) ==
      0) {
    const uint32_t index =
        slot_shift - IREE_HAL_AMD_FABRIC_TRANSIENT_BUFFER_MIN_SHIFT;
    iree_slim_mutex_lock(&pool->mutex);
    chunk->next = pool->available[index];
    pool->available[index] = chunk;
    iree_slim_mutex_unlock(&pool->mutex);
  }
}

iree_host_size_t iree_hal_amd_fabric_transient_buffer_slot_capacity(
    const iree_hal_amd_fabric_transient_buffer_chunk_t* chunk) {
  return (iree_host_size_t)1 << chunk->slot_shift;
}

iree_hal_amd_fabric_transient_buffer_pool_t*
iree_hal_amd_fabric_transient_buffer_slot_pool(
    const iree_hal_amd_fabric_transient_buffer_chunk_t* chunk) {
  return chunk->pool;
}
