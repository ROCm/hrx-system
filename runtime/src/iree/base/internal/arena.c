// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/debugging.h"
#include "iree/base/internal/math.h"

// Per-allocation and free-list zones can overwhelm traces of compiler work.
// Backing allocations, oversized allocations, and pool lifecycle operations
// remain visible without enabling these fine-grained operation zones.
#ifndef IREE_ARENA_TRACE_ENABLE_OPERATIONS
#define IREE_ARENA_TRACE_ENABLE_OPERATIONS 0
#endif  // IREE_ARENA_TRACE_ENABLE_OPERATIONS

#if IREE_ARENA_TRACE_ENABLE_OPERATIONS
#define IREE_ARENA_TRACE_ZONE_BEGIN IREE_TRACE_ZONE_BEGIN
#define IREE_ARENA_TRACE_ZONE_BEGIN_NAMED IREE_TRACE_ZONE_BEGIN_NAMED
#define IREE_ARENA_TRACE_ZONE_END IREE_TRACE_ZONE_END
#define IREE_ARENA_RETURN_AND_END_ZONE_IF_ERROR \
  IREE_RETURN_AND_END_ZONE_IF_ERROR
#else
#define IREE_ARENA_TRACE_ZONE_BEGIN(...)
#define IREE_ARENA_TRACE_ZONE_BEGIN_NAMED(...)
#define IREE_ARENA_TRACE_ZONE_END(...)
#define IREE_ARENA_RETURN_AND_END_ZONE_IF_ERROR(zone_id, ...) \
  IREE_RETURN_IF_ERROR(__VA_ARGS__)
#endif  // IREE_ARENA_TRACE_ENABLE_OPERATIONS

//===----------------------------------------------------------------------===//
// iree_arena_block_pool_t
//===----------------------------------------------------------------------===//

#if defined(IREE_SANITIZER_THREAD)
static void iree_arena_block_pool_release_tsan(iree_arena_block_t* block_head,
                                               iree_arena_block_t* block_tail) {
  iree_arena_block_t* block = block_head;
  while (block) {
    iree_arena_block_t* next = block->next;
    IREE_TSAN_RELEASE(block);
    if (block == block_tail) {
      break;
    }
    block = next;
  }
}
#endif  // IREE_SANITIZER_THREAD

void iree_arena_block_pool_initialize(iree_host_size_t total_block_size,
                                      iree_allocator_t block_allocator,
                                      iree_arena_block_pool_t* out_block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t aligned_total_block_size = 0;
  const bool can_align_total_block_size = iree_host_size_checked_align(
      total_block_size, iree_alignof(iree_arena_block_t),
      &aligned_total_block_size);
  IREE_ASSERT(can_align_total_block_size, "block size too large");
  if (IREE_LIKELY(can_align_total_block_size)) {
    total_block_size = aligned_total_block_size;
  }

  memset(out_block_pool, 0, sizeof(*out_block_pool));
  IREE_ASSERT(total_block_size >= sizeof(iree_arena_block_t),
              "block size too small for header");
  out_block_pool->total_block_size = total_block_size;
  out_block_pool->usable_block_size =
      total_block_size - sizeof(iree_arena_block_t);
  out_block_pool->block_allocator = block_allocator;
  iree_atomic_arena_block_slist_initialize(&out_block_pool->available_slist);
  IREE_STATISTICS({
    iree_atomic_store(&out_block_pool->statistics.block_system_allocation_count,
                      0, iree_memory_order_relaxed);
    iree_atomic_store(&out_block_pool->statistics.oversized_allocation_count, 0,
                      iree_memory_order_relaxed);
    iree_atomic_store(&out_block_pool->statistics.oversized_allocation_bytes, 0,
                      iree_memory_order_relaxed);
  });

  IREE_TRACE_ZONE_END(z0);
}

void iree_arena_block_pool_deinitialize(iree_arena_block_pool_t* block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Since all blocks must have been released we can just reuse trim (today) as
  // it doesn't retain any blocks.
  iree_arena_block_pool_trim(block_pool);
  iree_atomic_arena_block_slist_deinitialize(&block_pool->available_slist);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_arena_block_pool_preallocate(
    iree_arena_block_pool_t* block_pool, iree_host_size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);

  for (iree_host_size_t i = 0; i < count; ++i) {
    uint8_t* block_base = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc_uninitialized(block_pool->block_allocator,
                                                block_pool->total_block_size,
                                                (void**)&block_base));
    iree_arena_block_t* block =
        iree_arena_block_trailer(block_pool, block_base);
    IREE_STATISTICS(iree_atomic_fetch_add(
        &block_pool->statistics.block_system_allocation_count, 1,
        iree_memory_order_relaxed));
#if defined(IREE_SANITIZER_THREAD)
    IREE_TSAN_RELEASE(block);
#endif  // IREE_SANITIZER_THREAD
    iree_atomic_arena_block_slist_concat(&block_pool->available_slist, block,
                                         block);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_arena_block_pool_trim(iree_arena_block_pool_t* block_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_arena_block_t* head = NULL;
  iree_atomic_arena_block_slist_flush(
      &block_pool->available_slist,
      IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO, &head, NULL);
  while (head) {
    void* ptr = (uint8_t*)head - block_pool->usable_block_size;
    head = head->next;
    iree_allocator_free(block_pool->block_allocator, ptr);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_arena_block_pool_query_statistics(
    const iree_arena_block_pool_t* block_pool,
    iree_arena_block_pool_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_statistics);
  memset(out_statistics, 0, sizeof(*out_statistics));
  IREE_STATISTICS({
    out_statistics->block_system_allocation_count = (uint64_t)iree_atomic_load(
        &block_pool->statistics.block_system_allocation_count,
        iree_memory_order_relaxed);
    out_statistics->block_system_allocation_bytes =
        iree_math_saturating_mul_u64(
            out_statistics->block_system_allocation_count,
            block_pool->total_block_size);
    out_statistics->oversized_allocation_count = (uint64_t)iree_atomic_load(
        &block_pool->statistics.oversized_allocation_count,
        iree_memory_order_relaxed);
    out_statistics->oversized_allocation_bytes = (uint64_t)iree_atomic_load(
        &block_pool->statistics.oversized_allocation_bytes,
        iree_memory_order_relaxed);
  });
}

iree_status_t iree_arena_block_pool_acquire(iree_arena_block_pool_t* block_pool,
                                            iree_arena_block_t** out_block,
                                            void** out_ptr) {
  IREE_ARENA_TRACE_ZONE_BEGIN(z0);

  iree_arena_block_t* block =
      iree_atomic_arena_block_slist_pop(&block_pool->available_slist);

  if (!block) {
    // No blocks available; allocate one now.
    // Note that it's possible for there to be a race here where one thread
    // releases a block to the pool while we are trying to acquire one - in that
    // case we may end up allocating a block when perhaps we didn't need to but
    // that's fine - it's just one block and the contention means there's likely
    // to be a need for more anyway.
    IREE_TRACE_ZONE_BEGIN_NAMED(z_allocate, "iree_arena_block_pool_allocate");
    uint8_t* block_base = NULL;
    iree_status_t status = iree_allocator_malloc_uninitialized(
        block_pool->block_allocator, block_pool->total_block_size,
        (void**)&block_base);
    IREE_TRACE_ZONE_END(z_allocate);
    IREE_ARENA_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
    block = iree_arena_block_trailer(block_pool, block_base);
    *out_ptr = block_base;
    IREE_STATISTICS(iree_atomic_fetch_add(
        &block_pool->statistics.block_system_allocation_count, 1,
        iree_memory_order_relaxed));
  } else {
#if defined(IREE_SANITIZER_THREAD)
    IREE_TSAN_ACQUIRE(block);
#endif  // IREE_SANITIZER_THREAD
    *out_ptr = iree_arena_block_ptr(block_pool, block);
  }

  block->next = NULL;
  *out_block = block;

  // Payload starts a new allocation lifetime even when the backing allocator
  // or a previous owner initialized it. The free-list trailer remains defined.
  IREE_MSAN_ALLOCATED_MEMORY(*out_ptr, block_pool->usable_block_size);

  IREE_ARENA_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_arena_block_pool_release(iree_arena_block_pool_t* block_pool,
                                   iree_arena_block_t* block_head,
                                   iree_arena_block_t* block_tail) {
  IREE_ARENA_TRACE_ZONE_BEGIN(z0);
#if defined(IREE_SANITIZER_THREAD)
  // Arena blocks are custom-freed back to this pool and later reused without
  // malloc/free. Publish the ownership transfer so TSAN does not treat the next
  // user of the block as racing with the previous owner.
  iree_arena_block_pool_release_tsan(block_head, block_tail);
#endif  // IREE_SANITIZER_THREAD
  iree_atomic_arena_block_slist_concat(&block_pool->available_slist, block_head,
                                       block_tail);
  IREE_ARENA_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_arena_allocator_t
//===----------------------------------------------------------------------===//

void iree_arena_initialize(iree_arena_block_pool_t* block_pool,
                           iree_arena_allocator_t* out_arena) {
  memset(out_arena, 0, sizeof(*out_arena));
  out_arena->block_pool = block_pool;
}

void iree_arena_deinitialize(iree_arena_allocator_t* arena) {
  iree_arena_reset(arena);
}

void iree_arena_reset(iree_arena_allocator_t* arena) {
  IREE_ARENA_TRACE_ZONE_BEGIN(z0);

  // Snapshot block pool + per-arena state BEFORE clearing or releasing
  // anything. The arena struct itself may live within one of its own blocks
  // (e.g., when an embedded object owns its enclosing arena), in which case
  // once iree_arena_block_pool_release returns those blocks become available
  // for reuse by another thread and any subsequent write to the arena's
  // fields would corrupt the next owner's data. Clearing the fields up
  // front avoids that use-after-release.
  iree_arena_block_pool_t* block_pool = arena->block_pool;
  iree_arena_oversized_allocation_t* allocation_head = arena->allocation_head;
  iree_arena_block_t* block_head = arena->block_head;
  iree_arena_block_t* block_tail = arena->block_tail;

  arena->allocation_head = NULL;
  arena->block_head = NULL;
  arena->block_tail = NULL;
  arena->total_allocation_size = 0;
  arena->used_allocation_size = 0;
  arena->block_bytes_remaining = 0;

  if (allocation_head != NULL) {
    iree_arena_oversized_allocation_t* head = allocation_head;
    do {
      void* ptr = (void*)head;
      head = head->next;
      iree_allocator_free(block_pool->block_allocator, ptr);
    } while (head);
  }

  if (block_head != NULL) {
#if defined(IREE_SANITIZER_ADDRESS)
    iree_arena_block_t* block = block_head;
    while (block) {
      IREE_ASAN_UNPOISON_MEMORY_REGION(iree_arena_block_ptr(block_pool, block),
                                       block_pool->usable_block_size);
      block = block->next;
    }
#endif  // IREE_SANITIZER_ADDRESS
    iree_arena_block_pool_release(block_pool, block_head, block_tail);
  }

  IREE_ARENA_TRACE_ZONE_END(z0);
}

void iree_arena_transfer(iree_arena_allocator_t* source,
                         iree_arena_allocator_t* target) {
  IREE_ASSERT(source != target && source->block_pool == target->block_pool,
              "arena transfer requires distinct arenas sharing a block pool");
  if (source->allocation_head != NULL) {
    iree_arena_oversized_allocation_t* tail = source->allocation_head;
    while (tail->next != NULL) {
      tail = tail->next;
    }
    tail->next = target->allocation_head;
    target->allocation_head = source->allocation_head;
  }
  if (source->block_head != NULL) {
    source->block_tail->next = target->block_head;
    if (target->block_tail == NULL) {
      target->block_tail = source->block_tail;
    }
    target->block_head = source->block_head;
    target->block_bytes_remaining = source->block_bytes_remaining;
  }
  target->total_allocation_size += source->total_allocation_size;
  target->used_allocation_size += source->used_allocation_size;
  iree_arena_initialize(source->block_pool, source);
}

iree_arena_checkpoint_t iree_arena_checkpoint_save(
    iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(arena);
  return (iree_arena_checkpoint_t){
      .arena = arena,
      .allocation_head = arena->allocation_head,
      .block_head = arena->block_head,
      .total_allocation_size = arena->total_allocation_size,
      .used_allocation_size = arena->used_allocation_size,
      .block_bytes_remaining = arena->block_bytes_remaining,
  };
}

// Restores a checkpoint after the arena acquired blocks or oversized
// allocations. This path is kept out of line so same-block scratch rewinds do
// not carry list traversal and release machinery in their instruction working
// set.
//
// The checkpoint and arena may live in arena storage. The checkpoint is copied
// before any mutation, and the arena state is published before blocks become
// available to other threads so neither is accessed after release.
static IREE_ATTRIBUTE_NOINLINE IREE_ATTRIBUTE_COLD void
iree_arena_checkpoint_restore_acquired_storage(
    const iree_arena_checkpoint_t* checkpoint) {
  const iree_arena_checkpoint_t saved_state = *checkpoint;
  iree_arena_allocator_t* arena = saved_state.arena;

  iree_arena_oversized_allocation_t* released_allocation_head =
      arena->allocation_head;
  iree_arena_oversized_allocation_t* allocation = released_allocation_head;
  while (allocation != saved_state.allocation_head) {
    IREE_ASSERT(allocation != NULL);
    allocation = allocation->next;
  }

  iree_arena_block_t* released_block_head = arena->block_head;
  iree_arena_block_t* released_block_tail = NULL;
  iree_arena_block_t* block = released_block_head;
  while (block != saved_state.block_head) {
    IREE_ASSERT(block != NULL);
    released_block_tail = block;
    block = block->next;
  }
  if (released_block_tail != NULL) {
    released_block_tail->next = NULL;
  } else {
    released_block_head = NULL;
  }
  iree_arena_block_t* retained_block_tail =
      saved_state.block_head != NULL ? arena->block_tail : NULL;

  iree_arena_block_pool_t* block_pool = arena->block_pool;
  arena->allocation_head = saved_state.allocation_head;
  arena->block_head = saved_state.block_head;
  arena->block_tail = retained_block_tail;
  arena->total_allocation_size = saved_state.total_allocation_size;
  arena->used_allocation_size = saved_state.used_allocation_size;
  arena->block_bytes_remaining = saved_state.block_bytes_remaining;

  while (released_allocation_head != saved_state.allocation_head) {
    allocation = released_allocation_head;
    released_allocation_head = allocation->next;
    iree_allocator_free(block_pool->block_allocator, allocation);
  }

  if (released_block_head != NULL) {
#if defined(IREE_SANITIZER_ADDRESS)
    block = released_block_head;
    while (block != NULL) {
      IREE_ASAN_UNPOISON_MEMORY_REGION(iree_arena_block_ptr(block_pool, block),
                                       block_pool->usable_block_size);
      block = block->next;
    }
#endif  // IREE_SANITIZER_ADDRESS
    iree_arena_block_pool_release(block_pool, released_block_head,
                                  released_block_tail);
  }

  if (saved_state.block_head != NULL &&
      saved_state.block_bytes_remaining != 0) {
    void* free_ptr =
        (uint8_t*)saved_state.block_head - saved_state.block_bytes_remaining;
    IREE_ASAN_POISON_MEMORY_REGION(free_ptr, saved_state.block_bytes_remaining);
    IREE_MSAN_ALLOCATED_MEMORY(free_ptr, saved_state.block_bytes_remaining);
  }
}

void iree_arena_checkpoint_restore(const iree_arena_checkpoint_t* checkpoint) {
  IREE_ASSERT_ARGUMENT(checkpoint);
  IREE_ARENA_TRACE_ZONE_BEGIN(z0);

  iree_arena_allocator_t* arena = checkpoint->arena;
  IREE_ASSERT_ARGUMENT(arena);

  // Releasing acquired storage is uncommon for scratch checkpoints and may
  // publish the arena's own containing block. Keep that ownership transition
  // in the out-of-line path and return without touching the arena again.
  if (IREE_UNLIKELY(arena->allocation_head != checkpoint->allocation_head ||
                    arena->block_head != checkpoint->block_head)) {
    iree_arena_checkpoint_restore_acquired_storage(checkpoint);
    IREE_ARENA_TRACE_ZONE_END(z0);
    return;
  }

  // No ownership lists changed, so total storage and list links are already
  // identical to the checkpoint. Rewind only the bump-pointer accounting.
  const iree_host_size_t used_allocation_size =
      checkpoint->used_allocation_size;
  const iree_host_size_t block_bytes_remaining =
      checkpoint->block_bytes_remaining;
  iree_arena_block_t* block_head = checkpoint->block_head;
#if defined(IREE_SANITIZER_MEMORY)
  const iree_host_size_t discarded_length =
      block_bytes_remaining - arena->block_bytes_remaining;
#endif  // IREE_SANITIZER_MEMORY
  arena->used_allocation_size = used_allocation_size;
  arena->block_bytes_remaining = block_bytes_remaining;
  if (block_head != NULL && block_bytes_remaining != 0) {
    void* free_ptr = (uint8_t*)block_head - block_bytes_remaining;
    IREE_ASAN_POISON_MEMORY_REGION(free_ptr, block_bytes_remaining);
#if defined(IREE_SANITIZER_MEMORY)
    // Unused capacity is already uninitialized. Only the discarded allocations
    // need new shadow and origins; small scratch rewinds do not touch the tail.
    if (discarded_length != 0) {
      IREE_MSAN_ALLOCATED_MEMORY(free_ptr, discarded_length);
    }
#endif  // IREE_SANITIZER_MEMORY
  }

  IREE_ARENA_TRACE_ZONE_END(z0);
}

iree_status_t iree_arena_allocate(iree_arena_allocator_t* arena,
                                  iree_host_size_t byte_length,
                                  void** out_ptr) {
  *out_ptr = NULL;

  iree_arena_block_pool_t* block_pool = arena->block_pool;

  // Pad length allocated so that each pointer bump is always ending at an
  // aligned address and the next allocation will start aligned.
  iree_host_size_t aligned_length = 0;
  if (!iree_host_size_checked_align(byte_length, iree_max_align_t,
                                    &aligned_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "alignment overflow");
  }

  if (aligned_length > block_pool->usable_block_size) {
    // Oversized allocation that can't be handled by the block pool. We'll
    // allocate directly from the system allocator and track it ourselves for
    // freeing during reset.
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_arena_allocate_oversize");
    iree_host_size_t allocation_size = 0;
    if (!iree_host_size_checked_add(sizeof(iree_arena_oversized_allocation_t),
                                    byte_length, &allocation_size)) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "oversized allocation size overflow");
    }
    iree_arena_oversized_allocation_t* allocation = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_allocator_malloc_uninitialized(
            block_pool->block_allocator, allocation_size, (void**)&allocation));
    allocation->next = arena->allocation_head;
    arena->allocation_head = allocation;
    arena->total_allocation_size += allocation_size;
    arena->used_allocation_size += byte_length;
    IREE_STATISTICS({
      iree_atomic_fetch_add(&block_pool->statistics.oversized_allocation_count,
                            1, iree_memory_order_relaxed);
      iree_atomic_fetch_add(&block_pool->statistics.oversized_allocation_bytes,
                            (int64_t)allocation_size,
                            iree_memory_order_relaxed);
    });
    *out_ptr = (uint8_t*)allocation + sizeof(iree_arena_oversized_allocation_t);
    IREE_MSAN_ALLOCATED_MEMORY(*out_ptr, byte_length);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Acquired blocks and rewound capacity have uninitialized MSAN shadow, so
  // bump allocations inherit it without a per-allocation annotation.
  // Check to see if the current block (if any) has space - if not, get another.
  if (arena->block_head == NULL ||
      arena->block_bytes_remaining < aligned_length) {
    IREE_ARENA_TRACE_ZONE_BEGIN_NAMED(z0, "iree_arena_allocate_grow");
    iree_arena_block_t* block = NULL;
    void* ptr = NULL;
    IREE_ARENA_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_block_pool_acquire(arena->block_pool, &block, &ptr));
    block->next = arena->block_head;
    arena->block_head = block;
    if (!arena->block_tail) {
      arena->block_tail = block;
    }
    arena->total_allocation_size += block_pool->total_block_size;
    arena->block_bytes_remaining = block_pool->usable_block_size;
    IREE_ASAN_POISON_MEMORY_REGION(
        iree_arena_block_ptr(block_pool, arena->block_head),
        block_pool->usable_block_size);
    IREE_ARENA_TRACE_ZONE_END(z0);
  }

  // Slice out the allocation from the current block.
  void* ptr = (uint8_t*)arena->block_head - arena->block_bytes_remaining;
  IREE_ASAN_UNPOISON_MEMORY_REGION(ptr, aligned_length);
  arena->block_bytes_remaining -= aligned_length;
  arena->used_allocation_size += aligned_length;
  *out_ptr = ptr;
  return iree_ok_status();
}

iree_status_t iree_arena_allocate_aligned(iree_arena_allocator_t* arena,
                                          iree_host_size_t byte_length,
                                          iree_host_size_t min_alignment,
                                          void** out_ptr) {
  // When the requested alignment is at or below the arena's natural alignment
  // no padding is needed — every arena allocation is already aligned to
  // iree_max_align_t.
  if (min_alignment <= iree_max_align_t) {
    return iree_arena_allocate(arena, byte_length, out_ptr);
  }

  // Over-allocate by up to (alignment - 1) bytes so we can align the pointer
  // forward. The arena doesn't support individual frees, so the padding bytes
  // are simply wasted until the arena is reset.
  iree_host_size_t padded_length = 0;
  if (!iree_host_size_checked_add(byte_length, min_alignment - 1,
                                  &padded_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "aligned allocation size overflow");
  }
  void* raw_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, padded_length, &raw_ptr));
  *out_ptr =
      (void*)(((uintptr_t)raw_ptr + min_alignment - 1) & ~(min_alignment - 1));
  return iree_ok_status();
}

iree_status_t iree_arena_allocate_array(iree_arena_allocator_t* arena,
                                        iree_host_size_t count,
                                        iree_host_size_t element_size,
                                        void** out_ptr) {
  IREE_ARENA_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t byte_size = 0;
  if (!iree_host_size_checked_mul(count, element_size, &byte_size)) {
    IREE_ARENA_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "array size overflow: %" PRIhsz " * %" PRIhsz,
                            count, element_size);
  }
  iree_status_t status = iree_arena_allocate(arena, byte_size, out_ptr);
  IREE_ARENA_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_arena_allocate_array_aligned(iree_arena_allocator_t* arena,
                                                iree_host_size_t count,
                                                iree_host_size_t element_size,
                                                iree_host_size_t min_alignment,
                                                void** out_ptr) {
  IREE_ARENA_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t byte_size = 0;
  if (!iree_host_size_checked_mul(count, element_size, &byte_size)) {
    IREE_ARENA_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "array size overflow: %" PRIhsz " * %" PRIhsz,
                            count, element_size);
  }
  iree_status_t status =
      iree_arena_allocate_aligned(arena, byte_size, min_alignment, out_ptr);
  IREE_ARENA_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_arena_grow_array(iree_arena_allocator_t* arena,
                                    iree_host_size_t existing_count,
                                    iree_host_size_t minimum_capacity,
                                    iree_host_size_t element_size,
                                    iree_host_size_t* inout_capacity,
                                    void** inout_ptr) {
  if (*inout_ptr != NULL && existing_count < *inout_capacity &&
      minimum_capacity <= *inout_capacity) {
    return iree_ok_status();
  }
  IREE_ARENA_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t doubled_capacity = 0;
  if (!iree_host_size_checked_mul(*inout_capacity, 2, &doubled_capacity)) {
    IREE_ARENA_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
  }
  iree_host_size_t new_capacity = iree_max(minimum_capacity, doubled_capacity);
  void* new_ptr = NULL;
  iree_status_t status =
      iree_arena_allocate_array(arena, new_capacity, element_size, &new_ptr);
  if (iree_status_is_ok(status)) {
    if (*inout_ptr && existing_count > 0) {
      memcpy(new_ptr, *inout_ptr, existing_count * element_size);
    }
    *inout_ptr = new_ptr;
    *inout_capacity = new_capacity;
  }
  IREE_ARENA_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_arena_grow_array_block_fit(iree_arena_allocator_t* arena,
                                              iree_host_size_t existing_count,
                                              iree_host_size_t minimum_capacity,
                                              iree_host_size_t element_size,
                                              iree_host_size_t* inout_capacity,
                                              void** inout_ptr) {
  if (*inout_ptr != NULL && existing_count < *inout_capacity &&
      minimum_capacity <= *inout_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t doubled_capacity = 0;
  if (element_size != 0 &&
      iree_host_size_checked_mul(*inout_capacity, 2, &doubled_capacity)) {
    const iree_host_size_t block_capacity =
        iree_arena_block_pool_max_allocation_size(arena->block_pool) /
        element_size;
    if (*inout_capacity < block_capacity &&
        minimum_capacity <= block_capacity &&
        doubled_capacity > block_capacity) {
      void* new_ptr = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, block_capacity,
                                                     element_size, &new_ptr));
      if (*inout_ptr && existing_count > 0) {
        memcpy(new_ptr, *inout_ptr, existing_count * element_size);
      }
      *inout_ptr = new_ptr;
      *inout_capacity = block_capacity;
      return iree_ok_status();
    }
  }
  return iree_arena_grow_array(arena, existing_count, minimum_capacity,
                               element_size, inout_capacity, inout_ptr);
}

static iree_status_t iree_arena_allocator_ctl(void* self,
                                              iree_allocator_command_t command,
                                              const void* params,
                                              void** inout_ptr) {
  iree_arena_allocator_t* arena = (iree_arena_allocator_t*)self;
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC: {
      const iree_allocator_alloc_params_t* alloc_params =
          (const iree_allocator_alloc_params_t*)params;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, alloc_params->byte_length, inout_ptr));
      if (command == IREE_ALLOCATOR_COMMAND_CALLOC) {
        memset(*inout_ptr, 0, alloc_params->byte_length);
      }
      return iree_ok_status();
    }
    case IREE_ALLOCATOR_COMMAND_FREE: {
      // Do nothing: can't free from an arena.
      return iree_ok_status();
    }
    case IREE_ALLOCATOR_COMMAND_TRIM:
      iree_arena_block_pool_trim(arena->block_pool);
      return iree_allocator_trim(arena->block_pool->block_allocator);
    default:
      // NOTE: we could try to support IREE_ALLOCATOR_COMMAND_REALLOC, but
      // it requires the original size to be able to do properly (without
      // copying memory we shouldn't have access to). For this and other reasons
      // we very rarely realloc in IREE so having this limitation isn't too bad.
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported iree_arena_t allocator command");
  }
}

iree_allocator_t iree_arena_allocator(iree_arena_allocator_t* arena) {
  iree_allocator_t v = {
      .self = arena,
      .ctl = iree_arena_allocator_ctl,
  };
  return v;
}
