// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/signal.h"

static_assert(sizeof(iree_hal_amd_gpu_signal_t) <= 32,
              "two signal owners must fit in one cache line");
static_assert(offsetof(iree_hal_amd_gpu_signal_pool_t, returned) == 64,
              "publisher-local bookkeeping fits in one cache line");
static_assert(sizeof(iree_hal_amd_gpu_signal_pool_t) == 128,
              "concurrent returns have their own cache line");
static_assert(sizeof(iree_hal_amd_gpu_signal_block_t) <= 64,
              "signal block bookkeeping fits in one cache line");

static_assert(sizeof(iree_hal_amd_gpu_signal_frontiers_t) == 16,
              "frontier storage has two eight-byte header words");

static void iree_hal_amd_gpu_signal_frontiers_initialize(
    uint32_t capacity, uint32_t queue_count,
    iree_hal_amd_gpu_signal_frontiers_t* out_frontiers) {
  *out_frontiers = (iree_hal_amd_gpu_signal_frontiers_t){
      .capacity = capacity,
      .queue_count = queue_count,
  };
  memset(out_frontiers + 1, 0,
         (iree_host_size_t)capacity * queue_count * sizeof(uint64_t));
}

void iree_hal_amd_gpu_signal_frontiers_free(
    iree_hal_amd_gpu_signal_frontiers_t* frontiers,
    iree_hal_amd_gpu_signal_frontiers_t* initial,
    iree_allocator_t host_allocator) {
  while (frontiers != initial) {
    iree_hal_amd_gpu_signal_frontiers_t* previous = frontiers->previous;
    iree_allocator_free(host_allocator, frontiers);
    frontiers = previous;
  }
}

void iree_hal_amd_gpu_signal_pool_initialize(
    iree_hal_amd_gpu_signal_pool_notification_t notification,
    iree_hal_amd_gpu_signal_pool_t* out_pool) {
  *out_pool = (iree_hal_amd_gpu_signal_pool_t){.notification = notification};
}

void iree_hal_amd_gpu_signal_pool_deinitialize(
    iree_hal_amd_gpu_signal_pool_t* pool) {
  IREE_ASSERT_EQ(pool->block_count, 0, "native signal blocks remain attached");
}

// Only the exclusive publisher changes block membership and free cells.
static void iree_hal_amd_gpu_signal_block_link(
    iree_hal_amd_gpu_signal_block_t* block) {
  iree_hal_amd_gpu_signal_pool_t* pool = block->pool;
  block->available_next = pool->available;
  if (pool->available) {
    pool->available->available_previous = block;
  }
  pool->available = block;
}

static void iree_hal_amd_gpu_signal_block_unlink(
    iree_hal_amd_gpu_signal_block_t* block) {
  if (block->available_previous) {
    block->available_previous->available_next = block->available_next;
  } else {
    block->pool->available = block->available_next;
  }
  if (block->available_next) {
    block->available_next->available_previous = block->available_previous;
  }
  block->available_previous = block->available_next = NULL;
}

// Pending returns keep their blocks live until the publisher either reuses the
// cell or reconciles its free-list membership here. Only cold trim drains a
// complete batch; ordinary acquisition pops a single cell in constant work.
static void iree_hal_amd_gpu_signal_pool_reconcile(
    iree_hal_amd_gpu_signal_pool_t* pool) {
  iree_hal_amd_gpu_signal_t* batches[] = {
      pool->recycled, (iree_hal_amd_gpu_signal_t*)iree_atomic_exchange(
                          &pool->returned, 0, iree_memory_order_acq_rel)};
  pool->recycled = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(batches); ++i) {
    iree_hal_amd_gpu_signal_t* signal = batches[i];
    while (signal) {
      iree_hal_amd_gpu_signal_t* next = signal->state.next;
      iree_hal_amd_gpu_signal_block_t* block = signal->block;
      if (!block->free_head) {
        iree_hal_amd_gpu_signal_block_link(block);
      }
      signal->state.next = block->free_head;
      block->free_head = signal;
      --block->live_count;
      signal = next;
    }
  }
}

void iree_hal_amd_gpu_signal_block_initialize(
    iree_hal_amd_gpu_signal_pool_t* pool, const void* domain, void* storage,
    uint64_t device_address, uint32_t stride, uint32_t value_offset,
    iree_hal_amd_gpu_signal_t* signals, uint32_t capacity, uint32_t queue_count,
    void* frontier_storage, iree_hal_amd_gpu_signal_block_t* out_block) {
  iree_hal_amd_gpu_signal_frontiers_t* frontiers = frontier_storage;
  iree_hal_amd_gpu_signal_frontiers_initialize(capacity, queue_count,
                                               frontiers);
  *out_block = (iree_hal_amd_gpu_signal_block_t){
      .pool = pool,
      .free_head = capacity ? signals : NULL,
      .domain = domain,
      .signals = signals,
  };
  iree_atomic_store(&out_block->frontiers, (intptr_t)frontiers,
                    iree_memory_order_relaxed);
  for (uint32_t i = 0; i < capacity; ++i) {
    const uint64_t offset = (uint64_t)i * stride;
    uint64_t* value = (uint64_t*)((uint8_t*)storage + offset + value_offset);
    *value = 0;
    signals[i] = (iree_hal_amd_gpu_signal_t){
        .block = out_block,
        .device_address = device_address + offset,
        .host_value = value,
        .state.next = i + 1 < capacity ? &signals[i + 1] : NULL,
    };
  }
  ++pool->block_count;
  if (capacity) {
    iree_hal_amd_gpu_signal_block_link(out_block);
  }
}

iree_status_t iree_hal_amd_gpu_signal_block_reserve_frontiers(
    iree_hal_amd_gpu_signal_block_t* block, uint32_t queue_count,
    iree_allocator_t host_allocator) {
  iree_hal_amd_gpu_signal_frontiers_t* previous =
      iree_hal_amd_gpu_signal_block_frontiers(block);
  if (queue_count <= previous->queue_count) {
    return iree_ok_status();
  }
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(*previous), &total_size,
                         IREE_STRUCT_ARRAY_FIELD(previous->capacity,
                                                 queue_count, uint64_t, NULL)));
  iree_hal_amd_gpu_signal_frontiers_t* frontiers = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&frontiers));
  *frontiers = (iree_hal_amd_gpu_signal_frontiers_t){
      .previous = previous,
      .capacity = previous->capacity,
      .queue_count = queue_count,
  };
  const uint64_t* source = (const uint64_t*)(previous + 1);
  uint64_t* target = (uint64_t*)(frontiers + 1);
  for (uint32_t i = 0; i < previous->capacity; ++i) {
    memcpy(target + (iree_host_size_t)i * queue_count,
           source + (iree_host_size_t)i * previous->queue_count,
           previous->queue_count * sizeof(uint64_t));
  }
  iree_atomic_store(&block->frontiers, (intptr_t)frontiers,
                    iree_memory_order_release);
  return iree_ok_status();
}

bool iree_hal_amd_gpu_signal_block_try_deinitialize(
    iree_hal_amd_gpu_signal_block_t* block) {
  iree_hal_amd_gpu_signal_pool_t* pool = block->pool;
  iree_hal_amd_gpu_signal_pool_reconcile(pool);
  const bool unused = block->live_count == 0;
  if (unused) {
    if (block->free_head) {
      iree_hal_amd_gpu_signal_block_unlink(block);
    }
    --pool->block_count;
    block->pool = NULL;
  }
  return unused;
}

void iree_hal_amd_gpu_signal_block_deinitialize(
    iree_hal_amd_gpu_signal_block_t* block) {
  const bool unused = iree_hal_amd_gpu_signal_block_try_deinitialize(block);
  IREE_ASSERT(unused, "native signal backing still has live uses");
  (void)unused;
}

bool iree_hal_amd_gpu_signal_pool_has_available(
    iree_hal_amd_gpu_signal_pool_t* pool) {
  if (pool->available || pool->recycled) {
    return true;
  }
  // This RMW also publishes the caller's capacity enrollment. A later return
  // acquires it before checking the notification predicate; an earlier return
  // is consumed here. A plain load would leave a store/load missed-wake race.
  pool->recycled = (iree_hal_amd_gpu_signal_t*)iree_atomic_exchange(
      &pool->returned, 0, iree_memory_order_acq_rel);
  return pool->recycled != NULL;
}

iree_hal_amd_gpu_signal_t* iree_hal_amd_gpu_signal_pool_try_acquire(
    iree_hal_amd_gpu_signal_pool_t* pool, uint64_t initial_value) {
  iree_hal_amd_gpu_signal_block_t* block = pool->available;
  iree_hal_amd_gpu_signal_t* signal = NULL;
  if (block) {
    signal = block->free_head;
    block->free_head = signal->state.next;
    if (!block->free_head) {
      iree_hal_amd_gpu_signal_block_unlink(block);
    }
    ++block->live_count;
  } else {
    if (!pool->recycled) {
      pool->recycled = (iree_hal_amd_gpu_signal_t*)iree_atomic_exchange(
          &pool->returned, 0, iree_memory_order_acq_rel);
    }
    signal = pool->recycled;
    if (signal) {
      pool->recycled = signal->state.next;
      // A returned cell never left its block's live_count. Reuse only changes
      // its exclusive owner; there is no block-list update to perform.
    }
  }
  if (signal) {
    iree_atomic_ref_count_init(&signal->state.live.ref_count);
    iree_atomic_store(&signal->state.live.retirement, 0,
                      iree_memory_order_relaxed);
    *signal->host_value = initial_value;
  }
  return signal;
}

void iree_hal_amd_gpu_signal_publish_frontier(iree_hal_amd_gpu_signal_t* signal,
                                              const uint64_t* frontier) {
  const iree_hal_amd_gpu_signal_block_t* block = signal->block;
  const iree_hal_amd_gpu_signal_frontiers_t* frontiers =
      iree_hal_amd_gpu_signal_block_frontiers(block);
  uint64_t* target = (uint64_t*)(frontiers + 1) +
                     (signal - block->signals) * frontiers->queue_count;
  memcpy(target, frontier, frontiers->queue_count * sizeof(*target));
}

void iree_hal_amd_gpu_signal_join_frontier(
    const iree_hal_amd_gpu_signal_t* signal, uint32_t queue_count,
    uint64_t* frontier) {
  const iree_hal_amd_gpu_signal_block_t* block = signal->block;
  const iree_hal_amd_gpu_signal_frontiers_t* frontiers =
      iree_hal_amd_gpu_signal_block_frontiers(block);
  const uint64_t* source = (const uint64_t*)(frontiers + 1) +
                           (signal - block->signals) * frontiers->queue_count;
  const uint32_t count = iree_min(queue_count, frontiers->queue_count);
  for (uint32_t i = 0; i < count; ++i) {
    frontier[i] = iree_max(frontier[i], source[i]);
  }
}

void iree_hal_amd_gpu_signal_retain(iree_hal_amd_gpu_signal_t* signal) {
  if (signal) {
    iree_atomic_ref_count_inc(&signal->state.live.ref_count);
  }
}

void iree_hal_amd_gpu_signal_release(iree_hal_amd_gpu_signal_t* signal) {
  if (signal && iree_atomic_ref_count_dec(&signal->state.live.ref_count) == 1) {
    iree_hal_amd_gpu_signal_pool_t* pool = signal->block->pool;
    const iree_hal_amd_gpu_signal_pool_notification_t notification =
        pool->notification;
    intptr_t previous =
        iree_atomic_load(&pool->returned, iree_memory_order_relaxed);
    do {
      signal->state.next = (iree_hal_amd_gpu_signal_t*)previous;
    } while (!iree_atomic_compare_exchange_weak(
        &pool->returned, &previous, (intptr_t)signal, iree_memory_order_acq_rel,
        iree_memory_order_relaxed));
    // A block can be reclaimed as soon as its last slot returns. The enclosing
    // progress owner, not this block, owns the copied wake target. Neither the
    // signal nor pool may be accessed after the successful publication above.
    if (!previous && notification.fn) {
      notification.fn(notification.user_data);
    }
  }
}
