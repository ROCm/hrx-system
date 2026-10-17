// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/signal.h"

static_assert(sizeof(iree_hal_amd_gpu_pm4_signal_t) <= 32,
              "two signal owners must fit in one cache line");
static_assert(sizeof(iree_hal_amd_gpu_pm4_signal_pool_t) -
                      offsetof(iree_hal_amd_gpu_pm4_signal_pool_t, available) <=
                  64,
              "non-mutex pool bookkeeping fits in one cache line");
static_assert(sizeof(iree_hal_amd_gpu_pm4_signal_block_t) <= 64,
              "signal block bookkeeping fits in one cache line");

void iree_hal_amd_gpu_pm4_signal_pool_initialize(
    iree_hal_amd_gpu_pm4_signal_pool_notification_t notification,
    iree_hal_amd_gpu_pm4_signal_pool_t* out_pool) {
  *out_pool =
      (iree_hal_amd_gpu_pm4_signal_pool_t){.notification = notification};
  iree_slim_mutex_initialize(&out_pool->mutex);
}

void iree_hal_amd_gpu_pm4_signal_pool_deinitialize(
    iree_hal_amd_gpu_pm4_signal_pool_t* pool) {
  IREE_ASSERT_EQ(pool->block_count, 0, "native signal blocks remain attached");
  iree_slim_mutex_deinitialize(&pool->mutex);
}

// The pool mutex serializes availability with acquisition, release, and trim.
static void iree_hal_amd_gpu_pm4_signal_block_link(
    iree_hal_amd_gpu_pm4_signal_block_t* block) {
  iree_hal_amd_gpu_pm4_signal_pool_t* pool = block->pool;
  block->available_next = pool->available;
  if (pool->available) {
    pool->available->available_previous = block;
  }
  pool->available = block;
}

static void iree_hal_amd_gpu_pm4_signal_block_unlink(
    iree_hal_amd_gpu_pm4_signal_block_t* block) {
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

void iree_hal_amd_gpu_pm4_signal_block_initialize(
    iree_hal_amd_gpu_pm4_signal_pool_t* pool, const void* domain,
    uint64_t* values, uint64_t device_address,
    iree_hal_amd_gpu_pm4_signal_t* signals, uint32_t capacity,
    uint32_t queue_count, uint64_t* frontiers,
    iree_hal_amd_gpu_pm4_signal_block_t* out_block) {
  *out_block = (iree_hal_amd_gpu_pm4_signal_block_t){
      .pool = pool,
      .queue_count = queue_count,
      .free_head = capacity ? signals : NULL,
      .domain = domain,
      .signals = signals,
      .frontiers = frontiers,
  };
  for (uint32_t i = 0; i < capacity; ++i) {
    values[i] = 0;
    signals[i] = (iree_hal_amd_gpu_pm4_signal_t){
        .block = out_block,
        .device_address = device_address + i * sizeof(*values),
        .host_value = &values[i],
        .state.next = i + 1 < capacity ? &signals[i + 1] : NULL,
    };
  }
  iree_slim_mutex_lock(&pool->mutex);
  ++pool->block_count;
  if (capacity) {
    iree_hal_amd_gpu_pm4_signal_block_link(out_block);
  }
  iree_slim_mutex_unlock(&pool->mutex);
}

bool iree_hal_amd_gpu_pm4_signal_block_try_deinitialize(
    iree_hal_amd_gpu_pm4_signal_block_t* block) {
  iree_hal_amd_gpu_pm4_signal_pool_t* pool = block->pool;
  iree_slim_mutex_lock(&pool->mutex);
  const bool unused = block->live_count == 0;
  if (unused) {
    if (block->free_head) {
      iree_hal_amd_gpu_pm4_signal_block_unlink(block);
    }
    --pool->block_count;
    block->pool = NULL;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return unused;
}

void iree_hal_amd_gpu_pm4_signal_block_deinitialize(
    iree_hal_amd_gpu_pm4_signal_block_t* block) {
  const bool unused = iree_hal_amd_gpu_pm4_signal_block_try_deinitialize(block);
  IREE_ASSERT(unused, "native signal backing still has live uses");
  (void)unused;
}

bool iree_hal_amd_gpu_pm4_signal_pool_has_available(
    iree_hal_amd_gpu_pm4_signal_pool_t* pool) {
  iree_slim_mutex_lock(&pool->mutex);
  const bool available = pool->available != NULL;
  iree_slim_mutex_unlock(&pool->mutex);
  return available;
}

iree_hal_amd_gpu_pm4_signal_t* iree_hal_amd_gpu_pm4_signal_pool_try_acquire(
    iree_hal_amd_gpu_pm4_signal_pool_t* pool) {
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_amd_gpu_pm4_signal_block_t* block = pool->available;
  iree_hal_amd_gpu_pm4_signal_t* signal = NULL;
  if (block) {
    signal = block->free_head;
    block->free_head = signal->state.next;
    if (!block->free_head) {
      iree_hal_amd_gpu_pm4_signal_block_unlink(block);
    }
    ++block->live_count;
    iree_atomic_ref_count_init(&signal->state.live.ref_count);
    iree_atomic_store(&signal->state.live.retirement, 0,
                      iree_memory_order_relaxed);
  }
  iree_slim_mutex_unlock(&pool->mutex);
  if (signal) {
    *signal->host_value = 0;
  }
  return signal;
}

void iree_hal_amd_gpu_pm4_signal_publish_frontier(
    iree_hal_amd_gpu_pm4_signal_t* signal, const uint64_t* frontier) {
  const iree_hal_amd_gpu_pm4_signal_block_t* block = signal->block;
  uint64_t* target =
      block->frontiers + (signal - block->signals) * block->queue_count;
  memcpy(target, frontier, block->queue_count * sizeof(*target));
}

void iree_hal_amd_gpu_pm4_signal_join_frontier(
    const iree_hal_amd_gpu_pm4_signal_t* signal, uint64_t* frontier) {
  const iree_hal_amd_gpu_pm4_signal_block_t* block = signal->block;
  const uint64_t* source =
      block->frontiers + (signal - block->signals) * block->queue_count;
  for (uint32_t i = 0; i < block->queue_count; ++i) {
    frontier[i] = iree_max(frontier[i], source[i]);
  }
}

void iree_hal_amd_gpu_pm4_signal_retain(iree_hal_amd_gpu_pm4_signal_t* signal) {
  if (signal) {
    iree_atomic_ref_count_inc(&signal->state.live.ref_count);
  }
}

void iree_hal_amd_gpu_pm4_signal_release(
    iree_hal_amd_gpu_pm4_signal_t* signal) {
  if (signal && iree_atomic_ref_count_dec(&signal->state.live.ref_count) == 1) {
    iree_hal_amd_gpu_pm4_signal_block_t* block = signal->block;
    iree_hal_amd_gpu_pm4_signal_pool_t* pool = block->pool;
    iree_slim_mutex_lock(&pool->mutex);
    const bool became_available = pool->available == NULL;
    if (!block->free_head) {
      iree_hal_amd_gpu_pm4_signal_block_link(block);
    }
    signal->state.next = block->free_head;
    block->free_head = signal;
    --block->live_count;
    const iree_hal_amd_gpu_pm4_signal_pool_notification_t notification =
        pool->notification;
    iree_slim_mutex_unlock(&pool->mutex);
    // A block can be reclaimed as soon as its last slot returns. The enclosing
    // progress owner, not this block, owns the copied wake target.
    if (became_available && notification.fn) {
      notification.fn(notification.user_data);
    }
  }
}
