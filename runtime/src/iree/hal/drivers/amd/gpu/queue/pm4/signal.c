// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/signal.h"

static_assert(sizeof(iree_hal_amd_gpu_pm4_signal_t) <= 32,
              "two signal owners must fit in one cache line");

void iree_hal_amd_gpu_pm4_signal_pool_initialize(
    const void* domain, uint64_t* values, uint64_t device_address,
    iree_hal_amd_gpu_pm4_signal_t* signals, uint32_t capacity,
    iree_hal_amd_gpu_pm4_signal_pool_notification_t notification,
    iree_hal_amd_gpu_pm4_signal_pool_t* out_pool) {
  *out_pool = (iree_hal_amd_gpu_pm4_signal_pool_t){
      .free_head = capacity ? signals : NULL,
      .domain = domain,
      .notification = notification,
  };
  iree_slim_mutex_initialize(&out_pool->mutex);
  for (uint32_t i = 0; i < capacity; ++i) {
    values[i] = 0;
    signals[i] = (iree_hal_amd_gpu_pm4_signal_t){
        .pool = out_pool,
        .device_address = device_address + i * sizeof(*values),
        .host_value = &values[i],
        .state.next = i + 1 < capacity ? &signals[i + 1] : NULL,
    };
  }
}

void iree_hal_amd_gpu_pm4_signal_pool_deinitialize(
    iree_hal_amd_gpu_pm4_signal_pool_t* pool) {
  IREE_ASSERT_EQ(pool->live_count, 0,
                 "native signal backing still has live uses");
  iree_slim_mutex_deinitialize(&pool->mutex);
}

iree_hal_amd_gpu_pm4_signal_t* iree_hal_amd_gpu_pm4_signal_pool_try_acquire(
    iree_hal_amd_gpu_pm4_signal_pool_t* pool) {
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_amd_gpu_pm4_signal_t* signal = pool->free_head;
  if (signal) {
    pool->free_head = signal->state.next;
    ++pool->live_count;
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

void iree_hal_amd_gpu_pm4_signal_retain(iree_hal_amd_gpu_pm4_signal_t* signal) {
  if (signal) {
    iree_atomic_ref_count_inc(&signal->state.live.ref_count);
  }
}

void iree_hal_amd_gpu_pm4_signal_release(
    iree_hal_amd_gpu_pm4_signal_t* signal) {
  if (signal && iree_atomic_ref_count_dec(&signal->state.live.ref_count) == 1) {
    iree_hal_amd_gpu_pm4_signal_pool_t* pool = signal->pool;
    iree_slim_mutex_lock(&pool->mutex);
    const bool became_available = pool->free_head == NULL;
    signal->state.next = pool->free_head;
    pool->free_head = signal;
    --pool->live_count;
    const iree_hal_amd_gpu_pm4_signal_pool_notification_t notification =
        pool->notification;
    iree_slim_mutex_unlock(&pool->mutex);
    // An inactive pool can be reclaimed as soon as its last slot returns.
    // The enclosing progress owner, not this pool, owns the wake target.
    if (became_available && notification.fn) {
      notification.fn(notification.user_data);
    }
  }
}
