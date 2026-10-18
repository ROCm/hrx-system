// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/pool_history.h"

static void iree_hal_pool_history_free_captures(
    iree_async_frontier_join_t* captures) {
  while (captures) {
    iree_async_frontier_join_t* next = captures->next;
    iree_async_frontier_join_free(captures);
    captures = next;
  }
}

static void iree_hal_pool_history_supply(iree_hal_pool_history_t* history,
                                         iree_async_frontier_join_t* capture) {
  if (!capture) {
    return;
  }
  capture->next = history->reserves;
  history->reserves = capture;
  ++history->reserve_count;
}

static iree_async_frontier_join_t* iree_hal_pool_history_seal(
    iree_hal_pool_history_t* history) {
  iree_async_frontier_join_t* capture = history->open;
  if (capture) {
    history->open = NULL;
    --history->reserve_count;
    iree_atomic_store(&history->capture_pending, 0, iree_memory_order_release);
  }
  return capture;
}

static iree_async_frontier_join_t* iree_hal_pool_history_take_unused(
    iree_hal_pool_history_t* history, bool retain_idle) {
  const uint32_t retain_count =
      iree_max(history->backing_count, retain_idle ? 1u : 0u);
  iree_async_frontier_join_t* retired = NULL;
  while (history->reserves && history->reserve_count > retain_count) {
    iree_async_frontier_join_t* capture = history->reserves;
    history->reserves = capture->next;
    capture->next = retired;
    retired = capture;
    --history->reserve_count;
  }
  return retired;
}

static void iree_hal_pool_history_collapse_locked(
    iree_hal_pool_history_t* history, iree_async_frontier_t* target,
    const iree_async_frontier_t* source) {
  if (!history->open) {
    IREE_ASSERT(history->reserves,
                "acquisition secures a capture for every live backing");
    history->open = history->reserves;
    history->reserves = history->open->next;
    history->open->next = NULL;
    iree_async_frontier_join_begin(history->open);
    iree_atomic_store(&history->capture_pending, 1, iree_memory_order_release);
  }
  iree_async_frontier_join_merge(history->open, target);
  iree_async_frontier_join_merge(history->open, source);
  iree_async_frontier_initialize(target, 1);
  target->entries[0] = history->open->point;
}

iree_status_t iree_hal_pool_history_initialize(
    iree_async_frontier_tracker_t* tracker, uint8_t frontier_capacity,
    iree_hal_pool_history_t* out_history) {
  memset(out_history, 0, sizeof(*out_history));
  iree_slim_mutex_initialize(&out_history->mutex);
  out_history->frontier_capacity = frontier_capacity;
  return tracker ? iree_async_frontier_tracker_acquire_timeline(
                       tracker, &out_history->timeline)
                 : iree_ok_status();
}

void iree_hal_pool_history_deinitialize(iree_hal_pool_history_t* history) {
  iree_async_frontier_join_t* capture = iree_hal_pool_history_seal(history);
  if (capture) {
    iree_async_frontier_join_publish(capture);
  }
  iree_hal_pool_history_free_captures(history->reserves);
  iree_async_frontier_timeline_release(history->timeline);
  iree_slim_mutex_deinitialize(&history->mutex);
  memset(history, 0, sizeof(*history));
}

iree_status_t iree_hal_pool_history_prepare_backing(
    iree_hal_pool_history_t* history,
    const iree_async_frontier_t* initial_frontier,
    iree_async_single_frontier_t* compact_storage,
    const iree_async_frontier_t** out_frontier) {
  *out_frontier = initial_frontier;
  if (!history->timeline) {
    return iree_ok_status();
  }
  iree_async_frontier_join_t* prepared = NULL;
  for (;;) {
    iree_slim_mutex_lock(&history->mutex);
    iree_hal_pool_history_supply(history, prepared);
    prepared = NULL;
    const bool ready = history->reserve_count > history->backing_count;
    if (ready) {
      ++history->backing_count;
      if (initial_frontier &&
          initial_frontier->entry_count > history->frontier_capacity) {
        iree_async_frontier_t* compact =
            iree_async_single_frontier_as_frontier(compact_storage);
        iree_async_frontier_initialize(compact, 0);
        iree_hal_pool_history_collapse_locked(history, compact,
                                              initial_frontier);
        *out_frontier = compact;
      }
    }
    iree_slim_mutex_unlock(&history->mutex);
    if (ready) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        iree_async_frontier_join_allocate(history->timeline, &prepared));
  }
}

void iree_hal_pool_history_retire_backing(iree_hal_pool_history_t* history) {
  if (!history->timeline) {
    return;
  }
  iree_slim_mutex_lock(&history->mutex);
  --history->backing_count;
  iree_async_frontier_join_t* sealed = iree_hal_pool_history_seal(history);
  iree_async_frontier_join_t* retired =
      iree_hal_pool_history_take_unused(history, true);
  iree_slim_mutex_unlock(&history->mutex);
  if (sealed) {
    iree_async_frontier_join_publish(sealed);
  }
  iree_hal_pool_history_free_captures(retired);
}

void iree_hal_pool_history_collapse(void* user_data,
                                    iree_async_frontier_t* target,
                                    const iree_async_frontier_t* source) {
  iree_hal_pool_history_t* history = user_data;
  iree_slim_mutex_lock(&history->mutex);
  iree_hal_pool_history_collapse_locked(history, target, source);
  iree_slim_mutex_unlock(&history->mutex);
}

iree_status_t iree_hal_pool_history_seal_acquire(
    iree_hal_pool_history_t* history) {
  // Free-block publication follows the store of capture_pending. Claiming
  // those blocks therefore observes either their open capture or its seal.
  if (!iree_atomic_load(&history->capture_pending, iree_memory_order_acquire)) {
    return iree_ok_status();
  }
  iree_async_frontier_join_t* prepared = NULL;
  for (;;) {
    iree_slim_mutex_lock(&history->mutex);
    iree_hal_pool_history_supply(history, prepared);
    prepared = NULL;
    const bool ready =
        !history->open || history->reserve_count > history->backing_count;
    iree_async_frontier_join_t* sealed =
        ready ? iree_hal_pool_history_seal(history) : NULL;
    iree_slim_mutex_unlock(&history->mutex);
    if (sealed) {
      iree_async_frontier_join_publish(sealed);
    }
    if (ready) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        iree_async_frontier_join_allocate(history->timeline, &prepared));
  }
}

void iree_hal_pool_history_trim(iree_hal_pool_history_t* history) {
  iree_slim_mutex_lock(&history->mutex);
  iree_async_frontier_join_t* retired =
      iree_hal_pool_history_take_unused(history, false);
  iree_slim_mutex_unlock(&history->mutex);
  iree_hal_pool_history_free_captures(retired);
}
