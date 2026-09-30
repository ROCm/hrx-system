// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/allocation_preparation.h"

void iree_hal_streaming_allocation_preparation_initialize(
    iree_hal_streaming_allocation_preparation_t* out_preparation) {
  iree_slim_mutex_initialize(&out_preparation->mutex);
  iree_notification_initialize(&out_preparation->notification);
  out_preparation->active_count = 0;
  out_preparation->is_closing = false;
}

void iree_hal_streaming_allocation_preparation_deinitialize(
    iree_hal_streaming_allocation_preparation_t* preparation) {
  IREE_ASSERT_EQ(preparation->active_count, 0);
  iree_notification_deinitialize(&preparation->notification);
  iree_slim_mutex_deinitialize(&preparation->mutex);
}

bool iree_hal_streaming_allocation_preparation_try_acquire(
    iree_hal_streaming_allocation_preparation_t* preparation) {
  iree_slim_mutex_lock(&preparation->mutex);
  const bool acquired =
      !preparation->is_closing && preparation->active_count != SIZE_MAX;
  if (acquired) {
    ++preparation->active_count;
  }
  iree_slim_mutex_unlock(&preparation->mutex);
  return acquired;
}

void iree_hal_streaming_allocation_preparation_release(
    iree_hal_streaming_allocation_preparation_t* preparation) {
  iree_slim_mutex_lock(&preparation->mutex);
  IREE_ASSERT_GT(preparation->active_count, 0);
  --preparation->active_count;
  if (preparation->active_count == 0 && preparation->is_closing) {
    // The waiter observes zero under this mutex, so the notification post has
    // returned before allocation teardown can deinitialize it.
    iree_notification_post(&preparation->notification, IREE_ALL_WAITERS);
  }
  iree_slim_mutex_unlock(&preparation->mutex);
}

void iree_hal_streaming_allocation_preparation_begin_close(
    iree_hal_streaming_allocation_preparation_t* preparation) {
  iree_slim_mutex_lock(&preparation->mutex);
  IREE_ASSERT_FALSE(preparation->is_closing);
  preparation->is_closing = true;
  iree_slim_mutex_unlock(&preparation->mutex);
}

void iree_hal_streaming_allocation_preparation_ensure_closed(
    iree_hal_streaming_allocation_preparation_t* preparation) {
  iree_slim_mutex_lock(&preparation->mutex);
  preparation->is_closing = true;
  iree_slim_mutex_unlock(&preparation->mutex);
}

void iree_hal_streaming_allocation_preparation_reopen(
    iree_hal_streaming_allocation_preparation_t* preparation) {
  iree_slim_mutex_lock(&preparation->mutex);
  IREE_ASSERT_TRUE(preparation->is_closing);
  IREE_ASSERT_EQ(preparation->active_count, 0);
  preparation->is_closing = false;
  iree_slim_mutex_unlock(&preparation->mutex);
}

static bool iree_hal_streaming_allocation_preparation_is_idle(void* user_data) {
  iree_hal_streaming_allocation_preparation_t* preparation =
      (iree_hal_streaming_allocation_preparation_t*)user_data;
  iree_slim_mutex_lock(&preparation->mutex);
  const bool is_idle = preparation->active_count == 0;
  iree_slim_mutex_unlock(&preparation->mutex);
  return is_idle;
}

void iree_hal_streaming_allocation_preparation_await_idle(
    iree_hal_streaming_allocation_preparation_t* preparation) {
  iree_slim_mutex_lock(&preparation->mutex);
  IREE_ASSERT_TRUE(preparation->is_closing);
  iree_slim_mutex_unlock(&preparation->mutex);
  iree_notification_await(&preparation->notification,
                          iree_hal_streaming_allocation_preparation_is_idle,
                          preparation, iree_infinite_timeout());
}
