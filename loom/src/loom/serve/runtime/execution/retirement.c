// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/retirement.h"

void loom_serve_retirement_initialize(loom_serve_retirement_t* retirement) {
  iree_slim_mutex_initialize(&retirement->mutex);
  iree_notification_initialize(&retirement->notification);
  retirement->count = 0;
  retirement->wake = NULL;
  retirement->user_data = NULL;
}

static void retirement_release_view(void* user_data,
                                    iree_hal_buffer_t* buffer) {
  loom_serve_retirement_t* retirement = user_data;
  iree_slim_mutex_lock(&retirement->mutex);
  --retirement->count;
  iree_notification_post(&retirement->notification, IREE_ALL_WAITERS);
  if (!retirement->count && retirement->wake) {
    retirement->wake(retirement->user_data);
  }
  iree_slim_mutex_unlock(&retirement->mutex);
}

iree_status_t loom_serve_retirement_track(loom_serve_retirement_t* retirement,
                                          iree_hal_buffer_t** buffer,
                                          iree_allocator_t host_allocator) {
  iree_hal_buffer_t* view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_subspan_buffer_create_with_callback(
      *buffer, iree_hal_buffer_byte_offset(*buffer),
      iree_hal_buffer_byte_length(*buffer),
      (iree_hal_buffer_release_callback_t){retirement_release_view, retirement},
      host_allocator, &view));
  iree_slim_mutex_lock(&retirement->mutex);
  ++retirement->count;
  iree_slim_mutex_unlock(&retirement->mutex);
  iree_hal_buffer_release(*buffer);
  *buffer = view;
  return iree_ok_status();
}

void loom_serve_retirement_set_observer(loom_serve_retirement_t* retirement,
                                        void (*wake)(void* user_data),
                                        void* user_data) {
  iree_slim_mutex_lock(&retirement->mutex);
  retirement->wake = wake;
  retirement->user_data = user_data;
  iree_slim_mutex_unlock(&retirement->mutex);
}

bool loom_serve_retirement_is_idle(loom_serve_retirement_t* retirement) {
  iree_slim_mutex_lock(&retirement->mutex);
  const bool retired = retirement->count == 0;
  iree_slim_mutex_unlock(&retirement->mutex);
  return retired;
}

static bool retirement_is_idle(void* user_data) {
  return loom_serve_retirement_is_idle(user_data);
}

void loom_serve_retirement_await(loom_serve_retirement_t* retirement) {
  iree_notification_await(&retirement->notification, retirement_is_idle,
                          retirement, iree_infinite_timeout());
}

void loom_serve_retirement_deinitialize(loom_serve_retirement_t* retirement) {
  // Taking the mutex joins the final callback's access before destroying the
  // synchronization objects. The owner already observed actual retirement.
  iree_slim_mutex_lock(&retirement->mutex);
  IREE_ASSERT(!retirement->count);
  iree_slim_mutex_unlock(&retirement->mutex);
  iree_notification_deinitialize(&retirement->notification);
  iree_slim_mutex_deinitialize(&retirement->mutex);
}
