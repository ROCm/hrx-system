// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/retirement.h"

void loom_serve_retirement_initialize(loom_serve_retirement_t* retirement) {
  iree_slim_mutex_initialize(&retirement->mutex);
  iree_notification_initialize(&retirement->notification);
  retirement->count = 0;
}

static void retirement_release_view(void* user_data,
                                    iree_hal_buffer_t* buffer) {
  loom_serve_retirement_t* retirement = user_data;
  iree_slim_mutex_lock(&retirement->mutex);
  --retirement->count;
  iree_notification_post(&retirement->notification, IREE_ALL_WAITERS);
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

static bool retirement_is_idle(void* user_data) {
  loom_serve_retirement_t* retirement = user_data;
  iree_slim_mutex_lock(&retirement->mutex);
  const bool retired = retirement->count == 0;
  iree_slim_mutex_unlock(&retirement->mutex);
  return retired;
}

void loom_serve_retirement_await(loom_serve_retirement_t* retirement) {
  iree_notification_await(&retirement->notification, retirement_is_idle,
                          retirement, iree_infinite_timeout());
}

void loom_serve_retirement_deinitialize(loom_serve_retirement_t* retirement) {
  loom_serve_retirement_await(retirement);
  iree_notification_deinitialize(&retirement->notification);
  iree_slim_mutex_deinitialize(&retirement->mutex);
}
