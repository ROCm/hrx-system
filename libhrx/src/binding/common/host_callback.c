// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/host_callback.h"

#include <string.h>

#include "iree/hal/semaphore.h"

struct iree_hal_streaming_host_callback_job_t {
  // Caller, HAL invocation, and ready worker references protect the capture.
  iree_hal_resource_t resource;
  // Allocator copied from the owning frontend for terminal cleanup.
  iree_allocator_t host_allocator;
  // Borrowed registry owner, live through synchronous native device teardown.
  iree_hal_streaming_host_callback_queue_t* queue;
  // Next ready callback, linked only while queued under the owner's mutex.
  iree_hal_streaming_host_callback_job_t* next;
  // Application function or compatibility CPU copy invoked by the worker.
  void (*fn)(void* user_data);
  // Unowned function argument, optionally protected by captured_resource.
  void* user_data;
  // Retained callback payload released before publishing callback completion.
  iree_hal_resource_t* captured_resource;
  // Retained native callback completion list, populated only upon readiness.
  iree_hal_semaphore_list_t signals;
};

static void iree_hal_streaming_host_callback_destroy(
    iree_hal_resource_t* base_resource) {
  iree_hal_streaming_host_callback_job_t* callback = (void*)base_resource;
  iree_hal_resource_release(callback->captured_resource);
  iree_hal_semaphore_list_free(callback->signals, callback->host_allocator);
  // Native cancellation may release callback payloads after a failed stream's
  // semaphore becomes observable. Keep frontend device owners until every
  // native capture has actually released its payload and completion list.
  iree_hal_streaming_host_callback_queue_t* queue = callback->queue;
  iree_slim_mutex_lock(&queue->mutex);
  --queue->capture_count;
  iree_notification_post(&queue->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&queue->mutex);
  iree_allocator_free(callback->host_allocator, callback);
}

static const iree_hal_resource_vtable_t
    iree_hal_streaming_host_callback_vtable = {
        .destroy = iree_hal_streaming_host_callback_destroy,
};

static void iree_hal_streaming_host_callback_run(
    iree_hal_streaming_host_callback_job_t* callback) {
  IREE_TRACE_ZONE_BEGIN(z0);
  callback->fn(callback->user_data);
  // End captured CPU state before waking stream consumers. The worker still
  // owns the capture while signaling may synchronously dispatch continuations.
  iree_hal_resource_release(callback->captured_resource);
  callback->captured_resource = NULL;
  iree_status_t status =
      iree_hal_semaphore_list_signal(callback->signals, NULL);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(callback->signals, status);
  }
  iree_hal_resource_release(&callback->resource);
  IREE_TRACE_ZONE_END(z0);
}

static int iree_hal_streaming_host_callback_queue_main(void* user_data) {
  iree_hal_streaming_host_callback_queue_t* queue = user_data;
  while (true) {
    const iree_wait_token_t token =
        iree_notification_prepare_wait(&queue->notification);
    iree_slim_mutex_lock(&queue->mutex);
    iree_hal_streaming_host_callback_job_t* callback = queue->head;
    if (callback) {
      queue->head = callback->next;
      if (!queue->head) {
        queue->tail = NULL;
      }
    }
    const bool stopping = queue->stopping;
    iree_slim_mutex_unlock(&queue->mutex);
    if (callback || stopping) {
      iree_notification_cancel_wait(&queue->notification);
      if (!callback) {
        break;
      }
      iree_hal_streaming_host_callback_run(callback);
    } else {
      iree_notification_commit_wait(&queue->notification, token,
                                    IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
  }
  return 0;
}

void iree_hal_streaming_host_callback_queue_initialize(
    iree_allocator_t host_allocator,
    iree_hal_streaming_host_callback_queue_t* out_queue) {
  memset(out_queue, 0, sizeof(*out_queue));
  out_queue->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&out_queue->mutex);
  iree_notification_initialize(&out_queue->notification);
}

static bool iree_hal_streaming_host_callback_queue_is_idle(void* user_data) {
  iree_hal_streaming_host_callback_queue_t* queue = user_data;
  iree_slim_mutex_lock(&queue->mutex);
  const bool idle = queue->capture_count == 0;
  iree_slim_mutex_unlock(&queue->mutex);
  return idle;
}

void iree_hal_streaming_host_callback_queue_stop(
    iree_hal_streaming_host_callback_queue_t* queue) {
  iree_slim_mutex_lock(&queue->mutex);
  queue->stopping = true;
  iree_thread_t* thread = queue->thread;
  queue->thread = NULL;
  iree_notification_post(&queue->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&queue->mutex);
  iree_notification_await(&queue->notification,
                          iree_hal_streaming_host_callback_queue_is_idle, queue,
                          iree_infinite_timeout());
  iree_thread_release(thread);
}

void iree_hal_streaming_host_callback_queue_deinitialize(
    iree_hal_streaming_host_callback_queue_t* queue) {
  iree_hal_streaming_host_callback_queue_stop(queue);
  iree_notification_deinitialize(&queue->notification);
  iree_slim_mutex_deinitialize(&queue->mutex);
}

static iree_status_t iree_hal_streaming_host_callback_ready(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  iree_hal_streaming_host_callback_job_t* callback = user_data;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_clone(
      &context->signal_semaphore_list, callback->host_allocator,
      &callback->signals));
  iree_hal_streaming_host_callback_queue_t* queue = callback->queue;
  iree_slim_mutex_lock(&queue->mutex);
  const bool stopping = queue->stopping;
  if (!stopping) {
    iree_hal_resource_retain(&callback->resource);
    if (queue->tail) {
      queue->tail->next = callback;
    } else {
      queue->head = callback;
    }
    queue->tail = callback;
    iree_notification_post(&queue->notification, 1);
  }
  iree_slim_mutex_unlock(&queue->mutex);
  return iree_status_from_code(stopping ? IREE_STATUS_CANCELLED
                                        : IREE_STATUS_DEFERRED);
}

iree_status_t iree_hal_streaming_host_callback_prepare(
    iree_hal_streaming_host_callback_queue_t* queue, void (*fn)(void*),
    void* user_data, iree_hal_resource_t* resource,
    iree_hal_host_call_t* out_call) {
  iree_hal_streaming_host_callback_job_t* callback = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, sizeof(*callback), (void**)&callback));
  iree_hal_resource_initialize(&iree_hal_streaming_host_callback_vtable,
                               &callback->resource);
  callback->host_allocator = queue->host_allocator;
  callback->queue = queue;
  callback->fn = fn;
  callback->user_data = user_data;
  callback->captured_resource = resource;
  iree_hal_resource_retain(resource);
  iree_slim_mutex_lock(&queue->mutex);
  ++queue->capture_count;
  iree_status_t status = iree_ok_status();
  if (!queue->thread) {
    const iree_thread_create_params_t params = {
        .name = IREE_SV("hrx-host-call"),
    };
    status =
        iree_thread_create(iree_hal_streaming_host_callback_queue_main, queue,
                           params, queue->host_allocator, &queue->thread);
  }
  iree_slim_mutex_unlock(&queue->mutex);
  if (iree_status_is_ok(status)) {
    *out_call = iree_hal_make_host_call_with_resource(
        iree_hal_streaming_host_callback_ready, callback, &callback->resource);
  } else {
    iree_hal_resource_release(&callback->resource);
  }
  return status;
}
