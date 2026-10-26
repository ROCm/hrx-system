// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_HOST_CALLBACK_H_
#define LIBHRX_SRC_BINDING_COMMON_HOST_CALLBACK_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_host_callback_job_t
    iree_hal_streaming_host_callback_job_t;

// Frontend-owned FIFO of ready compatibility callouts. Native queues retain
// responsibility for dependency waiting and GPU visibility. Only application
// host functions and bulk compatibility CPU copies use this execution owner.
typedef struct iree_hal_streaming_host_callback_queue_t {
  // Allocator owning callback captures and the lazily started worker.
  iree_allocator_t host_allocator;
  // Protects worker creation and the ready FIFO; never held during callouts.
  iree_slim_mutex_t mutex;
  // Wakes the worker when ready callouts arrive or shutdown starts.
  iree_notification_t notification;
  // Worker owned by this frontend registry, or NULL before first use/after
  // join.
  iree_thread_t* thread;
  // First ready callback, protected by mutex.
  iree_hal_streaming_host_callback_job_t* head;
  // Last ready callback, or NULL when the FIFO is empty.
  iree_hal_streaming_host_callback_job_t* tail;
  // Live prepared captures, including native callbacks not yet ready/cancelled.
  // Shutdown joins these references before releasing frontend device owners.
  iree_host_size_t capture_count;
  // Stops ready admission and exits the worker after draining its FIFO.
  bool stopping;
} iree_hal_streaming_host_callback_queue_t;

// Initializes empty frontend-owned state without allocating or starting a
// thread.
void iree_hal_streaming_host_callback_queue_initialize(
    iree_allocator_t host_allocator,
    iree_hal_streaming_host_callback_queue_t* out_queue);

// Closes ready admission, joins native captures, and drains worker CPU tails.
// The application has stopped submitting frontend operations and synchronized
// its streams. Late native callbacks on failed queues are rejected. The queue
// object remains live until native device teardown has joined those producers.
void iree_hal_streaming_host_callback_queue_stop(
    iree_hal_streaming_host_callback_queue_t* queue);

// Releases an initialized queue after native callbacks have been joined.
// Stops the worker if needed; no callback may access the queue after return.
void iree_hal_streaming_host_callback_queue_deinitialize(
    iree_hal_streaming_host_callback_queue_t* queue);

// Prepares a one-shot HAL call on an open queue, starting the worker on
// first use. |resource| optionally retains |user_data| through callback
// execution or cancellation. Success returns an owned reference in
// out_call->resource; the caller releases it after queue_host_call returns.
// Failure changes no output. The HAL adapter clones its ready completion list
// and returns DEFERRED; only this queue's worker invokes |fn|. No device
// reference is retained.
iree_status_t iree_hal_streaming_host_callback_prepare(
    iree_hal_streaming_host_callback_queue_t* queue, void (*fn)(void*),
    void* user_data, iree_hal_resource_t* resource,
    iree_hal_host_call_t* out_call);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_HOST_CALLBACK_H_
