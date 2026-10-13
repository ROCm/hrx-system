// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_RETIREMENT_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_RETIREMENT_H_

#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Final ownership of buffer views lent to queues. Failed readiness may precede
// retirement of accepted operations; borrowed host payloads and virtual address
// mappings must survive that interval. Tracking wraps a root once, not on every
// dispatch. Nested views inherit the obligation without address indirection.
typedef struct loom_serve_retirement_t {
  // Serializes final notification with observation of zero outstanding views.
  iree_slim_mutex_t mutex;
  // Wakes the teardown owner after a view's final queue release.
  iree_notification_t notification;
  // Root views still retained by callers, commands, children or queues.
  iree_host_size_t count;
  // Optional nonblocking owner wake, invoked under mutex at final retirement.
  void (*wake)(void* user_data);
  // Borrowed wake payload, protected through observer detachment.
  void* user_data;
} loom_serve_retirement_t;

void loom_serve_retirement_initialize(loom_serve_retirement_t* retirement);

// Replaces an owned reference with a release-tracked view of the same range.
// On failure the input reference is unchanged. All consumers use this returned
// view or its children; passing the original allocation would bypass tracking.
// The retirement owner outlives all tracked views and serializes track/await.
iree_status_t loom_serve_retirement_track(loom_serve_retirement_t* retirement,
                                          iree_hal_buffer_t** buffer,
                                          iree_allocator_t host_allocator);

// Observes final view retirement without waiting. The callback only schedules
// an owner turn; it must not reenter this object, destroy resources or block.
// Set the observer before querying is_idle to avoid losing the final wake.
// NULL detaches and joins a callback's final access to user_data. The owner
// serializes tracking and terminal observation; callbacks may run concurrently.
void loom_serve_retirement_set_observer(loom_serve_retirement_t* retirement,
                                        void (*wake)(void* user_data),
                                        void* user_data);
bool loom_serve_retirement_is_idle(loom_serve_retirement_t* retirement);

// Joins actual final ownership, independently of semaphore success or failure.
// The caller first releases its tracked views and cached consumers. No new
// views may be tracked concurrently with this wait. Queue callbacks may run.
void loom_serve_retirement_await(loom_serve_retirement_t* retirement);

// Releases the notification primitives after is_idle or await proved all views
// retired. Never waits for views; the owner has stopped tracking new ones.
void loom_serve_retirement_deinitialize(loom_serve_retirement_t* retirement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_EXECUTION_RETIREMENT_H_
