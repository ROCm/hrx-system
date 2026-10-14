// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_NET_DEMO_WORKER_H_
#define IREE_HAL_NET_DEMO_WORKER_H_

#include "iree/async/api.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/net/channel_provider.h"
#include "iree/task/executor.h"

// CPU execution, network progress and shared memory owned by one demo process.
// Zero-initialize before initialize; deinitialize accepts partial construction.
typedef struct demo_worker_t {
  // CPU worker shared by the device's independently ordered operations.
  iree_task_executor_t* executor;
  // Native heap allocator retained until device teardown.
  iree_hal_allocator_t* allocator;
  // Caller-driven asynchronous progress service owner.
  iree_async_proactor_pool_t* services;
  // Borrowed progress service, polled by the main thread.
  iree_async_proactor_t* proactor;
  // Local CPU HAL device.
  iree_hal_device_t* device;
  // Local memory/progress topology, outliving every pool.
  iree_hal_device_group_t* group;
  // Borrowed exact queue; ordering is always explicit.
  iree_hal_queue_t* queue;
  // Qualified coherent host storage source.
  iree_hal_pool_t* storage;
  // Shared reusable physical slabs.
  iree_hal_pool_t* cache;
  // Elastic tensor suballocations shared by both tags and directions.
  iree_hal_pool_t* workspace;
  // Direct worker-to-worker TCP association.
  iree_hal_channel_t* channel;
  // Actual network retirement, delivered on the polling thread.
  bool drained;
  // Owned diagnostic delivered with channel retirement.
  iree_status_t drain_status;
} demo_worker_t;

iree_status_t demo_worker_initialize(
    const iree_hal_net_channel_options_t* options, demo_worker_t* worker);

// Drives one progress turn. A broken progress service cannot safely release
// outstanding callback storage, so a polling failure terminates the demo.
void demo_worker_poll(demo_worker_t* worker);

// Stops admission and joins actual channel retirement, preserving |status|.
// Called once before preparation teardown. Failed requests keep their storage
// until this drain and all of their operation observers have retired.
iree_status_t demo_worker_close(demo_worker_t* worker, iree_status_t status);

// All request observers and the channel must have retired before this call.
void demo_worker_deinitialize(demo_worker_t* worker);

// One independent terminal operation observer. Observing another operation's
// failure does not retire this one. Zero-initialize before initialize.
typedef struct demo_completion_t {
  // Retained HAL timeline reused across requests.
  iree_hal_semaphore_t* semaphore;
  // Borrowed host-visible timeline for asynchronous observation.
  iree_async_semaphore_t* async_semaphore;
  // Current expected timeline value.
  uint64_t value;
  // Stable observer storage through its callback.
  iree_async_semaphore_wait_operation_t wait;
  // Whether the observer still owns its callback storage.
  bool pending;
  // Owned result from the terminal observer.
  iree_status_t status;
} demo_completion_t;

iree_status_t demo_completion_initialize(demo_worker_t* worker,
                                         demo_completion_t* completion);
iree_status_t demo_completion_arm(demo_worker_t* worker,
                                  demo_completion_t* completion);

static inline iree_hal_semaphore_list_t demo_completion_point(
    demo_completion_t* completion) {
  return (iree_hal_semaphore_list_t){1, &completion->semaphore,
                                     &completion->value};
}

// Retires a prearmed observer when queue admission was rejected. Admitted work
// always owns its own terminal publication; the application never fails it.
iree_status_t demo_completion_admit(demo_completion_t* completion,
                                    iree_status_t status);

// Joins this observer even if another dependency already reported failure.
iree_status_t demo_completion_join(demo_worker_t* worker,
                                   demo_completion_t* completion);

#endif  // IREE_HAL_NET_DEMO_WORKER_H_
