// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_LIBHRX_RUNTIME_H_
#define LIBHRX_SRC_LIBHRX_RUNTIME_H_

#include "hrx_runtime.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_async_proactor_pool_t iree_async_proactor_pool_t;

// Construction-time service ownership. Queue submission uses device-local
// state and never acquires the driver construction mutex.
typedef struct hrx_runtime_s {
  // References held by callers and complete device groups.
  iree_atomic_ref_count_t ref_count;
  // Allocator for the runtime, group metadata, and native device construction.
  iree_allocator_t host_allocator;
  // Lazy NUMA-aware I/O services shared by this runtime's devices.
  iree_async_proactor_pool_t* proactor_pool;
  // Guards only task-driver creation and publication.
  iree_slim_mutex_t task_mutex;
  // Retained task driver sharing its executor across all task devices.
  iree_hal_driver_t* task_driver;
  // Immutable construction options; any CPU IDs refer to trailing owned
  // storage.
  hrx_runtime_options_t options;
} hrx_runtime_s;

// Acquires a driver from the appropriate execution/native connection owner.
iree_status_t hrx_runtime_acquire_driver(hrx_runtime_t runtime,
                                         iree_string_view_t scheme,
                                         iree_hal_driver_t** out_driver);
// Translates a borrowed immutable HRX sink. The sink and its callback state
// remain live until all users of the returned HAL sink have completed.
iree_hal_device_event_sink_t hrx_device_event_sink_to_hal(
    const hrx_device_event_sink_t* sink);
iree_hal_device_event_sink_t hrx_runtime_hal_event_sink(hrx_runtime_t runtime);

// Consumes a teardown failure through the runtime's configured event sink.
// |device| is NULL for group-owned resources without one execution device.
void hrx_runtime_report_failure(hrx_runtime_t runtime,
                                iree_hal_device_t* device,
                                iree_status_t status);

// Parent-scoped services for frontend operations that already retain a device.
iree_async_proactor_pool_t* hrx_device_proactor_pool(hrx_device_t device);
iree_hal_device_event_sink_t hrx_device_hal_event_sink(hrx_device_t device);

// Starts one explicit capture before publishing the device to callers. Devices
// in a group may share an output sink; different groups own independent sinks.
iree_status_t hrx_device_profile_begin(
    hrx_device_t device, const iree_hal_device_profiling_options_t* options);

// Drains completed profile records after a host synchronization boundary.
// Inactive or absent devices are a no-op. Serializes producer flushes and
// writes to the group's shared profile file, without holding execution locks
// or waiting for other queues. File errors are returned to the synchronizing
// caller.
iree_status_t hrx_runtime_profile_flush(hrx_device_t device);
iree_status_t hrx_device_profile_end(hrx_device_t device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_RUNTIME_H_
