// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_DEVICE_H_
#define EXPERIMENTAL_LOOM_SERVE_DEVICE_H_

#include "experimental/loom_serve/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// One serving device, its async I/O services, semaphore domain and ordered
// execution timelines. Model programs, weights and request state are owned by
// the caller and may share this device without sharing a serving policy.
typedef struct loom_serve_device_t loom_serve_device_t;

// Creates the device identified by a HAL URI and selects the first provisioned
// queue supporting each of dispatch and transfer. Missing roles fail creation.
// This establishes runtime ownership, not compiler support for the device.
// On failure no owner is returned and all partial resources are released.
iree_status_t loom_serve_device_create(iree_string_view_t uri,
                                       iree_allocator_t host_allocator,
                                       loom_serve_device_t** out_device);

// Releases the device and then its async services, without an implicit wait.
// The caller first drains execution while all borrowed host I/O is alive,
// ends profiling, and releases all model/JIT/VM resources referencing this
// domain. Those resources must not outlive the owner. NULL is permitted.
void loom_serve_device_destroy(loom_serve_device_t* device);

// Borrowed handles valid through owner destruction. The group establishes the
// semaphore namespace; the exact queues are also used by the execution object.
iree_hal_device_t* loom_serve_device_handle(const loom_serve_device_t* device);
iree_hal_device_group_t* loom_serve_device_group(
    const loom_serve_device_t* device);
iree_hal_queue_t* loom_serve_device_dispatch_queue(
    const loom_serve_device_t* device);
iree_hal_queue_t* loom_serve_device_transfer_queue(
    const loom_serve_device_t* device);
loom_serve_execution_t* loom_serve_device_execution(
    const loom_serve_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_DEVICE_H_
