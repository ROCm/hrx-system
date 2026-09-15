// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_KERNEL_QUEUE_H_
#define AMDF_SRC_KERNEL_QUEUE_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_kernel_queue_vtable_t {
  // Samples native progress and retires completed submissions.
  amdf_status_t (*query_status)(amdf_kernel_queue_t* queue,
                                amdf_kernel_queue_status_t* out_status);
  // Waits for one accepted submission without cancelling on timeout.
  amdf_status_t (*wait)(amdf_kernel_queue_t* queue, uint64_t submission,
                        uint64_t timeout_nanoseconds,
                        uint64_t poll_duration_nanoseconds);
  // Releases exact native queue state after all submissions have retired.
  amdf_status_t (*destroy_native)(amdf_kernel_queue_t* queue);
} amdf_kernel_queue_vtable_t;

struct amdf_kernel_queue_t {
  // Host allocator copied for direct terminal teardown.
  amdf_allocator_t host_allocator;
  // Implementation operations selected before the queue is published.
  const amdf_kernel_queue_vtable_t* vtable;
  // Device borrowed for the lifetime of this queue.
  amdf_device_t* device;
  // Immutable properties established before publication.
  amdf_kernel_queue_info_t info;
};

// Initializes an unpublished queue base and borrows its device.
amdf_status_t amdf_kernel_queue_initialize(
    amdf_kernel_queue_t* queue, const amdf_kernel_queue_vtable_t* vtable,
    amdf_device_t* device, const amdf_kernel_queue_info_t* info);

// Releases the device borrow held by a torn-down queue.
void amdf_kernel_queue_deinitialize(amdf_kernel_queue_t* queue);

// Copies immutable queue properties.
amdf_status_t AMDF_CALL amdf_kernel_queue_query_info(
    amdf_kernel_queue_t* queue, amdf_kernel_queue_info_t* out_info);

// Samples queue retirement and terminal state.
amdf_status_t AMDF_CALL amdf_kernel_queue_query_status(
    amdf_kernel_queue_t* queue, amdf_kernel_queue_status_t* out_status);

// Waits for one accepted submission with caller-selected polling.
amdf_status_t AMDF_CALL amdf_kernel_queue_wait(
    amdf_kernel_queue_t* queue, uint64_t submission,
    uint64_t timeout_nanoseconds, uint64_t poll_duration_nanoseconds);

// Destroys a queue with no remaining accepted work.
amdf_status_t AMDF_CALL amdf_kernel_queue_destroy(amdf_kernel_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_KERNEL_QUEUE_H_
