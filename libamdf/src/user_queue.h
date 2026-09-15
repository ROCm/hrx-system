// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_USER_QUEUE_H_
#define AMDF_SRC_USER_QUEUE_H_

#include "amdf/amdf.h"
#include "libamdf/src/child_tracker.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_user_queue_vtable_t {
  // Creates one producer-local mapping and publishes it only on success.
  amdf_status_t (*map)(amdf_user_queue_t* queue, amdf_device_t* producer_device,
                       amdf_user_queue_mapping_t** out_mapping);
  // Samples native progress and terminal state into private result storage.
  amdf_status_t (*query_status)(amdf_user_queue_t* queue,
                                amdf_user_queue_status_t* out_status);
  // Waits for one format-specific published index to be consumed.
  amdf_status_t (*wait_consumed)(amdf_user_queue_t* queue,
                                 uint64_t published_index,
                                 uint64_t timeout_nanoseconds,
                                 uint64_t poll_duration_nanoseconds);
  // Releases exact native queue state after all work has been consumed.
  amdf_status_t (*destroy_native)(amdf_user_queue_t* queue);
} amdf_user_queue_vtable_t;

struct amdf_user_queue_t {
  // Host allocator copied for direct terminal teardown.
  amdf_allocator_t host_allocator;
  // Implementation operations selected before the queue is published.
  const amdf_user_queue_vtable_t* vtable;
  // Device borrowed for the lifetime of this queue.
  amdf_device_t* device;
  // Immutable properties established before publication.
  amdf_user_queue_info_t info;
  // Number of live producer mappings borrowing this queue.
  amdf_child_tracker_t mappings;
};

typedef struct amdf_user_queue_mapping_vtable_t {
  // Releases exact native mapping state after producer access has stopped.
  amdf_status_t (*destroy_native)(amdf_user_queue_mapping_t* mapping);
} amdf_user_queue_mapping_vtable_t;

struct amdf_user_queue_mapping_t {
  // Host allocator copied for direct terminal teardown.
  amdf_allocator_t host_allocator;
  // Implementation operations selected before the mapping is published.
  const amdf_user_queue_mapping_vtable_t* vtable;
  // Queue borrowed for the lifetime of this mapping.
  amdf_user_queue_t* queue;
  // Producer device borrowed for a device mapping, or NULL for the host.
  amdf_device_t* producer_device;
  // Immutable producer-local addresses established before publication.
  amdf_user_queue_mapping_info_t info;
};

// Initializes an unpublished queue base and borrows its device.
amdf_status_t amdf_user_queue_initialize(amdf_user_queue_t* queue,
                                         const amdf_user_queue_vtable_t* vtable,
                                         amdf_device_t* device,
                                         const amdf_user_queue_info_t* info);

// Releases the device borrow held by a torn-down queue.
void amdf_user_queue_deinitialize(amdf_user_queue_t* queue);

// Initializes an unpublished mapping base and borrows its queue.
amdf_status_t amdf_user_queue_mapping_initialize(
    amdf_user_queue_mapping_t* mapping,
    const amdf_user_queue_mapping_vtable_t* vtable, amdf_user_queue_t* queue,
    amdf_device_t* producer_device, const amdf_user_queue_mapping_info_t* info);

// Releases the queue borrow held by a torn-down mapping.
void amdf_user_queue_mapping_deinitialize(amdf_user_queue_mapping_t* mapping);

// Copies immutable queue properties.
amdf_status_t AMDF_CALL amdf_user_queue_query_info(
    amdf_user_queue_t* queue, amdf_user_queue_info_t* out_info);

// Creates one host- or device-producer mapping.
amdf_status_t AMDF_CALL
amdf_user_queue_map(amdf_user_queue_t* queue, amdf_device_t* producer_device,
                    amdf_user_queue_mapping_t** out_mapping);

// Copies producer-local addresses from one mapping.
amdf_status_t AMDF_CALL
amdf_user_queue_mapping_query_info(amdf_user_queue_mapping_t* mapping,
                                   amdf_user_queue_mapping_info_t* out_info);

// Releases one producer-local mapping.
amdf_status_t AMDF_CALL
amdf_user_queue_mapping_destroy(amdf_user_queue_mapping_t* mapping);

// Samples queue progress and terminal state.
amdf_status_t AMDF_CALL amdf_user_queue_query_status(
    amdf_user_queue_t* queue, amdf_user_queue_status_t* out_status);

// Waits for one published index to be consumed.
amdf_status_t AMDF_CALL amdf_user_queue_wait_consumed(
    amdf_user_queue_t* queue, uint64_t published_index,
    uint64_t timeout_nanoseconds, uint64_t poll_duration_nanoseconds);

// Destroys a queue with no mappings or unconsumed work.
amdf_status_t AMDF_CALL amdf_user_queue_destroy(amdf_user_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_USER_QUEUE_H_
