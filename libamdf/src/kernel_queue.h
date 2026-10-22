// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_KERNEL_QUEUE_H_
#define AMDF_SRC_KERNEL_QUEUE_H_

#include "amdf/amdf.h"
#include "libamdf/src/child_tracker.h"
#include "libamdf/src/kernel_queue_dependency.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_kernel_queue_vtable_t {
  // Reads established retirement and cached terminal state without mutation,
  // native queries, locks, allocation or system calls.
  void (*query_status)(amdf_kernel_queue_t* queue,
                       amdf_kernel_queue_status_t* out_status);
  // Samples native progress and consumes available results without waiting
  // for another retirement owner. Reports observation errors separately from
  // the snapshot's cached terminal failure.
  amdf_status_t (*refresh_status)(amdf_kernel_queue_t* queue,
                                  amdf_kernel_queue_status_t* out_status);
  // Requests one wake into a validated, supported caller-owned event. No
  // descriptor is retained and no result consumption occurs here.
  amdf_status_t (*request_notification)(amdf_kernel_queue_t* queue,
                                        uint64_t submission,
                                        const amdf_native_event_t* event);
  // Refreshes native progress and performs checked retirement for one accepted
  // submission, without cancelling on timeout.
  amdf_status_t (*wait)(amdf_kernel_queue_t* queue, uint64_t submission,
                        uint64_t timeout_nanoseconds,
                        uint64_t poll_duration_nanoseconds);
  // API-domain BUSY rejects unretired work without native mutation. Every
  // other result consumes native ownership, preserving unsafe backing on
  // failure. A native-domain busy error is a consuming cleanup failure.
  amdf_status_t (*destroy_native)(amdf_kernel_queue_t* queue);
  // Applies a validated mapping vector using the existing publication owner.
  // On native acceptance publishes a nonzero point even if a later step fails;
  // that failure is also latched on the queue. Rejection leaves the point zero.
  amdf_status_t (*update_memory)(amdf_kernel_queue_t* queue,
                                 amdf_memory_reservation_t* reservation,
                                 uint32_t access_ordinal, uint32_t update_count,
                                 const amdf_memory_mapping_update_t* updates,
                                 bool* mutation_started,
                                 uint64_t* accepted_submission);
  // Reads the greatest accepted point without native work. Required when
  // EXTERNAL_SYNCHRONIZATION is enabled for this queue.
  uint64_t (*query_last_submission)(const amdf_kernel_queue_t* queue);
  // Borrows the immutable native progress identity provisioned at creation.
  const amdf_kernel_queue_native_progress_t* (*query_native_progress)(
      const amdf_kernel_queue_t* queue);
  // Imports a supported producer protocol into this consumer's native domain.
  // Cold and transactional: failure leaves out_native unchanged.
  amdf_status_t (*create_native_dependency)(
      amdf_kernel_queue_t* consumer,
      const amdf_kernel_queue_native_progress_t* progress,
      amdf_kernel_queue_native_dependency_t** out_native);
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
  // Cold dependency/session owners borrowing this queue.
  amdf_child_tracker_t children;
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

// Reads established queue retirement and cached terminal state.
amdf_status_t AMDF_CALL amdf_kernel_queue_query_status(
    amdf_kernel_queue_t* queue, amdf_kernel_queue_status_t* out_status);

// Refreshes checked progress without waiting for a particular submission.
amdf_status_t AMDF_CALL amdf_kernel_queue_refresh_status(
    amdf_kernel_queue_t* queue, amdf_kernel_queue_status_t* out_status);

// Requests one native-progress hint without retaining the caller's event.
amdf_status_t AMDF_CALL amdf_kernel_queue_request_notification(
    amdf_kernel_queue_t* queue, uint64_t submission,
    const amdf_native_event_t* event);

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
