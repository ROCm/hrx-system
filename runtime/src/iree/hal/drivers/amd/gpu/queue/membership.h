// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_MEMBERSHIP_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_MEMBERSHIP_H_

#include "iree/hal/drivers/amd/gpu/completion.h"
#include "iree/hal/drivers/amd/gpu/profile.h"
#include "iree/hal/drivers/amd/gpu/queue/backing.h"
#include "iree/hal/drivers/amd/gpu/queue/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold protocol ownership operations. Submission and retirement remain in the
// concrete queue; no operation passes through this table on the execution path.
typedef struct iree_hal_amd_gpu_queue_lifecycle_t {
  // Closes acceptance while preserving accepted dependencies and work.
  void (*close)(iree_hal_queue_t* queue);
  // Proves every accepted native/host owner has retired.
  bool (*is_idle)(iree_hal_queue_t* queue);
  // Joins native consumption and destroys queue-specific backing.
  amdf_status_t (*shutdown)(iree_hal_queue_t* queue);
  // Returns unused backing without blocking publication.
  void (*trim)(iree_hal_queue_t* queue);
  // Cold native scratch detachment, or NULL without firmware-managed scratch.
  amdf_status_t (*trim_scratch)(iree_hal_queue_t* queue);
  // Configures optional capture after joining the queue's retirement tail.
  iree_status_t (*set_profile)(iree_hal_queue_t* queue,
                               iree_hal_amd_gpu_profile_t* profile);
  // Wakes admission after the device latches a shader failure.
  void (*notify_failure)(iree_hal_queue_t* queue);
  // Borrows a device-registered axis until queue destruction.
  void (*assign_frontier)(iree_hal_queue_t* queue,
                          iree_async_frontier_tracker_t* tracker,
                          iree_async_axis_t axis);
  // Widens host dependency metadata under the exact publisher.
  iree_status_t (*reserve_frontiers)(iree_hal_queue_t* queue, uint32_t count,
                                     iree_notification_t* notification);
  // Arms the synchronous final-release join after source registration.
  void (*set_release)(iree_hal_queue_t* queue,
                      bool (*release)(void*, iree_hal_queue_t*),
                      void* user_data);
  // Completed coordinate after the accepted-work join.
  uint64_t (*completed_epoch)(iree_hal_queue_t* queue);
  // Borrows this queue's exact publication callback.
  iree_hal_amd_gpu_publication_source_t (*publication_source)(
      iree_hal_queue_t* queue);
  // Borrows this queue's exact retirement callback.
  iree_hal_amd_gpu_completion_source_t (*completion_source)(
      iree_hal_queue_t* queue);
} iree_hal_amd_gpu_queue_lifecycle_t;

typedef struct iree_hal_amd_gpu_queue_slot_t iree_hal_amd_gpu_queue_slot_t;
typedef struct iree_hal_amd_gpu_queue_sources_t
    iree_hal_amd_gpu_queue_sources_t;

// One cold construction request. Coordinates and backing belong to the device
// and survive queue replacement; the returned queue borrows them without
// retaining the device. Native allocations execute on the acquiring caller.
typedef iree_status_t (*iree_hal_amd_gpu_queue_create_fn_t)(
    void* user_data, uint32_t ordinal, uint32_t source_ordinal,
    uint32_t frontier_count, uint64_t initial_epoch,
    iree_hal_amd_gpu_signal_cache_t* signals,
    const iree_hal_queue_params_t* params, iree_hal_queue_t** out_queue);

// Device-owned cold queue membership. Native vector coordinates are reused only
// after synchronous removal proves completion. Their monotonically continuing
// epochs preserve old signal readers and resident-program reuse proofs. All
// arrays grow on acquisition; release uses already-secured source-table
// scratch. No steady-state submission or retirement traverses this registry or
// locks it.
typedef struct iree_hal_amd_gpu_queue_membership_t {
  // Borrowed native owner and diagnostic sink, live through final cleanup.
  iree_hal_amd_gpu_context_t* context;
  // Allocator for stable slots and cold source-table storage.
  iree_allocator_t host_allocator;
  // Borrowed existing workers; membership creates no threads.
  iree_hal_amd_gpu_publication_t* publication;
  // Borrowed sole retirement owner and source-lifetime handoff.
  iree_hal_amd_gpu_completion_t* completion;
  // Borrowed registry of resident recordings requiring publisher coordinates.
  iree_hal_amd_gpu_program_pool_t* programs;
  // Immutable exact protocol ownership contract.
  const iree_hal_amd_gpu_queue_lifecycle_t* lifecycle;
  // Cold native construction supplied by the device's placement policies.
  iree_hal_amd_gpu_queue_create_fn_t create;
  // Borrowed device passed to create.
  void* user_data;
  // Fixed device-owned provisioned queue count.
  uint32_t provisioned_count;
  // Borrowed immutable provisioned queue array, populated during construction.
  iree_hal_queue_t** provisioned;
  // Serializes acquisition, release, and cold changes without execution locks.
  iree_slim_mutex_t mutex;
  // Protects live slot pointers during bounded failure fanout only.
  iree_slim_mutex_t slots_mutex;
  // Stable slots retained until final device cleanup, newest first.
  iree_hal_amd_gpu_queue_slot_t* slots;
  // Number of allocated dynamic coordinates, including currently vacant slots.
  uint32_t slot_count;
  // Installed geometric dependency-vector extent; never shrinks.
  uint32_t frontier_count;
  // Device-owned callback-tail notification for serialized cold requests.
  iree_notification_t notification;
  // Prepared active/spare tables; spare storage makes release allocation-free.
  iree_hal_amd_gpu_queue_sources_t* sources;
  // Borrowed active capture, installed on new queues before they are exposed.
  iree_hal_amd_gpu_profile_t* profile;
  // Immutable non-compute publisher sources at provisioned_count + [0, 3).
  iree_hal_amd_gpu_publication_source_t publication_tail[3];
  // Number of non-compute completion callbacks following all queue callbacks.
  uint32_t completion_tail_count;
  // Immutable transfer/feedback/shutdown callbacks; waits are never borrowed.
  iree_hal_amd_gpu_completion_source_t completion_tail[3];
  // Device-owned group namespace, retained once for all allocated coordinates.
  struct {
    // Retained tracker, or NULL outside a group.
    iree_async_frontier_tracker_t* tracker;
    // Group-assigned device axis with zero queue-index bits.
    iree_async_axis_t base_axis;
    // Successfully registered contiguous coordinate count.
    uint32_t count;
  } frontier;
  // Only present in the completion table while a cold release is joining.
  struct {
    // Borrowed queue whose accepted work must finish before removal.
    iree_hal_queue_t* queue;
    // Release-published after status; callback tails use notification above.
    iree_atomic_int32_t completed;
    // Terminal native observation status, read after completed.
    amdf_status_t status;
  } join;
} iree_hal_amd_gpu_queue_membership_t;

void iree_hal_amd_gpu_queue_membership_initialize(
    iree_hal_amd_gpu_queue_membership_t* membership);

// Called after all queue owners and both worker callback tables have joined.
// Returns false if an application still owns an acquired queue.
bool iree_hal_amd_gpu_queue_membership_deinitialize(
    iree_hal_amd_gpu_queue_membership_t* membership);

// True when no acquired queue remains; provisioned queues are owned separately.
bool iree_hal_amd_gpu_queue_membership_is_empty(
    iree_hal_amd_gpu_queue_membership_t* membership);

iree_status_t iree_hal_amd_gpu_queue_membership_acquire(
    iree_hal_amd_gpu_queue_membership_t* membership,
    const iree_hal_queue_params_t* params, iree_hal_queue_t** out_queue);

// Assigns or clears the group namespace before public work. Registration is
// transactional; the device retires axes only at namespace teardown, so a
// released queue's completed epochs remain queryable after slot reuse.
iree_status_t iree_hal_amd_gpu_queue_membership_assign_frontier(
    iree_hal_amd_gpu_queue_membership_t* membership,
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t base_axis);

// Cold failure fanout over acquired queues; provisioned queues are separate.
void iree_hal_amd_gpu_queue_membership_notify_failure(
    iree_hal_amd_gpu_queue_membership_t* membership);
// Trims provisioned and acquired queues, including idle native scratch. Native
// detachment runs on the calling thread; ready publication remains independent.
amdf_status_t iree_hal_amd_gpu_queue_membership_trim(
    iree_hal_amd_gpu_queue_membership_t* membership);

// Begins/ends capture across all live queues. The public profiling contract
// joins submissions and externally serializes this call with acquisition and
// release. Failure clears every session borrow before returning.
iree_status_t iree_hal_amd_gpu_queue_membership_set_profile(
    iree_hal_amd_gpu_queue_membership_t* membership,
    iree_hal_amd_gpu_profile_t* profile);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_MEMBERSHIP_H_
