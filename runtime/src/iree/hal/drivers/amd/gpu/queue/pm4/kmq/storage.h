// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_KMQ_STORAGE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_KMQ_STORAGE_H_

#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/ring.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Finite executable command storage borrowed by native KERNEL submissions.
// The exclusive publisher owns preparation and acceptance. The completion
// worker observes accepted points only after the enclosing queue publishes its
// count, and returns credit only after checked native retirement. This owner
// is allocated only for KERNEL publication; USER queues pay no point-array
// cost.
typedef struct iree_hal_amd_gpu_pm4_kmq_storage_t {
  // Borrowed native device and API tables, live through native removal.
  const iree_hal_amd_gpu_context_t* context;
  // Allocator owning this structure and its trailing native-point array.
  iree_allocator_t host_allocator;
  // Owned queue, consumed on native removal except an API BUSY rejection.
  amdf_kernel_queue_t* queue;
  // Owned executable bytes and coherent host producer mapping.
  iree_hal_amd_fabric_memory_t memory;
  // Native command byte zero, used by resident-program continuations.
  uint64_t address;
  // Prepared immutable command that has not yet been accepted. A zero length
  // means no prepared publication. BUSY retains the complete byte range.
  amdf_gpu_kernel_command_t pending;
  // Reservation positions restored if native publication rejects this piece.
  struct {
    // Command cursor before preparing the pending piece, including wrap.
    uint64_t words;
    // Argument cursor before preparing the pending piece.
    uint64_t arguments;
  } rollback;
  // Opaque native points indexed by the enclosing queue's publication slots.
  uint64_t* points;
  // Power-of-two point-array capacity minus one.
  uint32_t capacity_mask;
  // Last accepted native point, owned by the publisher until joined cleanup.
  uint64_t last_submission;
  // Point armed by the single completion observer, or zero before any arm.
  uint64_t armed_submission;
  // Idle observer requests one host wake from the next successful publisher.
  iree_atomic_int32_t wake_required;
  // Sticky failed removal after the native queue handle was consumed.
  amdf_status_t removal_status;
} iree_hal_amd_gpu_pm4_kmq_storage_t;

// Allocates a partial owner before acquiring native resources. The caller
// always frees a non-NULL output, including after failed preparation. Failure
// never publishes out_ring. All capacities and family facts are already
// admitted by the enclosing PM4 queue. Runs only on the device service.
iree_status_t iree_hal_amd_gpu_pm4_kmq_storage_allocate(
    const iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    uint64_t command_capacity, uint32_t submission_capacity,
    const iree_hal_amd_fabric_memory_options_t* memory_options,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_kmq_storage_t** out_storage,
    iree_hal_amd_gpu_pm4_ring_t* out_ring);

// Attempts the already-prepared piece. Success records its opaque native point
// in position's slot and clears pending. Any failure leaves pending unchanged
// and grants no native borrow; the caller retains or rolls back its
// reservation.
amdf_status_t iree_hal_amd_gpu_pm4_kmq_storage_submit(
    iree_hal_amd_gpu_pm4_kmq_storage_t* storage, uint64_t position);

// Refreshes checked native retirement once, preserving terminal native errors.
// Completion readiness in PM4 memory never substitutes for this native join.
amdf_status_t iree_hal_amd_gpu_pm4_kmq_storage_refresh(
    iree_hal_amd_gpu_pm4_kmq_storage_t* storage, uint64_t* out_retired);

// Arms the oldest unchecked native point on the device's shared wake. GPU
// KERNEL notifications are fence notifications, independent of PM4 markers.
// Repeated observations of the same pending point do not add native waiters.
amdf_status_t iree_hal_amd_gpu_pm4_kmq_storage_arm(
    iree_hal_amd_gpu_pm4_kmq_storage_t* storage, uint64_t submission);

// Joins the last accepted point and removes the native queue before releasing
// its executable backing. All publishers and observers have already joined.
// Success frees storage; failure preserves it and every unproven native use.
// NULL is an empty owner. Runs only on the device service.
amdf_status_t iree_hal_amd_gpu_pm4_kmq_storage_free(
    iree_hal_amd_gpu_pm4_kmq_storage_t* storage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_KMQ_STORAGE_H_
