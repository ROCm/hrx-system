// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/submitted_signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_semaphore_t
//===----------------------------------------------------------------------===//

// Creates an AMDGPU HAL semaphore backed by an embedded async semaphore.
//
// Signal, query, and wait all delegate to the async semaphore infrastructure.
// The semaphore embeds iree_async_semaphore_t at offset 0 for toll-free
// bridging between HAL and async layers.
//
// |device| is stored as a back-pointer for type discrimination (checking
// whether a semaphore belongs to a specific logical device). Not retained.
//
// |queue_family_affinity| is the complete set of queue families that may use
// the semaphore. The implementation may optimize synchronization scope when
// all selected families belong to one physical device.
//
// |flags| controls semaphore behavior:
//   DEVICE_LOCAL: only signaled/waited by queues within this device. Enables
//     epoch-based hardware synchronization (barrier-value packets).
//   HOST_INTERRUPT: host may call iree_hal_semaphore_wait. Enables
//     interrupt-driven host blocking via HSA signal waits.
//   SINGLE_PRODUCER: signals come from one producer timeline, allowing the
//     implementation to treat the latest producer queue epoch as the complete
//     causal frontier for the latest payload value.
iree_status_t iree_hal_amdgpu_semaphore_create(
    iree_hal_amdgpu_logical_device_t* device, iree_async_proactor_t* proactor,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

// Returns true if |semaphore| is an AMDGPU semaphore.
bool iree_hal_amdgpu_semaphore_isa(iree_hal_semaphore_t* semaphore);

// Returns true if |semaphore| is an AMDGPU semaphore belonging to |device|.
// Used by the submission path to gate the epoch-based synchronization fast
// path: only semaphores local to the submitting device can use barrier-value
// packets on the device's queue epoch signals. Non-local semaphores (from
// other HAL devices, remoting, etc.) always use the software timepoint path.
bool iree_hal_amdgpu_semaphore_is_local(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_amdgpu_logical_device_t* device);

// Returns the AMDGPU semaphore creation flags. Caller must verify
// iree_hal_amdgpu_semaphore_isa() first.
iree_hal_semaphore_flags_t iree_hal_amdgpu_semaphore_flags(
    iree_hal_semaphore_t* semaphore);

// Returns the AMDGPU semaphore queue family compatibility domain. Caller must
// verify iree_hal_amdgpu_semaphore_isa() first.
iree_hal_queue_family_affinity_t
iree_hal_amdgpu_semaphore_queue_family_affinity(
    iree_hal_semaphore_t* semaphore);

// Returns true if |semaphore| has the strict private-stream contract used by
// HIP-on-HAL stream timelines:
//   - owned by |device|;
//   - device-local;
//   - single-producer; and
//   - not host-interrupt/export/timepoint-export capable.
//
// Such semaphores are still normal HAL timeline semaphores, but AMDGPU may use
// the single-producer proof to publish only the producer queue epoch on the
// signal hot path. Completion drain still advances the timeline value, but
// does not need to accumulate a multi-producer async frontier for the private
// stream handoff.
bool iree_hal_amdgpu_semaphore_has_private_stream_semantics(
    iree_hal_semaphore_t* semaphore,
    const iree_hal_amdgpu_logical_device_t* device);

// Returns the latest submitted-signal metadata on an AMDGPU semaphore.
// Caller must verify iree_hal_amdgpu_semaphore_isa() first.
iree_hal_submitted_signal_t* iree_hal_amdgpu_semaphore_submitted_signal(
    iree_hal_semaphore_t* semaphore);

// Publishes the submission-time frontier and submitted-signal metadata
// from |producer_axis| at (|producer_epoch|, |producer_value|).
//
// Merges |producer_frontier| into the semaphore's accumulated frontier under
// the semaphore mutex, then updates the submitted-signal metadata while holding
// that mutex so PRODUCER_FRONTIER_EXACT reflects the post-merge frontier
// precisely. Returns false if the frontier merge overflowed capacity; in that
// case the cache is cleared and callers must fall back to software waits for
// not-yet-complete values.
//
// Caller must verify iree_hal_amdgpu_semaphore_isa() first.
bool iree_hal_amdgpu_semaphore_publish_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t producer_axis,
    const iree_async_frontier_t* producer_frontier, uint64_t producer_epoch,
    uint64_t producer_value);

// Publishes a single-producer private-stream signal without accumulating the
// full semaphore frontier under the async semaphore mutex.
//
// Caller must prove iree_hal_amdgpu_semaphore_has_private_stream_semantics()
// and serialize all signals through |producer_axis|. The submitted metadata is
// updated as PRODUCER_FRONTIER_EXACT because waiting on the producer queue
// epoch is sufficient to observe the signaled payload's transitive
// dependencies.
void iree_hal_amdgpu_semaphore_publish_private_stream_signal(
    iree_hal_semaphore_t* semaphore, iree_async_axis_t producer_axis,
    uint64_t producer_epoch, uint64_t producer_value);

// Clears the semaphore's submitted-signal metadata.
//
// Caller must verify iree_hal_amdgpu_semaphore_isa() first.
void iree_hal_amdgpu_semaphore_clear_submitted_signal(
    iree_hal_semaphore_t* semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SEMAPHORE_H_
