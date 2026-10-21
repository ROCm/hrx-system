// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_QUEUE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_QUEUE_H_

#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/completion.h"
#include "iree/hal/drivers/amd/gpu/feedback.h"
#include "iree/hal/drivers/amd/gpu/profile.h"
#include "iree/hal/drivers/amd/gpu/publication.h"
#include "iree/hal/drivers/amd/gpu/queue/allocation.h"
#include "iree/hal/drivers/amd/gpu/queue/backing.h"
#include "iree/hal/drivers/amd/gpu/queue/file_staging.h"
#include "iree/hal/drivers/amd/gpu/queue/memory_mapping.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/scratch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Concrete PM4 publication construction. The parent has selected this exact
// protocol/family independently of the physical compiler target. No worker or
// default HAL allocation pool is created by a queue.
typedef struct iree_hal_amd_gpu_pm4_queue_options_t {
  // Exact native family selected independently of the compiler target.
  uint32_t family_ordinal;
  // Stable ordinal in the parent's provisioned queue array.
  uint32_t queue_ordinal;
  // Device-owned native cells and return wake target, live beyond this queue.
  iree_hal_amd_gpu_signal_cache_t* signals;
  // One admitted USER or KERNEL publication mode.
  amdf_queue_publication_modes_t publication_mode;
  // Power-of-two command storage capacity in bytes.
  uint64_t command_byte_capacity;
  // Borrowed native control/argument policy. It contains exactly this context's
  // device at access ordinal zero, with coherent host read/write and GPU
  // access. All pointed-to storage remains live through final queue cleanup.
  const iree_hal_amd_fabric_memory_options_t* memory;
  // Borrowed read/write resident policy for lazy private storage, no host map.
  const iree_hal_amd_fabric_memory_options_t* scratch_memory;
  // Native private-storage geometry, captured before accepting dispatches.
  iree_hal_amd_gpu_pm4_scratch_profile_t scratch_profile;
  // Borrowed device publisher, independent of native allocation and retirement.
  iree_hal_amd_gpu_publication_t* publication;
  // Borrowed device-shared allocation executor, independent of publication.
  iree_hal_amd_gpu_allocation_context_t* allocations;
  // Borrowed device-shared mapping admission and result publication.
  iree_hal_amd_gpu_memory_mapping_context_t* mappings;
  // Borrowed shared file-copy pipeline, independent of this packet protocol.
  iree_hal_amd_gpu_file_staging_t* file_staging;
  // Optional shared channel drained before host completion on the sole worker.
  iree_hal_amd_gpu_feedback_t* feedback;
  // Prepared physical cache-control encoding.
  iree_hal_amd_gpu_pm4_barrier_profile_t barriers;
  // Power-of-two number of simultaneously published native completion records.
  uint32_t retirement_capacity;
  // Initial GPU-readable argument ring capacity in bytes, a power of two.
  // Oversized operations await typed native acquisition outside publication.
  uint32_t argument_capacity;
  // Fixed CPU block size for captures and deferred wait records. Construction
  // rejects sizes unable to hold either header; oversized captures allocate.
  iree_host_size_t capture_block_size;
  // Number of CPU capture blocks to allocate during construction.
  iree_host_size_t capture_block_count;
} iree_hal_amd_gpu_pm4_queue_options_t;

// Returns the publication modes satisfying the PM4 dispatch, transfer, cache,
// and native signal contract for captured family facts. Zero means this queue
// implementation cannot execute that family. This cold query is independent of
// physical compiler identity and performs no native calls or allocation.
amdf_queue_publication_modes_t iree_hal_amd_gpu_pm4_queue_publication_modes(
    const amdf_queue_family_info_t* native);

// Returns true when this host can satisfy the PM4 publication memory model.
// This cold query performs no native calls or allocation.
bool iree_hal_amd_gpu_pm4_queue_host_is_supported(void);

// Creates one exact refcounted HAL queue. Context, family, and publication are
// borrowed from its parent device. Cold construction joins typed native backing
// acquisitions on its caller. The parent registers both publication_source and
// completion_source before exposing the returned queue and
// keeps that source registered until accepted work has retired. Success assigns
// one owning reference; failure leaves out_queue NULL.
iree_status_t iree_hal_amd_gpu_pm4_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_pm4_queue_options_t* options,
    const iree_hal_queue_params_t* params, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue);

// Assigns a group-owned completion axis before any public work is accepted.
// Registration is transactional. NULL aborts a prior assignment during failed
// group construction. The queue retains the tracker and retires its axis after
// the completion observer joins. Native submission never advances this axis.
iree_status_t iree_hal_amd_gpu_pm4_queue_assign_frontier(
    iree_hal_queue_t* queue, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis);

// Borrows a device-owned profiling session until idle end detaches it.
void iree_hal_amd_gpu_pm4_queue_set_profile(
    iree_hal_queue_t* queue, iree_hal_amd_gpu_profile_t* profile);

// Borrows the queue's bounded deferred publication source. The parent joins
// this source and all native acquisition callbacks before destroying the queue.
iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_pm4_queue_publication_source(iree_hal_queue_t* queue);

// Borrows the queue's bounded record observer for its device's sole completion
// worker. A native failure fails accepted work but retains unproven GPU uses.
iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_pm4_queue_completion_source(iree_hal_queue_t* queue);

// Wakes deferred admission after the shared feedback owner latches failure.
// Already-published native work retains its completion and cleanup ownership.
void iree_hal_amd_gpu_pm4_queue_notify_failure(iree_hal_queue_t* queue);

// Requests trim of unused captures, replaced argument generations, and idle
// signal backing under queue publication ownership. Requests coalesce and can
// complete inline; live readers keep their storage. This never waits for native
// execution or backing release. The parent remains live through publication.
void iree_hal_amd_gpu_pm4_queue_trim(iree_hal_queue_t* queue);

// Rejects subsequent public submissions after the parent has joined external
// API callers. Already accepted work retains its normal dependency semantics.
// This is a nonblocking state change; the parent wakes its native observer.
void iree_hal_amd_gpu_pm4_queue_close(iree_hal_queue_t* queue);

// Cold shutdown predicate after close. True proves every accepted capture and
// publishing caller has retired, including deferred host waits. It does not
// prove consumption of the final notification/padding tail.
bool iree_hal_amd_gpu_pm4_queue_is_idle(iree_hal_queue_t* queue);

// Joins the final native notification tail after this queue is idle, its native
// acquisition callbacks have joined, and publication/completion sources have
// made their final accesses. Peer consumers may retain device-owned signal
// backing until their own retirement. Cold native cleanup
// runs on the caller; backing releases use the allocation domain. Failure
// preserves the queue/context graph and must be handled by the parent without
// freeing it.
amdf_status_t iree_hal_amd_gpu_pm4_queue_shutdown(iree_hal_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_QUEUE_H_
