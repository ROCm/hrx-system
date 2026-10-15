// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_QUEUE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_QUEUE_H_

#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/completion.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/barrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Concrete USER-publication construction. The parent has selected this exact
// protocol/family independently of the physical compiler target. No worker or
// default HAL allocation pool is created by a queue.
typedef struct iree_hal_amd_gpu_pm4_umq_queue_options_t {
  // Qualified native family, ring size, priority, and SINGLE producer mode.
  amdf_gpu_user_queue_create_info_t native;
  // Borrowed native control/argument policy. It contains exactly this context's
  // device at access ordinal zero, with coherent host read/write and GPU
  // access. All pointed-to storage remains live through final queue cleanup.
  const iree_hal_amd_fabric_memory_options_t* memory;
  // Borrowed device-shared thread-backed service, including cold cleanup.
  iree_hal_memory_maintenance_t* service;
  // Prepared physical cache-control encoding.
  iree_hal_amd_gpu_pm4_barrier_profile_t barriers;
  // Power-of-two number of simultaneously published native completion records.
  uint32_t retirement_capacity;
  // Initial GPU-readable argument ring capacity in bytes, a power of two.
  // Oversized operations grow it on service before publication.
  uint32_t argument_capacity;
  // Fixed CPU capture block size; larger captures use the host allocator.
  iree_host_size_t capture_block_size;
  // Number of CPU capture blocks to allocate during construction.
  iree_host_size_t capture_block_count;
} iree_hal_amd_gpu_pm4_umq_queue_options_t;

// Creates one exact refcounted HAL queue. The context, family, and service are
// borrowed from its parent device. Native construction runs on service. The
// parent registers completion_source before exposing the returned queue and
// keeps that source registered until accepted work has retired. Success assigns
// one owning reference; failure leaves out_queue NULL.
iree_status_t iree_hal_amd_gpu_pm4_umq_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_pm4_umq_queue_options_t* options,
    const iree_hal_queue_params_t* params, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue);

// Assigns a group-owned completion axis before any public work is accepted.
// Registration is transactional. NULL aborts a prior assignment during failed
// group construction. The queue retains the tracker and retires its axis after
// the completion observer joins. Native submission never advances this axis.
iree_status_t iree_hal_amd_gpu_pm4_umq_queue_assign_frontier(
    iree_hal_queue_t* queue, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis);

// Borrows the queue's bounded record observer for its device's sole completion
// worker. A native failure fails accepted work but retains unproven GPU uses.
iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_pm4_umq_queue_completion_source(iree_hal_queue_t* queue);

// Trims unused capture blocks, replaced argument generations, and inactive
// signal backing on the device's shared service. Live publishers/readers keep
// their storage. This never waits for native execution to complete.
void iree_hal_amd_gpu_pm4_umq_queue_trim(iree_hal_queue_t* queue);

// Rejects subsequent public submissions after the parent has joined external
// API callers. Already accepted work retains its normal dependency semantics.
// This is a nonblocking state change; the parent wakes its native observer.
void iree_hal_amd_gpu_pm4_umq_queue_close(iree_hal_queue_t* queue);

// Cold shutdown predicate after close. True proves every accepted capture and
// publishing caller has retired, including deferred host waits. It does not
// prove consumption of the final notification/padding tail.
bool iree_hal_amd_gpu_pm4_umq_queue_is_idle(iree_hal_queue_t* queue);

// Joins service turns and the final native notification tail after every queue
// in this native domain is idle and the completion observer has made its final
// access. Peer consumers may retain this queue's signal backing until their own
// retirement. Runs outside service; native work is
// placed on service. Success permits final queue release. Failure preserves the
// queue/context graph and must be handled by the parent without freeing it.
amdf_status_t iree_hal_amd_gpu_pm4_umq_queue_shutdown(iree_hal_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_QUEUE_H_
