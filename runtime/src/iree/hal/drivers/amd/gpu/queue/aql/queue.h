// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_QUEUE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_QUEUE_H_

#include "iree/hal/drivers/amd/gpu/completion.h"
#include "iree/hal/drivers/amd/gpu/publication.h"
#include "iree/hal/drivers/amd/gpu/queue/allocation.h"
#include "iree/hal/drivers/amd/gpu/queue/memory_mapping.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact AQL family construction, independent of the physical shader ISA.
typedef struct iree_hal_amd_gpu_aql_queue_options_t {
  // Native family selected by the device.
  uint32_t family_ordinal;
  // Source index in the device's immutable queue array.
  uint32_t queue_ordinal;
  // Power-of-two native packet storage extent in bytes.
  uint64_t command_byte_capacity;
  // Borrowed coherent host-readable/writable control and argument policy.
  const iree_hal_amd_fabric_memory_options_t* memory;
  // Borrowed local private-storage policy with no host mapping.
  const iree_hal_amd_fabric_memory_options_t* scratch_memory;
  // Existing device deferred publisher.
  iree_hal_amd_gpu_publication_t* publication;
  // Shared queue-ordered allocation owner.
  iree_hal_amd_gpu_allocation_context_t* allocations;
  // Shared queue-ordered mapping admission.
  iree_hal_amd_gpu_memory_mapping_context_t* mappings;
  // Power-of-two simultaneously published completion capacity.
  uint32_t retirement_capacity;
  // Initial argument generation capacity, in power-of-two bytes.
  uint32_t argument_capacity;
  // Reusable deferred capture block extent in bytes.
  iree_host_size_t capture_block_size;
  // Initial number of reusable host capture blocks.
  iree_host_size_t capture_block_count;
} iree_hal_amd_gpu_aql_queue_options_t;

// Creates a refcounted queue borrowing its family, context, and service owners.
// The parent registers publication/completion sources before exposing it and
// joins accepted work before removing those sources. Cold construction waits
// only for its typed backing acquisitions. Failure leaves out_queue NULL.
iree_status_t iree_hal_amd_gpu_aql_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_aql_queue_options_t* options,
    const iree_hal_queue_params_t* params, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue);

iree_status_t iree_hal_amd_gpu_aql_queue_assign_frontier(
    iree_hal_queue_t* queue, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis);

iree_hal_amd_gpu_publication_source_t
iree_hal_amd_gpu_aql_queue_publication_source(iree_hal_queue_t* queue);
iree_hal_amd_gpu_completion_source_t
iree_hal_amd_gpu_aql_queue_completion_source(iree_hal_queue_t* queue);

// Coalesced, nonblocking trim of unused host/backing storage. Native scratch
// detachment is a separate cold idle operation because it enters the kernel.
void iree_hal_amd_gpu_aql_queue_trim(iree_hal_queue_t* queue);

// Cold public trim caller attempts exclusive idle publication, joins the final
// consumed packet, and detaches native scratch. Busy queues retain their
// binding. Native release is asynchronous; acquisition may resume after
// publication handoff. Failure is terminal and preserves unproven backing.
amdf_status_t iree_hal_amd_gpu_aql_queue_trim_scratch(iree_hal_queue_t* queue);

// Closes public acceptance; accepted dependencies keep their normal meaning.
void iree_hal_amd_gpu_aql_queue_close(iree_hal_queue_t* queue);

// After external callers join, proves all accepted host/native uses retired.
bool iree_hal_amd_gpu_aql_queue_is_idle(iree_hal_queue_t* queue);

// Cold native tail/mapping/queue teardown after every peer queue is idle and
// service/completion sources and native callbacks have joined. Failure keeps
// unproven native users and backing alive; the parent preserves the context.
amdf_status_t iree_hal_amd_gpu_aql_queue_shutdown(iree_hal_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_QUEUE_H_
