// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_ALLOCATION_QUEUE_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_ALLOCATION_QUEUE_H_

#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// A fabric-private native allocation domain shared by execution engines.
// Only ready native backing acquisition and release execute on its worker.
typedef struct iree_hal_amd_fabric_allocation_queue_t
    iree_hal_amd_fabric_allocation_queue_t;

// Creates a ref-counted native allocation queue. A concrete NUMA node pins the
// worker before it starts; UINT32_MAX leaves placement unspecified. Failure
// publishes no output. The fabric keeps one reference per selected node.
iree_status_t iree_hal_amd_fabric_allocation_queue_create(
    uint32_t node_id, iree_allocator_t host_allocator,
    iree_hal_amd_fabric_allocation_queue_t** out_queue);

void iree_hal_amd_fabric_allocation_queue_retain(
    iree_hal_amd_fabric_allocation_queue_t* queue);
void iree_hal_amd_fabric_allocation_queue_release(
    iree_hal_amd_fabric_allocation_queue_t* queue);

// Enqueues a prepared native slab batch. Its fabric provider and complete
// request remain live through completion. Submission allocates no storage and
// performs no native calls. No dependency wait belongs in this queue.
void iree_hal_amd_fabric_allocation_queue_enqueue(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_slab_operation_t* operation);

// Native backing used directly by an implementation, without HAL buffer or
// pool materialization. This has the same allocation owner as physical slabs.
// These descriptors carry no execution function or dependency wait.
typedef enum iree_hal_amd_fabric_memory_operation_type_e {
  IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_ACQUIRE,
  IREE_HAL_AMD_FABRIC_MEMORY_OPERATION_RELEASE,
} iree_hal_amd_fabric_memory_operation_type_t;

typedef struct iree_hal_amd_fabric_memory_operation_t {
  // Queue-owned linkage through terminal completion.
  struct iree_hal_amd_fabric_memory_operation_t* next;
  // Concrete native operation, set by the submission entry point.
  iree_hal_amd_fabric_memory_operation_type_t type;
  // Borrowed native API, live through terminal completion.
  const amdf_api_t* api;
  // Borrowed diagnostic destination for native cleanup failures.
  iree_hal_device_event_sink_t event_sink;
} iree_hal_amd_fabric_memory_operation_t;

typedef struct iree_hal_amd_fabric_memory_request_t {
  // Qualified native policy, borrowed through terminal completion.
  const iree_hal_amd_fabric_memory_options_t* options;
  // Requested usable extent in bytes, rounded by the native profile.
  uint64_t byte_length;
  // Required power-of-two alignment of native byte zero.
  uint64_t alignment;
} iree_hal_amd_fabric_memory_request_t;

typedef struct iree_hal_amd_fabric_memory_acquire_request_t {
  // Queue linkage and borrowed native API/diagnostics.
  iree_hal_amd_fabric_memory_operation_t operation;
  // Number of native requests and output backings.
  iree_host_size_t count;
  // Qualified geometry and policy for each native backing.
  const iree_hal_amd_fabric_memory_request_t* requests;
  // Private output array, published only by terminal completion.
  iree_hal_amd_fabric_memory_t* memories;
  // Receives the whole batch result; failure rolls back every output.
  iree_hal_slab_acquire_callback_t completion;
} iree_hal_amd_fabric_memory_acquire_request_t;

typedef struct iree_hal_amd_fabric_memory_release_request_t {
  // Queue linkage and borrowed native API/diagnostics.
  iree_hal_amd_fabric_memory_operation_t operation;
  // Number of native backings consumed by this release.
  iree_host_size_t count;
  // Retired native owners; release storage was secured during acquisition.
  iree_hal_amd_fabric_memory_t* memories;
  // Runs after every native release attempt, including diagnosed failures.
  iree_hal_slab_release_callback_t completion;
} iree_hal_amd_fabric_memory_release_request_t;

// Submits a ready native acquisition without allocating or calling the driver.
// The request, its arrays, options, and their native owners remain live through
// completion. Exactly one callback takes the result: all outputs are owned on
// success, and all outputs are empty on failure. Completion may free or reuse
// the request and may happen before this call returns. The caller makes no
// access to transferred request storage after submission.
//
// Completion publishes backing to its consumer; it does not run GPU work or
// wait for execution dependencies on the allocation worker. The queue retains
// itself through the callback; native object ownership stays with the caller.
void iree_hal_amd_fabric_allocation_queue_acquire_memory(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_amd_fabric_memory_acquire_request_t* request);

// Consumes retired backing without allocation or a driver call on the caller.
// The caller establishes retirement before submission and embeds or reserves
// the request and array during acquisition. Completion has the same lifetime
// contract as acquire and runs after every release attempt. Native cleanup
// failures are diagnosed; consumed descriptors are never retried.
void iree_hal_amd_fabric_allocation_queue_release_memory(
    iree_hal_amd_fabric_allocation_queue_t* queue,
    iree_hal_amd_fabric_memory_release_request_t* request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_ALLOCATION_QUEUE_H_
