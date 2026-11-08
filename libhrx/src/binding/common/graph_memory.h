// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_GRAPH_MEMORY_H_
#define LIBHRX_SRC_BINDING_COMMON_GRAPH_MEMORY_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;
typedef struct iree_hal_streaming_graph_allocation_t
    iree_hal_streaming_graph_allocation_t;

// Reserves one stable graph address without acquiring physical storage.
// Devices and accesses describe the complete admitted set; device zero owns
// physical placement. Nodes retain the allocation, and each accepted allocation
// also owns one reference until a graph free or an external HIP free.
iree_status_t iree_hal_streaming_graph_allocation_create(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_host_size_t device_count, const hrx_device_t* devices,
    const iree_hal_memory_access_t* accesses,
    iree_hal_streaming_graph_allocation_t** out_allocation, void** out_pointer);
void iree_hal_streaming_graph_allocation_retain(
    iree_hal_streaming_graph_allocation_t* allocation);
void iree_hal_streaming_graph_allocation_release(
    iree_hal_streaming_graph_allocation_t* allocation);

typedef uint32_t iree_hal_streaming_graph_allocation_flags_t;
enum iree_hal_streaming_graph_allocation_flag_bits_e {
  // End the current allocation lifetime.
  IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_NONE = 0,
  // Begin an allocation lifetime at the reserved address.
  IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_ALLOCATE = 1u << 0,
  // With ALLOCATE, first end any still-live preceding allocation.
  IREE_HAL_STREAMING_GRAPH_ALLOCATION_FLAG_AUTO_FREE = 1u << 1,
};

// Captures an allocation or free using the caller's timeline dependencies.
// The accepted point covers any accepted prefix even when a later queue call
// rejects. It is borrowed from the allocation or caller until launch cleanup.
iree_status_t iree_hal_streaming_graph_allocation_enqueue(
    iree_hal_streaming_graph_allocation_t* allocation, iree_hal_queue_t* queue,
    iree_hal_semaphore_list_t waits, iree_hal_semaphore_list_t signals,
    iree_hal_streaming_graph_allocation_flags_t flags,
    iree_hal_semaphore_t** out_accepted_semaphore,
    uint64_t* out_accepted_value);

// Acquires a graph allocation from the frontend's existing pointer table.
// Ordinary pointers return NULL; interior graph pointers are rejected.
iree_status_t iree_hal_streaming_graph_allocation_lookup(
    iree_hal_streaming_context_t* context, uint64_t pointer,
    iree_hal_streaming_graph_allocation_t** out_allocation);

// Ends the allocation on a stream, or synchronously when stream is NULL.
// The reservation remains live while a graph can replay the allocation node.
iree_status_t iree_hal_streaming_graph_allocation_free(
    iree_hal_streaming_graph_allocation_t* allocation,
    iree_hal_streaming_stream_t* stream);

// Counters follow accepted allocation/free operations. Used bytes count logical
// lifetimes; reserved bytes count retained page-aligned physical extents,
// including cached storage. They do not wait for native completion.
uint64_t iree_hal_streaming_graph_memory_used_current(
    iree_hal_streaming_device_t* device);
uint64_t iree_hal_streaming_graph_memory_used_high(
    iree_hal_streaming_device_t* device);
uint64_t iree_hal_streaming_graph_memory_reserved_current(
    iree_hal_streaming_device_t* device);
uint64_t iree_hal_streaming_graph_memory_reserved_high(
    iree_hal_streaming_device_t* device);
void iree_hal_streaming_graph_memory_reset_used_high(
    iree_hal_streaming_device_t* device);
void iree_hal_streaming_graph_memory_reset_reserved_high(
    iree_hal_streaming_device_t* device);

// Returns completed cached physical blocks. Pending blocks remain cached;
// trimming never waits for an independently progressing graph.
iree_status_t iree_hal_streaming_graph_memory_trim(
    iree_hal_streaming_device_t* device);
void iree_hal_streaming_graph_memory_deinitialize(
    iree_hal_streaming_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_GRAPH_MEMORY_H_
