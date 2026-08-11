// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_GRAPH_MEMORY_H_
#define LIBHRX_SRC_BINDING_COMMON_GRAPH_MEMORY_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/device.h"

typedef struct iree_hal_streaming_buffer_t iree_hal_streaming_buffer_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_device_t iree_hal_streaming_device_t;
typedef struct iree_hal_streaming_graph_memory_allocation_t
    iree_hal_streaming_graph_memory_allocation_t;

// Creates a stable device virtual address for a graph allocation. Physical
// backing is acquired and mapped only when its allocation node executes.
iree_status_t iree_hal_streaming_graph_memory_allocation_create(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_hal_streaming_graph_memory_allocation_t** out_allocation);

// Retains/releases an allocation record held by graph nodes and graph-owned
// pointer metadata. The final release tears down an unmapped reservation.
void iree_hal_streaming_graph_memory_allocation_retain(
    iree_hal_streaming_graph_memory_allocation_t* allocation);
void iree_hal_streaming_graph_memory_allocation_release(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Claims the reference owned by the returned device pointer. A successful
// claim must be followed by a release, or restored when the free fails.
bool iree_hal_streaming_graph_memory_allocation_claim_pointer_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation);
void iree_hal_streaming_graph_memory_allocation_restore_pointer_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Claims a returned-pointer reference only if its allocation node has never
// executed. Graph destruction uses this to retire an address that was created
// during graph construction but was never made usable by a launch.
bool iree_hal_streaming_graph_memory_allocation_claim_unexecuted_pointer_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Claims the single graph free-node slot for an allocation. The claim remains
// until that node is destroyed with its graph, even after the node executes.
bool iree_hal_streaming_graph_memory_allocation_try_claim_free_node_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation);
void iree_hal_streaming_graph_memory_allocation_release_free_node_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Returns the stable raw device virtual address for |allocation|.
void* iree_hal_streaming_graph_memory_allocation_device_pointer(
    const iree_hal_streaming_graph_memory_allocation_t* allocation);

// Returns whether |allocation| has physical backing mapped at its virtual
// address. The value is synchronized with map and unmap operations.
bool iree_hal_streaming_graph_memory_allocation_is_mapped(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Resolves a base graph allocation pointer to a retained allocation record.
// Interior pointers are intentionally rejected because graph free nodes require
// the allocation node's exact device pointer.
iree_status_t iree_hal_streaming_graph_memory_allocation_lookup(
    iree_hal_streaming_context_t* context, uint64_t ptr,
    iree_hal_streaming_graph_memory_allocation_t** out_allocation);

// Ordered callbacks used by graph allocation/free nodes. Returning the HAL
// status makes a mapping failure fail the submitted stream rather than being
// hidden behind a host callback.
iree_status_t iree_hal_streaming_graph_memory_allocation_map_host_call(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context);
iree_status_t iree_hal_streaming_graph_memory_allocation_unmap_host_call(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context);

// Releases an allocation through normal HIP free paths after prior device use
// has completed.
iree_status_t iree_hal_streaming_graph_memory_allocation_unmap(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Releases physical backing only when it remains mapped. This supports a graph
// relaunch that automatically frees unmatched allocation nodes while allowing
// a prior explicit free to leave the allocation already unmapped.
iree_status_t iree_hal_streaming_graph_memory_allocation_unmap_if_mapped(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Returns graph-memory statistics backed by actual physical allocations.
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
iree_status_t iree_hal_streaming_graph_memory_trim(
    iree_hal_streaming_device_t* device);

#endif  // LIBHRX_SRC_BINDING_COMMON_GRAPH_MEMORY_H_
