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
// pointer metadata. The final release tears down any mapping and reservation.
void iree_hal_streaming_graph_memory_allocation_retain(
    iree_hal_streaming_graph_memory_allocation_t* allocation);
void iree_hal_streaming_graph_memory_allocation_release(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Claims the returned-pointer reference for a synchronous external free. The
// allocation must have executed and must not have a graph free node.
bool iree_hal_streaming_graph_memory_allocation_claim_external_free_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Claims the returned-pointer reference for a stream-ordered external free.
// The allocation node may still be pending earlier on the same stream.
bool iree_hal_streaming_graph_memory_allocation_claim_async_free_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Claims the returned-pointer reference from its graph free node after the
// allocation node has executed.
bool iree_hal_streaming_graph_memory_allocation_claim_graph_free_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Completes or rolls back a successful returned-pointer reference claim.
void iree_hal_streaming_graph_memory_allocation_complete_pointer_reference(
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

// Returns the borrowed context that owns |allocation|'s virtual reservation.
// The allocation retains the context for its complete lifetime.
iree_hal_streaming_context_t*
iree_hal_streaming_graph_memory_allocation_context(
    const iree_hal_streaming_graph_memory_allocation_t* allocation);

// Returns whether |allocation| has physical backing mapped at its virtual
// address. The value is synchronized with map and unmap operations.
bool iree_hal_streaming_graph_memory_allocation_is_mapped(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Ensures a graph launch owns and publishes the stable returned-pointer
// reference. |out_did_publish| is true only when this call restored a pointer
// retired by an earlier external or graph-ordered free.
iree_status_t
iree_hal_streaming_graph_memory_allocation_prepare_pointer_for_launch(
    iree_hal_streaming_graph_memory_allocation_t* allocation,
    bool* out_did_publish);

// Rolls back a pointer publication performed while preparing a launch that was
// not submitted. A map accepted before a later submission failure is undone
// before the pointer reference is retired.
iree_status_t
iree_hal_streaming_graph_memory_allocation_rollback_launch_pointer(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Resolves a base graph allocation pointer to a retained allocation record.
// Interior pointers are intentionally rejected because graph free nodes require
// the allocation node's exact device pointer.
iree_status_t iree_hal_streaming_graph_memory_allocation_lookup(
    iree_hal_streaming_context_t* context, uint64_t ptr,
    iree_hal_streaming_graph_memory_allocation_t** out_allocation);

// Returns ordered map/unmap calls whose queue-owned resource keeps the graph
// allocation alive until the callback completes or is cancelled.
iree_hal_host_call_t iree_hal_streaming_graph_memory_allocation_map_call(
    iree_hal_streaming_graph_memory_allocation_t* allocation);
iree_hal_host_call_t iree_hal_streaming_graph_memory_allocation_unmap_call(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

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
