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

#ifdef __cplusplus
extern "C" {
#endif

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

// Retires and releases the returned-pointer reference only if its allocation
// node has never executed. The pointer is first unpublished and every admitted
// lookup is drained, so no lookup can retain the allocation after its final
// reference reaches zero. A no-op when the allocation executed or another free
// operation already owns the pointer reference.
iree_status_t
iree_hal_streaming_graph_memory_allocation_release_unexecuted_pointer_reference(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Returns whether the allocation reference state permits a graph free-node
// claim. Callers snapshot all three values under the allocation mutex.
static inline bool
iree_hal_streaming_graph_memory_reference_can_claim_free_node(
    bool has_pointer_reference, bool is_pointer_reference_claimed,
    bool has_free_node_reference) {
  return has_pointer_reference && !is_pointer_reference_claimed &&
         !has_free_node_reference;
}

// Claims the single graph free-node slot for an allocation while its returned
// pointer reference is live and unclaimed. The claim remains until that node is
// destroyed with its graph, even after the node executes.
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

// Returns true when an unmatched allocation remains mapped by a completed
// launch. A mapping left by a synchronously rejected launch is retry residue,
// not a completed live allocation, and is consumed by the next map callback.
bool iree_hal_streaming_graph_memory_allocation_has_live_unfreed_mapping(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Converts this launch attempt's fully successful map into retry residue after
// a later graph block is synchronously rejected. A prior residue is preserved
// when this attempt's map callback did not run.
void iree_hal_streaming_graph_memory_allocation_mark_failed_launch_retry(
    iree_hal_streaming_graph_memory_allocation_t* allocation);

// Ensures a graph launch owns and publishes the stable returned-pointer
// reference. Publications persist across launch submission failures so an
// accepted mapping prefix and its returned pointer remain one retryable state.
// Resets only this attempt's successful-map evidence; retry residue from an
// earlier rejected attempt remains available to the map callback.
iree_status_t
iree_hal_streaming_graph_memory_allocation_prepare_pointer_for_launch(
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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_GRAPH_MEMORY_H_
