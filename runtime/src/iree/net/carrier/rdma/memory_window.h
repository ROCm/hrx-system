// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Revocable RDMA memory window exports.
//
// Memory regions own expensive pinning and registration state. Memory windows
// are cheap export capabilities bound to a region and can be invalidated
// independently, matching iree_net_carrier_unregister_buffer semantics without
// deregistering the underlying iree_async_region_t.

#ifndef IREE_NET_CARRIER_RDMA_MEMORY_WINDOW_H_
#define IREE_NET_CARRIER_RDMA_MEMORY_WINDOW_H_

#include "iree/async/region.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/rdma/context.h"
#include "iree/net/carrier/rdma/queue_pair.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_memory_window_t iree_net_rdma_memory_window_t;

// Allocates an unbound Type 1 memory window in |context|'s protection domain.
IREE_API_EXPORT iree_status_t iree_net_rdma_memory_window_allocate(
    iree_net_rdma_context_t* context, iree_allocator_t host_allocator,
    iree_net_rdma_memory_window_t** out_memory_window);

// Deallocates |memory_window|, invalidating any active binding.
IREE_API_EXPORT void iree_net_rdma_memory_window_release(
    iree_net_rdma_memory_window_t* memory_window);

// Posts a signaled bind request for |memory_window| to |queue_pair|.
//
// The returned handle must not be sent to peers until the bind work request
// identified by |work_request_id| completes successfully on the send CQ.
IREE_API_EXPORT iree_status_t iree_net_rdma_memory_window_post_bind(
    iree_net_rdma_memory_window_t* memory_window,
    iree_net_rdma_queue_pair_t* queue_pair, struct ibv_mr* memory_region,
    void* base_ptr, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags, uint64_t work_request_id,
    iree_net_remote_handle_t* out_handle);

// Returns true when |handle| was exported by |memory_window|.
IREE_API_EXPORT bool iree_net_rdma_memory_window_matches_handle(
    const iree_net_rdma_memory_window_t* memory_window,
    iree_net_remote_handle_t handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_MEMORY_WINDOW_H_
