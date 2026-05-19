// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed-slot receive posting for RDMA RC queue pairs.
//
// This component owns the local receive WQE inventory: each posted receive
// holds one buffer-pool lease and one generation-checked WR table entry. The
// carrier completion dispatcher completes the WR table entry once per CQE and
// then routes RECV completions here to recover the receive buffer lease in
// O(1).

#ifndef IREE_NET_CARRIER_RDMA_RECEIVE_QUEUE_H_
#define IREE_NET_CARRIER_RDMA_RECEIVE_QUEUE_H_

#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"
#include "iree/net/carrier/rdma/queue_pair.h"
#include "iree/net/carrier/rdma/work_request_table.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_receive_slot_t iree_net_rdma_receive_slot_t;

typedef struct iree_net_rdma_receive_queue_t {
  // Borrowed queue pair receiving the posted WQEs.
  iree_net_rdma_queue_pair_t* queue_pair;

  // Borrowed WR table used to tag posted receive WQEs.
  iree_net_rdma_work_request_table_t* work_request_table;

  // Borrowed buffer pool supplying receive buffers.
  iree_async_buffer_pool_t* buffer_pool;

  // Host allocator used for slot storage.
  iree_allocator_t host_allocator;

  // Allocated receive slot storage.
  iree_net_rdma_receive_slot_t* slots;

  // Total number of receive slots.
  uint32_t capacity;

  // Number of receive slots currently available for posting.
  uint32_t available_capacity;

  // Number of receive slots currently posted to the native QP.
  uint32_t posted_count;

  // Index of the first free slot, or UINT32_MAX when full.
  uint32_t free_head;
} iree_net_rdma_receive_queue_t;

// Initializes a fixed-capacity receive queue.
//
// |queue_pair|, |work_request_table|, and |buffer_pool| are borrowed and must
// outlive |out_queue|. Callers must serialize access; this component contains
// no locks so the carrier can choose the correct synchronization boundary.
IREE_API_EXPORT iree_status_t iree_net_rdma_receive_queue_initialize(
    iree_net_rdma_queue_pair_t* queue_pair,
    iree_net_rdma_work_request_table_t* work_request_table,
    iree_async_buffer_pool_t* buffer_pool, uint32_t capacity,
    iree_allocator_t host_allocator, iree_net_rdma_receive_queue_t* out_queue);

// Deinitializes |queue| and releases locally held receive-buffer leases.
//
// Any posted receives must have completed, been flushed, or been made
// unreachable by QP destruction before deinitialization. This function reclaims
// the local bookkeeping for those WQEs and releases their leases; it cannot
// prevent a live QP from later producing stale completions.
IREE_API_EXPORT void iree_net_rdma_receive_queue_deinitialize(
    iree_net_rdma_receive_queue_t* queue);

// Returns the number of receives currently posted to the native QP.
IREE_API_EXPORT uint32_t iree_net_rdma_receive_queue_posted_count(
    const iree_net_rdma_receive_queue_t* queue);

// Returns the number of receive slots available for new postings.
IREE_API_EXPORT uint32_t iree_net_rdma_receive_queue_available_capacity(
    const iree_net_rdma_receive_queue_t* queue);

// Posts receives until |target_posted_count| is reached.
//
// |out_posted_count| receives the number of new WQEs successfully posted by
// this call. Earlier WQEs posted by the same call remain owned by the QP if a
// later posting fails.
IREE_API_EXPORT iree_status_t iree_net_rdma_receive_queue_replenish(
    iree_net_rdma_receive_queue_t* queue, uint32_t target_posted_count,
    uint32_t* out_posted_count);

// Completes one receive after the central WR table dispatcher reports a RECV.
//
// |completion| must be the result returned by
// iree_net_rdma_work_request_table_complete() for the CQE wr_id. On success,
// |out_lease| receives the posted buffer lease with its span length narrowed to
// |byte_length|. The caller owns the lease and must release it after delivering
// the receive payload.
IREE_API_EXPORT iree_status_t iree_net_rdma_receive_queue_complete(
    iree_net_rdma_receive_queue_t* queue,
    iree_net_rdma_work_request_completion_t completion,
    iree_host_size_t byte_length, iree_async_buffer_lease_t* out_lease);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_RECEIVE_QUEUE_H_
