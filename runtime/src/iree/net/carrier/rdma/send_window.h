// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Local send-queue and remote receive-credit admission for RDMA work requests.
//
// RDMA send-side work requests consume local SQ capacity until completion.
// Two-sided SEND and RDMA WRITE WITH IMMEDIATE also consume one peer receive
// credit because they require a pre-posted remote RECV WQE. This component
// keeps those two budgets together so carriers make one admission decision
// before touching the native QP.

#ifndef IREE_NET_CARRIER_RDMA_SEND_WINDOW_H_
#define IREE_NET_CARRIER_RDMA_SEND_WINDOW_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint8_t iree_net_rdma_send_window_acquire_flags_t;
enum iree_net_rdma_send_window_acquire_flag_bits_e {
  IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_NONE = 0u,

  // The work request consumes one remote receive credit.
  IREE_NET_RDMA_SEND_WINDOW_ACQUIRE_FLAG_REMOTE_RECV_CREDIT = 1u << 0,
};

typedef struct iree_net_rdma_send_window_t {
  // Maximum local send WQEs that may be outstanding.
  uint32_t send_queue_depth;

  // Local send WQEs posted and not yet completed.
  uint32_t posted_send_count;

  // Cumulative peer receive-credit grant most recently observed.
  uint32_t remote_recv_credit_limit;

  // Cumulative peer receive credits consumed by posted work requests.
  uint32_t remote_recv_credit_used;
} iree_net_rdma_send_window_t;

// Initializes a send window with the peer's initial receive-credit grant.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_window_initialize(
    uint32_t send_queue_depth, uint32_t initial_remote_recv_credits,
    iree_net_rdma_send_window_t* out_window);

// Refreshes the cumulative peer receive-credit grant.
//
// Credit grants are sequence numbers. Values older than the current observed
// grant are ignored so a stale read cannot expand the send window after wrap.
IREE_API_EXPORT void iree_net_rdma_send_window_refresh_remote_credits(
    iree_net_rdma_send_window_t* window, uint32_t remote_recv_credit_limit);

// Returns the currently available local send WQE slots.
IREE_API_EXPORT uint32_t iree_net_rdma_send_window_available_send_slots(
    const iree_net_rdma_send_window_t* window);

// Returns the currently available peer receive credits.
IREE_API_EXPORT uint32_t iree_net_rdma_send_window_available_recv_credits(
    const iree_net_rdma_send_window_t* window);

// Returns the work requests currently admissible for |flags|.
IREE_API_EXPORT uint32_t iree_net_rdma_send_window_available(
    const iree_net_rdma_send_window_t* window,
    iree_net_rdma_send_window_acquire_flags_t flags);

// Acquires one send admission before posting to the native QP.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_window_acquire(
    iree_net_rdma_send_window_t* window,
    iree_net_rdma_send_window_acquire_flags_t flags);

// Aborts a previously acquired admission when native posting fails.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_window_abort(
    iree_net_rdma_send_window_t* window,
    iree_net_rdma_send_window_acquire_flags_t flags);

// Completes one send-side work request and releases its local SQ slot.
IREE_API_EXPORT iree_status_t
iree_net_rdma_send_window_complete(iree_net_rdma_send_window_t* window);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_SEND_WINDOW_H_
