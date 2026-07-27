// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA-ready send payload preparation.
//
// RDMA SEND requires every native scatter-gather entry to reference registered
// memory. This component preserves already-registered caller spans and copies
// only unregistered bytes into one transport-owned registered staging buffer.
// The resulting span list preserves wire order and remains valid while the
// caller keeps registered source spans alive and retains staging_buffer_lease.

#ifndef IREE_NET_CARRIER_RDMA_SEND_PAYLOAD_H_
#define IREE_NET_CARRIER_RDMA_SEND_PAYLOAD_H_

#include "iree/async/buffer_pool.h"
#include "iree/async/span.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/rdma/connection_data.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_send_payload_t {
  // Registered spans to submit as native scatter-gather entries.
  iree_async_span_t spans[IREE_NET_RDMA_CONNECTION_DATA_MAX_SEND_SGE];

  // Number of valid entries in |spans|.
  iree_host_size_t span_count;

  // Total number of bytes represented by |spans|.
  iree_host_size_t byte_length;

  // Optional registered staging storage containing copied source bytes.
  iree_async_buffer_lease_t staging_buffer_lease;
} iree_net_rdma_send_payload_t;

// Prepares |source_spans| for an RDMA SEND.
//
// RDMA-registered source spans are borrowed directly until send completion.
// Non-empty unregistered spans must be CPU-accessible and are copied into one
// buffer acquired from |staging_pool|. Adjacent unregistered spans are
// coalesced into one output span without changing their wire representation.
//
// IREE_NET_SEND_FLAG_ZERO_COPY rejects any non-empty unregistered span before
// acquiring a staging buffer. Other send flags do not affect preparation.
//
// On success |out_payload| owns staging_buffer_lease when staging was needed.
// Call iree_net_rdma_send_payload_deinitialize() unless ownership of that lease
// is transferred to a reservation or in-flight work request.
IREE_API_EXPORT iree_status_t iree_net_rdma_send_payload_prepare(
    iree_async_span_list_t source_spans, iree_net_send_flags_t send_flags,
    iree_async_buffer_pool_t* staging_pool,
    iree_net_rdma_send_payload_t* out_payload);

// Releases staging storage retained by |payload| and clears the representation.
IREE_API_EXPORT void iree_net_rdma_send_payload_deinitialize(
    iree_net_rdma_send_payload_t* payload);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_SEND_PAYLOAD_H_
