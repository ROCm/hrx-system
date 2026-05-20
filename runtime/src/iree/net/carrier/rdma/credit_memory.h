// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Registered receive-credit memory for RDMA connection flow control.
//
// Each carrier publishes one small host counter in connection data. The peer
// writes cumulative receive-credit grants into this counter with one-sided RDMA
// writes after it reposts receive WQEs. The local sender can then refresh its
// send budget with a cache-coherent load and without a client->server->client
// round trip.

#ifndef IREE_NET_CARRIER_RDMA_CREDIT_MEMORY_H_
#define IREE_NET_CARRIER_RDMA_CREDIT_MEMORY_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/net/carrier/rdma/connection_data.h"
#include "iree/net/carrier/rdma/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_credit_memory_t iree_net_rdma_credit_memory_t;

// Creates local credit memory registered for peer RDMA writes.
//
// |initial_credit_limit| is the cumulative receive-credit grant visible to the
// local sender until the peer writes a newer value. The memory retains
// |context| and deregisters itself during release.
IREE_API_EXPORT iree_status_t iree_net_rdma_credit_memory_create(
    iree_net_rdma_context_t* context, uint32_t initial_credit_limit,
    iree_allocator_t host_allocator,
    iree_net_rdma_credit_memory_t** out_memory);

// Releases credit memory and deregisters the underlying memory region.
IREE_API_EXPORT void iree_net_rdma_credit_memory_release(
    iree_net_rdma_credit_memory_t* memory);

// Returns the remote memory descriptor to advertise in connection data.
IREE_API_EXPORT iree_net_rdma_remote_credit_memory_t
iree_net_rdma_credit_memory_remote(const iree_net_rdma_credit_memory_t* memory);

// Loads the latest cumulative peer receive-credit grant.
IREE_API_EXPORT uint32_t
iree_net_rdma_credit_memory_load(const iree_net_rdma_credit_memory_t* memory);

// Stores the locally observed cumulative peer receive-credit grant.
//
// This is used after decoding peer connection data to seed the send budget
// before the peer can write newer values, and by tests to model coherent device
// writes.
IREE_API_EXPORT void iree_net_rdma_credit_memory_store(
    iree_net_rdma_credit_memory_t* memory, uint32_t credit_limit);

// Stores |credit_limit| and returns an SGE for sending it to a peer.
IREE_API_EXPORT iree_status_t iree_net_rdma_credit_memory_store_sge(
    iree_net_rdma_credit_memory_t* memory, uint32_t credit_limit,
    struct ibv_sge* out_sge);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_CREDIT_MEMORY_H_
